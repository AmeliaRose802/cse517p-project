import os
import torch
import json
import numpy as np
import torch.nn.functional as F
from .character_transformer_model import CharacterTransformer
from .character_dataset import CharDatasetWrapper, build_vocab, CharDataset
from torch.utils.data import DataLoader
from torch.optim.lr_scheduler import LambdaLR
import multiprocessing
from torch.amp import autocast, GradScaler 
import time
from tqdm import tqdm
from torch.optim.lr_scheduler import OneCycleLR


class TransformerModelWrapper:
    def __init__(self, device, work_directory):
        """
        Load in trained model and vocab
        """

        # The model's files will be saved and loaded from this directory
        self.work_directory = work_directory

        #input size must be 32 characters. If it is more or less it is padded or truncated
        self.context_length = 32
        self.device = device
        self.vocab_file_name = "char_to_index.json"
        self.model_file_name = "character_transformer.pt"

        self.vocab_file_path = os.path.join(work_directory, self.vocab_file_name)
        self.model_file_path = os.path.join(work_directory, self.model_file_name)
        #Create the checkpoints folder if it doesn't exist
        os.makedirs(os.path.join(self.work_directory, "checkpoints"), exist_ok=True)

        self.model_checkpoint_path = f"{work_directory}/checkpoints/{self.model_file_name}"

        self.index_to_char = None
        self.char_to_index = None

    def load(self):
        
        # Load vocab which was saved before
        with open(self.vocab_file_path, "r", encoding="utf-8") as f:
            self.char_to_index = json.load(f)
        
        self.index_to_char = {i: ch for ch, i in self.char_to_index.items()}
        max_len = max(len(v) for v in self.index_to_char.values())

        self.index_to_char = np.array(list(self.index_to_char.values()), dtype=f'U{max_len}')

        vocab_size = len(self.char_to_index)
        
        #load the model
        self.model = CharacterTransformer(vocab_size).to(self.device)
        self.model.load_state_dict(torch.load(self.model_file_path, map_location=self.device))

        non_compiled_model = self.model
        
        if hasattr(torch, 'compile'):
            self.model = torch.compile(self.model)

        return non_compiled_model
       
    
    def embed_strings(self, inputs: list[str]):
        pad_token = self.char_to_index[' ']
        context_len = self.context_length
        vocab = self.char_to_index

        encoded = np.full((len(inputs), context_len), pad_token, dtype=np.int32)

        for i, s in enumerate(inputs):
            trimmed = s[-context_len:]
            indices = [vocab.get(c, pad_token) for c in trimmed]

            if len(indices) < context_len:
                indices = [pad_token] * (context_len - len(indices)) + indices

            encoded[i] = indices

        # Convert once to torch tensor and move to GPU
        return torch.from_numpy(encoded).to(self.device)

    def predict(self, input: list[str]):
        self.model.eval().half()
        input_tensor = self.embed_strings(input)

        space_token_id = self.char_to_index[' ']

        with torch.no_grad():
            logits = self.model(input_tensor)
            logits[:, space_token_id] = float('-inf')
            top3 = torch.topk(logits, k=3, dim=1).indices.cpu().tolist()

            res = ["".join(self.index_to_char[j] for j in row) for row in top3]
            
        return res
    
    from torch.utils.data import DataLoader
    from torch.optim.lr_scheduler import OneCycleLR

    def train(self, data_directory, continue_training: bool = True, dataset_fraction: float = 1.0, num_epochs: int = 3, lr: float = 1e-4, batch_size=1048):

        dataset = CharDatasetWrapper(self.device, data_directory, self.context_length, dataset_fraction)
        
        with open(self.vocab_file_path, "w", encoding="utf-8") as f:
            json.dump(dataset.vocab(), f, ensure_ascii=False, indent=2)

        # Ensure checkpoints directory exists
        os.makedirs(os.path.join(self.work_directory, "checkpoints"), exist_ok=True)

        # Prepare datasets
        num_workers = min(20, multiprocessing.cpu_count() - 2)
        train_loader = DataLoader(dataset.train_dataset(), batch_size=batch_size, shuffle=True, pin_memory=True, num_workers=num_workers,  prefetch_factor=2, persistent_workers=True)
        dev_loader = DataLoader(dataset.dev_dataset(), batch_size=batch_size, shuffle=True, pin_memory=True, num_workers=num_workers,  prefetch_factor=2, persistent_workers=True)

        original_model = None
        
        #If this flag is true, then load in and continue to train an existing model instead of creating a new one from scratch
        if continue_training:
            original_model = self.load()
        else:
            self.model = CharacterTransformer(dataset.vocab_size()).to(self.device)
            
            original_model = self.model
            
            if hasattr(torch, 'compile'):
                self.model = torch.compile(self.model)
            

        self.optimizer = torch.optim.AdamW(self.model.parameters(), lr=lr)
        self.loss_fn = torch.nn.CrossEntropyLoss()
        
        scaler = GradScaler()

        self.optimizer = torch.optim.AdamW(self.model.parameters(), lr=lr)
        self.loss_fn = torch.nn.CrossEntropyLoss()
        scheduler = OneCycleLR(
            self.optimizer,
            max_lr=lr,
            steps_per_epoch=len(train_loader),
            epochs=num_epochs,
            pct_start=0.1,
            anneal_strategy="cos",
            final_div_factor=1e4,
        )

        for epoch in range(num_epochs):
            self.model.train()
            total_train_loss = 0

            # Add progress bar
            for x_batch, y_batch in tqdm(train_loader, desc=f"Epoch {epoch+1}/{num_epochs}"):
                x_batch = x_batch.to(self.device)
                y_batch = y_batch.to(self.device)
                self.optimizer.zero_grad()

                # AMP forward
                with autocast(device_type=self.device.type):
                    logits = self.model(x_batch)
                    loss = self.loss_fn(logits, y_batch)

                # AMP backward
                scaler.scale(loss).backward()
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
                scaler.step(self.optimizer)
                scaler.update()
                scheduler.step()

                scaler.unscale_(self.optimizer)

                total_train_loss += loss.item()
                
            avg_train_loss = total_train_loss / len(train_loader)

            dev_loss = self.eval_loss(dev_loader)

            print(f"[train] Epoch {epoch + 1}/{num_epochs} - Train Loss: {avg_train_loss:.4f}. Dev Loss: {dev_loss:.4f}")

            torch.save(self.model.state_dict(), f"{self.model_checkpoint_path}.{epoch}")

        if hasattr(torch, 'compile'):
            # Need to save the Og model not the compiled version to avoid problems
            torch.save(original_model.state_dict(), self.model_file_path)
        else:
            torch.save(self.model.state_dict(), self.model_file_path)
        print(f"[train] Model saved to {self.model_file_path}")
    
    def eval_loss(self, dataloader: DataLoader):
        self.model.eval()
        total_dev_loss = 0
        with torch.no_grad():
            for x_batch, y_batch in dataloader:
                x_batch, y_batch = x_batch.to(self.device), y_batch.to(self.device)
                logits = self.model(x_batch)
                loss = self.loss_fn(logits, y_batch)
                total_dev_loss += loss.item()
                
        avg_dev_loss = total_dev_loss / len(dataloader)

        return avg_dev_loss
    
    def test_train(self, data_directory, model, continue_training: bool = True, dataset_fraction: float = 1.0, num_epochs: int = 3, lr: float = 1e-4, batch_size=1048, verbose = False):

        dataset = CharDatasetWrapper(self.device, data_directory, self.context_length, dataset_fraction)
        
        with open(self.vocab_file_path, "w", encoding="utf-8") as f:
            json.dump(dataset.vocab(), f, ensure_ascii=False, indent=2)

        # Ensure checkpoints directory exists
        os.makedirs(os.path.join(self.work_directory, "checkpoints"), exist_ok=True)

        # Prepare datasets
        num_workers = min(20, multiprocessing.cpu_count() - 2)
        train_loader = DataLoader(dataset.train_dataset(), batch_size=batch_size, shuffle=True, pin_memory=True, num_workers=num_workers,  prefetch_factor=2, persistent_workers=True)
        dev_loader = DataLoader(dataset.dev_dataset(), batch_size=batch_size, shuffle=True, pin_memory=True, num_workers=num_workers,  prefetch_factor=2, persistent_workers=True)

        self.model = model
            
        original_model = self.model
            
        # if hasattr(torch, 'compile'):
        #     self.model = torch.compile(self.model)
        
        self.optimizer = torch.optim.AdamW(self.model.parameters(), lr=lr)
        self.loss_fn = torch.nn.CrossEntropyLoss()
        
        scaler = GradScaler()

        self.optimizer = torch.optim.AdamW(self.model.parameters(), lr=lr)
        self.loss_fn = torch.nn.CrossEntropyLoss()
        scheduler = OneCycleLR(
            self.optimizer,
            max_lr=lr,
            steps_per_epoch=len(train_loader),
            epochs=num_epochs,
            pct_start=0.1,
            anneal_strategy="cos",
            final_div_factor=1e4,
        )

        for epoch in range(num_epochs):
            self.model.train()
            total_train_loss = 0

            # Add progress bar
            for x_batch, y_batch in tqdm(train_loader, desc=f"Epoch {epoch+1}/{num_epochs}"):
                x_batch = x_batch.to(self.device)
                y_batch = y_batch.to(self.device)
                self.optimizer.zero_grad()

                # AMP forward
                with autocast(device_type=self.device.type):
                    logits = self.model(x_batch)
                    loss = self.loss_fn(logits, y_batch)

                # AMP backward
                scaler.scale(loss).backward()
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
                scaler.step(self.optimizer)
                scaler.update()
                scheduler.step()

                scaler.unscale_(self.optimizer)

                total_train_loss += loss.item()
                
            avg_train_loss = total_train_loss / len(train_loader)

            

            print(f"[train] Epoch {epoch + 1}/{num_epochs} - Train Loss: {avg_train_loss:.4f}.")

            if verbose:
                dev_loss = self.eval_loss(dev_loader)
                print(f"Dev loss: {dev_loss}")

            torch.save(self.model.state_dict(), f"{self.model_checkpoint_path}.{epoch}")

        if hasattr(torch, 'compile'):
            # Need to save the Og model not the compiled version to avoid problems
            torch.save(original_model.state_dict(), self.model_file_path)
        else:
            torch.save(self.model.state_dict(), self.model_file_path)
        print(f"[train] Model saved to {self.model_file_path}")

        if not verbose:
            dev_loss = self.eval_loss(dev_loader)

        return dev_loss