import os
import torch
import json
import numpy as np
import torch.nn.functional as F
from .character_transformer_model import CharacterTransformer
from .vocab import init_vocab
from .character_dataset import CharDatasetWrapper
from torch.utils.data import DataLoader
import multiprocessing
from torch.amp import autocast, GradScaler 
from tqdm import tqdm
from torch.optim.lr_scheduler import OneCycleLR
from typing import Final
import time
from torch.utils.data import DataLoader
from torch.optim.lr_scheduler import OneCycleLR


class TransformerModelWrapper:
    def __init__(self, device, work_directory, context_length = 32, use_existing_vocab=True, character_set="all", enable_timing=True):
        """
        Load in trained model and vocab
        
        Args:
            device: The device to run the model on
            work_directory: Directory for model files
            context_length: Context length for the model
            use_existing_vocab: Whether to use an existing vocabulary if available
            character_set: Character set(s) to use - can be a single set name, a list of set names, or "all"
                          Valid options: "latin", "romance", "germanic", "cyrillic", "cjk", "devanagari", "arabic", "all"
            enable_timing: Whether to enable detailed timing measurements
        """
        # Start overall initialization timing
        init_start_time = time.time()
        
        # The model's files will be saved and loaded from this directory
        self.work_directory = work_directory
        self.character_set = character_set
        self.context_length = context_length
        self.enable_timing = enable_timing
        self.timing_stats = {}  # Store timing statistics

        self.device = device
        self.model_file_name = "character_transformer.pt"
        self.model_file_path = os.path.join(work_directory, self.model_file_name)
        
        # Set up vocab
        vocab_start_time = time.time()
        vocab_file_path = os.path.join(work_directory, "vocab_chars.csv")

        # If the vocab file already exists then we should load it in
        if os.path.exists(vocab_file_path) and use_existing_vocab:
            with open(vocab_file_path, "r", encoding="utf-8") as f:
                # Read characters from CSV file and build vocabulary dict
                # Use rstrip instead of strip to preserve leading spaces
                chars = []
                for line in f:
                    char = line.rstrip('\n\r')
                    # Special handling for the space character
                    if char == '' and len(chars) == 0:  # First line is empty
                        char = ' '  # This is actually a space character
                    chars.append(char)
                
                self.vocab = {char: idx for idx, char in enumerate(chars)}
                
                # Double-check the space character is properly included
                if ' ' not in self.vocab:
                    print(f"Warning: Space character not found in vocabulary after loading. Adding it manually.")
                    self.vocab[' '] = 0  # Force space to be at index 0
        else:
            # Fall back to original method if CSV doesn't exist
            #TODO: No longer have it write out in wrong format
            json_vocab_path = os.path.join(work_directory, "char_to_index.json")
            self.vocab = init_vocab(json_vocab_path, character_set)
        
        # Debug output to verify vocabulary loading
        print(f"Vocabulary size: {len(self.vocab)}")
        print(f"Space character in vocab: {' ' in self.vocab}, index: {self.vocab.get(' ', -1)}")
        
        vocab_end_time = time.time()

        # Ensure we have a space character for padding
        if ' ' not in self.vocab:
            self.vocab[' '] = len(self.vocab)  # Add space at the end if not present
        self.PAD_TOKEN: Final = self.vocab[' ']

        # Setup index to char mapping
        index_char_start_time = time.time()
        self.index_to_char = np.array(list(self.vocab.keys()))
        index_char_end_time = time.time()

        # Set up the model
        model_init_start_time = time.time()
        self.model = CharacterTransformer(len(self.vocab)).to(self.device)
        model_init_end_time = time.time()
        
        # Store timing information
        if self.enable_timing:
            self.timing_stats["init"] = {
                "total": time.time() - init_start_time,
                "vocab_setup": vocab_end_time - vocab_start_time,
                "index_to_char_setup": index_char_end_time - index_char_start_time,
                "model_init": model_init_end_time - model_init_start_time
            }
            print(f"[TIMING] Initialization breakdown:")
            print(f"  - Total init time: {self.timing_stats['init']['total']:.4f}s")
            print(f"  - Vocab setup: {self.timing_stats['init']['vocab_setup']:.4f}s")
            print(f"  - Index to char mapping: {self.timing_stats['init']['index_to_char_setup']:.4f}s")
            print(f"  - Model initialization: {self.timing_stats['init']['model_init']:.4f}s")
        
    def get_vocab_size(self):
        return len(self.vocab)
    
    def load(self):
        if self.enable_timing:
            load_start_time = time.time()
            
        self.model.load_state_dict(torch.load(self.model_file_path, map_location=self.device))
        
        if self.enable_timing:
            load_time = time.time() - load_start_time
            self.timing_stats["model_load"] = load_time
            print(f"[TIMING] Model loading took {load_time:.4f}s")
       
    def embed_strings(self, inputs: list[str]):
        if self.enable_timing:
            embed_start_time = time.time()
            encode_start_time = time.time()

        encoded = np.full((len(inputs), self.context_length), 0, dtype=np.int32)

        for i, s in enumerate(inputs):
            indices = [self.vocab.get(c, self.PAD_TOKEN) for c in s[-self.context_length:]]

            if len(indices) < self.context_length:
                indices = [self.PAD_TOKEN] * (self.context_length - len(indices)) + indices

            encoded[i] = indices

        if self.enable_timing:
            encode_end_time = time.time()
            to_tensor_start_time = time.time()
            
        # Convert once to torch tensor and move to GPU
        tensor_result = torch.from_numpy(encoded).to(self.device)
        
        if self.enable_timing:
            to_tensor_end_time = time.time()
            embed_end_time = time.time()
            
            self.timing_stats["embed_strings"] = {
                "total": embed_end_time - embed_start_time,
                "encoding": encode_end_time - encode_start_time,
                "to_tensor_and_device": to_tensor_end_time - to_tensor_start_time
            }
            print(f"[TIMING] String embedding breakdown:")
            print(f"  - Total embedding time: {self.timing_stats['embed_strings']['total']:.4f}s")
            print(f"  - String encoding: {self.timing_stats['embed_strings']['encoding']:.4f}s")
            print(f"  - Tensor conversion and device transfer: {self.timing_stats['embed_strings']['to_tensor_and_device']:.4f}s")
            
        return tensor_result

    def predict(self, input: list[str]):
        if self.enable_timing:
            predict_start_time = time.time()
            
        self.model.eval()
        
        if self.enable_timing:
            embed_start = time.time()
            
        input_tensor = self.embed_strings(input)
        
        if self.enable_timing:
            embed_end = time.time()
            inference_start = time.time()
    
        with torch.no_grad():
            with torch.autocast(device_type=self.device.type, dtype=torch.float16):
                logits = self.model(input_tensor)

                logits[:, self.PAD_TOKEN] = float('-inf')
                top3 = torch.topk(logits, k=3, dim=1).indices.cpu().tolist()

                if self.enable_timing:
                    inference_end = time.time()
                    postprocess_start = time.time()
                    
                res = ["".join(self.index_to_char[j] for j in row) for row in top3]
                
                if self.enable_timing:
                    postprocess_end = time.time()
                    predict_end_time = time.time()
                    
                    self.timing_stats["predict"] = {
                        "total": predict_end_time - predict_start_time,
                        "embedding": embed_end - embed_start,
                        "inference": inference_end - inference_start,
                        "postprocessing": postprocess_end - postprocess_start
                    }
                    print(f"[TIMING] Prediction breakdown (batch size: {len(input)}):")
                    print(f"  - Total prediction time: {self.timing_stats['predict']['total']:.4f}s")
                    print(f"  - Embedding time: {self.timing_stats['predict']['embedding']:.4f}s")
                    print(f"  - Inference time: {self.timing_stats['predict']['inference']:.4f}s")
                    print(f"  - Postprocessing time: {self.timing_stats['predict']['postprocessing']:.4f}s")
        
        return res
    
    def train(self, dataset, num_epochs: int = 3, lr: float = 1e-4, batch_size=24576, verbose=True, save_checkpoints=True, 
              use_bf16=True, gradient_accumulation_steps=1):
        """
        Train the model on the given dataset.
        
        Args:
            dataset: A CharDatasetWrapper instance containing the training and validation datasets
            num_epochs: Number of training epochs
            lr: Learning rate
            batch_size: Batch size per device - increased for H200's 141GB memory (was 24576 for H100)
            verbose: Whether to print training progress
            save_checkpoints: Whether to save model checkpoints
            use_bf16: Whether to use bfloat16 precision (optimal for H200)
            gradient_accumulation_steps: Number of steps to accumulate gradients
        """
        # Enable TF32 precision on H200 for faster matrix multiplication
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True
        
        if hasattr(torch, 'compile'):
            self.compiled_model = torch.compile(self.model)
        
        #Create the checkpoints folder if it doesn't exist
        if save_checkpoints:
            os.makedirs(os.path.join(self.work_directory, "checkpoints"), exist_ok=True)
            self.model_checkpoint_path = f"{self.work_directory}/checkpoints/{self.model_file_name}"

        num_workers = min(16, multiprocessing.cpu_count()) 
        persistent_workers = num_workers > 0
        
        train_loader = DataLoader(
            dataset.train_dataset(), 
            batch_size=batch_size, 
            shuffle=True, 
            pin_memory=True, 
            num_workers=num_workers, 
            prefetch_factor=4, 
            persistent_workers=persistent_workers,
            pin_memory_device=str(self.device) if str(self.device) != 'cpu' else ''
        )
        
        dev_loader = DataLoader(
            dataset.dev_dataset(), 
            batch_size=batch_size, 
            shuffle=True, 
            pin_memory=True, 
            num_workers=num_workers, 
            prefetch_factor=4, 
            persistent_workers=persistent_workers,
            pin_memory_device=str(self.device) if str(self.device) != 'cpu' else ''
        )
        
        self.optimizer = torch.optim.AdamW(self.model.parameters(), lr=lr)
        self.loss_fn = torch.nn.CrossEntropyLoss()
        
        # Choose precision based on hardware capabilities
        amp_dtype = torch.bfloat16 if use_bf16 and torch.cuda.is_bf16_supported() else torch.float16
        scaler = GradScaler(enabled=amp_dtype == torch.float16)  # Only use scaler with fp16, not needed for bf16
        
        scheduler = OneCycleLR(
            self.optimizer,
            max_lr=lr,
            steps_per_epoch=len(train_loader) // gradient_accumulation_steps,
            epochs=num_epochs,
            pct_start=0.1,
            anneal_strategy="cos",
            final_div_factor=1e4,
        )

        # Training loop
        for epoch in range(num_epochs):
            self.model.train()
            total_train_loss = 0
            train_start_time = time.time()

            # Track accumulated loss for gradient accumulation
            accumulated_loss = 0

            # Add progress bar
            with autocast(device_type=self.device.type, dtype=amp_dtype):
                for step, (x_batch, y_batch) in enumerate(tqdm(train_loader, desc=f"Epoch {epoch+1}/{num_epochs}")):
                    x_batch = x_batch.to(self.device, non_blocking=True)
                    y_batch = y_batch.to(self.device, non_blocking=True)
                    
                    # Only zero gradients when accumulation is complete
                    if step % gradient_accumulation_steps == 0:
                        self.optimizer.zero_grad()

                    # AMP forward
                    
                        logits = self.compiled_model(x_batch)
                        loss = self.loss_fn(logits, y_batch) / gradient_accumulation_steps

                    # Track loss for reporting
                    if verbose:
                        total_train_loss += loss.item() * gradient_accumulation_steps
                    
                    accumulated_loss += loss.item()

                    # AMP backward
                    if amp_dtype == torch.float16:
                        scaler.scale(loss).backward()
                        
                        # Only update weights after accumulation steps
                        if (step + 1) % gradient_accumulation_steps == 0 or (step + 1) == len(train_loader):
                            scaler.unscale_(self.optimizer)
                            torch.nn.utils.clip_grad_norm_(self.compiled_model.parameters(), max_norm=1.0)
                            scaler.step(self.optimizer)
                            scaler.update()
                            scheduler.step()
                    else:
                        # With bf16, no need for scaler
                        loss.backward()
                        
                        # Only update weights after accumulation steps
                        if (step + 1) % gradient_accumulation_steps == 0 or (step + 1) == len(train_loader):
                            torch.nn.utils.clip_grad_norm_(self.compiled_model.parameters(), max_norm=1.0)
                            self.optimizer.step()
                            scheduler.step()

                if verbose:
                    train_time = time.time() - train_start_time
                    avg_train_loss = total_train_loss / len(train_loader)
                    
                    dev_start_time = time.time()
                    dev_loss = self.eval_loss(dev_loader, amp_dtype)
                    dev_time = time.time() - dev_start_time
                    
                    print(f"[train] Epoch {epoch + 1}/{num_epochs} - Train Loss: {avg_train_loss:.4f} (time: {train_time:.2f}s), "
                        f"Dev Loss: {dev_loss:.4f} (time: {dev_time:.2f}s)")

                if save_checkpoints:
                    torch.save(self.model.state_dict(), f"{self.model_checkpoint_path}.{epoch}")
            
        torch.save(self.model.state_dict(), self.model_file_path)
        print(f"[train] Model saved to {self.model_file_path}")
    
    def eval_loss(self, dataloader: DataLoader, amp_dtype=torch.float16):
        self.model.eval()
        total_dev_loss = 0
        with torch.no_grad():
            for x_batch, y_batch in dataloader:
                x_batch = x_batch.to(self.device, non_blocking=True)
                y_batch = y_batch.to(self.device, non_blocking=True)
                
                with autocast(device_type=self.device.type, dtype=amp_dtype):
                    logits = self.model(x_batch)
                    loss = self.loss_fn(logits, y_batch)
                
                total_dev_loss += loss.item()
                
        avg_dev_loss = total_dev_loss / len(dataloader)

        return avg_dev_loss