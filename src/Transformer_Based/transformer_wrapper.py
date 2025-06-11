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

# Global model cache for persistent model across runs
_GLOBAL_MODEL_CACHE = {}

class TransformerModelWrapper:
    def __init__(self, device, work_directory, context_length = 32, use_existing_vocab=True, 
                 character_set="all", enable_timing=False, use_torchscript=False, 
                 use_quantization=False, use_mmap=True, reuse_model=False):
        """
        Load in trained model and vocab
        
        Args:
            device: The device to run the model on
            work_directory: Directory for model files
            context_length: Context length for the model
            use_existing_vocab: Whether to use an existing vocabulary if available
            character_set: Character set(s) to use - can be a single set name, a list of set names, or "all"
            enable_timing: Whether to enable detailed timing instrumentation
            use_torchscript: Whether to use TorchScript for faster loading
            use_quantization: Whether to use quantized model for reduced memory footprint
            use_mmap: Whether to use memory mapping for faster loading
            reuse_model: Whether to reuse cached model
        """
        self.enable_timing = enable_timing
        self.timing_stats = {}
        self._reset_timing_stats()
        self.use_torchscript = use_torchscript
        self.use_quantization = use_quantization
        self.use_mmap = use_mmap
        self.reuse_model = reuse_model
        
        init_start = time.time() if self.enable_timing else 0
        
        # The model's files will be saved and loaded from this directory
        self.work_directory = work_directory
        self.character_set = character_set
        self.context_length = context_length
        self.device = device
        
        # Model file paths
        self.model_file_name = "character_transformer.pt"
        self.model_file_path = os.path.join(work_directory, self.model_file_name)
        self.torchscript_model_path = os.path.join(work_directory, "character_transformer_script.pt")
        self.quantized_model_path = os.path.join(work_directory, "character_transformer_quantized.pt")
        
        # Set up vocab
        vocab_start = time.time() if self.enable_timing else 0
        vocab_file_path = os.path.join(work_directory, "char_to_index.json")

        # If the vocab file already exists then we should load it in
        if os.path.exists(vocab_file_path) and use_existing_vocab:
            with open(vocab_file_path, "r", encoding="utf-8") as f:
                self.vocab = json.load(f)
        else:
            self.vocab = init_vocab(vocab_file_path, character_set)
            
        if self.enable_timing:
            self.timing_stats['vocab_init_time'] = time.time() - vocab_start

        self.PAD_TOKEN: Final = self.vocab[' ']
        self.index_to_char = np.array(list(self.vocab.keys()))

        # Check if we can reuse an existing model from the cache
        model_cache_key = f"{work_directory}_{use_torchscript}_{use_quantization}"
        global _GLOBAL_MODEL_CACHE
        
        if self.reuse_model and model_cache_key in _GLOBAL_MODEL_CACHE:
            if self.enable_timing:
                print("Using cached model")
            self.model = _GLOBAL_MODEL_CACHE[model_cache_key]
        else:
            # Initialize the model architecture (but not weights)
            model_init_start = time.time() if self.enable_timing else 0
            self.model = CharacterTransformer(len(self.vocab)).to(self.device)
            if self.enable_timing:
                self.timing_stats['model_init_time'] = time.time() - model_init_start
        
        if self.enable_timing:
            self.timing_stats['total_init_time'] = time.time() - init_start
            print(f"Init timing: vocab={self.timing_stats['vocab_init_time']:.4f}s, "
                  f"model_init={self.timing_stats['model_init_time']:.4f}s, "
                  f"total={self.timing_stats['total_init_time']:.4f}s")
    
    def _reset_timing_stats(self):
        """Reset timing statistics"""
        self.timing_stats = {
            'vocab_init_time': 0,
            'model_init_time': 0,
            'total_init_time': 0,
            'load_time': 0,
            'embed_time': 0,
            'inference_time': 0,
            'postprocess_time': 0,
            'total_predict_time': 0
        }
    
    def load(self):
        """Load model weights with various optimized methods"""
        load_start = time.time() if self.enable_timing else 0
        
        # Check if we already have a cached model with weights loaded
        model_cache_key = f"{self.work_directory}_{self.use_torchscript}_{self.use_quantization}"
        global _GLOBAL_MODEL_CACHE
        
        if self.reuse_model and model_cache_key in _GLOBAL_MODEL_CACHE:
            # Model is already loaded from cache in __init__
            if self.enable_timing:
                self.timing_stats['load_time'] = time.time() - load_start
                print(f"Model reused from cache: {self.timing_stats['load_time']:.4f}s")
            return
        
        # Load based on the selected method
        if self.use_torchscript and os.path.exists(self.torchscript_model_path):
            # TorchScript model loading
            if self.enable_timing:
                print(f"Loading TorchScript model from {self.torchscript_model_path}")
            self.model = torch.jit.load(self.torchscript_model_path, map_location=self.device)
        
        elif self.use_quantization and os.path.exists(self.quantized_model_path):
            # Quantized model loading
            if self.enable_timing:
                print(f"Loading quantized model from {self.quantized_model_path}")
            self.model = torch.load(self.quantized_model_path, map_location=self.device)
        
        else:
            # Standard PyTorch model loading with optional memory mapping
            if self.use_mmap and hasattr(torch, 'load') and 'mmap' in torch.load.__code__.co_varnames:
                # Use memory mapping for faster loading if available
                if self.enable_timing:
                    print(f"Loading model with memory mapping from {self.model_file_path}")
                self.model.load_state_dict(torch.load(self.model_file_path, map_location=self.device, mmap=True))
            else:
                # Fallback to standard loading
                if self.enable_timing:
                    print(f"Loading model from {self.model_file_path}")
                self.model.load_state_dict(torch.load(self.model_file_path, map_location=self.device))
                
            # Optionally export to TorchScript for future use
            if self.use_torchscript and not os.path.exists(self.torchscript_model_path):
                if self.enable_timing:
                    script_start = time.time()
                    print("Exporting model to TorchScript format...")
                
                self.model.eval()
                traced_script_module = torch.jit.script(self.model)
                traced_script_module.save(self.torchscript_model_path)
                
                if self.enable_timing:
                    print(f"TorchScript export took {time.time() - script_start:.4f}s")
            
            # Optionally quantize the model for future use
            if self.use_quantization and not os.path.exists(self.quantized_model_path):
                if self.enable_timing:
                    quant_start = time.time()
                    print("Quantizing model...")
                
                self.model.eval()
                quantized_model = torch.quantization.quantize_dynamic(
                    self.model, {torch.nn.Linear}, dtype=torch.qint8
                )
                torch.save(quantized_model, self.quantized_model_path)
                
                if self.enable_timing:
                    print(f"Model quantization took {time.time() - quant_start:.4f}s")
        
        # Cache the model for future use
        if self.reuse_model:
            _GLOBAL_MODEL_CACHE[model_cache_key] = self.model
        
        if self.enable_timing:
            self.timing_stats['load_time'] = time.time() - load_start
            print(f"Model load time: {self.timing_stats['load_time']:.4f}s")
    
    def embed_strings(self, inputs: list[str]):
        embed_start = time.time() if self.enable_timing else 0
        encoded = np.full((len(inputs), self.context_length), 0, dtype=np.int32)

        for i, s in enumerate(inputs):
            indices = [self.vocab.get(c, self.PAD_TOKEN) for c in s[-self.context_length:]]

            if len(indices) < self.context_length:
                indices = [self.PAD_TOKEN] * (self.context_length - len(indices)) + indices

            encoded[i] = indices

        # Convert once to torch tensor and move to GPU
        result = torch.from_numpy(encoded).to(self.device)
        
        if self.enable_timing:
            self.timing_stats['embed_time'] = time.time() - embed_start
            
        return result

    def predict(self, input: list[str]):
        predict_start = time.time() if self.enable_timing else 0
        self.model.eval()
        
        # Step 1: Embedding
        embed_start = time.time() if self.enable_timing else 0
        input_tensor = self.embed_strings(input)
        if self.enable_timing:
            embed_time = time.time() - embed_start
            self.timing_stats['embed_time'] = embed_time
    
        with torch.no_grad():
            with torch.autocast(device_type=self.device.type, dtype=torch.float16):
                # Step 2: Model inference
                inference_start = time.time() if self.enable_timing else 0
                logits = self.model(input_tensor)
                if self.enable_timing:
                    inference_time = time.time() - inference_start
                    self.timing_stats['inference_time'] = inference_time
                
                # Step 3: Post-processing
                postprocess_start = time.time() if self.enable_timing else 0
                logits[:, self.PAD_TOKEN] = float('-inf')
                top3 = torch.topk(logits, k=3, dim=1).indices.cpu().tolist()
                res = ["".join(self.index_to_char[j] for j in row) for row in top3]
                if self.enable_timing:
                    postprocess_time = time.time() - postprocess_start
                    self.timing_stats['postprocess_time'] = postprocess_time
        
        if self.enable_timing:
            total_time = time.time() - predict_start
            self.timing_stats['total_predict_time'] = total_time
            examples_per_second = len(input) / total_time
            
            print(f"Prediction breakdown for {len(input)} examples:")
            print(f"  Embedding: {embed_time:.4f}s ({embed_time/total_time*100:.1f}%)")
            print(f"  Inference: {inference_time:.4f}s ({inference_time/total_time*100:.1f}%)")
            print(f"  Postprocess: {postprocess_time:.4f}s ({postprocess_time/total_time*100:.1f}%)")
            print(f"  Total: {total_time:.4f}s ({examples_per_second:.1f} examples/s)")
        
        return res
    
    def train(self, dataset, num_epochs: int = 3, lr: float = 1e-4, batch_size=49152, verbose=True, save_checkpoints=True, 
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

        # Prepare datasets with optimized settings for H200
        num_workers = min(128, multiprocessing.cpu_count())  # Increased for H200
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