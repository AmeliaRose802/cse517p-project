#!/usr/bin/env python
import os
import time
from argparse import ArgumentParser, ArgumentDefaultsHelpFormatter
from Transformer_Based.transformer_wrapper import TransformerModelWrapper
import torch

if __name__ == '__main__':
    parser = ArgumentParser(formatter_class=ArgumentDefaultsHelpFormatter)
    parser.add_argument('--work_dir', help='where model is stored', default='work')
    parser.add_argument('--export_torchscript', help='Export model to TorchScript format', action='store_true')
    parser.add_argument('--export_quantized', help='Export quantized model', action='store_true')
    parser.add_argument('--time', help='Measure export time', action='store_true')
    args = parser.parse_args()

    if not args.export_torchscript and not args.export_quantized:
        print("Please specify at least one export format: --export_torchscript or --export_quantized")
        exit(1)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    start_time = time.time() if args.time else 0
    
    print(f"Loading model from {args.work_dir}...")
    model_wrapper = TransformerModelWrapper(device, args.work_dir)
    model_wrapper.load()
    
    if args.time:
        load_time = time.time() - start_time
        print(f"Model loaded in {load_time:.2f}s")
    
    model = model_wrapper.model
    model.eval()  # Set to evaluation mode
    
    # Export to TorchScript
    if args.export_torchscript:
        script_start = time.time() if args.time else 0
        print("Exporting to TorchScript format...")
        
        torchscript_path = os.path.join(args.work_dir, "character_transformer_script.pt")
        traced_script_module = torch.jit.script(model)
        traced_script_module.save(torchscript_path)
        
        if args.time:
            script_time = time.time() - script_start
            print(f"TorchScript export completed in {script_time:.2f}s")
        print(f"TorchScript model saved to {torchscript_path}")
    
    # Export quantized model
    if args.export_quantized:
        quant_start = time.time() if args.time else 0
        print("Exporting quantized model...")
        
        quantized_path = os.path.join(args.work_dir, "character_transformer_quantized.pt")
        quantized_model = torch.quantization.quantize_dynamic(
            model, {torch.nn.Linear}, dtype=torch.qint8
        )
        torch.save(quantized_model, quantized_path)
        
        if args.time:
            quant_time = time.time() - quant_start
            print(f"Quantization completed in {quant_time:.2f}s")
        print(f"Quantized model saved to {quantized_path}")
    
    if args.time:
        total_time = time.time() - start_time
        print(f"Total export time: {total_time:.2f}s")
    
    print("Export completed successfully.")
