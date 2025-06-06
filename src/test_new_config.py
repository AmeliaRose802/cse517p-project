import os
import torch
import json
import pandas as pd
import time
import matplotlib.pyplot as plt
from tqdm import tqdm
from Transformer_Based.character_transformer_model import CharacterTransformer
from Transformer_Based.transformer_wrapper import TransformerModelWrapper
import copy
from torch.utils.data import DataLoader
import Transformer_Based.character_transformer_model as model_module
from Transformer_Based.character_dataset import CharDatasetWrapper

if __name__ == '__main__':
    # Configuration
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    work_directory = "model_experiments"
    data_directory = "./data/parsed_data"  # Update this to your data directory
    results_file = os.path.join(work_directory, "experiment_results.csv")

    # Create work directory if it doesn't exist
    os.makedirs(work_directory, exist_ok=True)

    # Define hyperparameter combinations to test
    experiments = [
        # Baseline model
        # {"embedding_dim": 128, "num_heads": 8, "num_layers": 8, "ff_dim": 512, "dropout": 0.1, "desc": "baseline"},
        
        # # # Vary embedding dimensions
        # {"embedding_dim": 256, "num_heads": 8, "num_layers": 8, "ff_dim": 512, "dropout": 0.1, "desc": "larger_embedding"},
        
        {"embedding_dim": 128, "num_heads": 4, "num_layers": 2, "ff_dim": 512, "dropout": 0.1, "desc": "current_model"},
        {"embedding_dim": 32, "num_heads": 4, "num_layers": 2, "ff_dim": 512, "dropout": 0.1, "desc": "tiny_embeddings"},

        # # Vary transformer layers
        {"embedding_dim": 128, "num_heads": 4, "num_layers": 4, "ff_dim": 512, "dropout": 0.1, "desc": "fewer_layers"},
        {"embedding_dim": 128, "num_heads": 4, "num_layers": 12, "ff_dim": 512, "dropout": 0.1, "desc": "more_layers"},
        
        # Vary attention heads+
        {"embedding_dim": 128, "num_heads": 4, "num_layers": 2, "ff_dim": 512, "dropout": 0.1, "desc": "fewer_heads"},
        {"embedding_dim": 128, "num_heads": 16, "num_layers": 2, "ff_dim": 512, "dropout": 0.1, "desc": "more_heads"},
        
        # Vary feedforward network
        {"embedding_dim": 128, "num_heads": 4, "num_layers": 2, "ff_dim": 256, "dropout": 0.1, "desc": "smaller_ffn"},
        {"embedding_dim": 128, "num_heads": 4, "num_layers": 2, "ff_dim": 1024, "dropout": 0.1, "desc": "larger_ffn"},
        
        # Vary dropout
        {"embedding_dim": 128, "num_heads": 4, "num_layers": 2, "ff_dim": 512, "dropout": 0.2, "desc": "higher_dropout"},
        {"embedding_dim": 128, "num_heads": 4, "num_layers": 2, "ff_dim": 512, "dropout": 0.3, "desc": "highest_dropout"},
        
        # Small and large model combinations
        {"embedding_dim": 256, "num_heads": 12, "num_layers": 12, "ff_dim": 1024, "dropout": 0.2, "desc": "large_model"},
        {"embedding_dim": 64, "num_heads": 4, "num_layers": 4, "ff_dim": 256, "dropout": 0.1, "desc": "small_model"},
    ]

    def run_experiment(params, experiment_id):
        """Run a single experiment with the given hyperparameters"""
        
        # Create experiment directory
        exp_dir = os.path.join(work_directory, f"exp_{experiment_id}_{params['desc']}")
        os.makedirs(exp_dir, exist_ok=True)
        
        # Save experiment parameters
        with open(os.path.join(exp_dir, "params.json"), "w") as f:
            json.dump(params, f, indent=2)
    
        try:
        # Create wrapper and model
            wrapper = TransformerModelWrapper(device, exp_dir)
            
            print(f"\nTraining experiment {experiment_id}: {params['desc']}")
            start_time = time.time()
            
            dataset_wrapper = CharDatasetWrapper(device, data_directory, wrapper.context_length, 0.01)
            vocab_size = dataset_wrapper.vocab_size()
            
            # Create model with the specified parameters
            # Only use parameters that exist in the CharacterTransformer constructor
            model = CharacterTransformer(
                vocab_size,
                embedding_dim=params["embedding_dim"],
                num_heads=params["num_heads"],
                num_layers=params["num_layers"],
                ff_dim=params["ff_dim"],
                dropout=params["dropout"]
            ).to(device)
            
            # Train with test_train function
            dev_loss = wrapper.test_train(
                data_directory=data_directory,
                model=model,  # Pass the model explicitly
                dataset_fraction=.1,
                num_epochs=2)
            
            training_time = time.time() - start_time
            
            # Count model parameters
            model_params = sum(p.numel() for p in model.parameters())
            
            return {
                "experiment_id": experiment_id,
                "description": params["desc"],
                "embedding_dim": params["embedding_dim"],
                "num_heads": params["num_heads"],
                "num_layers": params["num_layers"],
                "ff_dim": params["ff_dim"],
                "dropout": params["dropout"],
                "dev_loss": dev_loss,
                "training_time": training_time,
                "parameter_count": model_params
            }
    
        except Exception as e:
            print(f"Error in experiment {experiment_id}: {str(e)}")
            return None
        

    # Run all experiments
    results = []
    for i, params in enumerate(experiments):
        print(f"\n=== Running experiment {i+1}/{len(experiments)}: {params['desc']} ===")
        result = run_experiment(params, i+1)
        
        if result:
            results.append(result)
            # Save intermediate results
            df = pd.DataFrame(results)
            df.to_csv(results_file, index=False)
            print(f"Completed experiment {i+1}: {params['desc']} - Dev Loss: {result['dev_loss']:.4f}")

    # Display and visualize final results
    if results:
        df = pd.DataFrame(results)
        df = df.sort_values("dev_loss")
        print("\n=== Final Results (Sorted by Dev Loss) ===")
        print(df[["experiment_id", "description", "dev_loss", "parameter_count", "training_time"]])
        
        # Create visualizations
        plt.figure(figsize=(15, 10))
        
        # Dev loss vs parameter count
        plt.subplot(2, 2, 1)
        plt.scatter(df["parameter_count"], df["dev_loss"])
        for i, row in df.iterrows():
            plt.annotate(row["description"], (row["parameter_count"], row["dev_loss"]))
        plt.xlabel("Parameter Count")
        plt.ylabel("Dev Loss")
        plt.title("Dev Loss vs Model Size")
        
        # Dev loss vs training time
        plt.subplot(2, 2, 2)
        plt.scatter(df["training_time"], df["dev_loss"])
        for i, row in df.iterrows():
            plt.annotate(row["description"], (row["training_time"], row["dev_loss"]))
        plt.xlabel("Training Time (s)")
        plt.ylabel("Dev Loss")
        plt.title("Dev Loss vs Training Time")
        
        # Bar chart of model architectures
        plt.subplot(2, 1, 2)
        sorted_df = df.sort_values("dev_loss")
        plt.bar(sorted_df["description"], sorted_df["dev_loss"])
        plt.xticks(rotation=45, ha="right")
        plt.ylabel("Dev Loss")
        plt.title("Model Architecture Comparison")
        
        plt.tight_layout()
        plt.savefig(os.path.join(work_directory, "experiment_results.png"))
        
        print(f"\nExperiments complete. Results saved to {results_file}")
        print(f"Visualizations saved to {os.path.join(work_directory, 'experiment_results.png')}")
    else:
        print("No successful experiments to report.")