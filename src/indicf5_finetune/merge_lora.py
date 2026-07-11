"""
Utility to merge trained LoRA weights back into the base model weights.

Creates a standard F5-TTS checkpoint that can be loaded by the base codebase
without needing the LoRA classes or wrapping functions.
"""

import argparse
import os
import sys
import torch

def merge_dict(state_dict, scaling):
    merged_count = 0
    keys = list(state_dict.keys())
    for k in keys:
        if k.endswith(".lora_A"):
            prefix = k[:-len(".lora_A")]
            
            lora_A = state_dict[prefix + ".lora_A"]
            lora_B = state_dict[prefix + ".lora_B"]
            base_weight_key = prefix + ".original_linear.weight"
            base_bias_key = prefix + ".original_linear.bias"
            
            if base_weight_key not in state_dict:
                print(f"Warning: Could not find base weight key {base_weight_key} for {prefix}")
                continue
                
            # Compute delta: B @ A * scaling
            delta_w = lora_B @ lora_A * scaling
            
            # Add to base weight
            state_dict[base_weight_key] = state_dict[base_weight_key] + delta_w
            
            # Delete LoRA parameters
            del state_dict[prefix + ".lora_A"]
            del state_dict[prefix + ".lora_B"]
            
            # Rename base weight to standard key
            state_dict[prefix + ".weight"] = state_dict.pop(base_weight_key)
            if base_bias_key in state_dict:
                state_dict[prefix + ".bias"] = state_dict.pop(base_bias_key)
                
            merged_count += 1
            
    return merged_count

def main():
    parser = argparse.ArgumentParser(description="Merge LoRA weights back into base model weights")
    parser.add_argument("--checkpoint-path", required=True, help="Path to LoRA checkpoint")
    parser.add_argument("--output-path", required=True, help="Path to save merged checkpoint")
    parser.add_argument("--lora-rank", type=int, default=16, help="Rank of LoRA adapters")
    parser.add_argument("--lora-alpha", type=int, default=32, help="Alpha scaling parameter of LoRA adapters")
    args = parser.parse_args()

    if not os.path.exists(args.checkpoint_path):
        print(f"ERROR: Checkpoint not found at {args.checkpoint_path}")
        sys.exit(1)

    print(f"Loading checkpoint from {args.checkpoint_path}...")
    # Load on CPU to avoid using GPU memory
    checkpoint = torch.load(args.checkpoint_path, map_location="cpu")
    scaling = args.lora_alpha / args.lora_rank

    merged_total = 0

    # 1. Check in ema_model_state_dict
    if "ema_model_state_dict" in checkpoint:
        print("Merging in 'ema_model_state_dict'...")
        merged_total += merge_dict(checkpoint["ema_model_state_dict"], scaling)

    # 2. Check in model_state_dict
    if "model_state_dict" in checkpoint:
        print("Merging in 'model_state_dict'...")
        merged_total += merge_dict(checkpoint["model_state_dict"], scaling)

    # 3. Check at root level
    root_keys = [k for k in checkpoint.keys() if isinstance(k, str) and k.endswith(".lora_A")]
    if root_keys:
        print("Merging at root level...")
        merged_total += merge_dict(checkpoint, scaling)

    if merged_total == 0:
        print("WARNING: No LoRA weights found to merge. Checkpoint might already be merged.")
    else:
        print(f"Successfully merged {merged_total} LoRA projection layers!")

    # Update step in output path
    print(f"Saving merged checkpoint to {args.output_path}...")
    os.makedirs(os.path.dirname(os.path.abspath(args.output_path)), exist_ok=True)
    torch.save(checkpoint, args.output_path)
    print("SUCCESS: Merging complete!")

if __name__ == "__main__":
    main()
