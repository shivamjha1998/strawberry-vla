#!/usr/bin/env python3
"""
Convert a PEFT (HuggingFace) LoRA adapter to MLX-VLM format.

PEFT stores adapters as:
  adapter_config.json  (has "r", "lora_alpha", "lora_dropout")
  adapter_model.safetensors  (keys like "base_model.model.model.language_model.layers.0.self_attn.q_proj.lora_A.weight")

MLX-VLM expects:
  adapter_config.json  (has "rank", "alpha", "dropout")
  adapters.safetensors (keys like "language_model.model.layers.0.self_attn.q_proj.A")

Weight matrices also need transposing:
  PEFT lora_A.weight: (rank, input_dim)  -> MLX A: (input_dim, rank)
  PEFT lora_B.weight: (output_dim, rank) -> MLX B: (rank, output_dim)

Usage:
    python convert_adapter_to_mlx.py \
        --peft_path output/v2-20260308-003328/checkpoint-861 \
        --output_path output/v2-20260308-003328/checkpoint-861-mlx
"""

import argparse
import json
import os

import numpy as np
from safetensors.numpy import load_file, save_file


def convert_key(peft_key: str) -> str:
    """Convert a PEFT weight key to MLX-VLM format.

    PEFT key hierarchy (Qwen3-VL via ms-swift):
        base_model.model  = PeftModel wrapper
        .model            = Qwen3VLForConditionalGeneration
        .language_model   = LLM backbone
        .layers.0...      = transformer layers

    MLX-VLM hierarchy:
        Model.language_model = LanguageModel
        .model               = Qwen3VLModel
        .layers.0...         = transformer layers

    Example:
        base_model.model.model.language_model.layers.0.self_attn.q_proj.lora_A.weight
        -> language_model.model.layers.0.self_attn.q_proj.A
    """
    k = peft_key
    # Strip PEFT wrapper prefix and swap HF→MLX model hierarchy
    # HF:  base_model.model.model.language_model.layers.N...
    # MLX: language_model.model.layers.N...
    k = k.replace("base_model.model.model.language_model.", "language_model.model.", 1)
    # Strip .weight suffix
    if k.endswith(".weight"):
        k = k[: -len(".weight")]
    # Rename lora matrices
    k = k.replace(".lora_A", ".A")
    k = k.replace(".lora_B", ".B")
    return k


def main():
    parser = argparse.ArgumentParser(description="Convert PEFT LoRA adapter to MLX-VLM format")
    parser.add_argument("--peft_path", required=True, help="Path to PEFT adapter directory")
    parser.add_argument("--output_path", default=None, help="Output directory (default: <peft_path>-mlx)")
    args = parser.parse_args()

    peft_path = args.peft_path
    output_path = args.output_path or (peft_path.rstrip("/") + "-mlx")
    os.makedirs(output_path, exist_ok=True)

    # ── 1. Convert config ────────────────────────────────────────
    config_file = os.path.join(peft_path, "adapter_config.json")
    with open(config_file) as f:
        peft_config = json.load(f)

    rank = peft_config["r"]
    lora_alpha = peft_config.get("lora_alpha", rank)
    lora_dropout = peft_config.get("lora_dropout", 0.0)

    # MLX-VLM uses alpha as a direct multiplier in: y + alpha * (x @ A @ B)
    # PEFT uses scaling = lora_alpha / r
    mlx_alpha = lora_alpha / rank

    mlx_config = {
        "rank": rank,
        "alpha": mlx_alpha,
        "dropout": lora_dropout,
    }

    config_out = os.path.join(output_path, "adapter_config.json")
    with open(config_out, "w") as f:
        json.dump(mlx_config, f, indent=2)
    print(f"Config saved: {config_out}")
    print(f"  rank={rank}, alpha={mlx_alpha}, dropout={lora_dropout}")

    # ── 2. Convert weights ───────────────────────────────────────
    weights_file = os.path.join(peft_path, "adapter_model.safetensors")
    peft_weights = load_file(weights_file)
    print(f"\nLoaded {len(peft_weights)} weight tensors from {weights_file}")

    mlx_weights = {}
    for peft_key, tensor in peft_weights.items():
        mlx_key = convert_key(peft_key)

        # Transpose: PEFT stores (out, in) for Linear, MLX uses (in, out) directly
        tensor = tensor.T

        mlx_weights[mlx_key] = tensor

    # Show a few sample key conversions
    sample_keys = sorted(peft_weights.keys())[:4]
    print("\nSample key conversions:")
    for pk in sample_keys:
        mk = convert_key(pk)
        orig_shape = peft_weights[pk].shape
        new_shape = mlx_weights[mk].shape
        print(f"  {pk}")
        print(f"    -> {mk}  ({orig_shape} -> {new_shape})")

    weights_out = os.path.join(output_path, "adapters.safetensors")
    save_file(mlx_weights, weights_out)
    print(f"\nWeights saved: {weights_out} ({len(mlx_weights)} tensors)")

    # ── 3. Summary ───────────────────────────────────────────────
    print(f"\nConversion complete!")
    print(f"  PEFT source:  {peft_path}")
    print(f"  MLX output:   {output_path}")
    print(f"\nTo use with mlx-vlm:")
    print(f"  from mlx_vlm import load")
    print(f'  model, processor = load("mlx-community/Qwen3-VL-8B-Instruct-4bit",')
    print(f'                          adapter_path="{output_path}")')


if __name__ == "__main__":
    main()