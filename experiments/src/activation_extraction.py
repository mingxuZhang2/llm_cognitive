"""
Phase 1: Extract FFN intermediate activations from LLMs.

For each sample, records mean absolute activation per neuron across tokens,
then z-score normalizes across samples. Produces activation matrix A[n_neurons x n_samples].
"""

import os
import json
import torch
import numpy as np
from pathlib import Path
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer


def get_ffn_hook(layer_idx, storage, model_type="qwen"):
    """Create a forward hook that captures FFN intermediate activations."""
    def hook_fn(module, input, output):
        if isinstance(output, tuple):
            act = output[0]
        else:
            act = output
        mean_abs = act.abs().mean(dim=1)  # mean across tokens → [batch, ffn_dim]
        storage[layer_idx] = mean_abs.detach().cpu()
    return hook_fn


def get_ffn_modules(model):
    """Identify FFN intermediate modules across different architectures."""
    ffn_modules = []
    for i, layer in enumerate(model.model.layers):
        if hasattr(layer, 'mlp'):
            mlp = layer.mlp
            if hasattr(mlp, 'gate_proj'):
                ffn_modules.append((i, mlp.gate_proj))
            elif hasattr(mlp, 'fc1'):
                ffn_modules.append((i, mlp.fc1))
            elif hasattr(mlp, 'up_proj'):
                ffn_modules.append((i, mlp.up_proj))
            else:
                w_names = [n for n, _ in mlp.named_modules() if 'linear' in n.lower() or 'dense' in n.lower()]
                if w_names:
                    ffn_modules.append((i, getattr(mlp, w_names[0])))
    return ffn_modules


def extract_activations(
    model_name: str,
    samples: list[str],
    output_dir: str,
    max_length: int = 512,
    batch_size: int = 4,
    device: str = "cuda",
):
    """
    Extract FFN activations for all samples and save to disk.

    Args:
        model_name: HuggingFace model identifier
        samples: list of input texts
        output_dir: directory to save activation matrices
        max_length: max tokens per sample
        batch_size: inference batch size
        device: cuda or cpu
    """
    print(f"Loading model: {model_name}")
    tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        torch_dtype=torch.float16,
        device_map=device,
        trust_remote_code=True,
    )
    model.eval()

    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    ffn_modules = get_ffn_modules(model)
    n_layers = len(ffn_modules)
    ffn_dim = ffn_modules[0][1].out_features if hasattr(ffn_modules[0][1], 'out_features') else ffn_modules[0][1].weight.shape[0]

    print(f"Model has {n_layers} layers, FFN dim = {ffn_dim}")
    print(f"Total neurons: {n_layers * ffn_dim:,}")

    all_activations = []

    for batch_start in tqdm(range(0, len(samples), batch_size), desc="Extracting"):
        batch_texts = samples[batch_start:batch_start + batch_size]

        inputs = tokenizer(
            batch_texts,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=max_length,
        ).to(device)

        storage = {}
        hooks = []
        for layer_idx, module in ffn_modules:
            h = module.register_forward_hook(get_ffn_hook(layer_idx, storage))
            hooks.append(h)

        with torch.no_grad():
            model(**inputs)

        for h in hooks:
            h.remove()

        for b in range(len(batch_texts)):
            sample_act = []
            for layer_idx, _ in ffn_modules:
                act = storage[layer_idx][b]  # [ffn_dim]
                sample_act.append(act.numpy())
            all_activations.append(np.concatenate(sample_act))  # [n_layers * ffn_dim]

        del storage
        torch.cuda.empty_cache()

    A = np.stack(all_activations, axis=1)  # [n_neurons, n_samples]
    print(f"Activation matrix shape: {A.shape}")

    # Z-score normalization across samples (axis=1)
    mean = A.mean(axis=1, keepdims=True)
    std = A.std(axis=1, keepdims=True)
    std[std == 0] = 1.0
    A_norm = (A - mean) / std

    os.makedirs(output_dir, exist_ok=True)
    model_short = model_name.split("/")[-1]
    np.save(os.path.join(output_dir, f"{model_short}_activations.npy"), A_norm)

    meta = {
        "model": model_name,
        "n_layers": n_layers,
        "ffn_dim": ffn_dim,
        "n_neurons": A_norm.shape[0],
        "n_samples": A_norm.shape[1],
        "shape": list(A_norm.shape),
    }
    with open(os.path.join(output_dir, f"{model_short}_meta.json"), "w") as f:
        json.dump(meta, f, indent=2)

    print(f"Saved to {output_dir}/{model_short}_activations.npy")
    return A_norm, meta


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str, required=True)
    parser.add_argument("--stimuli", type=str, required=True, help="Path to stimuli jsonl")
    parser.add_argument("--output_dir", type=str, required=True)
    parser.add_argument("--max_length", type=int, default=512)
    parser.add_argument("--batch_size", type=int, default=4)
    args = parser.parse_args()

    with open(args.stimuli) as f:
        samples = [json.loads(line)["text"] for line in f]

    extract_activations(
        model_name=args.model,
        samples=samples,
        output_dir=args.output_dir,
        max_length=args.max_length,
        batch_size=args.batch_size,
    )
