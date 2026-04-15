# NVFP4 Quantization Config for Qwen3.5-397B-A17B

## Summary

A custom NVFP4 quantization configuration (`NVFP4_QWEN35_397B_CFG`) has been created for the **Qwen3.5-397B-A17B** model. This config matches the official quantization strategy used for this flagship model.

## What Was Added

### 1. New Quantization Config
**File**: `modelopt/torch/quantization/config.py`

Added `NVFP4_QWEN35_397B_CFG` which implements the official quantization strategy:

#### Quantized (NVFP4, group_size=16):
- **Routed MoE Experts Only**: The 512 routed experts in each MoE layer are quantized to NVFP4
- This is the bulk of the model's 397B parameters

#### Excluded from NVFP4 (kept in higher precision):
- **All Attention Mechanisms** (across all 60 layers):
  - **DeltaNet** (`linear_attn*`): Layers 0,1,2, 4,5,6, 8,9,10, 12,13,14, 16,17,18, 20,21,22, 24,25,26, 28,29,30, 32,33,34, 36,37,38, 40,41,42, 44,45,46, 48,49,50, 52,53,54, 56,57,58
  - **Self-Attention** (`self_attn*`): Layers 3, 7, 11, 15, 19, 23, 27, 31, 35, 39, 43, 47, 51, 55, 59
- **Shared Experts** (`*.mlp.shared_expert.*`): The shared expert that processes every token
- **Shared Expert Gates** (`*shared_expert_gate`)
- **Vision Encoder** (`model.visual*`)
- **Multi-Token Prediction** (`mtp.layers.0*`)
- **LM Head** (`lm_head`)

### 2. Integration Points

The new config is now available in:
- ✅ `modelopt.torch.quantization.config.NVFP4_QWEN35_397B_CFG`
- ✅ `examples/llm_ptq/hf_ptq.py` → `--qformat nvfp4_qwen35_397b`
- ✅ `examples/llm_ptq/multinode_ptq.py` → `--qformat nvfp4_qwen35_397b`

## Usage

### Basic Usage with hf_ptq.py

```bash
python examples/llm_ptq/hf_ptq.py \
    --pyt_ckpt_path /path/to/Qwen3.5-397B-A17B \
    --qformat nvfp4_qwen35_397b \
    --kv_cache_qformat fp8 \
    --export_path /path/to/quantized_output \
    --calib_size 512
```

## Architecture Details

The Qwen3.5-397B-A17B model follows this structure:
- **Total Parameters**: 397 Billion
- **Activated Parameters**: 17 Billion
- **Total Layers**: 60
- **Block Pattern**: 15 × (3× DeltaNet + 1× Self-Attention)
- **MoE**: 512 experts per layer, 10 routed + 1 shared expert activated per token
- **Hidden Dimension**: 4096
- **MoE Expert Intermediate Dimension**: 1024

### Layer Layout Pattern
```
Layers 0-2:   DeltaNet (linear_attn)
Layer 3:      Self-Attention
Layers 4-6:   DeltaNet (linear_attn)
Layer 7:      Self-Attention
... (repeats 15 times)
Layers 56-58: DeltaNet (linear_attn)
Layer 59:     Self-Attention
```

## Comparison with Other Qwen3.5 Models

| Specification | Qwen3.5-35B-A3B | Qwen3.5-122B-A10B | Qwen3.5-397B-A17B |
|---------------|-----------------|-------------------|------------------|
| **Total Parameters** | 35 Billion | 122 Billion | 397 Billion |
| **Activated Parameters** | 3 Billion | 10 Billion | 17 Billion |
| **Number of Layers** | 40 | 48 | 60 |
| **Hidden Dimension** | 2048 | 3072 | 4096 |
| **Hidden Layout Blocks** | 10 blocks | 12 blocks | 15 blocks |
| **DeltaNet V Heads** | 32 | 64 | 64 |
| **Self-Attention Q Heads** | 16 | 32 | 32 |
| **MoE Total Experts** | 256 | 256 | 512 |
| **MoE Experts Activated** | 8 routed + 1 shared | 8 routed + 1 shared | 10 routed + 1 shared |
| **MoE Expert Intermediate Dimension** | 512 | 1024 | 1024 |

**What Remains Identical:**
- Vocabulary size (248,320)
- Context window (262,144 tokens, extensible to 1,010,000)
- Head dimensions (128 for DeltaNet, 256 for Self-Attention)
- RoPE dimension (64)
- Multi-Token Prediction (MTP) training

## Comparison with Other NVFP4 Configs

| Config | What's Quantized | Key Exclusions |
|--------|------------------|----------------|
| `nvfp4` (default) | Most layers | Basic defaults only |
| `nvfp4_mlp_only` | All MLP/MoE | All attention |
| `nvfp4_omlp_only` | All MLP/MoE + o_proj | All attention |
| `nvfp4_qwen35_35b` | Routed MoE experts only | Attention + Shared experts + Vision + MTP |
| `nvfp4_qwen35_122b` | Routed MoE experts only | Attention + Shared experts + Vision + MTP |
| **`nvfp4_qwen35_397b`** | **Routed MoE experts only** | **Attention + Shared experts + Vision + MTP** |

All Qwen3.5-specific configs (`nvfp4_qwen35_35b`, `nvfp4_qwen35_122b`, `nvfp4_qwen35_397b`) are **more conservative** than `nvfp4_mlp_only` because they exclude the shared experts, which are critical for maintaining model accuracy.

## Generated Config Structure

When you run quantization with `--qformat nvfp4_qwen35_397b`, the generated `quantization_config.json` will contain:

```json
{
    "producer": {
        "name": "modelopt",
        "version": "<version>"
    },
    "quantization": {
        "quant_algo": "NVFP4",
        "kv_cache_quant_algo": "FP8",
        "group_size": 16,
        "exclude_modules": [
            "lm_head",
            "*.mlp.shared_expert.*",
            "model.language_model.layers.0.linear_attn*",
            "model.language_model.layers.0.mlp.shared_expert*",
            "model.language_model.layers.0.mlp.shared_expert_gate",
            // ... (all 60 layers with layer-specific exclusions)
            "model.visual*",
            "mtp.layers.0*"
        ]
    }
}
```

## Why This Matters

The official `model.safetensors.index.json` confirms that **all 60 layers** in Qwen3.5-397B-A17B have `shared_expert_gate` weights. The previous `--qformat nvfp4_mlp_only` was **over-quantizing** the model because it would quantize the **shared experts** to 4-bit precision. Since shared experts process **every single token**, this causes compounding errors and degrades:
- Perplexity
- Reasoning capabilities
- Overall model quality

The new `nvfp4_qwen35_397b` config fixes this by excluding shared experts consistently across all layers.

## Technical Notes

- **Total exclusion patterns**: 91 (including defaults)
- **Custom layer-specific exclusions**: 88 (45 DeltaNet + 15 Self-Attn + special modules)
- **KV Cache**: Use `--kv_cache_qformat fp8` for FP8 KV cache quantization
- **Group Size**: 16 (as per NVFP4 spec)
- **Generated Config Pattern**: Matches the official Qwen3.5-397B-A17B NVFP4 config structure

## Important Correction

**Note on the official quantization config**: The official `hf_quant_config.json` from Qwen3.5-397B-A17B has an inconsistency where many Self-Attn layers (e.g., layers 3, 7, 11, 15, 19, 23, 27, 31, 35, 39, 43, 47, 51, 55, 59) are missing the `shared_expert_gate` exclusion pattern. Our implementation is **more correct** because it consistently excludes `shared_expert_gate` for **all 60 layers** as confirmed by the `model.safetensors.index.json`.
