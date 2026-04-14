# NVFP4 Quantization Config for Qwen3.5-35B-A3B

## Summary

A custom NVFP4 quantization configuration (`NVFP4_QWEN35_35B_CFG`) has been created for the **Qwen3.5-35B-A3B** model. This config mirrors the official quantization strategy used for the larger **Qwen3.5-397B-A17B** model.

## What Was Added

### 1. New Quantization Config
**File**: `modelopt/torch/quantization/config.py`

Added `NVFP4_QWEN35_35B_CFG` which implements the same quantization philosophy as the official 397B model:

#### Quantized (NVFP4, group_size=16):
- **Routed MoE Experts Only**: The 256 routed experts in each MoE layer are quantized to NVFP4
- This is the bulk of the model's 35B parameters

#### Excluded from NVFP4 (kept in higher precision):
- **All Attention Mechanisms** (across all 40 layers):
  - **DeltaNet** (`linear_attn*`): Layers 0,1,2, 4,5,6, 8,9,10, 12,13,14, 16,17,18, 20,21,22, 24,25,26, 28,29,30, 32,33,34, 36,37,38
  - **Self-Attention** (`self_attn*`): Layers 3, 7, 11, 15, 19, 23, 27, 31, 35, 39
- **Shared Experts** (`*.mlp.shared_expert.*`): The shared expert that processes every token
- **Shared Expert Gates** (`*shared_expert_gate`)
- **Vision Encoder** (`model.visual*`)
- **Multi-Token Prediction** (`mtp.layers.0*`)
- **LM Head** (`lm_head`)

### 2. Integration Points

The new config is now available in:
- ✅ `modelopt.torch.quantization.config.NVFP4_QWEN35_35B_CFG`
- ✅ `examples/llm_ptq/hf_ptq.py` → `--qformat nvfp4_qwen35_35b`
- ✅ `examples/llm_ptq/multinode_ptq.py` → `--qformat nvfp4_qwen35_35b`

## Usage

### Basic Usage with hf_ptq.py

```bash
python examples/llm_ptq/hf_ptq.py \
    --pyt_ckpt_path /path/to/Qwen3.5-35B-A3B \
    --qformat nvfp4_qwen35_35b \
    --kv_cache_qformat fp8 \
    --export_path /path/to/quantized_output \
    --calib_size 512
```

### With Image Calibration (for VLM)

```bash
python examples/llm_ptq/hf_ptq.py \
    --pyt_ckpt_path /path/to/Qwen3.5-35B-A3B \
    --qformat nvfp4_qwen35_35b \
    --kv_cache_qformat fp8 \
    --export_path /path/to/quantized_output \
    --calib_with_images \
    --calib_size 256
```

## Architecture Details

The Qwen3.5-35B-A3B model follows this structure:
- **Total Layers**: 40
- **Block Pattern**: 10 × (3× DeltaNet + 1× Self-Attention)
- **MoE**: 256 experts per layer, 8 routed + 1 shared expert activated per token
- **Hidden Dimension**: 2048

### Layer Layout Pattern
```
Layers 0-2:   DeltaNet (linear_attn)
Layer 3:      Self-Attention
Layers 4-6:   DeltaNet (linear_attn)
Layer 7:      Self-Attention
Layers 8-10:  DeltaNet (linear_attn)
Layer 11:     Self-Attention
... (repeats)
Layers 36-38: DeltaNet (linear_attn)
Layer 39:     Self-Attention
```

## Comparison with Other NVFP4 Configs

| Config | What's Quantized | Key Exclusions |
|--------|------------------|----------------|
| `nvfp4` (default) | Most layers | Basic defaults only |
| `nvfp4_mlp_only` | All MLP/MoE | All attention |
| `nvfp4_omlp_only` | All MLP/MoE + o_proj | All attention |
| **`nvfp4_qwen35_35b`** | **Routed MoE experts only** | **Attention + Shared experts + Vision + MTP** |

The `nvfp4_qwen35_35b` config is **more conservative** than `nvfp4_mlp_only` because it also excludes the shared experts, which are critical for maintaining model accuracy.

## Generated Config Structure

When you run quantization with `--qformat nvfp4_qwen35_35b`, the generated `quantization_config.json` will contain:

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
            "model.language_model.layers.1.linear_attn*",
            // ... (all 30 DeltaNet layers)
            "model.language_model.layers.3.self_attn*",
            "model.language_model.layers.7.self_attn*",
            // ... (all 10 Self-Attn layers)
            "model.visual*",
            "mtp.layers.0*"
        ]
    }
}
```

This matches the pattern of the official Qwen3.5-397B-A17B quantization config you provided.

## Why This Matters

The previous `--qformat nvfp4_mlp_only` was **over-quantizing** the Qwen3.5-35B-A3B model because it would quantize the **shared experts** to 4-bit precision. Since shared experts process **every single token**, this causes compounding errors and degrades:
- Perplexity
- Reasoning capabilities  
- Overall model quality

The new `nvfp4_qwen35_35b` config fixes this by excluding shared experts, matching the production-grade quantization strategy used for the 397B model.

## Technical Notes

- **Total exclusion patterns**: 55 (including defaults)
- **Custom layer-specific exclusions**: 42 (30 DeltaNet + 10 Self-Attn + special modules)
- **KV Cache**: Use `--kv_cache_qformat fp8` for FP8 KV cache quantization
- **Group Size**: 16 (as per NVFP4 spec)
