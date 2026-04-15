# NVFP4 Quantization Config for Qwen3.5-122B-A10B

## Summary

A custom NVFP4 quantization configuration (`NVFP4_QWEN35_122B_CFG`) has been created for the **Qwen3.5-122B-A10B** model. This config mirrors the official quantization strategy used for the larger **Qwen3.5-397B-A17B** model.

## What Was Added

### 1. New Quantization Config
**File**: `modelopt/torch/quantization/config.py`

Added `NVFP4_QWEN35_122B_CFG` which implements the same quantization philosophy as the official 397B model:

#### Quantized (NVFP4, group_size=16):
- **Routed MoE Experts Only**: The 256 routed experts in each MoE layer are quantized to NVFP4
- This is the bulk of the model's 122B parameters

#### Excluded from NVFP4 (kept in higher precision):
- **All Attention Mechanisms** (across all 48 layers):
  - **DeltaNet** (`linear_attn*`): Layers 0,1,2, 4,5,6, 8,9,10, 12,13,14, 16,17,18, 20,21,22, 24,25,26, 28,29,30, 32,33,34, 36,37,38, 40,41,42, 44,45,46
  - **Self-Attention** (`self_attn*`): Layers 3, 7, 11, 15, 19, 23, 27, 31, 35, 39, 43, 47
- **Shared Experts** (`*.mlp.shared_expert.*`): The shared expert that processes every token
- **Shared Expert Gates** (`*shared_expert_gate`)
- **Vision Encoder** (`model.visual*`)
- **Multi-Token Prediction** (`mtp.layers.0*`)
- **LM Head** (`lm_head`)

### 2. Integration Points

The new config is now available in:
- ✅ `modelopt.torch.quantization.config.NVFP4_QWEN35_122B_CFG`
- ✅ `examples/llm_ptq/hf_ptq.py` → `--qformat nvfp4_qwen35_122b`
- ✅ `examples/llm_ptq/multinode_ptq.py` → `--qformat nvfp4_qwen35_122b`

## Usage

### Basic Usage with hf_ptq.py

```bash
python examples/llm_ptq/hf_ptq.py \
    --pyt_ckpt_path /path/to/Qwen3.5-122B-A10B \
    --qformat nvfp4_qwen35_122b \
    --kv_cache_qformat fp8 \
    --export_path /path/to/quantized_output \
    --calib_size 512
```

### With Image Calibration (for VLM)

```bash
python examples/llm_ptq/hf_ptq.py \
    --pyt_ckpt_path /path/to/Qwen3.5-122B-A10B \
    --qformat nvfp4_qwen35_122b \
    --kv_cache_qformat fp8 \
    --export_path /path/to/quantized_output \
    --calib_with_images \
    --calib_size 256
```

## Architecture Details

The Qwen3.5-122B-A10B model follows this structure:
- **Total Parameters**: 122 Billion
- **Activated Parameters**: 10 Billion
- **Total Layers**: 48
- **Block Pattern**: 12 × (3× DeltaNet + 1× Self-Attention)
- **MoE**: 256 experts per layer, 9 routed + 1 shared expert activated per token (note: 9 vs 8 for 35B)
- **Hidden Dimension**: 3072
- **MoE Expert Intermediate Dimension**: 1024

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
Layers 40-42: DeltaNet (linear_attn)
Layer 43:     Self-Attention
Layers 44-46: DeltaNet (linear_attn)
Layer 47:     Self-Attention
```

## Comparison with Qwen3.5-35B-A3B

| Specification | Qwen3.5-35B-A3B | Qwen3.5-122B-A10B |
|---------------|-----------------|-------------------|
| **Total Parameters** | 35 Billion | 122 Billion |
| **Activated Parameters** | 3 Billion | 10 Billion |
| **Number of Layers** | 40 | 48 |
| **Hidden Dimension** | 2048 | 3072 |
| **Hidden Layout Blocks** | 10 blocks | 12 blocks |
| **DeltaNet V Heads** | 32 | 64 |
| **Self-Attention Q Heads** | 16 | 32 |
| **MoE Expert Intermediate Dimension** | 512 | 1024 |
| **Experts Activated** | 8 routed + 1 shared | 8 routed + 1 shared |

**What Remains Identical:**
- MoE: 256 total experts per layer
- MoE routing strategy (8 routed + 1 shared activated)
- Vocabulary size (248,320)
- Context window (262,144 tokens, extensible to 1,010,000)
- Head dimensions (128 for DeltaNet, 256 for Self-Attention)
- RoPE dimension (64)

## Comparison with Other NVFP4 Configs

| Config | What's Quantized | Key Exclusions |
|--------|------------------|----------------|
| `nvfp4` (default) | Most layers | Basic defaults only |
| `nvfp4_mlp_only` | All MLP/MoE | All attention |
| `nvfp4_omlp_only` | All MLP/MoE + o_proj | All attention |
| `nvfp4_qwen35_35b` | Routed MoE experts only | Attention + Shared experts + Vision + MTP |
| **`nvfp4_qwen35_122b`** | **Routed MoE experts only** | **Attention + Shared experts + Vision + MTP** |

Both `nvfp4_qwen35_35b` and `nvfp4_qwen35_122b` configs are **more conservative** than `nvfp4_mlp_only` because they exclude the shared experts, which are critical for maintaining model accuracy.

## Generated Config Structure

When you run quantization with `--qformat nvfp4_qwen35_122b`, the generated `quantization_config.json` will contain:

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
            // ... (all 48 layers with layer-specific exclusions)
            "model.visual*",
            "mtp.layers.0*"
        ]
    }
}
```

This matches the pattern of the official Qwen3.5-397B-A17B quantization config.

## Why This Matters

The previous `--qformat nvfp4_mlp_only` was **over-quantizing** the Qwen3.5-122B-A10B model because it would quantize the **shared experts** to 4-bit precision. Since shared experts process **every single token**, this causes compounding errors and degrades:
- Perplexity
- Reasoning capabilities
- Overall model quality

The new `nvfp4_qwen35_122b` config fixes this by excluding shared experts, matching the production-grade quantization strategy used for the 397B model.

## Technical Notes

- **Total exclusion patterns**: 66 (including defaults)
- **Custom layer-specific exclusions**: 63 (36 DeltaNet + 12 Self-Attn + special modules)
- **KV Cache**: Use `--kv_cache_qformat fp8` for FP8 KV cache quantization
- **Group Size**: 16 (as per NVFP4 spec)
