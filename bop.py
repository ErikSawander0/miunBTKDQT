"""
compute_bops.py - Compute BOPs (bit-operations) for all model configurations.

BOPs = sum over all matmuls of (FLOPs × activation_bits × weight_bits)
For weights-only quantization, activations stay FP32 (or 16), weights vary.

We assume activations at 32 bits (since we're doing weights-only PTQ).

Usage:
    python compute_bops.py
"""

import json
import argparse


def compute_vit_layer_flops(hidden_size=768, mlp_ratio=4, seq_len=192, num_heads=12):
    """
    Compute FLOPs for one ViT transformer layer.
    Input: (batch, seq_len, hidden_size) where seq_len = (256/16)*(192/16) = 16*12 = 192

    Returns dict with per-component FLOPs (multiply-accumulate operations).
    """
    head_dim = hidden_size // num_heads
    mlp_hidden = hidden_size * mlp_ratio

    # QKV projection: 3 × (seq_len × hidden_size × hidden_size)
    qkv_flops = 3 * seq_len * hidden_size * hidden_size

    # Attention: Q @ K^T -> (seq_len × seq_len × head_dim) per head
    attn_flops = num_heads * seq_len * seq_len * head_dim

    # Attention @ V -> (seq_len × head_dim × seq_len) per head  [same cost]
    attn_v_flops = num_heads * seq_len * head_dim * seq_len

    # Output projection: seq_len × hidden_size × hidden_size
    out_proj_flops = seq_len * hidden_size * hidden_size

    # FFN: two linear layers
    ffn_up_flops = seq_len * hidden_size * mlp_hidden
    ffn_down_flops = seq_len * mlp_hidden * hidden_size

    return {
        "qkv": qkv_flops,
        "attn": attn_flops,
        "attn_v": attn_v_flops,
        "out_proj": out_proj_flops,
        "ffn_up": ffn_up_flops,
        "ffn_down": ffn_down_flops,
        "total": (qkv_flops + attn_flops + attn_v_flops +
                  out_proj_flops + ffn_up_flops + ffn_down_flops),
    }


def compute_patch_embed_flops(hidden_size=768, patch_size=16, in_channels=3,
                               img_h=256, img_w=192):
    """FLOPs for patch embedding (conv2d with kernel=patch_size, stride=patch_size)."""
    out_h = img_h // patch_size
    out_w = img_w // patch_size
    # Conv2d FLOPs: out_h * out_w * in_channels * kernel_h * kernel_w * out_channels
    return out_h * out_w * in_channels * patch_size * patch_size * hidden_size


def compute_decoder_flops(hidden_size=768, num_keypoints=17,
                           heatmap_h=64, heatmap_w=48):
    """
    Simple decoder: bilinear upsample (free) + 1×1 conv to num_keypoints.
    The conv operates on the upsampled feature map.
    """
    # 1x1 conv: heatmap_h * heatmap_w * hidden_size * num_keypoints
    return heatmap_h * heatmap_w * hidden_size * num_keypoints


def flops_to_bops(flops, activation_bits, weight_bits):
    """Convert FLOPs to BOPs."""
    return flops * activation_bits * weight_bits


def compute_model_bops(depth, layer_bits=None, activation_bits=32):
    """
    Compute total BOPs for a ViTPose model.

    Args:
        depth: number of transformer layers
        layer_bits: dict mapping layer_idx -> weight bit-width.
                    If None, all layers are 32-bit.
        activation_bits: bit-width for activations (32 for weights-only quant)

    Returns: dict with breakdown and total
    """
    if layer_bits is None:
        layer_bits = {}

    layer_flops = compute_vit_layer_flops()
    patch_flops = compute_patch_embed_flops()
    decoder_flops = compute_decoder_flops()

    # Patch embedding and decoder stay FP32
    patch_bops = flops_to_bops(patch_flops, activation_bits, 32)
    decoder_bops = flops_to_bops(decoder_flops, activation_bits, 32)

    layer_bops_list = []
    for i in range(depth):
        w_bits = layer_bits.get(i, 32)

        # Weight-dependent ops (linear layers): QKV, out_proj, FFN
        weight_flops = (layer_flops["qkv"] + layer_flops["out_proj"] +
                        layer_flops["ffn_up"] + layer_flops["ffn_down"])
        # Attention matmul is activation × activation, not weight-dependent
        attn_only_flops = layer_flops["attn"] + layer_flops["attn_v"]

        bops = (flops_to_bops(weight_flops, activation_bits, w_bits) +
                flops_to_bops(attn_only_flops, activation_bits, activation_bits))
        layer_bops_list.append(bops)

    total_bops = patch_bops + decoder_bops + sum(layer_bops_list)

    return {
        "total_bops": total_bops,
        "total_gbops": total_bops / 1e9,
        "patch_embed_bops": patch_bops,
        "decoder_bops": decoder_bops,
        "per_layer_bops": layer_bops_list,
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--depths", type=int, nargs="+", default=[3, 4, 6, 8, 10, 12])
    parser.add_argument("--output", default="bops_results.json")
    args = parser.parse_args()

    results = {}

    # Per-layer FLOPs breakdown (same for all depths)
    layer_flops = compute_vit_layer_flops()
    print("Per-layer FLOPs breakdown:")
    for k, v in layer_flops.items():
        print(f"  {k}: {v/1e6:.2f} MFLOPs")
    print(f"  Patch embed: {compute_patch_embed_flops()/1e6:.2f} MFLOPs")
    print(f"  Decoder: {compute_decoder_flops()/1e6:.2f} MFLOPs")
    print()

    configs = {
        "fp32": {},       # empty dict = all FP32
        "int8": "int8",
        "int4": "int4",
    }

    print(f"{'Depth':>6} {'FP32 GBOPs':>12} {'INT8 GBOPs':>12} {'INT4 GBOPs':>12} "
          f"{'INT8/FP32':>10} {'INT4/FP32':>10}")
    print("-" * 64)

    for depth in args.depths:
        depth_results = {}

        # FP32
        fp32 = compute_model_bops(depth)
        depth_results["fp32"] = {
            "total_gbops": fp32["total_gbops"],
            "total_bops": fp32["total_bops"],
        }

        # INT8
        int8_layer_bits = {i: 8 for i in range(depth)}
        int8 = compute_model_bops(depth, layer_bits=int8_layer_bits)
        depth_results["int8"] = {
            "total_gbops": int8["total_gbops"],
            "total_bops": int8["total_bops"],
        }

        # INT4
        int4_layer_bits = {i: 4 for i in range(depth)}
        int4 = compute_model_bops(depth, layer_bits=int4_layer_bits)
        depth_results["int4"] = {
            "total_gbops": int4["total_gbops"],
            "total_bops": int4["total_bops"],
        }

        results[f"depth_{depth}"] = depth_results

        ratio_8 = int8["total_gbops"] / fp32["total_gbops"]
        ratio_4 = int4["total_gbops"] / fp32["total_gbops"]

        print(f"{depth:>6} {fp32['total_gbops']:>12.2f} {int8['total_gbops']:>12.2f} "
              f"{int4['total_gbops']:>12.2f} {ratio_8:>10.3f} {ratio_4:>10.3f}")

    # Save
    with open(args.output, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nResults saved to {args.output}")

    # Also print the size vs bops comparison
    print(f"\nComparison of size and compute reduction:")
    print(f"{'Depth':>6} {'Config':>8} {'Size MB':>9} {'GBOPs':>10} {'AP':>8}")
    print("-" * 45)

    # Hardcode the AP and size values from previous runs for the summary
    # (This is just for display - the JSON has the raw BOPs data)
    known = {
        3:  {"fp32_mb": 84.39,  "int8_mb": 23.55, "fp32_ap": 0.4799, "int8_ap": 0.4743},
        4:  {"fp32_mb": 111.42, "int8_mb": 30.31, "fp32_ap": 0.6123, "int8_ap": 0.6091},
        6:  {"fp32_mb": 165.50, "int8_mb": 43.83, "fp32_ap": 0.7089, "int8_ap": 0.7061},
        8:  {"fp32_mb": 219.58, "int8_mb": 57.35, "fp32_ap": 0.7351, "int8_ap": 0.7328},
        10: {"fp32_mb": 273.65, "int8_mb": 70.87, "fp32_ap": 0.7436, "int8_ap": 0.7412},
    }
    for depth in args.depths:
        if depth in known:
            k = known[depth]
            fp32_bops = results[f"depth_{depth}"]["fp32"]["total_gbops"]
            int8_bops = results[f"depth_{depth}"]["int8"]["total_gbops"]
            print(f"{depth:>6} {'FP32':>8} {k['fp32_mb']:>9.2f} {fp32_bops:>10.2f} {k['fp32_ap']:>8.4f}")
            print(f"{'':>6} {'INT8':>8} {k['int8_mb']:>9.2f} {int8_bops:>10.2f} {k['int8_ap']:>8.4f}")


if __name__ == "__main__":
    main()
