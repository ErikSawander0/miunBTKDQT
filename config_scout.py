"""
greedy_mixed_precision.py - Find optimal mixed-precision configs via greedy
sensitivity-driven allocation, then evaluate them.

For each (deeper_model, shallower_model) pair, finds the mixed-precision
config for the deeper model that matches the shallower model's INT8 size
(and/or BOPs) while maximizing accuracy.

Algorithm:
  1. Start with all layers at INT8 (the "free" baseline)
  2. Greedily demote layers from INT8 -> INT4, picking the layer with
     the smallest AP drop per byte saved, until hitting the size target
  3. Evaluate the resulting config

Usage:
    python greedy_mixed_precision.py
    python greedy_mixed_precision.py --output greedy_results.json
"""

import argparse
import copy
import json
import os
import time

import numpy as np
import torch
from PIL import Image
from pycocotools.coco import COCO
from pycocotools.cocoeval import COCOeval
from tqdm import tqdm
from transformers import VitPoseForPoseEstimation, VitPoseImageProcessor

from config import LAYER_MAPPINGS
from Student import createStudent

EXTRA_LAYER_MAPPINGS = {2: {0: 0, 1: 11}}


# ── Model loading ────────────────────────────────────────────────────────────

def get_layer_mapping(depth):
    if depth in LAYER_MAPPINGS:
        return LAYER_MAPPINGS[depth]
    if depth in EXTRA_LAYER_MAPPINGS:
        return EXTRA_LAYER_MAPPINGS[depth]
    raise ValueError(f"No layer mapping for depth {depth}")


def load_student(depth, checkpoint_path, device):
    layer_mapping = get_layer_mapping(depth)
    teacher = VitPoseForPoseEstimation.from_pretrained("usyd-community/vitpose-base-simple")
    student = createStudent(depth, layer_mapping, teacher.state_dict())
    del teacher
    ckpt = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    student.load_state_dict(ckpt["model_state_dict"])
    student = student.to(device)
    student.eval()
    return student


# ── Size computation ─────────────────────────────────────────────────────────

def get_layer_param_counts(model, depth):
    """Get parameter count for each transformer layer (weights only)."""
    layer_params = {}
    for layer_idx in range(depth):
        layer = model.backbone.encoder.layer[layer_idx]
        count = 0
        for name, param in layer.named_parameters():
            if "weight" in name:
                count += param.numel()
        layer_params[layer_idx] = count
    return layer_params


def get_non_layer_param_count(model, depth):
    """Get parameter count for everything outside transformer layers."""
    layer_param_set = set()
    for layer_idx in range(depth):
        layer = model.backbone.encoder.layer[layer_idx]
        for param in layer.parameters():
            layer_param_set.add(id(param))

    count = 0
    for param in model.parameters():
        if id(param) not in layer_param_set:
            count += param.numel()
    return count


def compute_size_bytes(layer_params, non_layer_params, layer_bits):
    """
    Compute total model size in bytes.
    layer_params: dict of {layer_idx: param_count}
    non_layer_params: int, params outside layers (always FP32)
    layer_bits: dict of {layer_idx: bit_width}
    """
    total = non_layer_params * 4  # FP32 = 4 bytes
    for idx, count in layer_params.items():
        bits = layer_bits.get(idx, 32)
        total += count * bits / 8
    return total


# ── BOPs computation ─────────────────────────────────────────────────────────

def compute_layer_bops(weight_bits, activation_bits=32,
                       hidden_size=768, mlp_ratio=4, seq_len=192, num_heads=12):
    """Compute BOPs for one transformer layer at given bit-widths."""
    head_dim = hidden_size // num_heads
    mlp_hidden = hidden_size * mlp_ratio

    # Weight-dependent FLOPs (linear layers)
    qkv = 3 * seq_len * hidden_size * hidden_size
    out_proj = seq_len * hidden_size * hidden_size
    ffn_up = seq_len * hidden_size * mlp_hidden
    ffn_down = seq_len * mlp_hidden * hidden_size
    weight_flops = qkv + out_proj + ffn_up + ffn_down

    # Activation-only FLOPs (attention matmuls)
    attn_flops = 2 * num_heads * seq_len * seq_len * head_dim

    return (weight_flops * activation_bits * weight_bits +
            attn_flops * activation_bits * activation_bits)


def compute_total_bops(depth, layer_bits):
    """Compute total BOPs for a model config."""
    # Fixed components (patch embed + decoder, always FP32)
    patch_bops = (12 * 16 * 192 // 16 * 256 // 16 * 3 * 16 * 16 * 768) * 32 * 32
    decoder_bops = (64 * 48 * 768 * 17) * 32 * 32

    layer_bops = sum(
        compute_layer_bops(layer_bits.get(i, 32))
        for i in range(depth)
    )
    return patch_bops + decoder_bops + layer_bops


# ── Simulated quantization ──────────────────────────────────────────────────

def quantize_tensor(tensor, num_bits):
    qmin = -(2 ** (num_bits - 1))
    qmax = 2 ** (num_bits - 1) - 1
    abs_max = tensor.abs().max().clamp(min=1e-8)
    scale = abs_max / qmax
    quantized = (tensor / scale).round().clamp(qmin, qmax)
    return quantized * scale


def apply_mixed_precision(model, layer_bits):
    """Apply mixed-precision quantization to a model copy. Returns new model."""
    model_q = copy.deepcopy(model)
    for layer_idx, bits in layer_bits.items():
        if bits < 32:
            layer = model_q.backbone.encoder.layer[layer_idx]
            for name, param in layer.named_parameters():
                if "weight" in name:
                    param.data = quantize_tensor(param.data, bits)
    return model_q


# ── Greedy optimizer ─────────────────────────────────────────────────────────

def greedy_allocate(depth, layer_params, non_layer_params, sensitivity_data,
                    target_size_bytes):
    """
    Greedy bit-width allocation starting from all-INT8.

    At each step, demote the layer from INT8->INT4 that has the best
    ratio of (bytes saved / AP drop). Stop when we hit the target size
    or run out of layers to demote.

    Args:
        depth: number of layers
        layer_params: dict of {layer_idx: weight_param_count}
        non_layer_params: param count for non-layer components
        sensitivity_data: dict with keys like "int4_layerN" containing "ap_drop"
        target_size_bytes: target size in bytes to hit

    Returns:
        layer_bits: dict of {layer_idx: bit_width}
        allocation_log: list of steps taken
    """
    # Start: all INT8
    layer_bits = {i: 8 for i in range(depth)}
    current_size = compute_size_bytes(layer_params, non_layer_params, layer_bits)

    allocation_log = [{
        "step": 0,
        "action": "start (all INT8)",
        "layer_bits": dict(layer_bits),
        "size_bytes": current_size,
        "size_mb": current_size / (1024 * 1024),
        "estimated_ap_cost": 0.0,
    }]

    if current_size <= target_size_bytes:
        allocation_log[0]["note"] = "Already at or below target size"
        return layer_bits, allocation_log

    total_ap_cost = 0.0
    step = 1

    while current_size > target_size_bytes:
        best_layer = None
        best_efficiency = float('inf')  # AP drop per byte saved (lower = better)
        best_savings = 0

        for i in range(depth):
            if layer_bits[i] != 8:
                continue  # Already INT4 or can't go lower

            # Bytes saved by going INT8 -> INT4
            bytes_saved = layer_params[i] * (8 - 4) / 8  # = params * 0.5 bytes

            # AP cost from sensitivity data
            # Use the INT4 individual layer drop as proxy
            key = f"int4_layer{i}"
            if key not in sensitivity_data:
                continue
            ap_drop = sensitivity_data[key].get("ap_drop", float('inf'))

            # Efficiency: AP drop per byte saved
            if bytes_saved > 0:
                efficiency = ap_drop / bytes_saved
            else:
                efficiency = float('inf')

            if efficiency < best_efficiency:
                best_efficiency = efficiency
                best_layer = i
                best_savings = bytes_saved

        if best_layer is None:
            allocation_log.append({
                "step": step,
                "action": "STOPPED - no more layers to demote",
                "layer_bits": dict(layer_bits),
                "size_bytes": current_size,
                "size_mb": current_size / (1024 * 1024),
            })
            break

        # Demote best layer
        ap_cost = sensitivity_data[f"int4_layer{best_layer}"]["ap_drop"]
        layer_bits[best_layer] = 4
        current_size -= best_savings
        total_ap_cost += ap_cost

        allocation_log.append({
            "step": step,
            "action": f"layer {best_layer}: INT8 -> INT4",
            "layer_demoted": best_layer,
            "ap_cost_individual": ap_cost,
            "estimated_total_ap_cost": total_ap_cost,
            "bytes_saved": best_savings,
            "efficiency": best_efficiency,
            "layer_bits": dict(layer_bits),
            "size_bytes": current_size,
            "size_mb": current_size / (1024 * 1024),
        })
        step += 1

    return layer_bits, allocation_log


# ── Data loading & eval ──────────────────────────────────────────────────────

def load_samples(ann_file):
    with open(ann_file) as f:
        coco_data = json.load(f)
    images_info = {img["id"]: img for img in coco_data["images"]}
    samples = []
    for ann in coco_data["annotations"]:
        if ann.get("iscrowd", 0):
            continue
        kpts = np.array(ann["keypoints"]).reshape(-1, 3)
        if (kpts[:, 2] > 0).sum() < 1:
            continue
        img_info = images_info[ann["image_id"]]
        samples.append({
            "image_id": ann["image_id"],
            "ann_id": ann["id"],
            "bbox": ann["bbox"],
            "file_name": img_info["file_name"],
            "height": img_info["height"],
            "width": img_info["width"],
        })
    return samples


def preprocess_all(processor, samples, data_root):
    preprocessed = []
    for s in tqdm(samples, desc="  Preprocessing"):
        pil_img = Image.open(f"{data_root}/{s['file_name']}").convert("RGB")
        box = s["bbox"]
        inputs = processor(images=pil_img, boxes=[[box]], return_tensors="pt")
        preprocessed.append({
            "pixel_values": inputs["pixel_values"].squeeze(0),
            "box": box,
            "image_id": s["image_id"],
        })
    return preprocessed


@torch.no_grad()
def evaluate_batched(model, processor, preprocessed, device, batch_size=64):
    results = []
    n = len(preprocessed)
    for i in range(0, n, batch_size):
        batch = preprocessed[i : i + batch_size]
        pixel_values = torch.stack([s["pixel_values"] for s in batch]).to(device)
        outputs = model(pixel_values)
        heatmaps = outputs.heatmaps
        for j, s in enumerate(batch):
            single_heatmaps = heatmaps[j : j + 1]
            single_output = type(outputs)(heatmaps=single_heatmaps)
            boxes_for_image = [[s["box"]]]
            pose_results = processor.post_process_pose_estimation(
                single_output, boxes=boxes_for_image,
            )
            for person in pose_results[0]:
                keypoints = person["keypoints"].cpu().numpy()
                scores = person["scores"].cpu().numpy()
                kpts_flat = np.zeros((keypoints.shape[0], 3))
                kpts_flat[:, :2] = keypoints
                kpts_flat[:, 2] = scores
                results.append({
                    "image_id": int(s["image_id"]),
                    "category_id": 1,
                    "keypoints": kpts_flat.flatten().tolist(),
                    "score": float(np.mean(scores)),
                })
    return results


def run_coco_eval(coco_gt, results):
    if not results:
        return {}
    coco_dt = coco_gt.loadRes(results)
    coco_eval = COCOeval(coco_gt, coco_dt, "keypoints")
    coco_eval.evaluate()
    coco_eval.accumulate()
    coco_eval.summarize()
    stat_names = ["AP", "AP50", "AP75", "APM", "APL", "AR", "AR50", "AR75", "ARM", "ARL"]
    return {name: float(val) for name, val in zip(stat_names, coco_eval.stats)}


def bits_str(layer_bits, depth):
    """Pretty-print a bit-width assignment."""
    return "[" + ",".join(str(layer_bits.get(i, 32)) for i in range(depth)) + "]"


# ── Main ─────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint_root", default="./main_runs_fr")
    parser.add_argument("--sensitivity_file", default="sensitivity_results.json",
                        help="Output from sensitivity_analysis.py")
    parser.add_argument("--uniform_file", default="uniform_quant_results.json",
                        help="Output from uniform_quantization_eval.py")
    parser.add_argument("--val_data_root", default="./dataset/val2017")
    parser.add_argument("--val_ann_file", default="./dataset/annotations/person_keypoints_val2017.json")
    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--output", default="greedy_results.json")
    args = parser.parse_args()

    # Load previous results
    with open(args.sensitivity_file) as f:
        sensitivity = json.load(f)
    with open(args.uniform_file) as f:
        uniform = json.load(f)

    if torch.cuda.is_available():
        device = torch.device("cuda")
    elif torch.backends.mps.is_available():
        device = torch.device("mps")
    else:
        device = torch.device("cpu")
    print(f"Device: {device}")

    processor = VitPoseImageProcessor.from_pretrained("usyd-community/vitpose-base-simple")

    print("Loading annotations...")
    coco_gt = COCO(args.val_ann_file)
    samples = load_samples(args.val_ann_file)
    print(f"  {len(samples)} samples")

    print("Preprocessing all samples...")
    preprocessed = preprocess_all(processor, samples, args.val_data_root)
    print()

    # Define adjacent pairs: (deeper, shallower)
    pairs = [
        (10, 8),
        (8, 6),
        (6, 4),
        (4, 3),
    ]

    all_results = {}

    for deeper, shallower in pairs:
        deeper_key = f"depth_{deeper}"
        shallower_key = f"depth_{shallower}"

        if deeper_key not in sensitivity or shallower_key not in uniform:
            print(f"Skipping pair ({deeper}, {shallower}): missing data")
            continue

        # Target: match the shallower model's INT8 size
        target_size_bytes = uniform[shallower_key]["int8"]["size_bytes"]
        target_size_mb = target_size_bytes / (1024 * 1024)
        target_ap = uniform[shallower_key]["int8"]["AP"]

        print(f"{'='*60}")
        print(f"PAIR: depth-{deeper} -> depth-{shallower} INT8 size")
        print(f"  Target: {target_size_mb:.2f} MB, beat AP {target_ap:.4f}")
        print(f"{'='*60}")

        # Load deeper model
        ckpt_path = f"{args.checkpoint_root}/depth_{deeper}/best.pt"
        if not os.path.exists(ckpt_path):
            print(f"  Checkpoint not found, skipping.")
            continue

        model = load_student(deeper, ckpt_path, device)

        # Get param counts
        layer_params = get_layer_param_counts(model, deeper)
        non_layer_params = get_non_layer_param_count(model, deeper)

        print(f"  Layer params: {', '.join(f'L{i}={c:,}' for i, c in layer_params.items())}")
        print(f"  Non-layer params: {non_layer_params:,}")

        # Current INT8 size
        int8_bits = {i: 8 for i in range(deeper)}
        int8_size = compute_size_bytes(layer_params, non_layer_params, int8_bits)
        print(f"  Current INT8 size: {int8_size/(1024*1024):.2f} MB")
        print(f"  Need to reach: {target_size_mb:.2f} MB")
        print(f"  Gap: {(int8_size - target_size_bytes)/(1024*1024):.2f} MB")

        # Run greedy allocation
        sens_data = sensitivity[deeper_key]
        layer_bits, log = greedy_allocate(
            deeper, layer_params, non_layer_params, sens_data, target_size_bytes
        )

        print(f"\n  Greedy allocation:")
        for entry in log:
            if "layer_demoted" in entry:
                print(f"    Step {entry['step']}: demote layer {entry['layer_demoted']} "
                      f"(AP cost: {entry['ap_cost_individual']:.4f}, "
                      f"size: {entry['size_mb']:.2f} MB)")
            else:
                print(f"    {entry['action']} ({entry['size_mb']:.2f} MB)")

        final_bits = layer_bits
        final_size = compute_size_bytes(layer_params, non_layer_params, final_bits)
        final_bops = compute_total_bops(deeper, final_bits)

        print(f"\n  Final config: {bits_str(final_bits, deeper)}")
        print(f"  Final size: {final_size/(1024*1024):.2f} MB "
              f"(target: {target_size_mb:.2f} MB)")
        print(f"  Final BOPs: {final_bops/1e9:.2f} GBOPs")

        # Evaluate
        print(f"  Evaluating...")
        t0 = time.time()
        model_q = apply_mixed_precision(model, final_bits)
        model_q = model_q.to(device)
        model_q.eval()
        preds = evaluate_batched(model_q, processor, preprocessed, device, args.batch_size)
        metrics = run_coco_eval(coco_gt, preds)
        elapsed = time.time() - t0

        is_pareto = metrics["AP"] > target_ap
        print(f"  AP: {metrics['AP']:.4f} vs target {target_ap:.4f} "
              f"{'✓ PARETO IMPROVEMENT' if is_pareto else '✗ not Pareto'} "
              f"({elapsed:.1f}s)")

        pair_key = f"depth{deeper}_to_depth{shallower}"
        all_results[pair_key] = {
            "deeper_depth": deeper,
            "shallower_depth": shallower,
            "target_size_mb": target_size_mb,
            "target_size_bytes": target_size_bytes,
            "target_ap": target_ap,
            "target_bops": compute_total_bops(shallower, {i: 8 for i in range(shallower)}) / 1e9,
            "config": {str(k): v for k, v in final_bits.items()},
            "config_str": bits_str(final_bits, deeper),
            "actual_size_mb": final_size / (1024 * 1024),
            "actual_size_bytes": final_size,
            "actual_bops_gbops": final_bops / 1e9,
            "is_pareto_improvement": is_pareto,
            "ap_delta": metrics["AP"] - target_ap,
            "allocation_log": log,
            "metrics": metrics,
            "eval_time_seconds": round(elapsed, 1),
        }

        del model, model_q
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        print()

    # Save
    with open(args.output, "w") as f:
        json.dump(all_results, f, indent=2)
    print(f"\nResults saved to {args.output}")

    # Final summary
    print(f"\n{'='*60}")
    print("PARETO FRONTIER SUMMARY")
    print(f"{'='*60}")
    print(f"{'Pair':>20} {'Config':>20} {'Size MB':>9} {'GBOPs':>9} "
          f"{'AP':>8} {'Target AP':>10} {'Pareto?':>8}")
    print("-" * 88)
    for key, r in all_results.items():
        pareto = "✓" if r["is_pareto_improvement"] else "✗"
        print(f"{key:>20} {r['config_str']:>20} {r['actual_size_mb']:>9.2f} "
              f"{r['actual_bops_gbops']:>9.2f} {r['metrics']['AP']:>8.4f} "
              f"{r['target_ap']:>10.4f} {pareto:>8}")


if __name__ == "__main__":
    main()
