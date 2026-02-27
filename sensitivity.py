"""
sensitivity_analysis.py - Per-layer quantization sensitivity analysis.

For each model depth, quantize one layer at a time to INT8/INT4,
measure AP drop vs FP32 baseline. Uses HF pipeline for pre/post processing
with batched model forward passes.

Usage:
    python sensitivity_analysis.py
    python sensitivity_analysis.py --depths 6 8 --bit_widths 8 4
    python sensitivity_analysis.py --batch_size 64 --output sensitivity_results.json
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


# ── Simulated quantization ──────────────────────────────────────────────────

def quantize_tensor(tensor, num_bits):
    """
    Simulate symmetric per-tensor quantization.
    Quantize to num_bits, then dequantize back to float.
    """
    qmin = -(2 ** (num_bits - 1))
    qmax = 2 ** (num_bits - 1) - 1

    abs_max = tensor.abs().max().clamp(min=1e-8)
    scale = abs_max / qmax

    quantized = (tensor / scale).round().clamp(qmin, qmax)
    return quantized * scale


def quantize_layer_weights(model, layer_idx, num_bits):
    """
    Quantize all weight parameters in a specific transformer layer.
    Modifies model in-place.
    """
    layer = model.backbone.encoder.layer[layer_idx]
    for name, param in layer.named_parameters():
        if "weight" in name:
            param.data = quantize_tensor(param.data, num_bits)


def quantize_model_layer(model, layer_idx, num_bits, device):
    """
    Return a copy of the model with one layer quantized.
    Original model is not modified.
    """
    model_q = copy.deepcopy(model)
    quantize_layer_weights(model_q, layer_idx, num_bits)
    model_q = model_q.to(device)
    model_q.eval()
    return model_q


# ── Data loading ─────────────────────────────────────────────────────────────

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


def preprocess_all(processor, samples, data_root, device):
    """
    Preprocess all samples using HF processor.
    Returns list of (pixel_values_tensor, box, sample_meta) tuples.

    We preprocess individually (each has its own image/box) but store
    the tensors for batched forward passes later.
    """
    preprocessed = []
    for s in tqdm(samples, desc="  Preprocessing"):
        pil_img = Image.open(f"{data_root}/{s['file_name']}").convert("RGB")
        box = s["bbox"]
        boxes_for_image = [[box]]

        inputs = processor(images=pil_img, boxes=boxes_for_image, return_tensors="pt")
        pixel_values = inputs["pixel_values"].squeeze(0)  # (3, 256, 192)

        preprocessed.append({
            "pixel_values": pixel_values,
            "box": box,
            "image_id": s["image_id"],
        })
    return preprocessed


# ── Batched evaluation ───────────────────────────────────────────────────────

@torch.no_grad()
def evaluate_batched(model, processor, preprocessed, device, batch_size=64):
    """
    Run batched forward passes, then per-sample HF postprocessing.
    """
    results = []
    n = len(preprocessed)

    for i in range(0, n, batch_size):
        batch = preprocessed[i : i + batch_size]

        pixel_values = torch.stack([s["pixel_values"] for s in batch]).to(device)
        outputs = model(pixel_values)
        heatmaps = outputs.heatmaps  # (B, 17, 64, 48)

        # Postprocess each sample individually with its own box
        for j, s in enumerate(batch):
            single_heatmaps = heatmaps[j : j + 1]  # (1, 17, 64, 48)

            # Build a minimal output object for the processor
            single_output = type(outputs)(heatmaps=single_heatmaps)

            boxes_for_image = [[s["box"]]]
            pose_results = processor.post_process_pose_estimation(
                single_output,
                boxes=boxes_for_image,
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


# ── Main ─────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="Per-layer quantization sensitivity analysis"
    )
    parser.add_argument("--checkpoint_root", default="./main_runs_fr")
    parser.add_argument("--depths", type=int, nargs="+", default=[3, 4, 6, 8, 10])
    parser.add_argument("--bit_widths", type=int, nargs="+", default=[8, 4])
    parser.add_argument("--val_data_root", default="./dataset/val2017")
    parser.add_argument("--val_ann_file", default="./dataset/annotations/person_keypoints_val2017.json")
    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--output", default="sensitivity_results.json")
    args = parser.parse_args()

    if torch.cuda.is_available():
        device = torch.device("cuda")
    elif torch.backends.mps.is_available():
        device = torch.device("mps")
    else:
        device = torch.device("cpu")
    print(f"Device: {device}")

    print("Loading HF processor...")
    processor = VitPoseImageProcessor.from_pretrained("usyd-community/vitpose-base-simple")

    print("Loading annotations...")
    coco_gt = COCO(args.val_ann_file)
    samples = load_samples(args.val_ann_file)
    print(f"  {len(samples)} samples")

    print("Preprocessing all samples (one-time cost)...")
    preprocessed = preprocess_all(processor, samples, args.val_data_root, device)
    print()

    all_results = {}

    for depth in args.depths:
        ckpt_path = f"{args.checkpoint_root}/depth_{depth}/best.pt"
        if not os.path.exists(ckpt_path):
            print(f"[depth {depth}] Checkpoint not found, skipping.")
            continue

        print(f"{'='*60}")
        print(f"Depth {depth}: Loading model")
        print(f"{'='*60}")

        model_fp32 = load_student(depth, ckpt_path, device)
        depth_results = {}

        # FP32 baseline
        print(f"  Running FP32 baseline...")
        t0 = time.time()
        preds = evaluate_batched(model_fp32, processor, preprocessed, device, args.batch_size)
        metrics = run_coco_eval(coco_gt, preds)
        elapsed = time.time() - t0
        baseline_ap = metrics["AP"]
        depth_results["fp32_baseline"] = metrics
        print(f"  FP32 baseline: AP={baseline_ap:.4f}  ({elapsed:.1f}s)")

        # Per-layer sensitivity sweep
        for bits in args.bit_widths:
            print(f"\n  --- INT{bits} sweep ---")
            for layer_idx in range(depth):
                print(f"    Quantizing layer {layer_idx}/{depth-1} to INT{bits}...", end=" ")
                t0 = time.time()

                model_q = quantize_model_layer(model_fp32, layer_idx, bits, device)
                preds = evaluate_batched(model_q, processor, preprocessed, device, args.batch_size)
                metrics = run_coco_eval(coco_gt, preds)
                elapsed = time.time() - t0

                ap = metrics["AP"]
                ap_drop = baseline_ap - ap
                metrics["ap_drop"] = ap_drop

                key = f"int{bits}_layer{layer_idx}"
                depth_results[key] = metrics

                print(f"AP={ap:.4f}  (Δ={ap_drop:+.4f})  ({elapsed:.1f}s)")

                del model_q
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()

        all_results[f"depth_{depth}"] = depth_results

        del model_fp32
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

        # Print summary for this depth
        print(f"\n  Summary for depth {depth} (AP drop from {baseline_ap:.4f}):")
        print(f"  {'Layer':>6} ", end="")
        for bits in args.bit_widths:
            print(f"{'INT'+str(bits):>10} ", end="")
        print()
        print(f"  {'-'*6} ", end="")
        for _ in args.bit_widths:
            print(f"{'-'*10} ", end="")
        print()
        for layer_idx in range(depth):
            print(f"  {layer_idx:>6} ", end="")
            for bits in args.bit_widths:
                key = f"int{bits}_layer{layer_idx}"
                drop = depth_results[key]["ap_drop"]
                print(f"{drop:>+10.4f} ", end="")
            print()
        print()

    # Save all results
    with open(args.output, "w") as f:
        json.dump(all_results, f, indent=2)
    print(f"\nAll results saved to {args.output}")

    # Final cross-depth summary
    print(f"\n{'='*60}")
    print("CROSS-DEPTH SENSITIVITY SUMMARY (AP drop)")
    print(f"{'='*60}")
    for bits in args.bit_widths:
        print(f"\n  INT{bits}:")
        print(f"  {'Depth':>6} | ", end="")
        max_layers = max(args.depths)
        for l in range(max_layers):
            print(f"{'L'+str(l):>8} ", end="")
        print()
        print(f"  {'-'*6}-+-", end="")
        for _ in range(max_layers):
            print(f"{'-'*8}-", end="")
        print()

        for depth in args.depths:
            key = f"depth_{depth}"
            if key not in all_results:
                continue
            dr = all_results[key]
            print(f"  {depth:>6} | ", end="")
            for l in range(max_layers):
                lkey = f"int{bits}_layer{l}"
                if lkey in dr:
                    print(f"{dr[lkey]['ap_drop']:>+8.4f} ", end="")
                else:
                    print(f"{'':>8} ", end="")
            print()


if __name__ == "__main__":
    main()
