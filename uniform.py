"""
uniform_quantization_eval.py - Compute model sizes and evaluate uniform INT8
for all depths. Establishes the baseline Pareto frontier.

Usage:
    python uniform_quantization_eval.py
    python uniform_quantization_eval.py --depths 3 4 6 8 10 --output uniform_quant_results.json
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

def compute_model_size(model, layer_bits=None, default_bits=32):
    """
    Compute model size in bytes given per-component bit-widths.

    Args:
        model: the model
        layer_bits: dict mapping layer_idx -> bit_width for transformer layers.
                    If None, all layers use default_bits.
        default_bits: bit-width for non-layer parameters (embeddings, decoder, norms)

    Returns:
        dict with total size and per-component breakdown in bytes and MB
    """
    if layer_bits is None:
        layer_bits = {}

    breakdown = {
        "backbone_layers": {},
        "backbone_other": {"params": 0, "bits": default_bits, "bytes": 0},
        "decoder": {"params": 0, "bits": default_bits, "bytes": 0},
        "other": {"params": 0, "bits": default_bits, "bytes": 0},
    }

    for name, param in model.named_parameters():
        num_params = param.numel()

        # Determine which component this belongs to
        layer_idx = None
        if "backbone.encoder.layer." in name:
            # Extract layer index
            parts = name.split(".")
            for i, p in enumerate(parts):
                if p == "layer" and i + 1 < len(parts):
                    try:
                        layer_idx = int(parts[i + 1])
                        break
                    except ValueError:
                        pass

        if layer_idx is not None:
            bits = layer_bits.get(layer_idx, default_bits)
            size_bytes = num_params * bits / 8
            if layer_idx not in breakdown["backbone_layers"]:
                breakdown["backbone_layers"][layer_idx] = {
                    "params": 0, "bits": bits, "bytes": 0
                }
            breakdown["backbone_layers"][layer_idx]["params"] += num_params
            breakdown["backbone_layers"][layer_idx]["bytes"] += size_bytes
        elif "backbone" in name:
            size_bytes = num_params * default_bits / 8
            breakdown["backbone_other"]["params"] += num_params
            breakdown["backbone_other"]["bytes"] += size_bytes
        elif "head" in name:
            size_bytes = num_params * default_bits / 8
            breakdown["decoder"]["params"] += num_params
            breakdown["decoder"]["bytes"] += size_bytes
        else:
            size_bytes = num_params * default_bits / 8
            breakdown["other"]["params"] += num_params
            breakdown["other"]["bytes"] += size_bytes

    total_params = sum(p.numel() for p in model.parameters())
    total_bytes = (
        sum(v["bytes"] for v in breakdown["backbone_layers"].values())
        + breakdown["backbone_other"]["bytes"]
        + breakdown["decoder"]["bytes"]
        + breakdown["other"]["bytes"]
    )

    return {
        "total_params": total_params,
        "total_bytes": total_bytes,
        "total_mb": total_bytes / (1024 * 1024),
        "breakdown": breakdown,
    }


def compute_size_with_config(model, depth, config_name):
    """
    Compute size for common configurations.
    config_name: "fp32", "int8", "int4", or a dict of {layer_idx: bits}
    """
    if config_name == "fp32":
        return compute_model_size(model, layer_bits=None, default_bits=32)
    elif config_name == "int8":
        layer_bits = {i: 8 for i in range(depth)}
        return compute_model_size(model, layer_bits=layer_bits, default_bits=32)
    elif config_name == "int4":
        layer_bits = {i: 4 for i in range(depth)}
        return compute_model_size(model, layer_bits=layer_bits, default_bits=32)
    elif isinstance(config_name, dict):
        return compute_model_size(model, layer_bits=config_name, default_bits=32)
    else:
        raise ValueError(f"Unknown config: {config_name}")


# ── Simulated quantization ──────────────────────────────────────────────────

def quantize_tensor(tensor, num_bits):
    qmin = -(2 ** (num_bits - 1))
    qmax = 2 ** (num_bits - 1) - 1
    abs_max = tensor.abs().max().clamp(min=1e-8)
    scale = abs_max / qmax
    quantized = (tensor / scale).round().clamp(qmin, qmax)
    return quantized * scale


def quantize_all_layers(model, num_bits):
    """Quantize all transformer layer weights to num_bits. Modifies in-place."""
    for layer in model.backbone.encoder.layer:
        for name, param in layer.named_parameters():
            if "weight" in name:
                param.data = quantize_tensor(param.data, num_bits)


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


# ── Main ─────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint_root", default="./main_runs_fr")
    parser.add_argument("--depths", type=int, nargs="+", default=[3, 4, 6, 8, 10])
    parser.add_argument("--teacher", action="store_true",
                        help="Also evaluate teacher (depth 12)")
    parser.add_argument("--val_data_root", default="./dataset/val2017")
    parser.add_argument("--val_ann_file", default="./dataset/annotations/person_keypoints_val2017.json")
    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--output", default="uniform_quant_results.json")
    args = parser.parse_args()

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

    all_results = {}

    # ── Teacher ──────────────────────────────────────────────────────────
    if args.teacher:
        print(f"{'='*60}")
        print("TEACHER (depth 12)")
        print(f"{'='*60}")

        model = VitPoseForPoseEstimation.from_pretrained(
            "usyd-community/vitpose-base-simple"
        ).to(device)
        model.eval()

        # Sizes
        fp32_size = compute_size_with_config(model, 12, "fp32")
        int8_size = compute_size_with_config(model, 12, "int8")
        print(f"  FP32: {fp32_size['total_mb']:.2f} MB  ({fp32_size['total_params']:,} params)")
        print(f"  INT8: {int8_size['total_mb']:.2f} MB")

        # FP32 eval
        print("  Evaluating FP32...")
        t0 = time.time()
        preds = evaluate_batched(model, processor, preprocessed, device, args.batch_size)
        fp32_metrics = run_coco_eval(coco_gt, preds)
        print(f"  FP32 AP={fp32_metrics['AP']:.4f}  ({time.time()-t0:.1f}s)")

        # INT8 eval
        model_int8 = copy.deepcopy(model)
        quantize_all_layers(model_int8, 8)
        model_int8.to(device)
        print("  Evaluating INT8...")
        t0 = time.time()
        preds = evaluate_batched(model_int8, processor, preprocessed, device, args.batch_size)
        int8_metrics = run_coco_eval(coco_gt, preds)
        print(f"  INT8 AP={int8_metrics['AP']:.4f}  ({time.time()-t0:.1f}s)")

        all_results["depth_12"] = {
            "fp32": {
                "size_mb": fp32_size["total_mb"],
                "size_bytes": fp32_size["total_bytes"],
                "total_params": fp32_size["total_params"],
                **fp32_metrics,
            },
            "int8": {
                "size_mb": int8_size["total_mb"],
                "size_bytes": int8_size["total_bytes"],
                **int8_metrics,
            },
        }

        del model, model_int8
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    # ── Students ─────────────────────────────────────────────────────────
    for depth in args.depths:
        ckpt_path = f"{args.checkpoint_root}/depth_{depth}/best.pt"
        if not os.path.exists(ckpt_path):
            print(f"\n[depth {depth}] Checkpoint not found, skipping.")
            continue

        print(f"\n{'='*60}")
        print(f"DEPTH {depth}")
        print(f"{'='*60}")

        model = load_student(depth, ckpt_path, device)

        # Sizes
        fp32_size = compute_size_with_config(model, depth, "fp32")
        int8_size = compute_size_with_config(model, depth, "int8")
        int4_size = compute_size_with_config(model, depth, "int4")
        print(f"  FP32: {fp32_size['total_mb']:.2f} MB  ({fp32_size['total_params']:,} params)")
        print(f"  INT8: {int8_size['total_mb']:.2f} MB")
        print(f"  INT4: {int4_size['total_mb']:.2f} MB")

        # FP32 eval (use cached results from sanity check if you want, but
        # running again for consistency)
        print("  Evaluating FP32...")
        t0 = time.time()
        preds = evaluate_batched(model, processor, preprocessed, device, args.batch_size)
        fp32_metrics = run_coco_eval(coco_gt, preds)
        print(f"  FP32 AP={fp32_metrics['AP']:.4f}  ({time.time()-t0:.1f}s)")

        # INT8 eval
        model_int8 = copy.deepcopy(model)
        quantize_all_layers(model_int8, 8)
        model_int8.to(device)
        print("  Evaluating INT8...")
        t0 = time.time()
        preds = evaluate_batched(model_int8, processor, preprocessed, device, args.batch_size)
        int8_metrics = run_coco_eval(coco_gt, preds)
        print(f"  INT8 AP={int8_metrics['AP']:.4f}  ({time.time()-t0:.1f}s)")

        all_results[f"depth_{depth}"] = {
            "fp32": {
                "size_mb": fp32_size["total_mb"],
                "size_bytes": fp32_size["total_bytes"],
                "total_params": fp32_size["total_params"],
                **fp32_metrics,
            },
            "int8": {
                "size_mb": int8_size["total_mb"],
                "size_bytes": int8_size["total_bytes"],
                **int8_metrics,
            },
            "int4_size_only": {
                "size_mb": int4_size["total_mb"],
                "size_bytes": int4_size["total_bytes"],
            },
        }

        del model, model_int8
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    # Save
    with open(args.output, "w") as f:
        json.dump(all_results, f, indent=2)
    print(f"\nResults saved to {args.output}")

    # Summary table
    print(f"\n{'='*60}")
    print("SUMMARY")
    print(f"{'='*60}")
    print(f"{'Depth':>6} {'FP32 MB':>9} {'INT8 MB':>9} {'INT4 MB':>9} {'FP32 AP':>9} {'INT8 AP':>9} {'Δ AP':>8}")
    print("-" * 62)
    for key in sorted(all_results.keys(), key=lambda x: int(x.split("_")[1])):
        d = all_results[key]
        depth = key.split("_")[1]
        fp32_mb = d["fp32"]["size_mb"]
        int8_mb = d["int8"]["size_mb"]
        int4_mb = d.get("int4_size_only", {}).get("size_mb", 0)
        fp32_ap = d["fp32"]["AP"]
        int8_ap = d["int8"]["AP"]
        delta = int8_ap - fp32_ap
        print(f"{depth:>6} {fp32_mb:>9.2f} {int8_mb:>9.2f} {int4_mb:>9.2f} "
              f"{fp32_ap:>9.4f} {int8_ap:>9.4f} {delta:>+8.4f}")


if __name__ == "__main__":
    main()
