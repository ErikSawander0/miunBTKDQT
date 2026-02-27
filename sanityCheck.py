"""
sanity_check_hf.py - Evaluate teacher and all students using HuggingFace's
official VitPoseImageProcessor for preprocessing and postprocessing.

Usage:
    python sanity_check_hf.py
    python sanity_check_hf.py --depths 6 8 10 --teacher
    python sanity_check_hf.py --teacher --depths 3 4 6 8 10 --output sanity_results.json
"""

import argparse
import json
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


@torch.no_grad()
def evaluate_with_hf_pipeline(model, processor, samples, data_root, device):
    """Run eval using HF processor for both pre and post processing."""
    results = []
    debug_printed = False

    for s in tqdm(samples, desc="  Inference"):
        pil_img = Image.open(f"{data_root}/{s['file_name']}").convert("RGB")
        box = s["bbox"]  # [x, y, w, h] COCO format
        boxes_for_image = [[box]]  # [image_level[box_level[x, y, w, h]]]
        target_size = (s["height"], s["width"])

        inputs = processor(images=pil_img, boxes=boxes_for_image, return_tensors="pt").to(device)
        outputs = model(**inputs)

        pose_results = processor.post_process_pose_estimation(
            outputs,
            boxes=boxes_for_image,
        )

        if not debug_printed:
            print(f"\n  DEBUG - first sample:")
            print(f"    bbox: {box}")
            print(f"    boxes_for_image: {boxes_for_image}")
            print(f"    image size: {pil_img.size}")
            print(f"    heatmaps shape: {outputs.heatmaps.shape}")
            print(f"    heatmaps min/max: {outputs.heatmaps.min().item():.4f} / {outputs.heatmaps.max().item():.4f}")
            print(f"    pose_results structure: {type(pose_results)}, len={len(pose_results)}")
            if pose_results and pose_results[0]:
                p = pose_results[0][0]
                print(f"    first person keys: {p.keys()}")
                print(f"    keypoints shape: {p['keypoints'].shape}, scores shape: {p['scores'].shape}")
                print(f"    keypoints[:3]: {p['keypoints'][:3]}")
                print(f"    scores[:3]: {p['scores'][:3]}")
            else:
                print(f"    pose_results[0] is EMPTY: {pose_results}")
            debug_printed = True

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


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint_root", default="./main_runs_fr")
    parser.add_argument("--depths", type=int, nargs="+", default=[3, 4, 6, 8, 10])
    parser.add_argument("--teacher", action="store_true")
    parser.add_argument("--val_data_root", default="./dataset/val2017")
    parser.add_argument("--val_ann_file", default="./dataset/annotations/person_keypoints_val2017.json")
    parser.add_argument("--output", default="sanity_check_results.json")
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
    print(f"  {len(samples)} samples\n")

    all_results = {}

    # Teacher
    if args.teacher:
        print(f"{'='*60}")
        print("Evaluating TEACHER (HF pipeline)")
        print(f"{'='*60}")
        t0 = time.time()
        model = VitPoseForPoseEstimation.from_pretrained("usyd-community/vitpose-base-simple").to(device)
        model.eval()
        preds = evaluate_with_hf_pipeline(model, processor, samples, args.val_data_root, device)
        metrics = run_coco_eval(coco_gt, preds)
        elapsed = time.time() - t0
        metrics["eval_time_seconds"] = round(elapsed, 1)
        all_results["teacher_depth_12"] = metrics
        print(f"  AP={metrics['AP']:.4f}  AP50={metrics['AP50']:.4f}  "
              f"AP75={metrics['AP75']:.4f}  ({elapsed:.1f}s)\n")
        del model
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    # Students
    for depth in args.depths:
        ckpt_path = f"{args.checkpoint_root}/depth_{depth}/best.pt"
        import os
        if not os.path.exists(ckpt_path):
            print(f"[depth {depth}] Checkpoint not found, skipping.")
            all_results[f"depth_{depth}"] = {"error": "not found"}
            continue

        print(f"{'='*60}")
        print(f"Evaluating depth={depth} (HF pipeline)")
        print(f"{'='*60}")
        t0 = time.time()
        model = load_student(depth, ckpt_path, device)
        preds = evaluate_with_hf_pipeline(model, processor, samples, args.val_data_root, device)
        metrics = run_coco_eval(coco_gt, preds)
        elapsed = time.time() - t0
        metrics["eval_time_seconds"] = round(elapsed, 1)
        all_results[f"depth_{depth}"] = metrics
        print(f"  AP={metrics['AP']:.4f}  AP50={metrics['AP50']:.4f}  "
              f"AP75={metrics['AP75']:.4f}  ({elapsed:.1f}s)\n")
        del model
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    # Save
    with open(args.output, "w") as f:
        json.dump(all_results, f, indent=2)
    print(f"\nResults saved to {args.output}")

    # Summary
    print(f"\n{'Model':>12} {'AP':>8} {'AP50':>8} {'AP75':>8} {'APM':>8} {'APL':>8} {'AR':>8}")
    print("-" * 68)
    for key in all_results:
        m = all_results[key]
        if "error" in m:
            print(f"{key:>12} {'SKIP':>8}")
        else:
            print(f"{key:>12} {m['AP']:>8.4f} {m['AP50']:>8.4f} {m['AP75']:>8.4f} "
                  f"{m['APM']:>8.4f} {m['APL']:>8.4f} {m['AR']:>8.4f}")


if __name__ == "__main__":
    main()
