"""
evaluate_ap.py - Evaluate COCO keypoint AP for distilled ViTPose student models.

Usage:
    python evaluate_ap.py
    python evaluate_ap.py --teacher                          # include teacher baseline
    python evaluate_ap.py --teacher --depths 3 4 6 8 10
    python evaluate_ap.py --checkpoint_root ./main_runs_fr --output ap_results.json
"""

import argparse
import json
import os
import time
from pathlib import Path

import cv2
import numpy as np
import torch
from pycocotools.coco import COCO
from pycocotools.cocoeval import COCOeval
from tqdm import tqdm
from transformers import VitPoseForPoseEstimation

from config import LAYER_MAPPINGS
from Student import createStudent
from COCOPoseDataset import (
    COCOPoseDataset,
    Compose,
    TopDownAffine,
    ToTensor,
    get_affine_transform,
)

# ── Constants ────────────────────────────────────────────────────────────────

IMAGE_SIZE = (192, 256)   # (w, h)
HEATMAP_SIZE = (48, 64)   # (w, h)

# Depth 2 isn't in the default LAYER_MAPPINGS; add a fallback.
EXTRA_LAYER_MAPPINGS: dict[int, dict[int, int]] = {
    2: {0: 0, 1: 11},
}


# ── Helpers ──────────────────────────────────────────────────────────────────

def get_layer_mapping(depth: int) -> dict[int, int]:
    if depth in LAYER_MAPPINGS:
        return LAYER_MAPPINGS[depth]
    if depth in EXTRA_LAYER_MAPPINGS:
        return EXTRA_LAYER_MAPPINGS[depth]
    raise ValueError(f"No layer mapping defined for depth {depth}")


def load_student(depth: int, checkpoint_path: str, device: torch.device):
    """Instantiate a student and load trained weights from a checkpoint."""
    layer_mapping = get_layer_mapping(depth)

    # Need teacher weights only for architecture construction; they get overwritten.
    teacher = VitPoseForPoseEstimation.from_pretrained(
        "usyd-community/vitpose-base-simple"
    )
    student = createStudent(depth, layer_mapping, teacher.state_dict())
    del teacher

    ckpt = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    student.load_state_dict(ckpt["model_state_dict"])
    student = student.to(device)
    student.eval()
    return student


def decode_heatmaps_to_coords(heatmaps: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """
    Decode (B, K, H, W) heatmaps → (B, K, 2) coords in heatmap space + (B, K) scores.
    Uses argmax with quarter-offset shift (standard top-down approach).
    """
    B, K, H, W = heatmaps.shape
    coords = np.zeros((B, K, 2), dtype=np.float64)
    scores = np.zeros((B, K), dtype=np.float64)

    for b in range(B):
        for k in range(K):
            hm = heatmaps[b, k]
            idx = np.argmax(hm)
            py, px = np.unravel_index(idx, (H, W))
            score = float(hm[py, px])

            # Quarter-offset refinement
            px_f, py_f = float(px), float(py)
            if 1 < px < W - 1:
                diff_x = hm[py, px + 1] - hm[py, px - 1]
                px_f += 0.25 * np.sign(diff_x)
            if 1 < py < H - 1:
                diff_y = hm[int(py) + 1, px] - hm[int(py) - 1, px]
                py_f += 0.25 * np.sign(diff_y)

            coords[b, k] = [px_f, py_f]
            scores[b, k] = score

    return coords, scores


def transform_preds_to_original(
    coords: np.ndarray,
    centers: np.ndarray,
    scales: np.ndarray,
    heatmap_size: tuple[int, int] = HEATMAP_SIZE,
    image_size: tuple[int, int] = IMAGE_SIZE,
) -> np.ndarray:
    """
    Map predicted heatmap-space coords back to original image coordinates.

    coords:  (B, K, 2) in heatmap pixel space
    centers: (B, 2)
    scales:  (B, 2)

    Returns (B, K, 2) in original image space.
    """
    hm_w, hm_h = heatmap_size
    img_w, img_h = image_size
    B, K, _ = coords.shape
    out = np.zeros_like(coords)

    for b in range(B):
        # First: heatmap coords → input-crop coords (using UDP mapping)
        crop_coords = coords[b].copy()
        crop_coords[:, 0] = crop_coords[:, 0] * img_w / hm_w
        crop_coords[:, 1] = crop_coords[:, 1] * img_h / hm_h

        # Then: inverse affine from crop coords → original image coords
        trans_inv = get_affine_transform(
            center=centers[b],
            scale=scales[b],
            rot=0.0,
            output_size=image_size,
            inv=True,
            use_udp=True,
        )
        for k in range(K):
            pt = np.array([crop_coords[k, 0], crop_coords[k, 1], 1.0])
            out[b, k] = (trans_inv @ pt)[:2]

    return out


# ── Evaluation dataset (no heatmap target needed) ───────────────────────────

class COCOPoseEvalDataset(COCOPoseDataset):
    """
    Thin wrapper that also returns metadata needed for AP evaluation:
    image_id, ann_id, center, scale, bbox, area, and original score.
    """

    def __init__(self, data_root: str, ann_file: str):
        eval_transforms = Compose([
            TopDownAffine(image_size=IMAGE_SIZE, use_udp=True),
            ToTensor(),
        ])
        super().__init__(
            data_root=data_root,
            ann_file=ann_file,
            image_size=IMAGE_SIZE,
            heatmap_size=HEATMAP_SIZE,
            transforms=eval_transforms,
            min_keypoints=1,
        )

    def __getitem__(self, idx):
        sample = self.samples[idx]

        img_path = self.data_root / sample["image_file"]
        img = cv2.imread(str(img_path))
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

        center, scale = self._bbox_to_center_scale(sample["bbox"])

        results = {
            "img": img,
            "joints": sample["keypoints"].copy(),
            "center": center,
            "scale": scale,
            "rotation": 0.0,
            "img_width": sample["img_width"],
            "img_height": sample["img_height"],
            "aspect_ratio": self.aspect_ratio,
            "bbox": sample["bbox"].copy(),
            "image_file": sample["image_file"],
        }
        results = self.transforms(results)

        bbox = sample["bbox"]
        area = float(bbox[2] * bbox[3])

        return {
            "img": results["img"],
            "center": center.astype(np.float32),
            "scale": scale.astype(np.float32),
            "image_id": sample["image_id"],
            "ann_id": sample["ann_id"],
            "bbox": bbox,
            "area": area,
        }


def numpy_collate(batch):
    """Custom collate that keeps center/scale as numpy but stacks images."""
    elem = batch[0]
    out = {}
    for key in elem:
        vals = [d[key] for d in batch]
        if key == "img":
            out[key] = torch.stack(vals)
        elif key in ("center", "scale", "bbox"):
            out[key] = np.stack(vals)
        elif key in ("image_id", "ann_id"):
            out[key] = vals
        elif key == "area":
            out[key] = vals
        else:
            out[key] = vals
    return out


# ── Main evaluation logic ───────────────────────────────────────────────────

@torch.no_grad()
def evaluate_model(
    model,
    dataloader,
    coco_gt: COCO,
    device: torch.device,
) -> dict:
    """
    Run model on entire val set and compute COCO keypoint AP.
    Returns dict with AP, AP50, AP75, APM, APL, AR, etc.
    """
    model.eval()
    results = []

    for batch in tqdm(dataloader, desc="  Inference"):
        images = batch["img"].to(device)
        centers = batch["center"]   # (B, 2) numpy
        scales = batch["scale"]     # (B, 2) numpy
        image_ids = batch["image_id"]
        ann_ids = batch["ann_id"]
        areas = batch["area"]
        bboxes = batch["bbox"]

        out = model(images)
        heatmaps = out.heatmaps.cpu().numpy()  # (B, K, H, W)

        coords, scores = decode_heatmaps_to_coords(heatmaps)  # heatmap space
        orig_coords = transform_preds_to_original(coords, centers, scales)  # image space

        B, K, _ = orig_coords.shape
        for b in range(B):
            keypoints = np.zeros((K, 3), dtype=np.float64)
            keypoints[:, :2] = orig_coords[b]
            keypoints[:, 2] = scores[b]  # per-keypoint score

            mean_score = float(np.mean(scores[b]))
            bbox = bboxes[b]

            results.append({
                "image_id": int(image_ids[b]),
                "category_id": 1,
                "keypoints": keypoints.flatten().tolist(),
                "score": mean_score,
            })

    if not results:
        print("  WARNING: no predictions generated.")
        return {}

    # Run COCO evaluation
    coco_dt = coco_gt.loadRes(results)
    coco_eval = COCOeval(coco_gt, coco_dt, "keypoints")
    coco_eval.evaluate()
    coco_eval.accumulate()
    coco_eval.summarize()

    stat_names = [
        "AP", "AP50", "AP75", "APM", "APL",
        "AR", "AR50", "AR75", "ARM", "ARL",
    ]
    metrics = {name: float(val) for name, val in zip(stat_names, coco_eval.stats)}
    return metrics


# ── Entry point ──────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="Evaluate COCO keypoint AP for distilled ViTPose students"
    )
    parser.add_argument(
        "--checkpoint_root",
        type=str,
        default="./main_runs_fr",
        help="Root dir containing depth_N/best.pt checkpoints",
    )
    parser.add_argument(
        "--depths",
        type=int,
        nargs="+",
        default=[2, 3, 4, 6, 8, 10],
        help="Depths to evaluate",
    )
    parser.add_argument(
        "--val_data_root",
        type=str,
        default="./dataset/val2017",
    )
    parser.add_argument(
        "--val_ann_file",
        type=str,
        default="./dataset/annotations/person_keypoints_val2017.json",
    )
    parser.add_argument(
        "--batch_size",
        type=int,
        default=64,
    )
    parser.add_argument(
        "--num_workers",
        type=int,
        default=4,
    )
    parser.add_argument(
        "--teacher",
        action="store_true",
        help="Also evaluate the teacher (vitpose-base-simple, 12 layers)",
    )
    parser.add_argument(
        "--output",
        type=str,
        default="ap_results.json",
        help="Output JSON file for results",
    )
    args = parser.parse_args()

    # Device
    if torch.cuda.is_available():
        device = torch.device("cuda")
    elif torch.backends.mps.is_available():
        device = torch.device("mps")
    else:
        device = torch.device("cpu")
    print(f"Device: {device}")

    # Dataset & COCO ground truth (shared across all models)
    print("Loading validation data...")
    dataset = COCOPoseEvalDataset(args.val_data_root, args.val_ann_file)
    dataloader = torch.utils.data.DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=True,
        collate_fn=numpy_collate,
    )
    coco_gt = COCO(args.val_ann_file)
    print(f"  {len(dataset)} samples\n")

    all_results = {}

    # ── Evaluate teacher (full 12-layer vitpose-base) ────────────────────
    if args.teacher:
        print(f"{'='*60}")
        print("Evaluating TEACHER (vitpose-base-simple, depth=12)")
        print(f"{'='*60}")

        t0 = time.time()
        teacher = VitPoseForPoseEstimation.from_pretrained(
            "usyd-community/vitpose-base-simple"
        )
        teacher = teacher.to(device)
        teacher.eval()

        metrics = evaluate_model(teacher, dataloader, coco_gt, device)
        elapsed = time.time() - t0

        metrics["eval_time_seconds"] = round(elapsed, 1)
        all_results["teacher_depth_12"] = metrics

        print(f"  AP={metrics.get('AP', 0):.4f}  AP50={metrics.get('AP50', 0):.4f}  "
              f"AP75={metrics.get('AP75', 0):.4f}  AR={metrics.get('AR', 0):.4f}  "
              f"({elapsed:.1f}s)\n")

        del teacher
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    # ── Evaluate students ────────────────────────────────────────────────
    for depth in args.depths:
        ckpt_path = os.path.join(args.checkpoint_root, f"depth_{depth}", "best.pt")
        if not os.path.exists(ckpt_path):
            print(f"[depth {depth}] Checkpoint not found at {ckpt_path}, skipping.")
            all_results[f"depth_{depth}"] = {"error": "checkpoint not found"}
            continue

        print(f"{'='*60}")
        print(f"Evaluating depth={depth}  ({ckpt_path})")
        print(f"{'='*60}")

        t0 = time.time()
        model = load_student(depth, ckpt_path, device)
        metrics = evaluate_model(model, dataloader, coco_gt, device)
        elapsed = time.time() - t0

        metrics["eval_time_seconds"] = round(elapsed, 1)
        all_results[f"depth_{depth}"] = metrics

        print(f"  AP={metrics.get('AP', 0):.4f}  AP50={metrics.get('AP50', 0):.4f}  "
              f"AP75={metrics.get('AP75', 0):.4f}  AR={metrics.get('AR', 0):.4f}  "
              f"({elapsed:.1f}s)\n")

        # Free GPU memory
        del model
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    # Save results
    with open(args.output, "w") as f:
        json.dump(all_results, f, indent=2)
    print(f"\nResults saved to {args.output}")

    # Print summary table
    print(f"\n{'Model':>12} {'AP':>8} {'AP50':>8} {'AP75':>8} {'APM':>8} {'APL':>8} {'AR':>8}")
    print("-" * 68)
    if "teacher_depth_12" in all_results:
        m = all_results["teacher_depth_12"]
        print(f"{'teacher(12)':>12} {m['AP']:>8.4f} {m['AP50']:>8.4f} {m['AP75']:>8.4f} "
              f"{m['APM']:>8.4f} {m['APL']:>8.4f} {m['AR']:>8.4f}")
    for depth in args.depths:
        key = f"depth_{depth}"
        m = all_results.get(key, {})
        if "error" in m:
            print(f"{'depth_'+str(depth):>12} {'SKIP':>8}")
        else:
            print(f"{'depth_'+str(depth):>12} {m['AP']:>8.4f} {m['AP50']:>8.4f} {m['AP75']:>8.4f} "
                  f"{m['APM']:>8.4f} {m['APL']:>8.4f} {m['AR']:>8.4f}")


if __name__ == "__main__":
    main()
