"""
file: ./lr_sweep.py
LR sweep for depth-10 student distillation
"""

import torch
import torch.optim as optim
from transformers import VitPoseForPoseEstimation
from tqdm import tqdm
import json
import os

from COCOPoseDataset import build_train_dataloader, build_val_dataloader
from Student import createStudent
from FeatureExtractor import FeatureExtractor
from loss import task_loss, output_distill_loss, feature_distill_loss
import numpy as np
import random

SEED = 42


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    if torch.backends.mps.is_available():
        torch.mps.manual_seed(seed)


def train_one_epoch(
    teacher,
    student,
    teacher_extractor,
    student_extractor,
    train_loader,
    optimizer,
    device,
    layer_mapping,
    alpha=1.0,
    beta=0.5,
    step_log=None,
    global_step=0,
):
    student.train()
    teacher.eval()

    total_task = 0
    total_distill = 0
    total_feature = 0
    total_loss = 0
    num_batches = 0

    pbar = tqdm(train_loader, desc="Training")
    for batch in pbar:
        images = batch['img'].to(device)
        gt_heatmaps = batch['target'].to(device)

        teacher_extractor.clear()
        student_extractor.clear()

        with torch.no_grad():
            teacher_out = teacher(images)

        student_out = student(images)

        l_task = task_loss(student_out.heatmaps, gt_heatmaps)
        l_distill = output_distill_loss(student_out.heatmaps, teacher_out.heatmaps)
        l_feature = feature_distill_loss(
            student_extractor.features,
            teacher_extractor.features,
            layer_mapping
        )

        loss = l_task + alpha * l_distill + beta * l_feature

        optimizer.zero_grad()
        loss.backward()

        # Gradient norm before stepping
        grad_norm = torch.nn.utils.clip_grad_norm_(student.parameters(), max_norm=float('inf'))

        optimizer.step()

        total_task += l_task.item()
        total_distill += l_distill.item()
        total_feature += l_feature.item()  # type: ignore
        total_loss += loss.item()
        num_batches += 1
        global_step += 1

        # Per-step logging
        if step_log is not None:
            step_log.append({
                'step': global_step,
                'train_loss': loss.item(),
                'task_loss': l_task.item(),
                'distill_loss': l_distill.item(),
                'feature_loss': l_feature.item(), #type: ignore
                'grad_norm': grad_norm.item(),
            })

        pbar.set_postfix({
            'loss': f'{loss.item():.4f}',
            'task': f'{l_task.item():.4f}',
            'dist': f'{l_distill.item():.4f}',
            'feat': f'{l_feature.item():.4f}',  # type: ignore
            'gnorm': f'{grad_norm.item():.2f}',
        })

    epoch_metrics = {
        'loss': total_loss / num_batches,
        'task_loss': total_task / num_batches,
        'distill_loss': total_distill / num_batches,
        'feature_loss': total_feature / num_batches,
    }
    return epoch_metrics, global_step


@torch.no_grad()
def validate(student, teacher, teacher_extractor, student_extractor,
             val_loader, device, layer_mapping, alpha=1.0, beta=0.5):
    student.eval()
    teacher.eval()

    total_task = 0
    total_distill = 0
    total_feature = 0
    total_loss = 0
    num_batches = 0

    for batch in tqdm(val_loader, desc="Validating"):
        images = batch['img'].to(device)
        gt_heatmaps = batch['target'].to(device)

        teacher_extractor.clear()
        student_extractor.clear()

        teacher_out = teacher(images)
        student_out = student(images)

        l_task = task_loss(student_out.heatmaps, gt_heatmaps)
        l_distill = output_distill_loss(student_out.heatmaps, teacher_out.heatmaps)
        l_feature = feature_distill_loss(
            student_extractor.features,
            teacher_extractor.features,
            layer_mapping
        )
        loss = l_task + alpha * l_distill + beta * l_feature

        total_task += l_task.item()
        total_distill += l_distill.item()
        total_feature += l_feature.item()  # type: ignore
        total_loss += loss.item()
        num_batches += 1

    return {
        'val_loss': total_loss / num_batches,
        'val_task_loss': total_task / num_batches,
        'val_distill_loss': total_distill / num_batches,
        'val_feature_loss': total_feature / num_batches,
    }


def main():
    device = torch.device( "cuda" if torch.cuda.is_available() else "cpu") 
    device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
    print(f"Using device: {device}")

    depth = 10
    layer_mapping = {0: 0,
                     1: 1,
                     2: 2,
                     3: 3,
                     4: 4,
                     5: 5,
                     6: 6,
                     7: 7,
                     8: 9,
                     9: 11}
    lrs = [1e-5, 3e-5, 1e-4, 3e-4, 1e-3, 3e-3]
    alpha = 1.0
    beta = 0.5
    batch_size = 64
    epochs = 1

    os.makedirs('sweep_logs', exist_ok=True)

    print("Loading data...")
    train_loader = build_train_dataloader(
        data_root='./dataset/train2017',
        ann_file='./dataset/annotations/person_keypoints_train2017.json',
        batch_size=batch_size,
        num_workers=4,
    )
    val_loader = build_val_dataloader(
        data_root='./dataset/val2017',
        ann_file='./dataset/annotations/person_keypoints_val2017.json',
        batch_size=batch_size,
        num_workers=4,
    )

    print("Loading teacher...")
    teacher = VitPoseForPoseEstimation.from_pretrained("usyd-community/vitpose-base-simple")
    teacher = teacher.to(device)  # type: ignore
    teacher.eval()
    for p in teacher.parameters():
        p.requires_grad = False

    # Store summary across all LRs
    sweep_summary = []

    for lr in lrs:
        print(f"\n{'#'*60}")
        print(f"  LR = {lr}")
        print(f"{'#'*60}")

        # Reset seed and recreate student from scratch for each LR
        set_seed(SEED)

        student = createStudent(depth, layer_mapping, teacher.state_dict())
        student = student.to(device)  # type: ignore

        teacher_layers = list(layer_mapping.values())
        student_layers = list(layer_mapping.keys())
        teacher_extractor = FeatureExtractor(teacher, teacher_layers)
        student_extractor = FeatureExtractor(student, student_layers)

        optimizer = optim.AdamW(student.parameters(), lr=lr)

        step_log = []
        epoch_log = []
        global_step = 0
        best_val_loss = float('inf')

        for epoch in range(epochs):
            print(f"\n{'='*50}")
            print(f"LR={lr} | Epoch {epoch + 1}/{epochs}")
            print(f"{'='*50}")

            train_metrics, global_step = train_one_epoch(
                teacher=teacher,
                student=student,
                teacher_extractor=teacher_extractor,
                student_extractor=student_extractor,
                train_loader=train_loader,
                optimizer=optimizer,
                device=device,
                layer_mapping=layer_mapping,
                alpha=alpha,
                beta=beta,
                step_log=step_log,
                global_step=global_step,
            )

            val_metrics = validate(
                student=student,
                teacher=teacher,
                teacher_extractor=teacher_extractor,
                student_extractor=student_extractor,
                val_loader=val_loader,
                device=device,
                layer_mapping=layer_mapping,
                alpha=alpha,
                beta=beta,
            )

            best_val_loss = min(best_val_loss, val_metrics['val_loss'])

            epoch_entry = {
                'epoch': epoch + 1,
                **train_metrics,
                **val_metrics,
            }
            epoch_log.append(epoch_entry)

            print(f"Train - loss: {train_metrics['loss']:.4f}, "
                  f"task: {train_metrics['task_loss']:.4f}, "
                  f"distill: {train_metrics['distill_loss']:.4f}, "
                  f"feature: {train_metrics['feature_loss']:.4f}")
            print(f"Val   - loss: {val_metrics['val_loss']:.4f}, "
                  f"task: {val_metrics['val_task_loss']:.4f}, "
                  f"distill: {val_metrics['val_distill_loss']:.4f}, "
                  f"feature: {val_metrics['val_feature_loss']:.4f}")

        teacher_extractor.remove()
        student_extractor.remove()

        # Save per-LR logs
        lr_str = f"{lr:.0e}"
        with open(f'sweep_logs/depth{depth}_lr{lr_str}_steps.json', 'w') as f:
            json.dump(step_log, f)
        with open(f'sweep_logs/depth{depth}_lr{lr_str}_epochs.json', 'w') as f:
            json.dump(epoch_log, f)

        sweep_summary.append({
            'lr': lr,
            'best_val_loss': best_val_loss,
            'final_train_loss': epoch_log[-1]['loss'],
            'final_val_loss': epoch_log[-1]['val_loss'],
        })

    # Save sweep summary
    with open(f'sweep_logs/depth{depth}_summary.json', 'w') as f:
        json.dump(sweep_summary, f, indent=2)

    # Print summary table
    print(f"\n{'='*60}")
    print(f"  SWEEP SUMMARY (depth={depth})")
    print(f"{'='*60}")
    print(f"{'LR':>10} | {'Best Val Loss':>14} | {'Final Train':>12} | {'Final Val':>12}")
    print(f"{'-'*10}-+-{'-'*14}-+-{'-'*12}-+-{'-'*12}")
    for row in sweep_summary:
        print(f"{row['lr']:>10.0e} | {row['best_val_loss']:>14.4f} | "
              f"{row['final_train_loss']:>12.4f} | {row['final_val_loss']:>12.4f}")


if __name__ == '__main__':
    main()
