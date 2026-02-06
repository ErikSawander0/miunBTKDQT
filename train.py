"""
train.py - Train a single ViTPose student model via distillation.

Usage:
    python train.py --config configs/depth_6.json
    python train.py --depth 6  # uses defaults
"""

import argparse
import json
import os
import time
from pathlib import Path

import torch
import torch.optim as optim
from transformers import VitPoseForPoseEstimation
from tqdm import tqdm
import numpy as np
import random

from config import TrainConfig
from COCOPoseDataset import build_train_dataloader, build_val_dataloader
from Student import createStudent
from FeatureExtractor import FeatureExtractor
from loss import task_loss, output_distill_loss, feature_distill_loss


def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    if torch.backends.mps.is_available():
        torch.mps.manual_seed(seed)


def log_metrics(path: str, metrics: dict):
    """Append metrics as JSON line to file."""
    with open(path, 'a') as f:
        f.write(json.dumps(metrics) + '\n')


def save_checkpoint(path: str, epoch: int, model, optimizer, best_val_loss: float):
    """Save training checkpoint."""
    torch.save({
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'best_val_loss': best_val_loss,
    }, path)


def load_checkpoint(path: str, model, optimizer):
    """Load training checkpoint. Returns (start_epoch, best_val_loss)."""
    checkpoint = torch.load(path, weights_only=False)
    model.load_state_dict(checkpoint['model_state_dict'])
    optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    return checkpoint['epoch'], checkpoint['best_val_loss']


def train_one_epoch(
    teacher,
    student,
    teacher_extractor,
    student_extractor,
    train_loader,
    optimizer,
    device,
    layer_mapping: dict[int, int],
    alpha: float,
    beta: float,
) -> dict:
    """Train for one epoch. Returns metrics dict."""
    student.train()
    teacher.eval()

    total_task = 0.0
    total_distill = 0.0
    total_feature = 0.0
    total_loss = 0.0
    total_grad_norm = 0.0
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
        
        grad_norm = torch.nn.utils.clip_grad_norm_(student.parameters(), max_norm=float('inf'))
        
        optimizer.step()

        total_task += l_task.item()
        total_distill += l_distill.item()
        total_feature += l_feature.item()  # type: ignore
        total_loss += loss.item()
        total_grad_norm += grad_norm.item()
        num_batches += 1

        pbar.set_postfix({
            'loss': f'{loss.item():.4f}',
            'task': f'{l_task.item():.4f}',
            'dist': f'{l_distill.item():.4f}',
            'feat': f'{l_feature.item():.4f}',  # type: ignore
        })

    return {
        'train_loss': total_loss / num_batches,
        'train_task_loss': total_task / num_batches,
        'train_distill_loss': total_distill / num_batches,
        'train_feature_loss': total_feature / num_batches,
        'train_grad_norm': total_grad_norm / num_batches,
    }


@torch.no_grad()
def validate(
    teacher,
    student,
    teacher_extractor,
    student_extractor,
    val_loader,
    device,
    layer_mapping: dict[int, int],
    alpha: float,
    beta: float,
) -> dict:
    """Validate model. Returns metrics dict."""
    student.eval()
    teacher.eval()

    total_task = 0.0
    total_distill = 0.0
    total_feature = 0.0
    total_loss = 0.0
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


def train(config: TrainConfig, resume: bool = False):
    """Main training loop."""
    
    os.makedirs(config.run_dir, exist_ok=True)
    os.makedirs(config.checkpoint_dir, exist_ok=True)
    
    config.save()
    
    if torch.cuda.is_available():
        device = torch.device("cuda")
    elif torch.backends.mps.is_available():
        device = torch.device("mps")
    else:
        device = torch.device("cpu")
    print(f"Using device: {device}")
    
    set_seed(config.seed)
    
    print("Loading data...")
    train_loader = build_train_dataloader(
        data_root=config.train_data_root,
        ann_file=config.train_ann_file,
        batch_size=config.batch_size,
        num_workers=4,
    )
    val_loader = build_val_dataloader(
        data_root=config.val_data_root,
        ann_file=config.val_ann_file,
        batch_size=config.batch_size,
        num_workers=4,
    )
    
    print("Loading teacher...")
    teacher = VitPoseForPoseEstimation.from_pretrained("usyd-community/vitpose-base-simple")
    teacher = teacher.to(device)  # type: ignore
    teacher.eval()
    for p in teacher.parameters():
        p.requires_grad = False
    
    # Student
    print(f"Creating student (depth={config.depth})...")
    student = createStudent(config.depth, config.layer_mapping, teacher.state_dict())
    student = student.to(device)  # type: ignore
    
    teacher_layers = list(config.layer_mapping.values())
    student_layers = list(config.layer_mapping.keys())
    teacher_extractor = FeatureExtractor(teacher, teacher_layers)
    student_extractor = FeatureExtractor(student, student_layers)
    
    optimizer = optim.AdamW(student.parameters(), lr=config.lr)
    
    start_epoch = 0
    best_val_loss = float('inf')
    
    if resume:
        latest_ckpt = Path(config.checkpoint_dir) / 'latest.pt'
        if latest_ckpt.exists():
            print(f"Resuming from {latest_ckpt}...")
            start_epoch, best_val_loss = load_checkpoint(str(latest_ckpt), student, optimizer)
            start_epoch += 1  
            print(f"Resuming from epoch {start_epoch}, best_val_loss={best_val_loss:.4f}")
    
    print(f"\nStarting training for {config.epochs} epochs...")
    print(f"Logging to: {config.metrics_file}")
    print(f"Checkpoints: {config.checkpoint_dir}")
    
    for epoch in range(start_epoch, config.epochs):
        epoch_start = time.time()
        
        print(f"\n{'='*60}")
        print(f"Epoch {epoch + 1}/{config.epochs}")
        print(f"{'='*60}")
        
        train_metrics = train_one_epoch(
            teacher=teacher,
            student=student,
            teacher_extractor=teacher_extractor,
            student_extractor=student_extractor,
            train_loader=train_loader,
            optimizer=optimizer,
            device=device,
            layer_mapping=config.layer_mapping,
            alpha=config.alpha,
            beta=config.beta,
        )
        
        val_metrics = validate(
            teacher=teacher,
            student=student,
            teacher_extractor=teacher_extractor,
            student_extractor=student_extractor,
            val_loader=val_loader,
            device=device,
            layer_mapping=config.layer_mapping,
            alpha=config.alpha,
            beta=config.beta,
        )
        
        epoch_time = time.time() - epoch_start
        
        is_best = val_metrics['val_loss'] < best_val_loss
        if is_best:
            best_val_loss = val_metrics['val_loss']
        
        metrics = {
            'epoch': epoch + 1,
            'lr': config.lr,
            **train_metrics,
            **val_metrics,
            'best_val_loss': best_val_loss,
            'epoch_time_seconds': epoch_time,
        }
        log_metrics(config.metrics_file, metrics)
        
        print(f"Train - loss: {train_metrics['train_loss']:.4f}, "
              f"task: {train_metrics['train_task_loss']:.4f}, "
              f"distill: {train_metrics['train_distill_loss']:.4f}, "
              f"feature: {train_metrics['train_feature_loss']:.4f}")
        print(f"Val   - loss: {val_metrics['val_loss']:.4f}, "
              f"task: {val_metrics['val_task_loss']:.4f}, "
              f"distill: {val_metrics['val_distill_loss']:.4f}, "
              f"feature: {val_metrics['val_feature_loss']:.4f}")
        print(f"Time: {epoch_time:.1f}s | Best val loss: {best_val_loss:.4f}" + 
              (" *" if is_best else ""))
        
        # Checkpointing
        save_checkpoint(
            f"{config.checkpoint_dir}/latest.pt",
            epoch, student, optimizer, best_val_loss
        )
        
        if is_best:
            save_checkpoint(
                f"{config.checkpoint_dir}/best.pt",
                epoch, student, optimizer, best_val_loss
            )
        
        if (epoch + 1) % config.checkpoint_every == 0:
            save_checkpoint(
                f"{config.checkpoint_dir}/epoch_{epoch + 1}.pt",
                epoch, student, optimizer, best_val_loss
            )
    
    teacher_extractor.remove()
    student_extractor.remove()
    
    print(f"\nTraining complete! Best val loss: {best_val_loss:.4f}")
    print(f"Results saved to: {config.run_dir}")


def main():
    parser = argparse.ArgumentParser(description='Train ViTPose student via distillation')
    parser.add_argument('config', type=str, help='Path to config JSON file')
    parser.add_argument('--resume', action='store_true', help='Resume from latest checkpoint')
    args = parser.parse_args()
    
    config = TrainConfig.load(args.config)
    train(config, resume=args.resume)


if __name__ == '__main__':
    main()
