"""
file: ./train.py
Basic distillation training loop
"""
import torch
import torch.optim as optim
from transformers import VitPoseForPoseEstimation
from tqdm import tqdm

from COCOPoseDataset import build_train_dataloader, build_val_dataloader
from Student import createStudent
from FeatureExtractor import FeatureExtractor
from loss import task_loss, output_distill_loss, feature_distill_loss

import time

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
        
        # Clear stored features
        teacher_extractor.clear()
        student_extractor.clear()
        
        # Teacher forward (no grad)
        with torch.no_grad():
            teacher_out = teacher(images)
        
        # Student forward
        student_out = student(images)
        
        # Compute losses
        l_task = task_loss(student_out.heatmaps, gt_heatmaps)
        l_distill = output_distill_loss(student_out.heatmaps, teacher_out.heatmaps)
        l_feature = feature_distill_loss(
            student_extractor.features,
            teacher_extractor.features,
            layer_mapping
        )
        
        loss = l_task + alpha * l_distill + beta * l_feature
        
        # Backward
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        # Track losses
        total_task += l_task.item()
        total_distill += l_distill.item()
        total_feature += l_feature.item() 
        total_loss += loss.item()
        num_batches += 1
        
        pbar.set_postfix({
            'loss': f'{loss.item():.4f}',
            'task': f'{l_task.item():.4f}',
            'dist': f'{l_distill.item():.4f}',
            'feat': f'{l_feature.item():.4f}',
        })
    
    return {
        'loss': total_loss / num_batches,
        'task_loss': total_task / num_batches,
        'distill_loss': total_distill / num_batches,
        'feature_loss': total_feature / num_batches,
    }


@torch.no_grad()
def validate(student, val_loader, device):
    student.eval()
    
    total_loss = 0
    num_batches = 0
    
    for batch in tqdm(val_loader, desc="Validating"):
        images = batch['img'].to(device)
        gt_heatmaps = batch['target'].to(device)
        
        student_out = student(images)
        loss = task_loss(student_out.heatmaps, gt_heatmaps)
        
        total_loss += loss.item()
        num_batches += 1
    
    return {'val_loss': total_loss / num_batches}


def main():
    # Config - hardcoded for now, pull out to config file later
    device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
    print(f"Using device: {device}")
    
    # Hyperparams
    depth = 3
    layer_mapping = {0: 0, 1: 4, 2: 11}
    batch_size = 64
    lr = 1e-4
    epochs = 10
    alpha = 1.0  # output distillation weight
    beta = 0.5   # feature distillation weight
    
    # Data
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
    print(f"Train samples: {len(train_loader.dataset)}, Val samples: {len(val_loader.dataset)}")
    
    # Models
    print("Loading teacher...")
    teacher = VitPoseForPoseEstimation.from_pretrained("usyd-community/vitpose-base-simple")
    teacher = teacher.to(device)
    teacher.eval()
    for p in teacher.parameters():
        p.requires_grad = False
    
    print(f"Creating student (depth={depth})...")
    student = createStudent(depth, layer_mapping, teacher.state_dict())
    student = student.to(device)
    
    # Feature extractors
    teacher_layers = list(layer_mapping.values())
    student_layers = list(layer_mapping.keys())
    teacher_extractor = FeatureExtractor(teacher, teacher_layers)
    student_extractor = FeatureExtractor(student, student_layers)
    
    # Optimizer
    optimizer = optim.AdamW(student.parameters(), lr=lr)
    
    # Training loop
    best_val_loss = float('inf')
    
    for epoch in range(epochs):
        print(f"\n{'='*50}")
        print(f"Epoch {epoch + 1}/{epochs}")
        print(f"{'='*50}")
        
        # Train
        train_metrics = train_one_epoch(
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
        )
        print(f"Train - loss: {train_metrics['loss']:.4f}, "
              f"task: {train_metrics['task_loss']:.4f}, "
              f"distill: {train_metrics['distill_loss']:.4f}, "
              f"feature: {train_metrics['feature_loss']:.4f}")
        
        # Validate
        val_metrics = validate(student, val_loader, device)
        print(f"Val - loss: {val_metrics['val_loss']:.4f}")
        
        elapsed = time.time() - start_time
        print(f"Epoch {epoch + 1} complete - elapsed: {elapsed/60:.1f} min")
        # Save best
        if val_metrics['val_loss'] < best_val_loss:
            best_val_loss = val_metrics['val_loss']
            torch.save({
                'epoch': epoch,
                'model_state_dict': student.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'val_loss': best_val_loss,
                'config': {
                    'depth': depth,
                    'layer_mapping': layer_mapping,
                }
            }, f'checkpoints/best_depth{depth}.pt')
            print(f"Saved new best model (val_loss: {best_val_loss:.4f})")
    
    # Cleanup
    teacher_extractor.remove()
    student_extractor.remove()
    
    total_time = time.time() - start_time
    print(f"\nTotal training time: {total_time/60:.1f} min ({total_time/3600:.2f} hours)")
    print("\nDone!")


if __name__ == '__main__':
    import os

    start_time = time.time()


    os.makedirs('checkpoints', exist_ok=True)
    main()
