"""
file: ./loss.py
"""
import torch.nn.functional as F

def task_loss(student_heatmaps, gt_heatmaps):
    return F.mse_loss(student_heatmaps, gt_heatmaps)

def output_distill_loss(student_heatmaps, teacher_heatmaps):
    return F.mse_loss(student_heatmaps, teacher_heatmaps)

def feature_distill_loss(student_features, teacher_features, layer_mapping):
    loss = 0
    for s_idx, t_idx in layer_mapping.items():
        loss += F.mse_loss(student_features[s_idx], teacher_features[t_idx])
    return loss / len(layer_mapping)  # average over layers
