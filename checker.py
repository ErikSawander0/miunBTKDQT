"""
file: ./inference.py
Quick inference test comparing student and teacher models side by side
"""
import json
import torch
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
from transformers import VitPoseForPoseEstimation, AutoImageProcessor
from Student import createStudent

COCO_SKELETON = [
    (0, 1), (0, 2), (1, 3), (2, 4),
    (5, 6), (5, 7), (7, 9), (6, 8), (8, 10),
    (5, 11), (6, 12), (11, 12),
    (11, 13), (13, 15), (12, 14), (14, 16),
]

def load_student_from_checkpoint(checkpoint_path, device):
    checkpoint = torch.load(checkpoint_path, map_location=device)
    
    depth = checkpoint['config']['depth']
    layer_mapping = checkpoint['config']['layer_mapping']
    
    teacher = VitPoseForPoseEstimation.from_pretrained("usyd-community/vitpose-base-simple")
    student = createStudent(depth, layer_mapping, teacher.state_dict())
    
    student.load_state_dict(checkpoint['model_state_dict'])
    student = student.to(device)
    student.eval()
    
    return student

def load_teacher(device):
    teacher = VitPoseForPoseEstimation.from_pretrained("usyd-community/vitpose-base-simple")
    teacher = teacher.to(device)
    teacher.eval()
    return teacher

def draw_pose_on_ax(ax, img, keypoints, scores, title, threshold=0.0):
    ax.imshow(img)
    
    for i, j in COCO_SKELETON:
        if scores[i] > threshold and scores[j] > threshold:
            ax.plot(
                [keypoints[i][0], keypoints[j][0]],
                [keypoints[i][1], keypoints[j][1]],
                'g-', linewidth=2
            )
    
    for kpt, score in zip(keypoints, scores):
        if score > threshold:
            ax.plot(kpt[0], kpt[1], 'ro', markersize=5)
    
    ax.set_title(title, fontsize=14, fontweight='bold')
    ax.axis('off')

def run_inference(checkpoint_path, sample_idx=0, save_path='inference_output.png'):
    device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
    print(f"Using device: {device}")
    
    # Load both models
    print("Loading student model...")
    student = load_student_from_checkpoint(checkpoint_path, device)
    
    print("Loading teacher model (ViTPose-B)...")
    teacher = load_teacher(device)
    
    processor = AutoImageProcessor.from_pretrained("usyd-community/vitpose-base-simple")
    
    # Load annotation to get real bbox
    with open('./dataset/annotations/person_keypoints_val2017.json') as f:
        coco = json.load(f)
    
    images = {img['id']: img for img in coco['images']}
    ann = coco['annotations'][sample_idx]
    img_info = images[ann['image_id']]
    
    bbox = ann['bbox']  # [x, y, w, h]
    image_path = f"./dataset/val2017/{img_info['file_name']}"
    image = Image.open(image_path).convert("RGB")
    print(f"Loaded image: {img_info['file_name']}")
    
    # Prepare inputs
    boxes = [[bbox]]
    inputs = processor(image, boxes=boxes, return_tensors="pt")
    inputs = {k: v.to(device) for k, v in inputs.items()}
    
    # Run inference on both models
    with torch.no_grad():
        print("Running student inference...")
        student_output = student(**inputs)
        
        print("Running teacher inference...")
        teacher_output = teacher(**inputs)
    
    # Post-process results
    student_results = processor.post_process_pose_estimation(student_output, boxes=boxes)
    teacher_results = processor.post_process_pose_estimation(teacher_output, boxes=boxes)
    
    student_keypoints = student_results[0][0]['keypoints'].cpu().numpy()
    student_scores = student_results[0][0]['scores'].cpu().numpy()
    
    teacher_keypoints = teacher_results[0][0]['keypoints'].cpu().numpy()
    teacher_scores = teacher_results[0][0]['scores'].cpu().numpy()
    
    # Create side-by-side comparison
    fig, axes = plt.subplots(1, 2, figsize=(16, 10))
    
    draw_pose_on_ax(axes[0], image, student_keypoints, student_scores, 
                    "Student (Distilled)", threshold=0.1)
    draw_pose_on_ax(axes[1], image, teacher_keypoints, teacher_scores, 
                    "Teacher (ViTPose-B)", threshold=0.1)
    
    plt.tight_layout()
    fig.savefig(save_path, bbox_inches='tight', dpi=150)
    print(f"Saved comparison to: {save_path}")
    plt.close()

if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--checkpoint', default='checkpoints/best_depth3.pt')
    parser.add_argument('--sample-idx', type=int, default=0)
    parser.add_argument('--save-path', default='inference_output.png')
    args = parser.parse_args()
    
    run_inference(args.checkpoint, args.sample_idx, args.save_path)
