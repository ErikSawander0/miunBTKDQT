"""
file: ./COCOPoseDataset.py
ViTPose Training Dataset and Transforms

Replicates MMPose training pipeline:
- TopDownRandomFlip
- TopDownHalfBodyTransform  
- TopDownGetRandomScaleRotation
- TopDownAffine (with UDP)
- TopDownGenerateTarget (UDP encoding)

COCO format expected:
./dataset/
    annotations/
        person_keypoints_train2017.json
    train2017/
        *.jpg
"""

import json
import random
from pathlib import Path
from typing import Optional, Tuple, List, Dict, Any

import cv2
import numpy as np
import torch
from torch.utils.data import Dataset


# =============================================================================
# COCO Keypoint Constants
# =============================================================================

COCO_KEYPOINT_NAMES = [
    'nose', 'left_eye', 'right_eye', 'left_ear', 'right_ear',
    'left_shoulder', 'right_shoulder', 'left_elbow', 'right_elbow',
    'left_wrist', 'right_wrist', 'left_hip', 'right_hip',
    'left_knee', 'right_knee', 'left_ankle', 'right_ankle'
]

# Flip pairs: (left_idx, right_idx)
COCO_FLIP_PAIRS = [
    (1, 2),   # eyes
    (3, 4),   # ears
    (5, 6),   # shoulders
    (7, 8),   # elbows
    (9, 10),  # wrists
    (11, 12), # hips
    (13, 14), # knees
    (15, 16), # ankles
]

# Upper body indices (for half-body transform)
COCO_UPPER_BODY_IDS = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
COCO_LOWER_BODY_IDS = [11, 12, 13, 14, 15, 16]


# =============================================================================
# Affine Transform Utilities
# =============================================================================

def get_affine_transform(
    center: np.ndarray,
    scale: np.ndarray,
    rot: float,
    output_size: Tuple[int, int],
    shift: np.ndarray = np.array([0., 0.]),
    inv: bool = False,
    use_udp: bool = True
) -> np.ndarray:
    """
    Get affine transform matrix.
    
    Args:
        center: Center of the bounding box (x, y)
        scale: Scale of the bounding box (w, h) in pixels
        rot: Rotation angle in degrees
        output_size: (width, height) of output
        shift: Shift ratio (0-1) relative to scale
        inv: If True, return inverse transform
        use_udp: Use Unbiased Data Processing (recommended)
    
    Returns:
        2x3 affine transform matrix
    """
    output_w, output_h = output_size
    scale_x, scale_y = scale
    
    shift = np.array(shift)
    src_w = scale_x
    src_h = scale_y
    dst_w = output_w
    dst_h = output_h
    
    rot_rad = np.deg2rad(rot)
    
    if use_udp:
        # UDP: unbiased data processing
        # Maps src corners to dst corners more accurately
        src_dir = _rotate_point(np.array([0., src_h * -0.5]), rot_rad)
        dst_dir = np.array([0., dst_h * -0.5])
        
        src_points = np.zeros((3, 2), dtype=np.float32)
        dst_points = np.zeros((3, 2), dtype=np.float32)
        
        src_points[0, :] = center + scale * shift
        src_points[1, :] = center + src_dir + scale * shift
        src_points[2, :] = _get_3rd_point(src_points[0, :], src_points[1, :])
        
        dst_points[0, :] = np.array([dst_w * 0.5, dst_h * 0.5])
        dst_points[1, :] = np.array([dst_w * 0.5, dst_h * 0.5]) + dst_dir
        dst_points[2, :] = _get_3rd_point(dst_points[0, :], dst_points[1, :])
    else:
        # Standard approach
        src_dir = _rotate_point(np.array([0., src_w * -0.5]), rot_rad)
        dst_dir = np.array([0., dst_w * -0.5])
        
        src_points = np.zeros((3, 2), dtype=np.float32)
        dst_points = np.zeros((3, 2), dtype=np.float32)
        
        src_points[0, :] = center + scale * shift
        src_points[1, :] = center + src_dir + scale * shift
        src_points[2, :] = _get_3rd_point(src_points[0, :], src_points[1, :])
        
        dst_points[0, :] = np.array([dst_w * 0.5, dst_h * 0.5])
        dst_points[1, :] = np.array([dst_w * 0.5, dst_h * 0.5]) + dst_dir
        dst_points[2, :] = _get_3rd_point(dst_points[0, :], dst_points[1, :])
    
    if inv:
        trans = cv2.getAffineTransform(dst_points, src_points)
    else:
        trans = cv2.getAffineTransform(src_points, dst_points)
    
    return trans


def _rotate_point(pt: np.ndarray, angle_rad: float) -> np.ndarray:
    """Rotate a point by angle (in radians)."""
    cos_a = np.cos(angle_rad)
    sin_a = np.sin(angle_rad)
    return np.array([
        pt[0] * cos_a - pt[1] * sin_a,
        pt[0] * sin_a + pt[1] * cos_a
    ])


def _get_3rd_point(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """Get 3rd point for affine transform (perpendicular)."""
    direction = a - b
    return b + np.array([-direction[1], direction[0]])


def affine_transform_point(pt: np.ndarray, trans: np.ndarray) -> np.ndarray:
    """Apply affine transform to a single point."""
    pt_homogeneous = np.array([pt[0], pt[1], 1.])
    new_pt = trans @ pt_homogeneous
    return new_pt[:2]


# =============================================================================
# Transform Classes
# =============================================================================

class TopDownRandomFlip:
    """Random horizontal flip with keypoint swapping."""
    
    def __init__(self, flip_prob: float = 0.5, flip_pairs: List[Tuple[int, int]] = None):
        self.flip_prob = flip_prob
        self.flip_pairs = flip_pairs or COCO_FLIP_PAIRS
    
    def __call__(self, results: Dict[str, Any]) -> Dict[str, Any]:
        if random.random() > self.flip_prob:
            results['flipped'] = False
            return results
        
        results['flipped'] = True
        
        # Flip center horizontally
        img_width = results['img_width']
        center = results['center'].copy()
        center[0] = img_width - 1 - center[0]
        results['center'] = center
        
        # Swap left/right keypoint labels (but don't change coords yet - 
        # the actual coordinate flip happens with the image in TopDownAffine)
        joints = results['joints'].copy()
        for left_idx, right_idx in self.flip_pairs:
            joints[left_idx], joints[right_idx] = joints[right_idx].copy(), joints[left_idx].copy()
        
        results['joints'] = joints
        
        return results


class TopDownAffine:
    """
    Apply affine transformation to crop and resize the person.
    Also transforms keypoints to the new coordinate system.
    """
    
    def __init__(self, image_size: Tuple[int, int] = (192, 256), use_udp: bool = True):
        self.image_size = image_size  # (w, h)
        self.use_udp = use_udp
    
    def __call__(self, results: Dict[str, Any]) -> Dict[str, Any]:
        img = results['img']
        joints = results['joints'].copy()
        center = results['center']
        scale = results['scale']
        rotation = results.get('rotation', 0.)
        flipped = results.get('flipped', False)
        
        # If flipped, we need to flip the source image first
        # (or equivalently, adjust the transform)
        if flipped:
            img = img[:, ::-1, :].copy()  # flip horizontally
            img_width = results['img_width']
            # Flip keypoint x-coordinates
            joints[:, 0] = img_width - 1 - joints[:, 0]
        
        # Get affine transform matrix
        trans = get_affine_transform(
            center=center,
            scale=scale,
            rot=rotation,
            output_size=self.image_size,
            use_udp=self.use_udp
        )
        
        # Warp image
        img = cv2.warpAffine(
            img,
            trans,
            self.image_size,
            flags=cv2.INTER_LINEAR
        )
        
        # Transform keypoints
        for i in range(len(joints)):
            if joints[i, 2] > 0:
                joints[i, :2] = affine_transform_point(joints[i, :2], trans)
        
        results['img'] = img
        results['joints'] = joints
        results['trans'] = trans
        
        return results

class TopDownHalfBodyTransform:
    """
    Randomly use only upper or lower body keypoints to define the crop.
    This helps the model learn from partial occlusions.
    """
    
    def __init__(
        self,
        num_joints_half_body: int = 8,
        prob_half_body: float = 0.3,
        upper_body_ids: List[int] = None, #type: ignore
        lower_body_ids: List[int] = None  #type: ignore
    ):
        self.num_joints_half_body = num_joints_half_body
        self.prob_half_body = prob_half_body
        self.upper_body_ids = upper_body_ids or COCO_UPPER_BODY_IDS
        self.lower_body_ids = lower_body_ids or COCO_LOWER_BODY_IDS
    
    def __call__(self, results: Dict[str, Any]) -> Dict[str, Any]:
        joints = results['joints']  # (num_kpts, 3)
        
        # Count visible joints
        visible_mask = joints[:, 2] > 0
        num_visible = visible_mask.sum()
        
        if num_visible < self.num_joints_half_body:
            return results
        
        if random.random() > self.prob_half_body:
            return results
        
        # Separate upper and lower body visible joints
        upper_visible = [i for i in self.upper_body_ids if joints[i, 2] > 0]
        lower_visible = [i for i in self.lower_body_ids if joints[i, 2] > 0]
        
        # Choose which half to use
        if len(upper_visible) > 2 and len(lower_visible) > 2:
            selected_ids = upper_visible if random.random() < 0.5 else lower_visible
        elif len(upper_visible) > 2:
            selected_ids = upper_visible
        elif len(lower_visible) > 2:
            selected_ids = lower_visible
        else:
            return results
        
        # Recompute center and scale based on selected keypoints
        selected_joints = joints[selected_ids, :2]
        
        center = selected_joints.mean(axis=0)
        
        x_min, y_min = selected_joints.min(axis=0)
        x_max, y_max = selected_joints.max(axis=0)
        
        w = x_max - x_min
        h = y_max - y_min
        
        # Add padding
        aspect_ratio = results['aspect_ratio']  # width / height
        if w > aspect_ratio * h:
            h = w / aspect_ratio
        elif w < aspect_ratio * h:
            w = h * aspect_ratio
        
        # Scale with padding factor
        scale = np.array([w, h], dtype=np.float32) * 1.5
        
        results['center'] = center
        results['scale'] = scale
        
        return results


class TopDownGetRandomScaleRotation:
    """Apply random scale and rotation augmentation."""
    
    def __init__(self, rot_factor: float = 40, scale_factor: float = 0.5, rot_prob: float = 0.6):
        self.rot_factor = rot_factor
        self.scale_factor = scale_factor
        self.rot_prob = rot_prob
    
    def __call__(self, results: Dict[str, Any]) -> Dict[str, Any]:
        scale = results['scale'].copy()
        
        # Scale augmentation
        scale_factor = self.scale_factor
        scale *= np.clip(np.random.randn() * scale_factor + 1, 1 - scale_factor, 1 + scale_factor)
        
        # Rotation augmentation
        if random.random() < self.rot_prob:
            rotation = np.clip(
                np.random.randn() * self.rot_factor,
                -self.rot_factor * 2,
                self.rot_factor * 2
            )
        else:
            rotation = 0.
        
        results['scale'] = scale
        results['rotation'] = rotation
        
        return results



class ToTensor:
    """Convert image to tensor and normalize."""
    
    def __init__(
        self,
        mean: Tuple[float, ...] = (0.485, 0.456, 0.406),
        std: Tuple[float, ...] = (0.229, 0.224, 0.225)
    ):
        self.mean = np.array(mean, dtype=np.float32).reshape(1, 1, 3)
        self.std = np.array(std, dtype=np.float32).reshape(1, 1, 3)
    
    def __call__(self, results: Dict[str, Any]) -> Dict[str, Any]:
        img = results['img'].astype(np.float32) / 255.0
        img = (img - self.mean) / self.std
        
        # HWC -> CHW
        img = img.transpose(2, 0, 1)
        results['img'] = torch.from_numpy(img).float()
        
        return results


class TopDownGenerateTarget:
    """
    Generate target heatmaps from keypoints.
    Uses UDP (Unbiased Data Processing) encoding.
    """
    
    def __init__(
        self,
        sigma: float = 2,
        image_size: Tuple[int, int] = (192, 256),
        heatmap_size: Tuple[int, int] = (48, 64),
        use_udp: bool = True
    ):
        """
        Args:
            sigma: Gaussian sigma for heatmap generation
            image_size: (width, height) of input image
            heatmap_size: (width, height) of output heatmaps
            use_udp: Use Unbiased Data Processing
        """
        self.sigma = sigma
        self.image_size = np.array(image_size)  # (w, h)
        self.heatmap_size = np.array(heatmap_size)  # (w, h)
        self.use_udp = use_udp
    
    def __call__(self, results: Dict[str, Any]) -> Dict[str, Any]:
        joints = results['joints']  # (num_kpts, 3)
        num_joints = len(joints)
        
        heatmap_w, heatmap_h = self.heatmap_size
        
        target = np.zeros((num_joints, heatmap_h, heatmap_w), dtype=np.float32)
        target_weight = np.zeros((num_joints, 1), dtype=np.float32)
        
        # Compute scale factor from image to heatmap
        feat_stride = self.image_size / self.heatmap_size  # (w_stride, h_stride)
        
        for joint_id in range(num_joints):
            if joints[joint_id, 2] <= 0:
                continue
            
            target_weight[joint_id] = 1.
            
            # Map joint position to heatmap coordinates
            if self.use_udp:
                # UDP encoding
                mu_x = joints[joint_id, 0] / feat_stride[0]
                mu_y = joints[joint_id, 1] / feat_stride[1]
            else:
                mu_x = joints[joint_id, 0] / feat_stride[0] - 0.5
                mu_y = joints[joint_id, 1] / feat_stride[1] - 0.5
            
            # Check bounds
            if mu_x < 0 or mu_x >= heatmap_w or mu_y < 0 or mu_y >= heatmap_h:
                target_weight[joint_id] = 0.
                continue
            
            # Generate Gaussian
            x = np.arange(0, heatmap_w, 1, np.float32)
            y = np.arange(0, heatmap_h, 1, np.float32)
            y = y[:, np.newaxis]
            
            target[joint_id] = np.exp(
                -((x - mu_x) ** 2 + (y - mu_y) ** 2) / (2 * self.sigma ** 2)
            )
        
        results['target'] = torch.from_numpy(target).float()
        results['target_weight'] = torch.from_numpy(target_weight).float()
        
        return results


class Compose:
    """Compose multiple transforms."""
    
    def __init__(self, transforms: List):
        self.transforms = transforms
    
    def __call__(self, results: Dict[str, Any]) -> Dict[str, Any]:
        for t in self.transforms:
            results = t(results)
        return results


# =============================================================================
# Dataset
# =============================================================================

class COCOPoseDataset(Dataset):
    """
    COCO Keypoint Dataset for top-down pose estimation.
    
    Each sample is a single person crop with their keypoints.
    """
    
    def __init__(
        self,
        data_root: str,
        ann_file: str,
        image_size: Tuple[int, int] = (192, 256),
        heatmap_size: Tuple[int, int] = (48, 64),
        transforms: Optional[Compose] = None,
        min_keypoints: int = 1,
        bbox_padding: float = 1.25
    ):
        """
        Args:
            data_root: Path to image directory (e.g., './dataset/train2017')
            ann_file: Path to annotation file (e.g., './dataset/annotations/person_keypoints_train2017.json')
            image_size: (width, height) of output crops
            heatmap_size: (width, height) of target heatmaps
            transforms: Transform pipeline (if None, uses default training transforms)
            min_keypoints: Minimum visible keypoints to include a sample
            bbox_padding: Padding factor for bounding boxes
        """
        self.data_root = Path(data_root)
        self.image_size = image_size
        self.heatmap_size = heatmap_size
        self.bbox_padding = bbox_padding
        self.aspect_ratio = image_size[0] / image_size[1]  # w / h
        
        # Load annotations
        self.samples = self._load_annotations(ann_file, min_keypoints)
        
        # Setup transforms
        if transforms is None:
            self.transforms = self._default_transforms()
        else:
            self.transforms = transforms
    
    def _load_annotations(self, ann_file: str, min_keypoints: int) -> List[Dict]:
        """Load and parse COCO annotations."""
        with open(ann_file, 'r') as f:
            coco = json.load(f)
        
        # Build image lookup
        images = {img['id']: img for img in coco['images']}
        
        samples = []
        for ann in coco['annotations']:
            # Skip crowd annotations
            if ann.get('iscrowd', 0):
                continue
            
            # Parse keypoints: [x1, y1, v1, x2, y2, v2, ...]
            kpts = np.array(ann['keypoints']).reshape(-1, 3).astype(np.float32)
            
            # Count visible keypoints (v > 0)
            num_visible = (kpts[:, 2] > 0).sum()
            if num_visible < min_keypoints:
                continue
            
            # Get bbox: [x, y, w, h]
            bbox = np.array(ann['bbox'], dtype=np.float32)
            
            # Get image info
            img_info = images[ann['image_id']]
            
            samples.append({
                'image_file': img_info['file_name'],
                'image_id': ann['image_id'],
                'ann_id': ann['id'],
                'bbox': bbox,
                'keypoints': kpts,
                'img_width': img_info['width'],
                'img_height': img_info['height'],
            })
        
        return samples
    
    def _default_transforms(self) -> Compose:
        """Create default training transforms matching MMPose config."""
        return Compose([
            TopDownRandomFlip(flip_prob=0.5),
            TopDownHalfBodyTransform(num_joints_half_body=8, prob_half_body=0.3),
            TopDownGetRandomScaleRotation(rot_factor=40, scale_factor=0.5),
            TopDownAffine(image_size=self.image_size, use_udp=True),
            ToTensor(),
            TopDownGenerateTarget(
                sigma=2,
                image_size=self.image_size,
                heatmap_size=self.heatmap_size,
                use_udp=True
            ),
        ])
    
    def _bbox_to_center_scale(self, bbox: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """
        Convert bbox [x, y, w, h] to center and scale.
        Scale is adjusted for aspect ratio and padding.
        """
        x, y, w, h = bbox
        
        center = np.array([x + w / 2, y + h / 2], dtype=np.float32)
        
        # Adjust for aspect ratio
        if w > self.aspect_ratio * h:
            h = w / self.aspect_ratio
        elif w < self.aspect_ratio * h:
            w = h * self.aspect_ratio
        
        # Apply padding
        scale = np.array([w, h], dtype=np.float32) * self.bbox_padding
        
        return center, scale
    
    def __len__(self) -> int:
        return len(self.samples)
    
    def __getitem__(self, idx: int) -> Dict[str, Any]:
        sample = self.samples[idx]
        
        # Load image
        img_path = self.data_root / sample['image_file']
        img = cv2.imread(str(img_path))
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        
        # Prepare results dict
        center, scale = self._bbox_to_center_scale(sample['bbox'])
        
        results = {
            'img': img,
            'joints': sample['keypoints'].copy(),
            'center': center,
            'scale': scale,
            'rotation': 0.,
            'img_width': sample['img_width'],
            'img_height': sample['img_height'],
            'aspect_ratio': self.aspect_ratio,
            'bbox': sample['bbox'].copy(),
            'image_file': sample['image_file'],
        }
        
        # Apply transforms
        results = self.transforms(results)
        
        return {
            'img': results['img'],
            'target': results['target'],
            'target_weight': results['target_weight'],
            # Keep metadata for debugging / combined losses
            'joints': results['joints'],
            'center': results['center'],
            'scale': results['scale'],
            'rotation': results.get('rotation', 0.),
            'image_file': results['image_file'],
        }


# =============================================================================
# Factory function for convenience
# =============================================================================

def build_train_dataloader(
    data_root: str = './dataset/train2017',
    ann_file: str = './dataset/annotations/person_keypoints_train2017.json',
    batch_size: int = 32,
    num_workers: int = 4,
    **kwargs
) -> torch.utils.data.DataLoader: #type: ignore
    """Build training dataloader with default settings."""
    dataset = COCOPoseDataset(data_root=data_root, ann_file=ann_file, **kwargs)
    
    return torch.utils.data.DataLoader( #type: ignore
        dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=True,
        drop_last=True,
    )


def build_val_dataloader(
    data_root: str = './dataset/val2017',
    ann_file: str = './dataset/annotations/person_keypoints_val2017.json',
    batch_size: int = 32,
    num_workers: int = 4,
    **kwargs
) -> torch.utils.data.DataLoader: #type: ignore
    """Build validation dataloader (no augmentation except affine)."""
    
    val_transforms = Compose([
        TopDownAffine(image_size=kwargs.get('image_size', (192, 256)), use_udp=True),
        ToTensor(),
        TopDownGenerateTarget(
            sigma=2,
            image_size=kwargs.get('image_size', (192, 256)),
            heatmap_size=kwargs.get('heatmap_size', (48, 64)),
            use_udp=True
        ),
    ])
    
    dataset = COCOPoseDataset(
        data_root=data_root,
        ann_file=ann_file,
        transforms=val_transforms,
        **kwargs
    )
    
    return torch.utils.data.DataLoader( #type: ignore
        dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True,
    )


# =============================================================================
# Quick test / example usage
# =============================================================================

if __name__ == '__main__':
    # Test with a dummy sample to verify transforms work
    print("Testing transforms...")
    
    # Create dummy data
    dummy_img = np.random.randint(0, 255, (480, 640, 3), dtype=np.uint8)
    dummy_joints = np.array([
        [320, 100, 2],  # nose
        [310, 90, 2],   # left_eye
        [330, 90, 2],   # right_eye
        [300, 95, 2],   # left_ear
        [340, 95, 2],   # right_ear
        [280, 200, 2],  # left_shoulder
        [360, 200, 2],  # right_shoulder
        [250, 300, 2],  # left_elbow
        [390, 300, 2],  # right_elbow
        [240, 400, 2],  # left_wrist
        [400, 400, 2],  # right_wrist
        [290, 350, 2],  # left_hip
        [350, 350, 2],  # right_hip
        [280, 450, 0],  # left_knee (invisible)
        [360, 450, 0],  # right_knee (invisible)
        [270, 550, 0],  # left_ankle (invisible)
        [370, 550, 0],  # right_ankle (invisible)
    ], dtype=np.float32)
    
    results = {
        'img': dummy_img,
        'joints': dummy_joints,
        'center': np.array([320., 300.]),
        'scale': np.array([200., 350.]),
        'rotation': 0.,
        'img_width': 640,
        'img_height': 480,
        'aspect_ratio': 192 / 256,
        'bbox': np.array([220., 50., 200., 500.]),
        'image_file': 'test.jpg',
    }
    
    # Test each transform
    transforms = Compose([
        TopDownRandomFlip(flip_prob=0.5),
        TopDownHalfBodyTransform(num_joints_half_body=8, prob_half_body=0.3),
        TopDownGetRandomScaleRotation(rot_factor=40, scale_factor=0.5),
        TopDownAffine(image_size=(192, 256), use_udp=True),
        ToTensor(),
        TopDownGenerateTarget(sigma=2, image_size=(192, 256), heatmap_size=(48, 64), use_udp=True),
    ])
    
    out = transforms(results)
    
    print(f"  img shape: {out['img'].shape}")  # Should be (3, 256, 192)
    print(f"  target shape: {out['target'].shape}")  # Should be (17, 64, 48)
    print(f"  target_weight shape: {out['target_weight'].shape}")  # Should be (17, 1)
    print(f"  target max: {out['target'].max():.3f}")  # Should be ~1.0
    print(f"  visible joints: {(out['target_weight'] > 0).sum()}")
    
    print("\nAll transforms working!")
