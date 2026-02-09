"""
Training configuration for ViTPose distillation.
"""

from dataclasses import dataclass, field, asdict
import json
from typing import Optional


# Layer mappings for each student depth
# Maps student_layer_idx -> teacher_layer_idx
LAYER_MAPPINGS: dict[int, dict[int, int]] = {
    10: {0: 0, 1: 1, 2: 2, 3: 3, 4: 4, 5: 5, 6: 6, 7: 7, 8: 9, 9: 11},
    8:  {0: 0, 1: 1, 2: 2, 3: 4, 4: 6, 5: 8, 6: 9, 7: 11},
    6:  {0: 0, 1: 2, 2: 4, 3: 6, 4: 9, 5: 11},
    4:  {0: 0, 1: 3, 2: 7, 3: 11},
    3:  {0: 0, 1: 5, 2: 11},
}


@dataclass
class TrainConfig:
    """Configuration for a single training run."""
    
    depth: int
    
    # training hyperparams
    lr: float = 2e-4
    alpha: float = 0.9      # output distillation weight
    beta: float = 0.1       # feature distillation weight
    batch_size: int = 64
    epochs: int = 50
    seed: int = 42
    
    # Data paths
    train_data_root: str = './dataset/train2017'
    train_ann_file: str = './dataset/annotations/person_keypoints_train2017.json'
    val_data_root: str = './dataset/val2017'
    val_ann_file: str = './dataset/annotations/person_keypoints_val2017.json'
    
    # Output
    output_dir: str = './runs'
    checkpoint_every: int = 5  # checkpoint every N epoch
    
    layer_mapping: dict[int, int] = field(default_factory=dict)
    
    def __post_init__(self):
        if not self.layer_mapping:
            if self.depth not in LAYER_MAPPINGS:
                raise ValueError(f"depth must be one of {list(LAYER_MAPPINGS.keys())}")
            self.layer_mapping = LAYER_MAPPINGS[self.depth]
    
    @property
    def run_dir(self) -> str:
        return f"{self.output_dir}/depth_{self.depth}"
    
    @property
    def checkpoint_dir(self) -> str:
        return f"{self.run_dir}/checkpoints"
    
    @property
    def metrics_file(self) -> str:
        return f"{self.run_dir}/metrics.jsonl"
    
    @property
    def config_file(self) -> str:
        return f"{self.run_dir}/config.json"
    
    def save(self, path: Optional[str] = None):
        """Save config to JSON."""
        path = path or self.config_file
        with open(path, 'w') as f:
            json.dump(asdict(self), f, indent=2)
    
    @classmethod
    def load(cls, path: str) -> 'TrainConfig':
        """Load config from JSON."""
        with open(path) as f:
            data = json.load(f)
        if 'layer_mapping' in data and data['layer_mapping']:
            data['layer_mapping'] = {int(k): int(v) for k, v in data['layer_mapping'].items()}
        return cls(**data)


def generate_configs(
    depths: list[int],
    output_dir: str = './runs',
    **overrides
) -> list[TrainConfig]:
    """Generate configs for multiple depths."""
    configs = []
    for depth in depths:
        cfg = TrainConfig(depth=depth, output_dir=output_dir, **overrides)
        configs.append(cfg)
    return configs
