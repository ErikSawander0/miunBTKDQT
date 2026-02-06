"""
make_configs.py - Generate config files for all depths.

Usage:
    python make_configs.py                     # generate configs with defaults
    python make_configs.py --epochs 100        # override epochs
    python make_configs.py --depths 6 4        # specific depths only
"""

import argparse
import os
from config import TrainConfig, LAYER_MAPPINGS


def main():
    parser = argparse.ArgumentParser(description='Generate training configs')
    parser.add_argument('--depths', type=int, nargs='+', 
                        default=list(LAYER_MAPPINGS.keys()),
                        help='Depths to generate configs for')
    parser.add_argument('--epochs', type=int, default=100, help='Epochs')
    parser.add_argument('--lr', type=float, default=2e-4, help='Learning rate')
    parser.add_argument('--alpha', type=float, default=1.0, help='Distillation loss weight')
    parser.add_argument('--beta', type=float, default=0.5, help='Feature loss weight')
    parser.add_argument('--output-dir', type=str, default='./runs', help='Output directory')
    parser.add_argument('--config-dir', type=str, default='./configs', help='Where to save configs')
    args = parser.parse_args()
    
    os.makedirs(args.config_dir, exist_ok=True)
    
    for depth in args.depths:
        if depth not in LAYER_MAPPINGS:
            print(f"Skipping invalid depth {depth}")
            continue
        
        config = TrainConfig(
            depth=depth,
            epochs=args.epochs,
            lr=args.lr,
            alpha=args.alpha,
            beta=args.beta,
            output_dir=args.output_dir,
        )
        
        path = f"{args.config_dir}/depth_{depth}.json"
        config.save(path)
        print(f"Created {path}")
    
    print(f"\nGenerated {len(args.depths)} configs in {args.config_dir}/")
    print(f"\nTo train, run:")
    print(f"  python train.py {args.config_dir}/depth_10.json")
    print(f"\nOr use the bash script:")
    print(f"  for cfg in {args.config_dir}/*.json; do python train.py $cfg; done")


if __name__ == '__main__':
    main()
