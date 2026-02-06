"""
analyze.py - Quick analysis of training runs.

Usage:
    python analyze.py                    # show summary of all runs
    python analyze.py --depth 6          # show details for one run
    python analyze.py --plot             # generate plots
"""

import argparse
import json
from pathlib import Path

import pandas as pd


def load_metrics(run_dir: Path) -> pd.DataFrame:
    """Load metrics.jsonl into a DataFrame."""
    metrics_file = run_dir / 'metrics.jsonl'
    if not metrics_file.exists():
        return pd.DataFrame()
    
    records = []
    with open(metrics_file) as f:
        for line in f:
            records.append(json.loads(line))
    
    return pd.DataFrame(records)


def load_config(run_dir: Path) -> dict:
    """Load config.json."""
    config_file = run_dir / 'config.json'
    if not config_file.exists():
        return {}
    
    with open(config_file) as f:
        return json.load(f)


def summarize_all(output_dir: str):
    """Print summary table of all runs."""
    output_path = Path(output_dir)
    
    if not output_path.exists():
        print(f"No runs found in {output_dir}")
        return
    
    rows = []
    for run_dir in sorted(output_path.iterdir()):
        if not run_dir.is_dir():
            continue
        
        config = load_config(run_dir)
        df = load_metrics(run_dir)
        
        if df.empty:
            status = 'no metrics'
            epochs = 0
            best_val = float('nan')
            final_val = float('nan')
        else:
            epochs = len(df)
            best_val = df['best_val_loss'].min()
            final_val = df['val_loss'].iloc[-1]
            status = 'complete' if epochs >= config.get('epochs', 0) else f'running ({epochs} epochs)'
        
        rows.append({
            'run': run_dir.name,
            'depth': config.get('depth', '?'),
            'epochs': epochs,
            'best_val_loss': best_val,
            'final_val_loss': final_val,
            'status': status,
        })
    
    if not rows:
        print(f"No runs found in {output_dir}")
        return
    
    print(f"\n{'='*80}")
    print(f"  TRAINING RUNS SUMMARY ({output_dir})")
    print(f"{'='*80}")
    print(f"{'Run':<15} | {'Depth':>5} | {'Epochs':>6} | {'Best Val':>10} | {'Final Val':>10} | Status")
    print(f"{'-'*15}-+-{'-'*5}-+-{'-'*6}-+-{'-'*10}-+-{'-'*10}-+-{'-'*15}")
    
    for row in sorted(rows, key=lambda x: x['depth'] if isinstance(x['depth'], int) else 999):
        print(f"{row['run']:<15} | {row['depth']:>5} | {row['epochs']:>6} | "
              f"{row['best_val_loss']:>10.4f} | {row['final_val_loss']:>10.4f} | {row['status']}")


def show_run_details(output_dir: str, depth: int):
    """Show detailed metrics for a single run."""
    run_dir = Path(output_dir) / f'depth_{depth}'
    
    if not run_dir.exists():
        print(f"Run not found: {run_dir}")
        return
    
    config = load_config(run_dir)
    df = load_metrics(run_dir)
    
    print(f"\n{'='*60}")
    print(f"  DEPTH {depth} RUN DETAILS")
    print(f"{'='*60}")
    
    print("\nConfig:")
    for k, v in config.items():
        if k != 'layer_mapping':
            print(f"  {k}: {v}")
    
    if df.empty:
        print("\nNo metrics recorded yet.")
        return

    print(f"\nTraining progress: {len(df)}/{config.get('epochs', '?')} epochs")
    print(f"Best val loss: {df['best_val_loss'].min():.4f} (epoch {df['val_loss'].idxmin() + 1})") #type: ignore
    print(f"Final val loss: {df['val_loss'].iloc[-1]:.4f}")
    
    print(f"\nLast 5 epochs:")
    print(df[['epoch', 'train_loss', 'val_loss', 'best_val_loss']].tail().to_string(index=False))


def plot_runs(output_dir: str):
    """Generate comparison plots."""
    try:
        import matplotlib.pyplot as plt
    except ImportError:
        print("matplotlib not installed. Run: pip install matplotlib")
        return
    
    output_path = Path(output_dir)
    
    _, axes = plt.subplots(1, 2, figsize=(14, 5))
    
    for run_dir in sorted(output_path.iterdir()):
        if not run_dir.is_dir():
            continue
        
        config = load_config(run_dir)
        df = load_metrics(run_dir)
        
        if df.empty:
            continue
        
        depth = config.get('depth', '?')
        label = f"depth={depth}"
        
        axes[0].plot(df['epoch'], df['train_loss'], label=label)
        axes[1].plot(df['epoch'], df['val_loss'], label=label)
    
    axes[0].set_xlabel('Epoch')
    axes[0].set_ylabel('Train Loss')
    axes[0].set_title('Training Loss by Depth')
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)
    
    axes[1].set_xlabel('Epoch')
    axes[1].set_ylabel('Val Loss')
    axes[1].set_title('Validation Loss by Depth')
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    plot_path = output_path / 'training_curves.png'
    plt.savefig(plot_path, dpi=150)
    print(f"Plot saved to: {plot_path}")
    
    plt.show()


def main():
    parser = argparse.ArgumentParser(description='Analyze training runs')
    parser.add_argument('--output-dir', type=str, default='./runs', help='Output directory')
    parser.add_argument('--depth', type=int, help='Show details for specific depth')
    parser.add_argument('--plot', action='store_true', help='Generate plots')
    args = parser.parse_args()
    
    if args.depth:
        show_run_details(args.output_dir, args.depth)
    elif args.plot:
        plot_runs(args.output_dir)
    else:
        summarize_all(args.output_dir)


if __name__ == '__main__':
    main()
