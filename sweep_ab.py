# sweep_alpha_beta.py
from config import TrainConfig
import os

depth = 6
epochs = 3

combos = [
    (1.0, 0.5),   # your current default
    (1.0, 1.0),   # equal feature weight
    (0.5, 0.5),   # less distillation pressure
    (1.0, 0.25),  # less feature weight
    (2.0, 0.5),   # more distillation
]

os.makedirs('configs/alpha_beta_sweep', exist_ok=True)

for alpha, beta in combos:
    cfg = TrainConfig(
        depth=depth,
        epochs=epochs,
        alpha=alpha,
        beta=beta,
        output_dir=f'./runs/alpha_beta_sweep/a{alpha}_b{beta}',
    )
    path = f'configs/alpha_beta_sweep/a{alpha}_b{beta}.json'
    cfg.save(path)
    print(f"Created {path}")
