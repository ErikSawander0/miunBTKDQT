"""
plot_pareto.py - Plot the full Pareto frontier: size vs AP
Reads from uniform_quant_results.json, greedy_fp32_int8_results.json,
and greedy_results.json.

Usage:
    python plot_pareto.py
"""

import json
import matplotlib.pyplot as plt
import matplotlib
import numpy as np

matplotlib.rcParams.update({
    "font.family": "serif",
    "font.size": 11,
    "axes.titlesize": 14,
    "axes.labelsize": 12,
    "legend.fontsize": 9,
    "figure.dpi": 150,
})

# ── Load data ────────────────────────────────────────────────────────────────

with open("uniform_quant_results.json") as f:
    uniform = json.load(f)

with open("greedy_fp32_int8_results.json") as f:
    mixed_fp32_int8 = json.load(f)

with open("greedy_results.json") as f:
    mixed_int8_int4 = json.load(f)

# ── Collect points ───────────────────────────────────────────────────────────

fp32_points = []
int8_points = []
mixed_win_points = []
mixed_fail_points = []

# FP32 and INT8 baselines
for key, data in uniform.items():
    depth = int(key.split("_")[1])
    fp32_points.append({
        "depth": depth,
        "size": data["fp32"]["size_mb"],
        "ap": data["fp32"]["AP"],
    })
    int8_points.append({
        "depth": depth,
        "size": data["int8"]["size_mb"],
        "ap": data["int8"]["AP"],
    })

# Mixed FP32/INT8 (all wins)
for key, data in mixed_fp32_int8.items():
    mixed_win_points.append({
        "depth": data["deeper_depth"],
        "size": data["actual_size_mb"],
        "ap": data["metrics"]["AP"],
        "config": data["config_str"],
        "vs": f"d{data['shallower_depth']} FP32",
    })

# Mixed INT8/INT4 (all fails)
for key, data in mixed_int8_int4.items():
    mixed_fail_points.append({
        "depth": data["deeper_depth"],
        "size": data["actual_size_mb"],
        "ap": data["metrics"]["AP"],
        "config": data["config_str"],
        "vs": f"d{data['shallower_depth']} INT8",
    })

# Sort by size for line plots
fp32_points.sort(key=lambda x: x["size"])
int8_points.sort(key=lambda x: x["size"])

# ── Plot ─────────────────────────────────────────────────────────────────────

fig, ax = plt.subplots(1, 1, figsize=(10, 7))

# Colors
C_FP32 = "#2c3e50"
C_INT8 = "#2980b9"
C_WIN = "#27ae60"
C_FAIL = "#c0392b"

# FP32 baseline curve
ax.plot([p["size"] for p in fp32_points],
        [p["ap"] for p in fp32_points],
        "o-", color=C_FP32, markersize=8, linewidth=2,
        label="FP32 (baseline)", zorder=3)
for p in fp32_points:
    ax.annotate(f'd{p["depth"]}',
                (p["size"], p["ap"]),
                textcoords="offset points", xytext=(8, -4),
                fontsize=8, color=C_FP32)

# INT8 uniform curve
ax.plot([p["size"] for p in int8_points],
        [p["ap"] for p in int8_points],
        "s-", color=C_INT8, markersize=8, linewidth=2,
        label="Uniform INT8", zorder=3)
for p in int8_points:
    ax.annotate(f'd{p["depth"]}',
                (p["size"], p["ap"]),
                textcoords="offset points", xytext=(8, -4),
                fontsize=8, color=C_INT8)

# Mixed FP32/INT8 wins
for p in mixed_win_points:
    ax.plot(p["size"], p["ap"], "^", color=C_WIN, markersize=10,
            markeredgecolor="white", markeredgewidth=1, zorder=4)
    ax.annotate(f'd{p["depth"]} mix',
                (p["size"], p["ap"]),
                textcoords="offset points", xytext=(8, 4),
                fontsize=8, color=C_WIN, fontweight="bold")
# Single legend entry
ax.plot([], [], "^", color=C_WIN, markersize=10,
        markeredgecolor="white", markeredgewidth=1,
        label="Mixed FP32/INT8 (Pareto ✓)")

# Mixed INT8/INT4 fails
for p in mixed_fail_points:
    ax.plot(p["size"], p["ap"], "x", color=C_FAIL, markersize=9,
            markeredgewidth=2, zorder=4)
# Single legend entry
ax.plot([], [], "x", color=C_FAIL, markersize=9, markeredgewidth=2,
        label="Mixed INT8/INT4 (Pareto ✗)")

# Arrows connecting mixed FP32/INT8 wins to their FP32 targets
for p in mixed_win_points:
    shallower_depth = int(p["vs"].split("d")[1].split(" ")[0])
    target = next(fp for fp in fp32_points if fp["depth"] == shallower_depth)
    ax.annotate("",
                xy=(target["size"], target["ap"]),
                xytext=(p["size"], p["ap"]),
                arrowprops=dict(arrowstyle="->", color=C_WIN,
                                linestyle="--", alpha=0.4, lw=1.2))

# Arrows connecting mixed INT8/INT4 fails to their INT8 targets
for p in mixed_fail_points:
    shallower_depth = int(p["vs"].split("d")[1].split(" ")[0])
    target = next(i8 for i8 in int8_points if i8["depth"] == shallower_depth)
    ax.annotate("",
                xy=(target["size"], target["ap"]),
                xytext=(p["size"], p["ap"]),
                arrowprops=dict(arrowstyle="->", color=C_FAIL,
                                linestyle="--", alpha=0.4, lw=1.2))

ax.set_xlabel("Model Size (MB)")
ax.set_ylabel("COCO AP")
ax.set_title("Pareto Frontier: Depth Reduction vs Mixed-Precision Quantization")
ax.legend(loc="lower right", framealpha=0.9)
ax.grid(True, alpha=0.3)
ax.set_xlim(15, 290)
ax.set_ylim(0.15, 0.78)

plt.tight_layout()
plt.savefig("pareto_frontier.png", dpi=200, bbox_inches="tight")
plt.savefig("pareto_frontier.pdf", bbox_inches="tight")
print("Saved pareto_frontier.png and pareto_frontier.pdf")
