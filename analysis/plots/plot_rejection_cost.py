from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt

FIGURES_DIR = Path(__file__).resolve().parents[1] / "results" / "figures"

# -----------------------------
# Style preset (as requested)
# -----------------------------
plt.rcParams.update({
    "font.family": "serif",
    "mathtext.fontset": "cm",
    "font.size": 12,
    "axes.labelsize": 13,
    "axes.titlesize": 14,
    "xtick.labelsize": 11,
    "ytick.labelsize": 11,
    "legend.fontsize": 11,
    "axes.spines.top": False,
    "axes.spines.right": False,
})

# -----------------------------
# Data (Allen–Cahn, 15% outliers)
# -----------------------------
rej_cost = np.array([0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0], dtype=float)
mae = np.array([0.11655, 0.088816, 0.12473, 0.15568, 0.14423, 0.10382, 0.11448, 0.095592, 0.78751, 0.78076], dtype=float)
mse = np.array([0.11166, 0.091611, 0.11424, 0.15199, 0.13637, 0.10025, 0.10621, 0.095081, 0.64723, 0.64234], dtype=float)

# Best points (lower is better)
best_mae_idx = int(np.argmin(mae))
best_mse_idx = int(np.argmin(mse))

# -----------------------------
# Plot (two stacked subfigures)
# -----------------------------
fig, axes = plt.subplots(2, 1, figsize=(7.2, 6.2), dpi=150, sharex=True)

# MAE
ax = axes[0]
ax.plot(rej_cost, mae, marker="o", linewidth=2, markersize=5, label="rMAE")
ax.scatter([rej_cost[best_mae_idx]], [mae[best_mae_idx]], marker="*", s=160, zorder=3, label="Best rMAE")
ax.set_ylabel("rMAE")
ax.grid(True, alpha=0.25, linestyle="--")
ax.legend(loc="upper left", frameon=False)

# MSE
ax = axes[1]
ax.plot(rej_cost, mse, marker="s", linewidth=2, markersize=5, label="rMSE")
ax.scatter([rej_cost[best_mse_idx]], [mse[best_mse_idx]], marker="*", s=160, zorder=3, label="Best rMSE")
ax.set_xlabel(r"Rejection cost $\lambda_{\mathrm{rej}}$")
ax.set_ylabel("rMSE")
ax.legend(loc="upper left", frameon=False)

# Nice x-ticks
axes[1].set_xticks(rej_cost)

# Tight layout
fig.tight_layout(rect=[0, 0, 1, 0.965])

FIGURES_DIR.mkdir(parents=True, exist_ok=True)
output_path = FIGURES_DIR / "rej_cost_sensitivity_allencahn15.pdf"
plt.savefig(output_path, bbox_inches="tight")
plt.close(fig)
print(f"Figure saved to {output_path}")
