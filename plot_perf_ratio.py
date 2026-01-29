import matplotlib.pyplot as plt
import numpy as np

# -----------------------------
# Configuration
# -----------------------------
outlier_ratios = ["5%", "10%", "15%"]   # three outlier ratios
x = np.arange(len(outlier_ratios))
methods = ["LAD-PINN", "OrPINN(q=1.9)", "OrPINN(q=2.9)", "naPINN"]
benchmarks = ["2D Burgers", "2D Allen–Cahn", "2D $\lambda-\omega$ RD"]

# Marker / style settings
markers = {
    # "PINN": "o",
    "LAD-PINN": "s",
    "OrPINN(q=1.9)": "^",
    "OrPINN(q=2.9)": "D",
    "naPINN": "*"   # star for naPINN
}
linestyles = {
    # "PINN": "-",
    "LAD-PINN": "--",
    "OrPINN(q=1.9)": "-.",
    "OrPINN(q=2.9)": ":",
    "naPINN": "-"
}
colors = {
    # "PINN":     "#0072B2",  # blue
    "LAD-PINN": "#E69F00",  # orange
    "OrPINN(q=1.9)":   "#009E73",  # green
    "OrPINN(q=2.9)":   "#56B4E9",
    "naPINN":   "#D55E00",  # vermillion (stands out well)
}

# -----------------------------
# rMSE placeholders (FILL THESE)
# Shape: [benchmark][method] -> list of 3 values
# -----------------------------
rmse_mean = {
    "2D Allen–Cahn": {
        # "PINN":     [0.224, 0.377, 0.547],
        "LAD-PINN":  [0.194, 0.216, 0.241],
        "OrPINN(q=1.9)": [0.158, 0.267, 0.373],
        "OrPINN(q=2.9)": [0.132, 0.155, 0.180],
        "naPINN":   [0.089, 0.092, 0.091],
    },
    "2D Burgers": {
        # "PINN":     [0.267, 0.488, 0.682],
        "LAD-PINN":  [0.233, 0.272, 0.296],
        "OrPINN(q=1.9)":    [0.219, 0.379, 0.545],
        "OrPINN(q=2.9)":    [0.151, 0.195, 0.234],
        "naPINN":   [0.101, 0.108, 0.127],
    },
    "2D $\lambda-\omega$ RD": {
        # "PINN":     [0.178, 0.292, 0.457],
        "LAD-PINN": [0.164, 0.181, 0.199],
        "OrPINN(q=1.9)": [0.109, 0.150, 0.201],
        "OrPINN(q=2.9)": [0.140, 0.151, 0.163],
        "naPINN":   [0.092, 0.092, 0.095],
    },
}

rmse_std = {
    "2D Allen–Cahn": {
        # "PINN":     [0.018, 0.021, 0.025],
        "LAD-PINN":  [0.013, 0.012, 0.009],
        "OrPINN(q=1.9)": [0.015, 0.044, 0.020],
        "OrPINN(q=2.9)": [0.010, 0.014, 0.012],
        "naPINN":    [0.008, 0.011, 0.006],
    },
    "2D Burgers": {
        # "PINN":     [0.022, 0.033, 0.049],
        "LAD-PINN":  [0.049, 0.055, 0.040],
        "OrPINN(q=1.9)": [0.022, 0.028, 0.052],
        "OrPINN(q=2.9)": [0.036, 0.036, 0.042],
        "naPINN":    [0.039, 0.023, 0.049],
    },
    "2D $\lambda-\omega$ RD": {
        # "PINN":     [0.010, 0.011, 0.014],
        "LAD-PINN":  [0.013, 0.012, 0.010],
        "OrPINN(q=1.9)": [0.007, 0.009, 0.012],
        "OrPINN(q=2.9)": [0.009, 0.009, 0.009],
        "naPINN":    [0.007, 0.007, 0.006],
    },
}

# -----------------------------
# Plot
# -----------------------------
plt.rcParams.update({
    "font.family": "serif",           # Serif fonts for body text
    "mathtext.fontset": "cm",         # Computer Modern for math (LaTeX look)
    "font.size": 12,
    "axes.labelsize": 13,
    "axes.titlesize": 14,
    "xtick.labelsize": 13,
    "ytick.labelsize": 13,
    "legend.fontsize": 11,
    "axes.spines.top": False,
    "axes.spines.right": False,
})
    
fig, axes = plt.subplots(1, 3, figsize=(15, 4), sharey=True)

for ax, bench in zip(axes, benchmarks):
    for method in methods:
        y = np.array(rmse_mean[bench][method], dtype=float)
        yerr = np.array(rmse_std[bench][method], dtype=float)
        
        ax.errorbar(
            x, y, yerr=yerr,
            color=colors[method],
            marker=markers[method],
            linestyle=linestyles[method],
            linewidth=1.8,
            markersize=12 if method == "naPINN" else 7,
            markeredgewidth=1.0,
            markeredgecolor="white",
            capsize=4,           # errorbar cap size
            elinewidth=1.2,      # errorbar line width
            alpha=0.95,
            label=method,
            zorder=3 if method == "naPINN" else 2,
        )

    ax.set_title(bench, fontsize=12)
    ax.set_xlabel("Outlier ratio")
    ax.set_ylim(0.05, 0.62)
    ax.set_xticks(x)
    ax.set_xticklabels(outlier_ratios)
    # ax.grid(True, linestyle=":", linewidth=0.8, alpha=0.45, zorder=0)

axes[0].set_ylabel("rMSE")

# Legend (single shared legend)
handles, labels = axes[0].get_legend_handles_labels()
fig.legend(
    handles,
    labels,
    loc="upper center",
    ncol=4,
    frameon=False,
    bbox_to_anchor=(0.5, 1.0),
    handlelength=2.4,
    columnspacing=1.6,
)

plt.tight_layout(rect=[0, 0, 1, 0.93])
plt.savefig("performance_ratio_plot.pdf", bbox_inches="tight", dpi=300)
plt.show()
