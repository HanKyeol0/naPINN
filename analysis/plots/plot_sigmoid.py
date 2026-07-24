from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt

FIGURES_DIR = Path(__file__).resolve().parents[1] / "results" / "figures"


def plot_sigmoid(output_path=None):
    output_path = Path(output_path) if output_path else FIGURES_DIR / "sigmoid_curve.pdf"
    output_path.parent.mkdir(parents=True, exist_ok=True)
    x = np.linspace(-6.0, 6.0, 600)
    y = 1.0 / (1.0 + np.exp(-x))

    plt.rcParams["font.family"] = "serif"
    plt.rcParams["axes.spines.top"] = False
    plt.rcParams["axes.spines.right"] = False

    fig, ax = plt.subplots(figsize=(8, 5))
    ax.plot(x, y, color="#9C4A8B", linewidth=10)

    ax.set_xlabel("log-likelihood z-score \n " + r"($\log{q}$)", fontsize=45)
    ax.set_ylabel("Weight", fontsize=45)
    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_xlim(x.min(), x.max())
    ax.set_ylim(0.0, 1.05)

    plt.tight_layout()
    plt.savefig(output_path, bbox_inches="tight", dpi=300)
    print(f"Figure saved to {output_path}")
    plt.close(fig)


if __name__ == "__main__":
    plot_sigmoid()
