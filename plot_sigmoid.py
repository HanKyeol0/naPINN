import numpy as np
import matplotlib.pyplot as plt


def plot_sigmoid(output_path="sigmoid_curve.svg"):
    x = np.linspace(-6.0, 6.0, 600)
    y = 1.0 / (1.0 + np.exp(-x))

    plt.rcParams["font.family"] = "serif"
    plt.rcParams["axes.spines.top"] = False
    plt.rcParams["axes.spines.right"] = False

    fig, ax = plt.subplots(figsize=(8, 5))
    ax.plot(x, y, color="#9C4A8B", linewidth=10)

    ax.set_xlabel("log-likelihood z-score ($\log{q}$)", fontsize=35)
    ax.set_ylabel("Sigmoid Weight", fontsize=35)
    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_xlim(x.min(), x.max())
    ax.set_ylim(0.0, 1.05)

    plt.tight_layout()
    plt.savefig(output_path, transparent=True, bbox_inches="tight")
    print(f"Figure saved to {output_path}")
    plt.show()


if __name__ == "__main__":
    plot_sigmoid()
