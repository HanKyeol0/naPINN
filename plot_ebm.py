import os
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker

def plot_noise_pdfs_overlay(
    true_pdf_path: str,
    ebm_pdf_path_a: str,
    ebm_pdf_path_b: str,
    label_a: str = "EBM (t₁)",
    label_b: str = "EBM (t₂)",
    title: str = "Noise PDF: True vs. EBM (two time steps)",
    out_path_pdf: str | None = None,
    out_path_png: str | None = None,
    xlim: tuple[float, float] | None = None,
    ylim: tuple[float, float] | None = None,
    dpi: int = 300,
):
    """
    Load one true pdf and two ebm pdf snapshots, plot in one figure.
    Enhanced for publication-quality aesthetics.
    """

    # ---------- Load ----------
    d_true = np.load(true_pdf_path)
    r_true = d_true["r_grid"].astype(np.float64)
    pdf_true = d_true["pdf_true"].astype(np.float64)

    d_a = np.load(ebm_pdf_path_a)
    r_a = d_a["r_grid"].astype(np.float64)
    pdf_a = d_a["pdf_ebm"].astype(np.float64)

    d_b = np.load(ebm_pdf_path_b)
    r_b = d_b["r_grid"].astype(np.float64)
    pdf_b = d_b["pdf_ebm"].astype(np.float64)

    # ---------- Align grids (interpolate if needed) ----------
    def _interp_to_true(r_src, y_src):
        if r_src.shape == r_true.shape and np.allclose(r_src, r_true):
            return y_src
        # ensure increasing x for interp
        idx = np.argsort(r_src)
        return np.interp(r_true, r_src[idx], y_src[idx])

    pdf_a_on_true = _interp_to_true(r_a, pdf_a)
    pdf_b_on_true = _interp_to_true(r_b, pdf_b)

    # ---------- Normalize safety ----------
    pdf_true = np.clip(pdf_true, 0.0, None)
    pdf_a_on_true = np.clip(pdf_a_on_true, 0.0, None)
    pdf_b_on_true = np.clip(pdf_b_on_true, 0.0, None)

    # ---------- Choose plot limits ----------
    if xlim is None:
        if "R_range" in d_true:
            rr = float(np.asarray(d_true["R_range"]))
            xlim = (-rr, rr)
        else:
            xlim = (float(r_true.min()), float(r_true.max()))

    if ylim is None:
        ymax = float(
            max(pdf_true.max(), pdf_a_on_true.max(), pdf_b_on_true.max()) * 1.08
        )
        ylim = (0.0, ymax)

    # ---------- Figure style ----------
    # Elegant, academic styling
    plt.rcParams.update({
        "font.family": "serif",           # Serif fonts for body text
        "mathtext.fontset": "cm",         # Computer Modern for math (LaTeX look)
        "font.size": 12,
        "axes.labelsize": 13,
        "axes.titlesize": 14,
        "xtick.labelsize": 11,
        "ytick.labelsize": 11,
        "legend.fontsize": 11,
        "axes.spines.top": False,
        "axes.spines.right": False,
    })

    fig, ax = plt.subplots(1, 1, figsize=(7.0, 4.5), dpi=dpi)

    # Colors (High contrast, colorblind safe)
    c_true = "#333333"      # Dark Grey
    c_fill = "#E0E0E0CB"      # Light Grey Fill
    c_a = "#0077BB"         # Cyan/Blue (Muted)
    c_b = "#CC3311"         # Vivid Red/Orange

    # 1. True PDF: Filled area + Line
    # This visually separates "Ground Truth" from "Estimations"
    ax.fill_between(r_true, pdf_true, color=c_fill, alpha=0.6, label=None, zorder=1)
    ax.plot(r_true, pdf_true, color=c_true, lw=2.5, label="True noise PDF", zorder=2)

    # 2. EBM PDFs
    # t1: Initialization (Dashed, Blue)
    ax.plot(r_true, pdf_a_on_true, color=c_a, lw=2.5, ls="--", 
            label=label_a, zorder=3, alpha=0.9)
    
    # t2: Final Training (Dash-Dot or Solid, Red)
    # Using a distinct line style helps distinguish the two learned steps
    ax.plot(r_true, pdf_b_on_true, color=c_b, lw=2.5, ls="--", 
            label=label_b, zorder=4)

    # ---------- Cosmetics ----------
    ax.set_xlabel(r"Residual value $r$")
    ax.set_ylabel(r"Probability Density $p(r)$")
    
    ax.set_xlim(*xlim)
    ax.set_ylim(*ylim)
    
    # Optional: Add a vertical line at 0 for reference in residual plots
    # ax.axvline(0, color='black', alpha=0.1, lw=1, zorder=0)

    # Legend
    legend = ax.legend(frameon=False, loc="upper right")
    
    # Adjust layout
    plt.tight_layout()

    # ---------- Save ----------
    if out_path_pdf is not None:
        os.makedirs(os.path.dirname(out_path_pdf), exist_ok=True)
        fig.savefig(out_path_pdf, bbox_inches="tight")
    if out_path_png is not None:
        os.makedirs(os.path.dirname(out_path_png), exist_ok=True)
        fig.savefig(out_path_png, bbox_inches="tight", dpi=dpi)

    plt.close(fig)
    return out_path_pdf, out_path_png

if __name__ == "__main__":
    # Example usage (paths preserved)
    exp_dir = "outputs/burgers2d/burgers2d_mlp_a8_na_rewind"
    true_pdf_file = f"{exp_dir}/after_init_true_pdf.npz"
    ebm_pdf_file_t1 = f"{exp_dir}/after_init_ebm_pdf.npz"
    ebm_pdf_file_t2 = f"{exp_dir}/eval_ep20000_ebm_pdf.npz"

    plot_noise_pdfs_overlay(
        true_pdf_path=true_pdf_file,
        ebm_pdf_path_a=ebm_pdf_file_t1,
        ebm_pdf_path_b=ebm_pdf_file_t2,
        label_a="After EBM initialization",
        label_b="After naPINN training",
        title="Noise PDF: True vs. EBM",
        out_path_pdf=f"{exp_dir}/noise_pdf_comparison.pdf",
        out_path_png=f"{exp_dir}/noise_pdf_comparison.png",
        xlim=(-1.0, 1.0),
        ylim=(0.0, 6.3),
        dpi=300,
    )