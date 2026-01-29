import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# Optional smoother (best): Savitzky–Golay
try:
    from scipy.signal import savgol_filter
    _HAS_SAVGOL = True
except Exception:
    _HAS_SAVGOL = False


def set_pub_style():
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


def _moving_average(y: np.ndarray, window: int) -> np.ndarray:
    window = int(max(1, window))
    if window <= 1:
        return y
    pad = window // 2
    ypad = np.pad(y, (pad, pad), mode="edge")
    kernel = np.ones(window, dtype=np.float64) / window
    return np.convolve(ypad, kernel, mode="valid")


def smooth_xy(x: np.ndarray, y: np.ndarray, method: str = "savgol", window: int = 31, poly: int = 3):
    x = np.asarray(x, dtype=np.float64)
    y = np.asarray(y, dtype=np.float64)

    if len(y) < 5:
        return x, y

    if method == "savgol" and _HAS_SAVGOL:
        w = int(window)
        if w % 2 == 0:
            w += 1
        if w >= len(y):
            w = len(y) - 1 if (len(y) - 1) % 2 == 1 else len(y) - 2
        w = max(w, 5)
        p = min(int(poly), w - 2)
        y_s = savgol_filter(y, window_length=w, polyorder=p, mode="interp")
        return x, y_s

    return x, _moving_average(y, window=max(3, window))


def _plot_gate_param_on_ax(
    ax: plt.Axes,
    csv_path: str,
    ylabel: str,
    curve_names: list[str] | None = None,
    dpi: int = 300,
    smooth: bool = True,
    smooth_method: str = "savgol",
    smooth_window: int = 31,
    smooth_poly: int = 3,
    legend_position: str = "False",
    legend_bbox: tuple[float, float] = (1.02, 0.6),
):
    """
    Plot Step vs 3 curves on a provided axis (ax).
    """
    df = pd.read_csv(csv_path)

    if df.shape[1] < 4:
        raise ValueError(f"Expected >=4 columns (Step + 3 curves). Got columns: {list(df.columns)}")

    step_col = "Step" if "Step" in df.columns else ("step" if "step" in df.columns else df.columns[0])
    y_cols = list(df.columns[1:4])

    df[step_col] = pd.to_numeric(df[step_col], errors="coerce")
    for c in y_cols:
        df[c] = pd.to_numeric(df[c], errors="coerce")

    df = df[np.isfinite(df[step_col].to_numpy())].copy()
    df = df.sort_values(step_col)

    x_all = df[step_col].to_numpy()

    if curve_names is not None:
        if len(curve_names) != 3:
            raise ValueError(f"curve_names must have length 3, got {len(curve_names)}")
        labels = curve_names
    else:
        labels = y_cols

    colors = ["#0077BB", "#CC3311", "#009988"]
    linestyles = ["-", "--", "-."]

    for i, (col, lab) in enumerate(zip(y_cols, labels)):
        y_all = df[col].to_numpy()

        m = np.isfinite(x_all) & np.isfinite(y_all)  # drop missing values per-curve
        x = x_all[m]
        y = y_all[m]
        if len(x) == 0:
            continue

        if smooth:
            x, y = smooth_xy(x, y, method=smooth_method, window=smooth_window, poly=smooth_poly)

        ax.plot(
            x, y,
            color=colors[i % len(colors)],
            lw=2.5,
            ls=linestyles[i % len(linestyles)],
            label=lab,
            alpha=0.95,
        )

    ax.set_ylabel(ylabel)
    ax.set_xlim(float(x_all.min()), float(x_all.max()))
    # ax.grid(True, alpha=0.18, linestyle="--", linewidth=0.8)

    if legend_position.lower() == "true":
        ax.legend(frameon=False, loc="center right", bbox_to_anchor=legend_bbox, borderaxespad=0.0)
    else:
        ax.legend(frameon=False, loc="best")


def plot_two_gate_params_in_one_figure(
    steepness_csv: str,
    cutoff_csv: str,
    out_path_pdf: str | None,
    out_path_png: str | None,
    curve_names: list[str] | None = None,
    dpi: int = 300,
    smooth: bool = True,
    smooth_method: str = "savgol",
    smooth_window: int = 31,
    smooth_poly: int = 3,
    # Font size controls (requested)
    xlabel_fs: int = 15,
    ylabel_fs: int = 15,
    # Legend anchor for the RIGHT subplot (cutoff)
    cutoff_legend_bbox: tuple[float, float] = (1.02, 0.6),
):
    set_pub_style()

    fig, axes = plt.subplots(2, 1, figsize=(7.0, 5.0), dpi=dpi, sharex=False, sharey=False)

    # Top: cutoff (legend fixed to right side)
    _plot_gate_param_on_ax(
        ax=axes[0],
        csv_path=cutoff_csv,
        ylabel=r"Cutoff $\tau$",
        curve_names=curve_names,
        dpi=dpi,
        smooth=smooth,
        smooth_method=smooth_method,
        smooth_window=smooth_window,
        smooth_poly=smooth_poly,
        legend_position="True",
        legend_bbox=cutoff_legend_bbox,
    )

    # Bottom: steepness
    _plot_gate_param_on_ax(
        ax=axes[1],
        csv_path=steepness_csv,
        ylabel=r"Steepness $a$",
        curve_names=curve_names,
        dpi=dpi,
        smooth=smooth,
        smooth_method=smooth_method,
        smooth_window=smooth_window,
        smooth_poly=smooth_poly,
        legend_position="False",   # no fixed right-side legend here
    )
    # Increase label font sizes (requested)
    axes[1].set_xlabel("Step", fontsize=xlabel_fs)
    for ax in axes:
        ax.set_ylabel(ax.get_ylabel(), fontsize=ylabel_fs)

    plt.tight_layout(h_pad=1.2)

    if out_path_pdf is not None:
        os.makedirs(os.path.dirname(out_path_pdf), exist_ok=True)
        fig.savefig(out_path_pdf, bbox_inches="tight")
    if out_path_png is not None:
        os.makedirs(os.path.dirname(out_path_png), exist_ok=True)
        fig.savefig(out_path_png, bbox_inches="tight", dpi=dpi)

    plt.close(fig)
    return out_path_pdf, out_path_png


if __name__ == "__main__":
    steepness_csv = "steepness.csv"
    cutoff_csv    = "cutoff.csv"
    out_dir = "figures"
    os.makedirs(out_dir, exist_ok=True)

    names = ["15%", "10%", "5%"]  # set your legend names here

    plot_two_gate_params_in_one_figure(
        steepness_csv=steepness_csv,
        cutoff_csv=cutoff_csv,
        out_path_pdf=os.path.join(out_dir, "gate_params_two_panel.pdf"),
        out_path_png=os.path.join(out_dir, "gate_params_two_panel.png"),
        curve_names=names,
        smooth=True,
        smooth_method="savgol",
        smooth_window=31,
        smooth_poly=3,
        dpi=300,
        xlabel_fs=13,   # <-- bigger x-label
        ylabel_fs=13,   # <-- bigger y-label
        cutoff_legend_bbox=(1.02, 0.62),  # <-- slightly higher legend
    )

    print("Saved plots to:", out_dir)
