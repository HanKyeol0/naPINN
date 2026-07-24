import os

import matplotlib.pyplot as plt
import numpy as np

def _ensure_dir(p): os.makedirs(p, exist_ok=True)

def save_plots_2d(x, y, u_true, u_pred, out_dir, prefix):
    _ensure_dir(out_dir)
    err = np.abs(u_true - u_pred)
    paths = {}
    for name, arr in [("true", u_true), ("pred", u_pred), ("abs_error", err)]:
        fig = plt.figure()
        plt.imshow(arr, origin="lower", extent=[x.min(), x.max(), y.min(), y.max()])
        plt.colorbar(label=name)
        plt.xlabel("x"); plt.ylabel("y"); plt.title(f"{prefix} {name}")
        path = os.path.join(out_dir, f"{prefix}_{name}.png")
        fig.savefig(path, dpi=150, bbox_inches="tight"); plt.close(fig)
        paths[f"{prefix}_{name}"] = path
    return paths
