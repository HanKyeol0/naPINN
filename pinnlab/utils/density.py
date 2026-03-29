"""Factory for density estimators (EBM, GMM, KDE).

All estimators expose a common interface:
    - forward(r)       -> [N, 1] log-density
    - train_step(res)  -> (nll, nll_mean) detached
    - pointwise_weights(res)  -> [N, 1]
    - gated_weights(res, ...)  -> [N, 1]
    - data_weight(res, kind)   -> [N, 1]
    - optimizer            (has .state_dict / .load_state_dict)
    - state_dict / load_state_dict  (nn.Module default)
"""

from pinnlab.utils.ebm import EBM, EBM2D, TrainableGMM
from pinnlab.utils.kde import KDE


def create_density_estimator(ebm_cfg: dict, input_dim: int, device):
    """Instantiate a density estimator from the experiment config.

    Args:
        ebm_cfg: The ``ebm:`` section of the experiment YAML.
        input_dim: Dimensionality of residuals (1 or 2).
        device: torch device.

    Returns:
        An ``nn.Module`` with the common density-estimator interface.
    """
    kind = ebm_cfg.get("density_estimator", "ebm").lower()

    if kind == "ebm":
        ebm_kind = ebm_cfg.get("kind", "1D")
        if ebm_kind == "2D" and input_dim == 2:
            cls = EBM2D
        else:
            cls = EBM
        return cls(
            hidden_dim=ebm_cfg.get("hidden_dim", 32),
            depth=ebm_cfg.get("depth", 3),
            num_grid=ebm_cfg.get("num_grid", 256),
            max_range_factor=ebm_cfg.get("max_range_factor", 2.5),
            lr=ebm_cfg.get("lr", 1e-3),
            input_dim=input_dim,
            device=device,
        )

    elif kind == "gmm":
        return TrainableGMM(
            n_components=int(ebm_cfg.get("n_components", 5)),
            input_dim=input_dim,
            device=device,
            lr=ebm_cfg.get("lr", 1e-3),
        )

    elif kind == "kde":
        return KDE(
            max_samples=int(ebm_cfg.get("max_samples", 2048)),
            bandwidth=ebm_cfg.get("bandwidth", "silverman"),
            input_dim=input_dim,
            device=device,
        )

    else:
        raise ValueError(
            f"Unknown density_estimator: '{kind}'. "
            f"Choose from: ebm, gmm, kde"
        )
