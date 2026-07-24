"""Factory for the EBM, GMM, and KDE residual-density estimators."""

from pinnlab.utils.ebm import EBM, TrainableGMM
from pinnlab.utils.kde import KDE


def create_density_estimator(ebm_cfg: dict, input_dim: int, device):
    """Instantiate a density estimator from the experiment config.

    Args:
        ebm_cfg: The ``ebm:`` section of the experiment YAML.
        input_dim: Residual dimensionality. The paper experiments use one.
        device: torch device.

    Returns:
        An ``nn.Module`` with the common density-estimator interface.
    """
    kind = ebm_cfg.get("density_estimator", "ebm").lower()

    if kind == "ebm":
        return EBM(
            hidden_dim=ebm_cfg.get("hidden_dim", 32),
            depth=ebm_cfg.get("depth", 3),
            num_grid=ebm_cfg.get("num_grid", 256),
            lr=ebm_cfg.get("lr", 1e-3),
            input_dim=input_dim,
            device=device,
        )

    elif kind == "gmm":
        return TrainableGMM(
            n_components=int(ebm_cfg.get("n_components", 3)),
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
