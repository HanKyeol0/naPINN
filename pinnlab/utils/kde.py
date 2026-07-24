import torch
import torch.nn as nn


class KDE(nn.Module):
    """Kernel Density Estimation for residual distribution.

    Non-parametric density estimator that maintains a buffer of recent residuals
    and uses Gaussian kernels to estimate log-density.

    Because KDE is non-parametric, ``train_step`` simply updates the internal
    sample buffer (no gradient-based optimisation).  A bandwidth parameter is
    maintained via Silverman's rule-of-thumb and optionally smoothed with EMA.
    """

    def __init__(
        self,
        max_samples: int = 2048,
        bandwidth: str = "silverman",
        input_dim: int = 1,
        device: torch.device | str = "cpu",
        **_kwargs,
    ):
        super().__init__()
        self.device = torch.device(device)
        self.input_dim = input_dim
        self.max_samples = max_samples
        self.bw_rule = bandwidth

        # Sample buffer (not a parameter — just persistent state)
        self.register_buffer("samples", torch.zeros(max_samples, input_dim, device=self.device))
        self.register_buffer("n_stored", torch.tensor(0, dtype=torch.long, device=self.device))
        self.register_buffer("_bandwidth", torch.tensor(1.0, device=self.device))

        # Dummy optimizer (no learnable params, but keeps the interface uniform)
        self.optimizer = _DummyOptimizer()

        self.to(self.device)
        print(f"[KDE] Initialized {input_dim}D KDE (max_samples={max_samples})")

    # ------------------------------------------------------------------
    # Bandwidth
    # ------------------------------------------------------------------
    def _silverman_bw(self, x: torch.Tensor) -> torch.Tensor:
        """Silverman's rule: h = 0.9 * min(std, IQR/1.34) * n^{-1/5}."""
        n = x.shape[0]
        if n < 2:
            return torch.tensor(1.0, device=self.device)
        std = x.std(dim=0).mean()
        # IQR approximation
        sorted_x = x.sort(dim=0).values
        q25 = sorted_x[max(int(0.25 * n) - 1, 0)].mean()
        q75 = sorted_x[min(int(0.75 * n), n - 1)].mean()
        iqr = (q75 - q25) / 1.34
        spread = torch.min(std, iqr) if iqr > 0 else std
        h = 0.9 * spread * (n ** (-0.2))
        return torch.clamp(h, min=1e-6)

    # ------------------------------------------------------------------
    # Buffer management
    # ------------------------------------------------------------------
    def _update_buffer(self, res: torch.Tensor):
        """Replace buffer contents with the most recent samples."""
        res = res.detach().to(self.device, dtype=torch.float32)
        if res.dim() == 1:
            res = res.unsqueeze(-1)
        n = min(res.shape[0], self.max_samples)
        self.samples[:n] = res[:n]
        self.n_stored.fill_(n)
        self._bandwidth = self._silverman_bw(self.samples[:n])

    # ------------------------------------------------------------------
    # Forward / log-density
    # ------------------------------------------------------------------
    def forward(self, r: torch.Tensor) -> torch.Tensor:
        """Return log-density estimates for query points *r*.

        Args:
            r: ``[N, input_dim]`` or ``[N, 1]`` query tensor.

        Returns:
            ``[N, 1]`` log-density values.
        """
        if r.dim() == 1:
            r = r.unsqueeze(-1)
        r = r.to(self.device, dtype=torch.float32)

        ns = int(self.n_stored.item())
        if ns == 0:
            return torch.zeros(r.shape[0], 1, device=self.device)

        samp = self.samples[:ns]  # [M, D]
        h = self._bandwidth

        # Gaussian kernel: K(u) = exp(-0.5 * ||u||^2)
        # diff shape: [N, M, D]
        diff = (r.unsqueeze(1) - samp.unsqueeze(0)) / h
        log_kernel = -0.5 * (diff ** 2).sum(dim=-1)  # [N, M]

        # log-sum-exp over M samples
        log_density = torch.logsumexp(log_kernel, dim=1) - torch.log(
            torch.tensor(float(ns), device=self.device) * h * torch.tensor(2 * torch.pi, device=self.device).sqrt()
        )
        return log_density.unsqueeze(-1)  # [N, 1]

    # ------------------------------------------------------------------
    # train_step  (interface-compatible with EBM)
    # ------------------------------------------------------------------
    def train_step(self, res: torch.Tensor):
        """Update the sample buffer (no gradient optimisation).

        Returns ``(nll, nll_mean)`` like EBM for logging compatibility.
        """
        res = res.detach()
        if res.dim() == 1:
            res = res.unsqueeze(-1)
        self._update_buffer(res)

        # Compute NLL on the *current* batch for logging
        with torch.no_grad():
            log_p = self.forward(res)
            nll = -log_p.squeeze(-1)
            nll_mean = nll.mean()
        return nll, nll_mean

class _DummyOptimizer:
    """Placeholder so that ``kde.optimizer.state_dict()`` works."""

    def state_dict(self):
        return {}

    def load_state_dict(self, state_dict):
        pass
