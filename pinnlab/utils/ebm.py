import torch
import torch.nn as nn
import torch.nn.functional as F

class EBM(nn.Module):
    """Simple 1D Energy-Based Model for residuals r in R.

    This model learns an unnormalized log-density log q_theta(r) via a small MLP.
    We train it by minimizing an approximate negative log-likelihood (NLL)
    using 1D numerical integration to estimate the partition function.

    The partition function is only needed during training; for per-point
    weighting we only need relative log-densities, so we can skip Z there.
    """
    def __init__(
        self,
        hidden_dim: int = 32,
        depth: int = 3,
        num_grid: int = 256,
        max_range_factor: float = 2.5,
        lr: float = 1e-3,
        device: torch.device | str = "cpu",
    ):
        super().__init__()
        self.device = torch.device(device)
        self.num_grid = num_grid
        self.max_range_factor = max_range_factor

        # Small MLP: 1 → hidden_dim → ... → hidden_dim → 1
        layers = []
        in_dim = 1 # input shape: [r, 1]
        for _ in range(depth - 1):
            layers.append(nn.Linear(in_dim, hidden_dim))
            layers.append(nn.Tanh())
            in_dim = hidden_dim
        layers.append(nn.Linear(in_dim, 1))
        self.net = nn.Sequential(*layers)

        self.to(self.device)
        self.optimizer = torch.optim.Adam(self.parameters(), lr=lr)

    def forward(self, r: torch.Tensor) -> torch.Tensor:
        """Return unnormalized log-density log q_theta(r).

        r: tensor of shape [..., 1] or [...]
        returns: same shape as r (broadcasted), containing log q_theta(r)
        """
        if r.dim() == 1:
            r = r.unsqueeze(-1)
            
        return self.net(r)

    @torch.no_grad()
    def _make_grid(self, res: torch.Tensor) -> torch.Tensor:
        """Make an integration grid based on residual range.

        We use a symmetric interval [-R, R] where R is proportional to the
        max absolute residual in the batch, scaled by max_range_factor.
        """
        res = res.view(-1)
        if res.numel() == 0:
            R = 1.0
        else:
            R = res.abs().max().item()
            if R <= 0:
                R = 1.0
        R = self.max_range_factor * R
        grid = torch.linspace(-R, R, self.num_grid, device=self.device)
        return grid

    def mean_nll(self, res: torch.Tensor) -> torch.Tensor:
        """Approximate mean negative log-likelihood of residuals.

        NLL(r) = -log q_theta(r) + log Z_theta
        where Z_theta = ∫ exp(log q_theta(s)) ds is approximated via
        1D trapezoidal integration on a grid.
        """
        res = res.detach().to(device=self.device, dtype=torch.float32).view(-1, 1)
        
        # data term: -log q_theta(r)
        log_q_res = self.forward(res).squeeze(-1) # [N]

        # partition term log Z
        grid = self._make_grid(res)
        grid_input = grid.unsqueeze(-1)  # [G,1]
        log_q_grid = self.forward(grid_input).squeeze(-1)
        m = log_q_grid.max()
        unnorm = torch.exp(log_q_grid - m)
        Z = torch.trapezoid(unnorm, grid)
        logZ = torch.log(Z + 1e-12) + m

        nll = -log_q_res + logZ
        nll_mean = nll.mean()
        return nll, nll_mean

    def train_step(self, res: torch.Tensor) -> torch.Tensor:
        """Perform one optimization step on a batch of residuals.

        Returns detached scalar NLL value (for logging).
        """
        self.train()
        nll, nll_mean = self.mean_nll(res)
        self.optimizer.zero_grad()
        nll_mean.backward()
        self.optimizer.step()
        return nll.detach(), nll_mean.detach()

    @torch.no_grad()
    def data_weight(self, res: torch.Tensor, kind: str = "pw") -> torch.Tensor:
        if kind == "pw":
            return self.pointwise_weights(res)
        elif kind == "inverse":
            return self.inverse_pointwise_weights(res)
        else:
            raise ValueError(f"Unknown data weight kind: {kind}")
    
    def pointwise_weights(self, res: torch.Tensor) -> torch.Tensor:
        """Compute noise-adaptive weights for each residual.

        We use w_i ∝ exp(log q_theta(r_i)) and normalize such that
        mean(w_i) ≈ 1 (so the global scale of the data loss is preserved).

        These weights can be treated as reliability weights: points
        that are more likely under the learned noise distribution
        receive larger weight, and outliers receive smaller weight.
        """
        res = res.detach().to(device=self.device, dtype=torch.float32)
        orig_shape = res.shape
        res_flat = res.view(-1, 1)
        log_q = self.forward(res_flat).squeeze(-1)
        log_q = log_q - log_q.max()  # shift for stability
        w = torch.exp(log_q)
        w = w / (w.mean() + 1e-8)
        return w.view(orig_shape)
    
    def inverse_pointwise_weights(self, res: torch.Tensor):
        res = res.detach().to(self.device)
        eps = 1e-8
        orig_shape = res.shape
        res_flat = res.view(-1, 1)

        # log q(r)
        log_q = self.forward(res_flat).squeeze(-1)          # [N]
        # 수치 안정화
        log_q = log_q - log_q.max()

        # q(r) ≈ exp(log_q)
        q = torch.exp(log_q)                                # [N]

        # 1 / q(r)
        w_raw = 1.0 / (q + eps)

        # 평균을 1로 정규화
        w = w_raw / (w_raw.mean() + 1e-8)
        return w.view(orig_shape)

class ResidualWeightNet(nn.Module):
    """
    Small auxiliary network that maps residuals r -> per-point positive weights w(r).

    - Input:  residual tensor of shape [N, 1] or [N]
    - Output: weights of shape [N, 1], positive and normalized to mean ≈ 1
    """
    def __init__(self, hidden_dim: int = 32, depth: int = 2, device: str | torch.device = "cpu"):
        super().__init__()
        self.device = torch.device(device)

        layers = []
        in_dim = 1
        for _ in range(max(depth - 1, 0)):
            layers.append(nn.Linear(in_dim, hidden_dim))
            layers.append(nn.Tanh())
            in_dim = hidden_dim
        layers.append(nn.Linear(in_dim, 1))
        self.net = nn.Sequential(*layers)

        self.to(self.device)

    def forward(self, res: torch.Tensor) -> torch.Tensor:
        """
        res: [N] or [N,1] residuals (can be detached in the caller).

        Returns w: [N,1] positive weights with mean ≈ 1.
        """
        if res.dim() == 1:
            res = res.unsqueeze(-1)  # [N] -> [N,1]
        r = res.to(self.device, dtype=torch.float32)

        # free-form MLP
        w_raw = self.net(r)

        # Ensure positivity and avoid zero:
        w_pos = F.softplus(w_raw) + 1e-6  # [N,1]

        # Normalize to mean ~1, but don't send gradients through the denominator
        denom = w_pos.mean().detach() + 1e-8
        w_norm = w_pos / denom

        return w_norm
