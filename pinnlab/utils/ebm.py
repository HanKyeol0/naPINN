import torch
import torch.distributions as D
import torch.nn as nn
import torch.nn.functional as F


class EBM(nn.Module):
    """One-dimensional residual density estimator used by naPINN."""

    def __init__(
        self,
        hidden_dim: int = 32,
        depth: int = 3,
        num_grid: int = 256,
        lr: float = 1e-3,
        input_dim: int = 1,
        device: torch.device | str = "cpu",
    ):
        super().__init__()
        self.device = torch.device(device)
        self.num_grid = num_grid

        layers = []
        in_dim = input_dim
        for _ in range(depth - 1):
            layers.extend((nn.Linear(in_dim, hidden_dim), nn.Tanh()))
            in_dim = hidden_dim
        layers.append(nn.Linear(in_dim, 1))
        self.net = nn.Sequential(*layers)

        self.to(self.device)
        self.optimizer = torch.optim.Adam(self.parameters(), lr=lr)

    def forward(self, residual: torch.Tensor) -> torch.Tensor:
        return self.net(residual)

    @torch.no_grad()
    def make_grid(self) -> torch.Tensor:
        return torch.linspace(-10.0, 10.0, self.num_grid, device=self.device)

    def mean_nll(self, residual: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        residual = residual.detach().to(
            device=self.device, dtype=torch.float32
        ).view(-1, 1)

        grid = self.make_grid()
        log_q_grid = self.forward(grid.unsqueeze(-1)).squeeze(-1)
        maximum = log_q_grid.max()
        partition = torch.trapezoid(torch.exp(log_q_grid - maximum), grid)
        log_partition = torch.log(partition) + maximum

        log_q_residual = self.forward(residual).squeeze(-1)
        nll = -log_q_residual + log_partition
        return nll, nll.mean()

    def train_step(self, residual: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        self.train()
        nll, mean_nll = self.mean_nll(residual)
        self.optimizer.zero_grad()
        mean_nll.backward()
        self.optimizer.step()
        return nll.detach(), mean_nll.detach()


class TrainableGMM(nn.Module):
    """Trainable scalar Gaussian mixture used in the estimator ablation."""

    def __init__(
        self,
        n_components: int = 3,
        input_dim: int = 1,
        device: torch.device | str = "cpu",
        lr: float = 1e-2,
    ):
        super().__init__()
        self.device = torch.device(device)
        self.mix_logits = nn.Parameter(torch.zeros(n_components, device=self.device))
        self.means = nn.Parameter(
            torch.randn(n_components, input_dim, device=self.device) * 0.1
        )
        self.log_stds = nn.Parameter(
            torch.zeros(n_components, input_dim, device=self.device)
        )
        self.optimizer = torch.optim.Adam(self.parameters(), lr=lr)
        self.to(self.device)

    def get_distribution(self):
        mixture = D.Categorical(logits=self.mix_logits)
        components = D.Independent(
            D.Normal(self.means, torch.exp(self.log_stds)), 1
        )
        return D.MixtureSameFamily(mixture, components)

    def forward(self, residual: torch.Tensor) -> torch.Tensor:
        if residual.dim() == 1:
            residual = residual.unsqueeze(-1)
        return self.get_distribution().log_prob(residual).unsqueeze(-1)

    def train_step(self, residual: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        self.train()
        residual = residual.detach().to(self.device)
        if residual.dim() == 1:
            residual = residual.unsqueeze(-1)

        nll = -self.forward(residual).squeeze(-1)
        mean_nll = nll.mean()
        self.optimizer.zero_grad()
        mean_nll.backward()
        self.optimizer.step()
        return nll.detach(), mean_nll.detach()


class QuantileThresholdGate(nn.Module):
    """Residual-quantile gate used by the estimator-free ablation."""

    def __init__(
        self,
        quantile: float = 0.95,
        steepness: float = 10.0,
        device: torch.device | str = "cpu",
    ):
        super().__init__()
        self.quantile = quantile
        self.steepness = steepness
        self.device = torch.device(device)

    def forward(self, residual: torch.Tensor):
        magnitude = residual.abs().view(-1)
        with torch.no_grad():
            cutoff = torch.quantile(magnitude, self.quantile)
        weights = torch.sigmoid(self.steepness * (cutoff - magnitude))
        return weights.view(-1, 1), torch.tensor(0.0, device=self.device)


class LearnableThresholdGate(nn.Module):
    """Learnable residual-magnitude gate used by the estimator-free ablation."""

    def __init__(
        self,
        init_threshold: float = 1.0,
        init_steepness: float = 10.0,
        rejection_cost: float = 0.5,
        device: torch.device | str = "cpu",
    ):
        super().__init__()
        self.raw_threshold = nn.Parameter(
            torch.tensor(init_threshold, dtype=torch.float32, device=device)
        )
        self.raw_steepness = nn.Parameter(
            torch.tensor(init_steepness, dtype=torch.float32, device=device)
        )
        self.rejection_cost = rejection_cost

    def forward(self, residual: torch.Tensor):
        magnitude = residual.abs().view(-1)
        threshold = F.softplus(self.raw_threshold)
        steepness = F.softplus(self.raw_steepness)
        weights = torch.sigmoid(steepness * (threshold - magnitude))
        rejection_loss = self.rejection_cost * (1.0 - weights).mean()
        return weights.view(-1, 1), rejection_loss


class TrainableLikelihoodGate(nn.Module):
    """Trainable likelihood gate and rejection regularizer from the paper."""

    def __init__(
        self,
        init_cutoff_sigma: float = 2.0,
        init_steepness: float = 30.0,
        device: torch.device | str = "cpu",
        rejection_cost: float = 0.5,
    ):
        super().__init__()
        self.cutoff_alpha = nn.Parameter(
            torch.tensor(init_cutoff_sigma, dtype=torch.float32, device=device)
        )
        self.steepness = nn.Parameter(
            torch.tensor(init_steepness, dtype=torch.float32, device=device)
        )
        self.rejection_cost = rejection_cost

    def forward(self, log_density: torch.Tensor):
        with torch.no_grad():
            mean = log_density.mean()
            std = log_density.std() + 1e-6
        z_score = (log_density - mean) / std

        cutoff = F.softplus(self.cutoff_alpha)
        steepness = F.softplus(self.steepness)
        weights = torch.sigmoid(steepness * (z_score + cutoff))
        rejection_loss = self.rejection_cost * (1.0 - weights).mean()
        return weights, rejection_loss
