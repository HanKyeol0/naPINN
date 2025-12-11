import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.distributions as D

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
        input_dim: int = 1,
        device: torch.device | str = "cpu",
    ):
        super().__init__()
        self.device = torch.device(device)
        self.num_grid = num_grid
        self.max_range_factor = max_range_factor

        # Small MLP: 1 → hidden_dim → ... → hidden_dim → 1
        layers = []
        in_dim = input_dim # input shape: [r, 1]
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

class EBM2D(nn.Module):
    """Simple 2D Energy-Based Model for residuals r in R^2.

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
        input_dim: int = 1,
        device: torch.device | str = "cpu",
    ):
        super().__init__()
        self.device = torch.device(device)
        self.num_grid = num_grid
        self.max_range_factor = max_range_factor

        # Small MLP: 2 → hidden_dim → ... → hidden_dim → 1
        layers = []
        in_dim = input_dim
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

    def _make_grid_2d(self, res: torch.Tensor, grid_size: int = 100) -> tuple[torch.Tensor, float]:
        """
        Creates a 2D grid for integration.
        Returns:
            grid_flat: [grid_size*grid_size, 2]
            cell_area: scalar area of one grid cell (dx * dy)
        """
        # Determine range based on current batch residuals
        with torch.no_grad():
            res_abs = res.abs()
            # Use the max extent in either u or v direction to keep aspect ratio or separate
            R_u = res_abs[:, 0].max().item()
            R_v = res_abs[:, 1].max().item()
            
            # Avoid zero range
            R_u = max(R_u * self.max_range_factor, 1.0)
            R_v = max(R_v * self.max_range_factor, 1.0)

        # Create 1D linspaces
        u_grid = torch.linspace(-R_u, R_u, grid_size, device=self.device)
        v_grid = torch.linspace(-R_v, R_v, grid_size, device=self.device)

        # Create meshgrid
        # indexing='ij' ensures (u, v) ordering matches standard matrix indexing
        uu, vv = torch.meshgrid(u_grid, v_grid, indexing='ij') 
        
        # Flatten to [N_grid, 2]
        grid_flat = torch.stack([uu.flatten(), vv.flatten()], dim=1)
        
        # Calculate cell area for integration: dx * dy
        dx = (2 * R_u) / (grid_size - 1)
        dy = (2 * R_v) / (grid_size - 1)
        cell_area = dx * dy
        
        return grid_flat, cell_area

    def mean_nll(self, res: torch.Tensor) -> tuple[torch.Tensor, float]:
        """
        Computes NLL with 2D numerical integration for Z.
        """
        res = res.detach().to(self.device, dtype=torch.float32) # [N, 2]
        
        # 1. Unnormalized log-probability of data: log q(r)
        # forward returns [N, 1], squeeze to [N]
        log_q_data = self.forward(res).squeeze(-1) 
        
        # 2. Estimate Partition Function Z via 2D Integration
        # We need to evaluate the EBM on the whole grid to find the volume
        grid_flat, cell_area = self._make_grid_2d(res, grid_size=80) # 80x80 is usually enough
        
        log_q_grid = self.forward(grid_flat).squeeze(-1) # [GridSize^2]
        
        # Log-Sum-Exp trick for numerical stability
        # Z = sum(exp(log_q)) * area
        # log Z = log(sum(exp(log_q))) + log(area)
        m = log_q_grid.max()
        # sum(exp(x - m))
        sum_exp = torch.sum(torch.exp(log_q_grid - m)) 
        log_Z = m + torch.log(sum_exp) + torch.log(torch.tensor(cell_area, device=self.device))
        
        # 3. NLL = -log_q_data + log_Z
        nll = -log_q_data + log_Z
        
        return nll, nll.mean()

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
    
    def gated_weights(self, res: torch.Tensor, alpha: float = 2.0, steepness: float = 5.0) -> torch.Tensor:
        """
        Computes weights using a Soft Sigmoid Gate on Log-Likelihood.
        
        Logic:
           - Standard 'pw' weighting (w ~ p(x)) penalizes tails too aggressively.
           - This method creates a 'plateau' of trust. If a point's log-likelihood 
             is within 'alpha' std-devs of the mean, weight is ~1.0. 
             If it drops below that, weight slides to 0.0.
             
        Args:
            res: Residuals [N, 2]
            alpha: Threshold factor. Cutoff = Mean_LL - alpha * Std_LL.
                   Higher alpha = more tolerant (includes more tails).
            steepness: How sharp the transition is from weight 1 to 0.
        """
        res = res.detach().to(device=self.device, dtype=torch.float32)
        orig_shape = res.shape # [N, 2]
        
        # 1. Compute Unnormalized Log-Likelihoods (Energy)
        # We don't need Z (partition function) because it's constant for the batch
        log_q = self.forward(res) # [N, 1]
        log_q = log_q.squeeze(-1) # Flatten to [N] for stats
        
        # 2. Compute Dynamic Threshold based on Batch Stats
        # We use robust stats (median/quantiles) if possible, but mean/std is faster/stable
        with torch.no_grad():
            mu_ll = log_q.mean()
            sigma_ll = log_q.std()
            
            # The Cutoff: Points below this likelihood are considered "suspicious"
            # e.g., if alpha=2.0, we keep the top ~95% of the probability mass (roughly)
            cutoff = mu_ll - alpha * sigma_ll

        # 3. Apply Sigmoid Gate
        # w = 1 / (1 + exp(-steepness * (log_q - cutoff)))
        # If log_q > cutoff, exp is small neg, w -> 1
        # If log_q < cutoff, exp is large pos, w -> 0
        w = torch.sigmoid(steepness * (log_q - cutoff))
        
        # 4. Normalize?
        # In this specific 'gating' philosophy, we arguably usually want weights 
        # to be exactly 1.0 for good data, not scaled to mean=1. 
        # However, to keep learning rates consistent with your previous code:
        w = w / (w.mean() + 1e-8)
        
        return w.view(-1, 1)  # [N, 1]

    @torch.no_grad()
    def data_weight(self, res: torch.Tensor, kind: str = "pw") -> torch.Tensor:
        if kind == "pw":
            return self.pointwise_weights(res)
        elif kind == "inverse":
            return self.inverse_pointwise_weights(res)
        elif kind == "gated":
            # You can tune alpha via config if you pass it down, 
            # but standard deviations of 2.0-3.0 are usually good outlier boundaries.
            return self.gated_weights(res, alpha=2.5, steepness=10.0)
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
        log_q = self.forward(res)
        log_q = log_q - log_q.max()  # shift for stability
        w = torch.exp(log_q)
        w = w / (w.mean() + 1e-8)
        return w
    
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
    
class TrainableGMM(nn.Module):
    def __init__(self, n_components=3, input_dim=2, device='cpu', lr=1e-2):
        super().__init__()
        self.device = device
        self.n_components = n_components
        self.input_dim = input_dim
        
        # 1. Mixture weights (logits)
        self.mix_logits = nn.Parameter(torch.zeros(n_components, device=device))
        
        # 2. Means (initialized near 0, but slightly spread)
        self.means = nn.Parameter(torch.randn(n_components, input_dim, device=device) * 0.1)
        
        # 3. Covariances (Diagonal for stability, parameterized by log_std)
        # Initializing one narrow mode (clean data) and wider modes (outliers)
        self.log_stds = nn.Parameter(torch.zeros(n_components, input_dim, device=device))
        
        self.optimizer = torch.optim.Adam(self.parameters(), lr=lr)
        self.to(device)

    def get_distribution(self):
        mix = D.Categorical(logits=self.mix_logits)
        # Diagonal covariance multivariate gaussians
        comp = D.Independent(D.Normal(self.means, torch.exp(self.log_stds)), 1)
        gmm = D.MixtureSameFamily(mix, comp)
        return gmm

    def forward(self, x):
        """Returns log probability log p(x)"""
        gmm = self.get_distribution()
        return gmm.log_prob(x) # [N]

    def train_step(self, res):
        """Minimize Negative Log Likelihood"""
        self.train()
        res = res.detach().to(self.device)
        
        log_prob = self.forward(res)
        nll = -log_prob
        nll_mean = nll.mean()
        
        self.optimizer.zero_grad()
        nll_mean.backward()
        self.optimizer.step()
        
        return nll.detach(), nll_mean.detach()

    @torch.no_grad()
    def pointwise_weights(self, res):
        """
        Standardize weights: w = p(x) / mean(p(x))
        """
        res = res.to(self.device)
        log_prob = self.forward(res)
        
        # Shift for stability inside exp (though less critical for GMM than raw MLP)
        # Using raw likelihoods usually works well for GMM
        prob = torch.exp(log_prob)
        
        # Normalize so mean weight is 1.0
        w = prob / (prob.mean() + 1e-8)
        
        # Return [N, 1] shape
        return w.view(-1, 1)
    
class TrainableLikelihoodGate(nn.Module):
    """
    Learns a soft cutoff for outlier rejection based on Log-Likelihood.
    
    Forward:
        1. Takes Log-Likelihoods (log_q) from the EBM.
        2. Standardizes them (Z-score) using batch statistics.
        3. Applies a Sigmoid Gate with trainable Cutoff and Steepness.
        4. Normalizes weights so mean(w) = 1.
    """
    def __init__(self, init_cutoff_sigma: float = 2.0, init_steepness: float = 5.0, device="cpu"):
        super().__init__()
        self.device = device
        
        # 1. Cutoff (alpha): 
        # Points below (Mean - alpha * Std) will be rejected.
        # We initialize it to ~2.0 sigma.
        self.cutoff_alpha = nn.Parameter(torch.tensor(float(init_cutoff_sigma), device=device))
        
        # 2. Steepness (beta): 
        # Controls how sharp the transition is from "trust" to "reject".
        self.steepness = nn.Parameter(torch.tensor(float(init_steepness), device=device))
        
        self.to(device)

    def forward(self, log_q: torch.Tensor) -> torch.Tensor:
        # log_q shape: [N] or [N, 1]
        
        # --- 1. Robust Standardization ---
        # We use batch stats to make the cutoff relative to the current noise level
        with torch.no_grad():
            mu = log_q.mean()
            sigma = log_q.std() + 1e-6
            
            # Detach stats so we don't try to backprop through the batch mean calculation
            # We only want to learn WHERE to cut, not move the mean.
            mu = mu.detach()
            sigma = sigma.detach()

        # Z-scored likelihoods
        z_scores = (log_q - mu) / sigma
        
        # --- 2. Gating Function ---
        # We want to keep points where z > -alpha
        # So we want sigmoid( z - (-alpha) ) -> sigmoid( z + alpha )
        # To make "alpha" intuitive (positive value = std devs below mean), we use:
        # Gate = Sigmoid( steepness * (z_score + cutoff_alpha) )
        
        # If z_score = -2.0 and cutoff_alpha = 2.0 -> input is 0 -> weight 0.5
        # If z_score = -1.0 (better) -> input is positive -> weight ~1.0
        # If z_score = -3.0 (worse) -> input is negative -> weight ~0.0
        
        # Softplus on params to ensure they stay positive
        alpha = torch.nn.functional.softplus(self.cutoff_alpha)
        beta = torch.nn.functional.softplus(self.steepness)
        
        raw_w = torch.sigmoid(beta * (z_scores + alpha))
        
        # --- 3. Normalization (Constraint: Mean=1) ---
        # This prevents the trivial solution where the model just outputs all zeros.
        w = raw_w / (raw_w.mean() + 1e-8)
        
        return w