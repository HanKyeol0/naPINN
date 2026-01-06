# pinnlab/experiments/allencahn2d.py
import math, os
import sys
from sklearn.metrics import confusion_matrix
import seaborn as sns
import torch
import numpy as np
from matplotlib import pyplot as plt
import imageio
from tqdm import trange
from pinnlab.experiments.base import BaseExperiment, make_leaf, grad_sum
from pinnlab.data.geometries import Rectangle, linspace_2d
from pinnlab.data.noise import get_noise
from pinnlab.utils.plotting import save_plots_2d
from pinnlab.utils.ebm import EBM, ResidualWeightNet, TrainableLikelihoodGate
from pinnlab.utils.data_loss import (
    data_loss_mse,
    data_loss_l1,
    data_loss_q_gaussian,
)

class AllenCahn2D(BaseExperiment):
    r"""
    Allen–Cahn (2D, time-dependent) with manufactured forcing:

        u_t - ε^2 (u_xx + u_yy) + (u^3 - u) = f(x,y,t),  (x,y)∈[xa,xb]×[ya,yb], t∈[t0,t1]

    Manufactured solution:
        u*(x,y,t) = sin(π x) sin(π y) cos(ω t)

    Then
        f = u*_t - ε^2 Δu* + (u*^3 - u*)
    with Dirichlet BC/IC from u*.

    cfg:
      domain: {x: [xa, xb], y: [ya, yb], t: [t0, t1]}
      eps: 0.01
      omega: 2.0
    """

    def __init__(self, cfg, device):
        super().__init__(cfg, device)
        self.device = device
        
        xa, xb = cfg["domain"]["x"]; ya, yb = cfg["domain"]["y"]
        self.t0, self.t1 = cfg["domain"]["t"]
        self.rect  = Rectangle(xa, xb, ya, yb, device)
        
        # PDE parameters
        self.pde_cfg = cfg.get("pde", {}) or {}
        self.true_eps = float(self.pde_cfg.get("eps", 0.01))
        self.true_omega = float(self.pde_cfg.get("omega", 2.0))
        self.learn_eps = self.pde_cfg.get("learn_eps", True)
        
        if self.learn_eps:
            init_eps = float(self.pde_cfg.get("init_eps", 0.0))
            print(f"[AllenCahn2D] PDE Parameter ε (eps) is TRAINABLE. Init: {init_eps}, True: {self.true_eps}")
            self.eps = torch.nn.Parameter(torch.tensor(init_eps, dtype=torch.float32, device=device))
        else:
            print(f"[AllenCahn2D] PDE Parameter ε (eps) is FIXED. Value: {self.true_eps}")
            self.eps = self.true_eps
            
        # set omega fixed for now
        self.omega = self.true_omega
        
        self.sampling_mode = cfg.get("sampling_mode", "random") # random / grid
        if self.sampling_mode == "grid":
            g = cfg.get("grid", {})
            nx, ny, nt = int(g.get("nx", 100)), int(g.get("ny", 100)), int(g.get("nt", 100))
            
            Xg, Yg = linspace_2d(self.rect.xa, self.rect.xb, self.rect.ya, self.rect.yb, nx, ny, device)
            XY = torch.stack([Xg.reshape(-1), Yg.reshape(-1)], dim=1)  # [nx*ny, 2]
            self._XY = XY
            self._T = torch.linspace(self.t0, self.t1, nt, device=device) # [nt]
            
            # boundary subset (any edge)
            xa, xb, ya, yb = self.rect.xa, self.rect.xb, self.rect.ya, self.rect.yb
            mask_b = XY[:,0].eq(xa) | XY[:,0].eq(xb) | XY[:,1].eq(ya) | XY[:,1].eq(yb)
            self._XYb = XY[mask_b]  # [nb, 2]
            
        noise_cfg = cfg.get("noise", None)
        self.use_data = bool(noise_cfg.get("enabled", False))
        self.noise_cfg = noise_cfg
        self.n_data_total = int(noise_cfg.get("n_train", 0))
        self.n_data_batch = int(
            noise_cfg.get("batch_size", 1000)
        )
        
        self.extra_noise_cfg = noise_cfg.get("extra_noise", {})
        self.use_extra_noise = bool(self.extra_noise_cfg.get("enabled", False))
        self.extra_noise_mask = None
        print(f"[AllenCahn2D] use_extra_noise = {self.use_extra_noise}")
        
        self.X_data = None
        self.y_data = None
        self.y_clean = None
        self.noise_model = None
        
        if self.use_data and self.n_data_total > 0:
            print(f"Initializing noisy data with {self.n_data_total} points.")
            self._init_noisy_dataset()

        ebm_cfg = cfg.get("ebm", {}) or {}
        self.use_ebm = bool(ebm_cfg.get("enabled", False))
        self.use_nll = bool(ebm_cfg.get("use_nll", False))
        self.ebm_init_train_epochs = int(ebm_cfg["init_train_epochs"])
        
        if self.use_ebm:
            self.ebm = EBM(
                hidden_dim=ebm_cfg.get("hidden_dim", 32),
                depth=ebm_cfg.get("depth", 3),
                num_grid=ebm_cfg.get("num_grid", 256),
                max_range_factor=ebm_cfg.get("max_range_factor", 2.5),
                lr=ebm_cfg.get("lr", 1e-3),
                input_dim=1,
                device=device,
            )
            self.running_std = torch.tensor(1.0, device=device)
            self.momentum = 0.05
        else:
            self.ebm = None

        self.use_phase = bool(cfg.get("phase", {}).get("enabled", False))
        
        data_loss_cfg = cfg.get("data_loss", {}) or {}
        self.data_loss_kind = data_loss_cfg.get("kind", "mse")       
        # q-Gaussian settings
        self.q_gauss_q = float(data_loss_cfg.get("q", 1.2))
        beta_val = data_loss_cfg.get("beta", None)
        self.q_gauss_beta = float(beta_val) if beta_val is not None else None
        
        data_lb_cfg = cfg.get("data_loss_balancer", {})
        self.use_data_loss_balancer = bool(data_lb_cfg.get("use_loss_balancer", False))
        self.data_loss_balancer_kind = data_lb_cfg.get("kind", "pw")  # 'pw', 'inverse', ...
        
        self.gate_module = None
        if self.data_loss_balancer_kind == "gated_trainable":
            self.rejection_cost = float(data_lb_cfg.get("rejection_cost", 0.5))
            self.gate_module = TrainableLikelihoodGate(device=device, rejection_cost=self.rejection_cost)
        
        # --- Learnable per-point data weights via auxiliary MLP ----
        self.weight_net = None
        if self.use_data_loss_balancer and self.data_loss_balancer_kind == "mlp":
            wn_cfg = data_lb_cfg.get("weight_net", {}) or {}
            dlb_hidden_dim = int(wn_cfg.get("hidden_dim", 32)) # dlb: data loss balancer
            dlb_depth = int(wn_cfg.get("depth", 2))
            self.weight_net = ResidualWeightNet(
                hidden_dim=dlb_hidden_dim,
                depth=dlb_depth,
                device=device,
            )
            print(f"[AllenCahn2D] Using learnable per-point data weights "
                  f"with MLP(hidden_dim={dlb_hidden_dim}, depth={dlb_depth}).")
        
        print(f"[AllenCahn2D] use_ebm = {self.use_ebm}, use_nll = {self.use_nll}")
        print(f"[AllenCahn2D] use_phase = {self.use_phase}")
        print(f"[AllenCahn2D] data_loss_kind = {self.data_loss_kind}")
        print(f"[AllenCahn2D] use_data_loss_balancer = {self.use_data_loss_balancer}")
        print(f"[AllenCahn2D] data_loss_balancer_kind = {self.data_loss_balancer_kind}")
        
        # ----- Optional trainable offset θ0 for non-zero mean noise (PINN-off style) -----
        offset_cfg = cfg.get("offset", {}) or {}
        self.use_offset = bool(offset_cfg.get("enabled", False))
        if self.use_offset:
            init = float(offset_cfg.get("init", 0.0))
            # scalar parameter θ0 that will only be used in the DATA term
            self.offset = torch.nn.Parameter(
                torch.tensor(init, dtype=torch.float32, device=device)
            )
            print(f"[AllenCahn2D] Using trainable data offset θ0, init={init}")
        else:
            self.offset = None
            
    def state_dict(self):
        # --- MODIFICATION 2: Save learned parameters ---
        state = {
            'running_std': self.running_std,
            'offset': self.offset,
        }
        if self.learn_eps:
            state['eps'] = self.eps
            
        if self.use_offset and self.offset is not None:
            state['offset'] = self.offset
            
        if self.use_ebm and self.ebm is not None:
            state['ebm'] = self.ebm.state_dict()
            state['ebm_optimizer'] = self.ebm.optimizer.state_dict()

        # Save Gate state if it exists
        if self.gate_module is not None:
            state['gate_module'] = self.gate_module.state_dict()
            
        if self.weight_net is not None:
            state['weight_net'] = self.weight_net.state_dict()

        return state
    
    def load_state_dict(self, state_dict):
        """
        Loads the experiment's optimization state.
        """
        if 'running_std' in state_dict:
            self.running_std.copy_(state_dict['running_std'].to(self.device))
            print(f"[AllenCahn2D] Loaded running_std: {self.running_std.item():.4f}")
            
        if 'offset' in state_dict and self.offset is not None:
            with torch.no_grad():
                self.offset.copy_(state_dict['offset'].to(self.device))
                
        if 'eps' in state_dict and self.learn_eps:
            with torch.no_grad():
                self.eps.copy_(state_dict['eps'].to(self.device))
                print(f"[AllenCahn2D] Loaded learned eps: {self.eps.item():.6f}")
                
        if 'ebm' in state_dict and self.ebm is not None:
            self.ebm.load_state_dict(state_dict['ebm'])
            if 'ebm_optimizer' in state_dict:
                self.ebm.optimizer.load_state_dict(state_dict['ebm_optimizer'])
            
        if 'gate_module' in state_dict and self.gate_module is not None:
            self.gate_module.load_state_dict(state_dict['gate_module'])
            
        if 'weight_net' in state_dict and self.weight_net is not None:
            self.weight_net.load_state_dict(state_dict['weight_net'])

    # ----- manufactured truth -----
    def u_star(self, x, y, t):
        return torch.sin(math.pi * x) * torch.sin(math.pi * y) * torch.cos(self.omega * t)

    def f(self, x, y, t):
        u   = self.u_star(x, y, t)
        ut  = -self.omega * torch.sin(math.pi * x) * torch.sin(math.pi * y) * torch.sin(self.omega * t)
        lap = -2.0 * (math.pi**2) * self.u_star(x, y, t)  # Δu* = u_xx + u_yy
        return ut - (self.true_eps**2) * lap + (u**3 - u)

    def _init_noisy_dataset(self):
        """
        Generate a fixed set of noisy measurements:
            X_data ~ Uniform(domain x time)
            y_clean = u_star(X_data)
            y_data  = y_clean + epsilon,     epsilon ~ noise distribution (from PINN-EBM)
        """
        kind = self.noise_cfg.get("kind", "3G")     # 'G', 'u', '3G', ...
        n = self.n_data_total

        # Sample input locations
        XY = self.rect.sample(n)  # [n, 2] in (x, y)
        t = torch.rand(n, 1, device=self.device) * (self.t1 - self.t0) + self.t0
        X = torch.cat([XY, t], dim=1)  # [n, 3]

        with torch.no_grad():
            u_clean = self.u_star(X[:, 0:1], X[:, 1:2], X[:, 2:3])  # [n, 1]
            
        base_dtype = u_clean.dtype

        # Determine noise scale "f" as in PINN-EBM: a factor times average magnitude of u*
        base_scale = float(self.noise_cfg.get("scale", 0.1))  # relative to mean |u|
        mean_level = float(u_clean.abs().mean().detach().cpu())
        self.sigma_local = base_scale * mean_level
        
        # Build noise distribution; this uses the PINN-EBM function
        self.noise_model = get_noise(kind, f=1.0, pars=0)
        z = self.noise_model.sample(n).float().to(self.device, dtype=base_dtype).view(-1, 1)  # [n, 1]
        eps = z * self.sigma_local
        
        # Outliers
        self.extra_noise_mask = torch.zeros(n, dtype=torch.bool)
        if self.use_extra_noise:
            n_extra = int(self.extra_noise_cfg.get("n_points", 0))
            print(f"Adding extra noise to {n_extra} points.")
            idx = torch.randperm(n, device=self.device)[:n_extra]
            self.outlier_indices = idx.cpu().numpy()
            
            scale_min = float(self.extra_noise_cfg.get("scale_min", 5.0))
            scale_max = float(self.extra_noise_cfg.get("scale_max", 10.0))

            # scale factor sampling for each noise point: scale factor ~ Uniform(scale_min, scale_max)
            factors = torch.empty(n_extra, 1, device=self.device, dtype=base_dtype).uniform_(scale_min, scale_max)
            f_outlier = base_scale * mean_level

            # amplitude_i in [scale_min * f, scale_max * f]
            amp = factors * f_outlier
            signs = torch.randint(0, 2, amp.shape, device=self.device, dtype=amp.dtype) * 2 - 1
            eps[idx] = signs * amp

        y_noisy = u_clean + eps

        self.X_data = X
        self.y_clean = u_clean
        self.y_data = y_noisy

    def _data_loss(self, residual: torch.Tensor) -> torch.Tensor:
        """
        Map residuals r = y_noisy - u_pred (maybe with offset)
        to per-point losses ℓ_i according to data_loss_kind:

            'mse'        → ℓ_i = r_i^2             (vanilla PINN)
            'l1'         → ℓ_i = |r_i|             (LAD-PINN)
            'q_gaussian' → Tsallis q-Gaussian NLL  (OrPINN)
        """
        kind = self.data_loss_kind
        if kind == "mse":
            return data_loss_mse(residual)
        elif kind == "L1":
            return data_loss_l1(residual)
        elif kind == "q_gaussian":
            return data_loss_q_gaussian(
                residual,
                q=self.q_gauss_q,
                beta=self.q_gauss_beta,
            )
        return residual.pow(2)

    # ----- batching -----
    def sample_batch(self, n_f: int, n_b: int, n_0: int):
        device = self.rect.device

        # interior (x,y,t)
        XY  = self.rect.sample(n_f)
        t_f = torch.rand(n_f, 1, device=device) * (self.t1 - self.t0) + self.t0
        X_f = torch.cat([XY, t_f], dim=1)

        # boundary on 4 edges (Dirichlet from u*)
        nb = max(1, n_b // 4)
        xa, xb, ya, yb = self.rect.xa, self.rect.xb, self.rect.ya, self.rect.yb
        t_b = lambda m: torch.rand(m,1,device=device)*(self.t1-self.t0)+self.t0

        # x=xa|xb
        yL = torch.rand(nb,1,device=device)*(yb-ya)+ya; TL = t_b(nb)
        yR = torch.rand(nb,1,device=device)*(yb-ya)+ya; TR = t_b(nb)
        Xb_L = torch.cat([torch.full_like(yL, xa), yL, TL], dim=1)
        Xb_R = torch.cat([torch.full_like(yR, xb), yR, TR], dim=1)

        # y=ya|yb
        xB = torch.rand(nb,1,device=device)*(xb-xa)+xa; TB = t_b(nb)
        xT = torch.rand(nb,1,device=device)*(xb-xa)+xa; TT = t_b(nb)
        Xb_B = torch.cat([xB, torch.full_like(xB, ya), TB], dim=1)
        Xb_T = torch.cat([xT, torch.full_like(xT, yb), TT], dim=1)

        X_b = torch.cat([Xb_L, Xb_R, Xb_B, Xb_T], dim=0)
        u_b = self.u_star(X_b[:,0:1], X_b[:,1:2], X_b[:,2:3])

        # initial condition at t = t0
        XY0 = self.rect.sample(n_0)
        X_0 = torch.cat([XY0, torch.full((n_0,1), self.t0, device=device)], dim=1)
        u0  = self.u_star(X_0[:,0:1], X_0[:,1:2], X_0[:,2:3])

        batch = {"X_f": X_f, "X_b": X_b, "u_b": u_b, "X_0": X_0, "u0": u0}
        
        if self.use_data and self.X_data is not None and self.n_data_batch > 0:
            n = self.X_data.size(0)
            k = min(self.n_data_batch, n)
            idx = torch.randint(0, n, (k,), device=self.device)
            batch["X_d"] = self.X_data[idx]
            batch["y_d"] = self.y_data[idx]

        return batch
    
    def initialize_EBM(self, model):
        use_tty = sys.stdout.isatty()
        pbar_ebm = trange(
            self.ebm_init_train_epochs,
            desc="Initialize EBM",
            ncols=120,
            dynamic_ncols=True,
            leave=False,
            disable=not use_tty
        )
        n_data = self.X_data.shape[0]
        k = min(self.n_data_batch, n_data)
        print("[EBM Init] Training EBM on initial residuals...")
        for ep in pbar_ebm:
            idx_d = torch.randint(0, n_data, (k,), device=self.device)
            X_d = self.X_data[idx_d]
            y_d = self.y_data[idx_d]
            pred = model(X_d)
            residual = y_d - pred
            if self.ebm_kind == "1D":
                residual = residual.view(-1, 1)
            with torch.no_grad():
                batch_std = residual.std()
                currend_std_clamped = torch.clamp(batch_std, min=1e-6, max=self.running_std * 10)
                self.running_std.mul(1 - self.momentum).add_(currend_std_clamped * self.momentum)
            residual_scaled = residual / self.running_std
            nll_ebm, nll_ebm_mean = self.ebm.train_step(residual_scaled.detach())

    # ----- losses -----
    def pde_residual_loss(self, model, batch):
        X = make_leaf(batch["X_f"])                # (x,y,t)
        x, y, t = X[:,0:1], X[:,1:2], X[:,2:3]
        u   = model(X)
        du  = grad_sum(u, X)                       # (u_x, u_y, u_t)
        u_x, u_y, u_t = du[:,0:1], du[:,1:2], du[:,2:3]
        d2u_x = grad_sum(u_x, X)                   # (u_xx, u_xy, u_xt)
        d2u_y = grad_sum(u_y, X)                   # (u_yx, u_yy, u_yt)
        u_xx, u_yy = d2u_x[:,0:1], d2u_y[:,1:2]
        res = u_t - (self.eps**2) * (u_xx + u_yy) + (u**3 - u) - self.f(x, y, t)
        return res.pow(2)

    def boundary_loss(self, model, batch):
        return (model(batch["X_b"]) - batch["u_b"]).pow(2)

    def initial_loss(self, model, batch):
        return (model(batch["X_0"]) - batch["u0"]).pow(2)

    def data_loss(self, model, batch, phase=1):
        if "X_d" not in batch or "y_d" not in batch:
            return torch.tensor(0.0, device=self.device)

        X_d = batch["X_d"] # [N,3]
        y_d = batch["y_d"] # [N,1] noisy measurements

        # Raw PINN prediction (this is what PDE/BC/IC see)
        pred = model(X_d)  # [N, 1]

        # For the DATA term, optionally add scalar offset θ0
        if getattr(self, "use_offset", False) and self.offset is not None:
            pred = pred + self.offset  # broadcast θ0

        # Residuals for data and for EBM
        residual = (y_d - pred)       # [N, 1] (data - model), used as "noise"
        data_loss_value = self._data_loss(residual) # per-point losses [N, 1]
        
        with torch.no_grad():
            batch_std = residual.std()
            
            if model.training and phase!=1:
                currend_std_clamped = torch.clamp(batch_std, min=1e-6, max=self.running_std * 10)
                self.running_std.mul_(1 - self.momentum).add_(currend_std_clamped * self.momentum)
        
        residual_scaled = residual / self.running_std
        
        if phase == 0:
            if self.ebm is not None and residual.numel() > 0:
                # Detach so EBM training does not backprop through PINN/θ0
                nll_ebm, nll_ebm_mean = self.ebm.train_step(residual_scaled.detach())
                batch["ebm_nll"] = nll_ebm_mean
                
                if self.use_data_loss_balancer:
                    # Query weights using SCALED residuals
                    w, gate_reg_loss = self._get_weights(residual_scaled.detach())
                    weighted_loss = (w * data_loss_value).mean()
                    total_loss = weighted_loss + gate_reg_loss
                else:
                    total_loss = data_loss_value.mean()
                return total_loss
            
        elif phase == 1:
            return data_loss_value.view(-1)
        
        elif phase == 2:
            if self.ebm is not None and residual.numel() > 0:
                # Detach so EBM training does not backprop through PINN/θ0
                nll_ebm, nll_ebm_mean = self.ebm.train_step(residual.detach())
                batch["ebm_nll"] = nll_ebm_mean
                
                loss_metric = nll_ebm if self.use_nll else data_loss_value

                if self.use_data_loss_balancer:
                    # Query weights using SCALED residuals
                    w, gate_reg_loss = self._get_weights(residual_scaled.detach())
                    weighted_loss = (w * loss_metric).mean()
                    total_loss = weighted_loss + gate_reg_loss
                else:
                    total_loss = loss_metric.mean()
                return total_loss

        return torch.tensor(0.0, device=self.device)
    
    def extra_params(self):
        """Experiment-specific trainable parameters (e.g., θ0, weight_net)."""
        params = []
        if isinstance(getattr(self, "offset", None), torch.nn.Parameter):
            params.append(self.offset)
        if getattr(self, "weight_net", None) is not None:
            params.extend(list(self.weight_net.parameters()))
        return params

    # ----- eval / plots -----
    def relative_l2_on_grid(self, model, grid_cfg):
        model.eval()
        nx, ny, nt = grid_cfg["nx"], grid_cfg["ny"], grid_cfg["nt"]
        Xg, Yg = linspace_2d(self.rect.xa, self.rect.xb, self.rect.ya, self.rect.yb, nx, ny, self.rect.device)
        ts = torch.linspace(self.t0, self.t1, nt, device=self.rect.device)
        idxs = [0, nt//2, nt-1] if nt >= 3 else list(range(nt))
        rels = []
        with torch.no_grad():
            for ti in idxs:
                T = torch.full_like(Xg, ts[ti])
                XYT = torch.stack([Xg.reshape(-1), Yg.reshape(-1), T.reshape(-1)], 1)
                U_pred = model(XYT).reshape(nx, ny)
                U_true = self.u_star(Xg, Yg, T)
                rel = torch.linalg.norm((U_pred - U_true).reshape(-1)) / torch.linalg.norm(U_true.reshape(-1))
                rels.append(rel.item())
        return float(np.mean(rels))

    def plot_final(self, model, grid_cfg, out_dir):
        nx, ny, nt = grid_cfg["nx"], grid_cfg["ny"], grid_cfg["nt"]
        Xg, Yg = linspace_2d(self.rect.xa, self.rect.xb, self.rect.ya, self.rect.yb, nx, ny, self.rect.device)
        ts = torch.linspace(self.t0, self.t1, nt, device=self.rect.device)
        figs = {}
        with torch.no_grad():
            for label, ti in zip(["t0","tmid","t1"], [0, nt//2, nt-1]):
                T = torch.full_like(Xg, ts[ti])
                XYT = torch.stack([Xg.reshape(-1), Yg.reshape(-1), T.reshape(-1)], 1)
                U_pred = model(XYT).reshape(nx, ny).cpu().numpy()
                U_true = self.u_star(Xg, Yg, T).cpu().numpy()
                figs.update(save_plots_2d(Xg.cpu().numpy(), Yg.cpu().numpy(), U_true, U_pred, out_dir, f"allencahn2d_{label}"))
        return figs
    
    def make_video(self, model, grid, out_dir, fps=10, filename="final_evolution.mp4", phase=0):
        """
        Make a video over t ∈ [t0, t1] with 4 panels:

            [0,0] Noisy data over entire domain: u*(x,y,t) + ε(x,y,t)
            [0,1] True solution u*(x,y,t)
            [1,0] Predicted solution u_hat(x,y,t)
            [1,1] Absolute error |u* - u_hat|

        - Color scales for u / noisy panels are consistent across time.
        - Error color scale is also global across time.
        - If noise is disabled (no self.noise_model), the noisy panel = true solution.

        Args:
            model: trained PINN model
            grid: dict with keys {"nx","ny","nt"}
            out_dir: directory to save video
            fps: frames per second
            filename: video file name (default: final_evolution.mp4)

        Returns:
            Full path to the main evolution video.
        """
        os.makedirs(out_dir, exist_ok=True)
        nx = int(grid.get("nx", 100))
        ny = int(grid.get("ny", 100))
        nt = int(grid.get("nt", 100))

        # Build spatial grid
        Xg, Yg = linspace_2d(
            self.rect.xa, self.rect.xb,
            self.rect.ya, self.rect.yb,
            nx, ny, self.rect.device
        )
        ts = torch.linspace(self.t0, self.t1, nt, device=self.rect.device)

        extent = [float(self.rect.xa), float(self.rect.xb),
                float(self.rect.ya), float(self.rect.yb)]

        # Check if we have a noise model
        has_noise_model = getattr(self, "noise_model", None) is not None

        # ---------- First pass: determine global color ranges ----------
        vmin, vmax = None, None
        err_max = 0.0

        model.eval()
        with torch.no_grad():
            for tval in ts:
                T = torch.full_like(Xg, tval)
                XYT = torch.stack(
                    [Xg.reshape(-1), Yg.reshape(-1), T.reshape(-1)], dim=1
                )  # [nx*ny, 3]

                U_pred = model(XYT).reshape(nx, ny)    # [nx, ny]
                U_true = self.u_star(Xg, Yg, T)        # [nx, ny]

                if has_noise_model:
                    eps = self.noise_model.sample(nx * ny).to(self.rect.device)
                    eps = eps.view(nx, ny)
                    U_noisy = U_true + eps
                else:
                    U_noisy = U_true

                # Global color range for all "solution-like" panels
                umin = min(U_true.min().item(), U_pred.min().item(), U_noisy.min().item())
                umax = max(U_true.max().item(), U_pred.max().item(), U_noisy.max().item())
                vmin = umin if vmin is None else min(vmin, umin)
                vmax = umax if vmax is None else max(vmax, umax)

                # Error scale
                err_max = max(err_max, (U_true - U_pred).abs().max().item())

        # Safety
        if err_max <= 0:
            err_max = 1e-6

        # ---------- Second pass: render frames ----------
        frames = []
        with torch.no_grad():
            for tval in ts:
                T = torch.full_like(Xg, tval)
                XYT = torch.stack(
                    [Xg.reshape(-1), Yg.reshape(-1), T.reshape(-1)], dim=1
                )

                U_pred = model(XYT).reshape(nx, ny).cpu().numpy()
                U_true = self.u_star(Xg, Yg, T).cpu().numpy()

                if has_noise_model:
                    eps = self.noise_model.sample(nx * ny).to(self.rect.device)
                    eps = eps.view(nx, ny).cpu().numpy()
                    U_noisy = U_true + eps
                else:
                    U_noisy = U_true  # fallback: no noise → same as true

                U_err = np.abs(U_true - U_pred)

                fig = plt.figure(figsize=(12, 12), dpi=120)
                plt.suptitle(f"t = {float(tval):.5f}")

                # [0,0] Noisy data over whole domain
                ax1 = plt.subplot(2, 2, 1)
                im1 = ax1.imshow(
                    U_noisy.T, origin="lower", extent=extent,
                    vmin=vmin, vmax=vmax, aspect="auto"
                )
                ax1.set_title("Noisy data: u*(x,y,t) + ε")
                ax1.set_xlabel("x"); ax1.set_ylabel("y")
                fig.colorbar(im1, ax=ax1, fraction=0.046, pad=0.04)

                # [0,1] True solution
                ax2 = plt.subplot(2, 2, 2)
                im2 = ax2.imshow(
                    U_true.T, origin="lower", extent=extent,
                    vmin=vmin, vmax=vmax, aspect="auto"
                )
                ax2.set_title("True solution u*(x,y,t)")
                ax2.set_xlabel("x"); ax2.set_ylabel("y")
                fig.colorbar(im2, ax=ax2, fraction=0.046, pad=0.04)

                # [1,0] Predicted solution
                ax3 = plt.subplot(2, 2, 3)
                im3 = ax3.imshow(
                    U_pred.T, origin="lower", extent=extent,
                    vmin=vmin, vmax=vmax, aspect="auto"
                )
                ax3.set_title("Predicted solution û(x,y,t)")
                ax3.set_xlabel("x"); ax3.set_ylabel("y")
                fig.colorbar(im3, ax=ax3, fraction=0.046, pad=0.04)

                # [1,1] Absolute error
                ax4 = plt.subplot(2, 2, 4)
                im4 = ax4.imshow(
                    U_err.T, origin="lower", extent=extent,
                    vmin=0.0, vmax=err_max, aspect="auto"
                )
                ax4.set_title("|Error| = |u* - û|")
                ax4.set_xlabel("x"); ax4.set_ylabel("y")
                fig.colorbar(im4, ax=ax4, fraction=0.046, pad=0.04)

                fig.tight_layout(rect=[0, 0.03, 1, 0.95])

                # Convert figure to frame
                fig.canvas.draw()
                w, h = fig.canvas.get_width_height()
                buf = np.frombuffer(fig.canvas.tostring_rgb(), dtype=np.uint8)
                frame = buf.reshape(h, w, 3)
                frames.append(frame.copy())
                plt.close(fig)

        # ---------- Write video ----------
        ext = os.path.splitext(filename)[1].lower()
        path = os.path.join(out_dir, filename)
        if ext == ".gif":
            imageio.mimsave(path, frames, fps=fps)
        elif ext == ".mp4":
            writer = imageio.get_writer(path, fps=fps, codec="libx264", quality=7)
            for fr in frames:
                writer.append_data(fr)
            writer.close()
        else:
            raise ValueError(f"Unsupported video format: {ext}")

        if phase == 2:
            try:
                self._make_noise_videos(model, grid, out_dir, fps, filename)
            except Exception as e:
                print(f"[make_video] Warning: noise video failed with error: {e}")
        return path
    
    def _make_noise_videos(self, model, grid, out_dir, fps, filename):
        """
        Create a single video visualizing noise over the whole domain.

        For each time slice t_k, the figure has 4 panels:

            [0,0]  True noise field ε*(x,y,t_k) sampled on the full grid
            [0,1]  Residual field r(x,y,t_k) = y_noisy - u_pred

                   where y_noisy = u*(x,y,t_k) + ε*(x,y,t_k)

            [1,0]  Histograms of ε* and r on the same axes
            [1,1]  True noise pdf vs EBM pdf on the same axes

        This satisfies:
          - True vs predicted noise distributions in the SAME figure.
          - Noise distribution measured over the WHOLE domain, not just sampled data points.
        """
        # Need both true noise model and EBM to make this meaningful
        if getattr(self, "noise_model", None) is None:
            return None, None
        if getattr(self, "ebm", None) is None:
            # You could still visualize true noise only, but here we require both.
            return None, None

        base, ext = os.path.splitext(filename)
        if ext == "":
            ext = ".mp4"

        os.makedirs(out_dir, exist_ok=True)

        nx = int(grid.get("nx", 50))
        ny = int(grid.get("ny", 50))
        nt = int(grid.get("nt", 50))

        # Spatial grid
        Xg, Yg = linspace_2d(
            self.rect.xa, self.rect.xb,
            self.rect.ya, self.rect.yb,
            nx, ny, self.rect.device
        )
        ts = torch.linspace(self.t0, self.t1, nt, device=self.rect.device)

        extent = [float(self.rect.xa), float(self.rect.xb),
                  float(self.rect.ya), float(self.rect.yb)]

        frames = []

        model.eval()
        with torch.no_grad():
            for tval in ts:
                # Build full space-time grid
                T = torch.full_like(Xg, tval)
                XYT = torch.stack(
                    [Xg.reshape(-1), Yg.reshape(-1), T.reshape(-1)], dim=1
                )  # [nx*ny, 3]

                # True solution and prediction on full grid
                U_true = self.u_star(Xg, Yg, T)                  # [nx, ny]
                U_pred = model(XYT).reshape(nx, ny)              # [nx, ny]

                # Sample true noise on full grid (whole domain)
                eps_true_flat = self.noise_model.sample(nx * ny).to(self.rect.device)
                eps_true = eps_true_flat.view(nx, ny)            # [nx, ny]

                # Noisy observations and residuals
                Y_noisy = U_true + eps_true                      # [nx, ny]
                R_field = (Y_noisy - U_pred)                     # [nx, ny]

                # Flatten for 1D distributions
                eps_flat = eps_true.detach().cpu().numpy().reshape(-1)
                res_flat = R_field.detach().cpu().numpy().reshape(-1)

                # Range for plots / pdf grid
                max_val = max(
                    float(np.max(np.abs(eps_flat))) if eps_flat.size > 0 else 0.0,
                    float(np.max(np.abs(res_flat))) if res_flat.size > 0 else 0.0,
                    1e-3,
                )
                R = 1.5 * max_val
                r_grid = np.linspace(-R, R, 200, dtype=np.float32)
                r_torch = torch.from_numpy(r_grid).float().to(self.rect.device)

                # True noise pdf
                pdf_true = None
                if hasattr(self.noise_model, "pdf"):
                    r_cpu = torch.from_numpy(r_grid).float()  # CPU tensor
                    pdf_true_tensor = self.noise_model.pdf(r_cpu)  # all CPU ops inside noise.py
                    if isinstance(pdf_true_tensor, torch.Tensor):
                        pdf_true = pdf_true_tensor.detach().cpu().numpy()
                    else:
                        pdf_true = np.asarray(pdf_true_tensor)
                        
                # EBM pdf (from log q_theta)
                with torch.no_grad():
                    log_q = self.ebm(r_torch.unsqueeze(-1)).squeeze(-1)  # [200]
                    log_q = log_q - log_q.max()  # shift for numerical stability
                    pdf_unn = torch.exp(log_q)   # unnormalized
                    Z = torch.trapezoid(pdf_unn, r_torch)
                    pdf_ebm = (pdf_unn / (Z + 1e-12)).cpu().numpy()

                # ---- Build figure ----
                fig, axes = plt.subplots(2, 2, figsize=(12, 12), dpi=120)
                fig.suptitle(f"Noise distributions at t = {float(tval):.4f}")

                # [0,0] True noise field ε*(x,y,t)
                im0 = axes[0, 0].imshow(
                    eps_true.cpu().numpy().T,
                    origin="lower",
                    extent=extent,
                    vmin=-R,
                    vmax=R,
                    aspect="auto",
                    cmap="coolwarm",
                )
                axes[0, 0].set_title("True noise field ε*(x,y,t)")
                axes[0, 0].set_xlabel("x")
                axes[0, 0].set_ylabel("y")
                fig.colorbar(im0, ax=axes[0, 0], fraction=0.046, pad=0.04)

                # [0,1] Residual field r(x,y,t)
                im1 = axes[0, 1].imshow(
                    R_field.cpu().numpy().T,
                    origin="lower",
                    extent=extent,
                    vmin=-R,
                    vmax=R,
                    aspect="auto",
                    cmap="coolwarm",
                )
                axes[0, 1].set_title("Residual field r(x,y,t) = y_noisy - u_pred")
                axes[0, 1].set_xlabel("x")
                axes[0, 1].set_ylabel("y")
                fig.colorbar(im1, ax=axes[0, 1], fraction=0.046, pad=0.04)

                # [1,0] Histograms of ε* and r on same axes
                axes[1, 0].hist(
                    eps_flat, bins=40, density=True,
                    alpha=0.5, label="true noise sample ε*",
                )
                axes[1, 0].hist(
                    res_flat, bins=40, density=True,
                    alpha=0.5, label="residual sample r",
                )
                axes[1, 0].set_xlim(-R, R)
                axes[1, 0].set_xlabel("value")
                axes[1, 0].set_ylabel("density")
                axes[1, 0].set_title("Empirical distributions over whole domain")
                axes[1, 0].legend(loc="upper right", fontsize=8)

                # [1,1] True pdf vs EBM pdf on same axes
                if pdf_true is not None:
                    axes[1, 1].plot(
                        r_grid, pdf_true,
                        label="true noise pdf",
                        linewidth=1.5,
                    )
                axes[1, 1].plot(
                    r_grid, pdf_ebm,
                    label="EBM pdf",
                    linestyle="--",
                    linewidth=1.5,
                )
                axes[1, 1].set_xlim(-R, R)
                axes[1, 1].set_xlabel("value")
                axes[1, 1].set_ylabel("density")
                axes[1, 1].set_title("True vs EBM noise pdf")
                axes[1, 1].legend(loc="upper right", fontsize=8)

                fig.tight_layout(rect=[0, 0.03, 1, 0.95])

                # Convert figure to frame
                fig.canvas.draw()
                w, h = fig.canvas.get_width_height()
                buf = np.frombuffer(fig.canvas.tostring_rgb(), dtype=np.uint8)
                frame = buf.reshape(h, w, 3)
                frames.append(frame.copy())
                plt.close(fig)

        if not frames:
            return None, None

        # Save video
        path_noise = os.path.join(out_dir, f"{base}_noise_dist{ext}")
        if ext == ".gif":
            imageio.mimsave(path_noise, frames, fps=fps)
        elif ext == ".mp4":
            writer = imageio.get_writer(path_noise, fps=fps, codec="libx264", quality=7)
            for fr in frames:
                writer.append_data(fr)
            writer.close()
        else:
            raise ValueError(f"Unsupported video format: {ext}")

        # For backward-compatibility, return (path_true, path_ebm) style if you want:
        return path_noise, None

    def evaluate_gate_performance(self, model, out_dir, filename_prefix=None):
        """
        Evaluates and visualizes the Trainable Gate's ability to distinguish outliers.
        Generates a Sigmoid Plot and a Confusion Matrix.
        """
        if self.gate_module is None or self.ebm is None:
            print("[Evaluate] Gate or EBM not available. Skipping gate evaluation.")
            return

        print("[Evaluate] Analyzing Gate Performance on all measurement data...")
        
        model.eval()
        self.gate_module.eval()
        self.ebm.eval()
        
        # 1. Get Residuals for ALL data
        # We process in one large batch (or chunk if memory is tight, but 5k points is fine)
        with torch.no_grad():
            pred = model(self.X_data)
            if self.use_offset and self.offset is not None:
                pred = pred + self.offset
            
            # Raw residuals [N, 2]
            residual = self.y_data - pred
            
            # Flatten to 1D for EBM [2N, 1]
            # NOTE: We must track which indices are outliers in the FLATTENED array.
            # Original outliers are indices `idx` in range [0, N).
            # In flattened array [u0, v0, u1, v1...], outlier i affects 2*i and 2*i+1.
            res_flat = residual.view(-1, 1)
            
            # 2. Standardization
            res_scaled = res_flat / self.running_std
            
            # 3. Get EBM Log-Likelihoods (Energy)
            log_q = self.ebm(res_scaled).squeeze(-1) # [2N]
            
            # 4. Get Gate Weights & Z-scores
            # We access gate internals to reproduce the Z-score logic for plotting
            mu = log_q.mean()
            sigma = log_q.std() + 1e-6
            z_scores = (log_q - mu) / sigma
            
            # Learned parameters
            alpha = torch.nn.functional.softplus(self.gate_module.cutoff_alpha).item()
            beta = torch.nn.functional.softplus(self.gate_module.steepness).item()
            
            # Calculate final weights [2N]
            # w = sigmoid(beta * (z + alpha))
            weights = torch.sigmoid(beta * (z_scores + alpha))
            
            # Move to CPU
            z_cpu = z_scores.cpu().numpy()
            w_cpu = weights.cpu().numpy()
            
            # 5. Prepare Labels (Normal vs Outlier)
            N = self.y_data.shape[0]
            labels = np.zeros(2 * N, dtype=int) # 0 = Normal
            
            if len(self.outlier_indices) > 0:
                # Mark outlier indices (both u and v components)
                # idx i corresponds to 2*i and 2*i+1 in flattened array
                outlier_idx_u = self.outlier_indices * 2
                outlier_idx_v = self.outlier_indices * 2 + 1
                labels[outlier_idx_u] = 1 # 1 = Outlier
                labels[outlier_idx_v] = 1
        
        # --- PLOT 1: Sigmoid Decision Boundary ---
        fig, ax = plt.subplots(figsize=(10, 6))
        
        # A. Plot Learned Sigmoid Curve
        # Generate z range for smooth curve
        z_grid = np.linspace(z_cpu.min() - 0.5, z_cpu.max() + 0.5, 500)
        w_curve = 1.0 / (1.0 + np.exp(-beta * (z_grid + alpha)))
        
        ax.plot(z_grid, w_curve, 'k--', linewidth=2, label=f'Learned Gate (α={alpha:.2f}, β={beta:.2f})')
        
        # B. Scatter Data Points
        # Normal Points (Green)
        mask_norm = (labels == 0)
        ax.scatter(z_cpu[mask_norm], w_cpu[mask_norm], c='green', alpha=0.3, s=10, label='Normal')
        
        # Outlier Points (Red)
        mask_out = (labels == 1)
        ax.scatter(z_cpu[mask_out], w_cpu[mask_out], c='red', alpha=0.6, s=15, label='Outlier')
        
        # Decorate
        ax.axvline(-alpha, color='gray', linestyle=':', label='Cutoff Threshold')
        ax.set_xlabel("Log-Likelihood Z-Score")
        ax.set_ylabel("Assigned Weight (Probability of Validity)")
        ax.set_title("Gate Optimization Result: Weights vs. Likelihood")
        ax.legend(loc='lower right')
        ax.grid(True, alpha=0.3)
        
        save_path = os.path.join(out_dir, f"{filename_prefix}_gate_sigmoid_analysis.png")
        plt.savefig(save_path, dpi=150)
        plt.close(fig)
        
        # --- PLOT 2: Confusion Matrix ---
        # Classification Rule: Weight < 0.5 => REJECTED (Predicted Outlier)
        #                      Weight >= 0.5 => ACCEPTED (Predicted Normal)
        
        # Map to standard Confusion Matrix format:
        # Class 0: Negative (Normal Data)
        # Class 1: Positive (Outlier Data)
        
        # Prediction: 1 if w < 0.5 (Rejected), 0 if w >= 0.5 (Accepted)
        preds = (w_cpu < 0.5).astype(int)
        
        cm = confusion_matrix(labels, preds) 
        # Structure of cm:
        # [[TN, FP],
        #  [FN, TP]]
        # TN: Normal classified as Normal (Accepted)
        # FP: Normal classified as Outlier (Rejected - False Alarm)
        # FN: Outlier classified as Normal (Accepted - Missed Detection)
        # TP: Outlier classified as Outlier (Rejected - Success)

        fig, ax = plt.subplots(figsize=(6, 5))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=ax,
                    xticklabels=['Accepted (Normal)', 'Rejected (Outlier)'],
                    yticklabels=['True Normal', 'True Outlier'])
        ax.set_xlabel("Gate Prediction")
        ax.set_ylabel("Ground Truth")
        ax.set_title("Outlier Rejection Confusion Matrix")
        
        cm_path = os.path.join(out_dir, f"{filename_prefix}_gate_confusion_matrix.png")
        plt.savefig(cm_path, dpi=150)
        plt.close(fig)
        
        print(f"[Evaluate] Plots saved to {out_dir}")
        return {
            "gate/sigmoid": save_path,
            "gate/confusion": cm_path
        }