import math, os
import numpy as np
from matplotlib import pyplot as plt
import imageio
import torch

from pinnlab.experiments.base import BaseExperiment, make_leaf, grad_sum
from pinnlab.data.geometries import Rectangle, linspace_2d
from pinnlab.data.noise import get_noise
from pinnlab.utils.plotting import save_plots_2d
from pinnlab.utils.ebm import EBM, ResidualWeightNet
from pinnlab.utils.data_loss import (
    data_loss_mse,
    data_loss_l1,
    data_loss_q_gaussian,
)

class NavierStokesCylinderFlow(BaseExperiment):
    """
    2D incompressible Navier–Stokes (x,y,t), Taylor–Green vortex (periodic in x,y).

    Momentum:
        u_t + u u_x + v u_y + (1/ρ) p_x - ν (u_xx + u_yy) = 0
        v_t + u v_x + v v_y + (1/ρ) p_y - ν (v_xx + v_yy) = 0
    Continuity:
        u_x + v_y = 0

    Analytic (TGV) solution:
        u*(x,y,t) =  U0 cos(kx) sin(ky) exp(-2 ν k^2 t)
        v*(x,y,t) = -U0 sin(kx) cos(ky) exp(-2 ν k^2 t)
        p*(x,y,t) =  ρ U0^2 / 4 [cos(2kx) + cos(2ky)] exp(-4 ν k^2 t)

    We enforce:
      - periodic BCs (paired equality at x=xa/xb and y=ya/yb)
      - IC at t = t0 from the analytic solution
      - PDE residual (momentum + continuity) at interior collocation points
    """

    def __init__(self, cfg, device):
        super().__init__(cfg, device)
        self.device = device
        
        # ---- Fluid parameters ----
        self.nu   = float(cfg.get("nu", 0.01))
        self.rho  = float(cfg.get("rho", 1.0))

        w = cfg.get("weights", {}) or {}
        self.w_mom = float(w.get("momentum", 1.0))
        self.w_div = float(w.get("continuity", 1.0))
        
        self.npz_path = cfg.get("npz_path", "cylinder_Re100_random_data.npz")
        data = np.load(self.npz_path)
        
        # for key in ["u_data", "v_data", "p_data", "u_grid", "v_grid", "p_grid"]:
        #     arr = data[key]
        #     print(f"{key}: shape={arr.shape}, NaN_count={np.isnan(arr).sum()}")
        #     print(f"  nanmin={np.nanmin(arr)}, nanmax={np.nanmax(arr)}")

        self.x_grid, self.y_grid, self.t_grid = data["x_grid"], data["y_grid"], data["t_grid"]

        print("grid sizes:", len(self.x_grid), len(self.y_grid), len(self.t_grid))

        self.Xg, self.Yg = np.meshgrid(self.x_grid, self.y_grid, indexing="xy")   # each [Ny, Nx]

        XY_res_np = np.stack(
            [self.Xg.reshape(-1), self.Yg.reshape(-1)], axis=-1
        )  # [Nx*Ny, 2]
        
        cx, cy, R = 0.2, 0.2, 0.05
        mask = (XY_res_np[:, 0] - cx) ** 2 + (XY_res_np[:, 1] - cy) ** 2 > (R + 1e-6) ** 2
        XY_res_np = XY_res_np[mask]

        self.X_res = torch.tensor(XY_res_np, dtype=torch.float32, device=device)
        self.T_res = torch.tensor(self.t_grid, dtype=torch.float32, device=device)
        self.u_res = torch.tensor(data["u_grid"], dtype=torch.float32, device=device)
        self.v_res = torch.tensor(data["v_grid"], dtype=torch.float32, device=device)
        self.p_res = torch.tensor(data["p_grid"], dtype=torch.float32, device=device)

        X_data_np = np.stack(
            [data["x_data"], data["y_data"], data["t_data"]], axis=-1
        )
        Y_data_np = np.stack(
            [data["u_data"], data["v_data"], data["p_data"]], axis=-1
        )
        
        self.X_data = torch.tensor(X_data_np, dtype=torch.float32, device=device)
        self.u_data = torch.tensor(Y_data_np[:,0:1], dtype=torch.float32, device=device)
        self.v_data = torch.tensor(Y_data_np[:,1:2], dtype=torch.float32, device=device)
        self.p_data = torch.tensor(Y_data_np[:,2:3], dtype=torch.float32, device=device)

        noise_cfg = cfg.get("noise", None)
        self.use_data = bool(noise_cfg.get("enabled", False))
        self.noise_cfg = noise_cfg
        self.n_data_total = len(self.u_data)
        self.n_data_batch = int(
            noise_cfg.get("batch_size", 1000)
        )
        if self.n_data_batch > self.n_data_total:
            raise ValueError(f"[NavierStokes2D] n_data_batch ({self.n_data_batch}) > n_data_total ({self.n_data_total})")
        
        self.extra_noise_cfg = noise_cfg.get("extra_noise", {})
        self.use_extra_noise = bool(self.extra_noise_cfg.get("enabled", False))
        self.extra_noise_mask = None
        print(f"[NavierStokes2D] use_extra_noise = {self.use_extra_noise}")

        self.noise_model = None
        
        if self.use_data and self.n_data_total > 0:
            print(f"Initializing noisy data with {self.n_data_total} points.")
            self._init_noisy_dataset()

        ebm_cfg = cfg.get("ebm", {}) or {}
        self.use_ebm = bool(ebm_cfg.get("enabled", False))
        self.use_nll = bool(ebm_cfg.get("use_nll", False))
        if self.use_ebm:
            self.ebm_u = EBM(
                hidden_dim=ebm_cfg.get("hidden_dim", 32),
                depth=ebm_cfg.get("depth", 3),
                num_grid=ebm_cfg.get("num_grid", 256),
                max_range_factor=ebm_cfg.get("max_range_factor", 2.5),
                lr=ebm_cfg.get("lr", 1e-3),
                device=device,
            )
            self.ebm_v = EBM(
                hidden_dim=ebm_cfg.get("hidden_dim", 32),
                depth=ebm_cfg.get("depth", 3),
                num_grid=ebm_cfg.get("num_grid", 256),
                max_range_factor=ebm_cfg.get("max_range_factor", 2.5),
                lr=ebm_cfg.get("lr", 1e-3),
                device=device,
            )
            self.ebm_p = EBM(
                hidden_dim=ebm_cfg.get("hidden_dim", 32),
                depth=ebm_cfg.get("depth", 3),
                num_grid=ebm_cfg.get("num_grid", 256),
                max_range_factor=ebm_cfg.get("max_range_factor", 2.5),
                lr=ebm_cfg.get("lr", 1e-3),
                device=device,
            )
        else:
            self.ebm_u, self.ebm_v, self.ebm_p = None, None, None

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
        
        # --- Learnable per-point data weights via auxiliary MLP ----
        self.weight_net = None
        if self.use_data_loss_balancer and self.data_loss_balancer_kind == "mlp":
            wn_cfg = data_lb_cfg.get("weight_net", {}) or {}
            dlb_hidden_dim = int(wn_cfg.get("hidden_dim", 32)) # dlb: data loss balancer
            dlb_depth = int(wn_cfg.get("depth", 2))
            self.weight_net_u = ResidualWeightNet(
                hidden_dim=dlb_hidden_dim,
                depth=dlb_depth,
                device=device,
            )
            self.weight_net_v = ResidualWeightNet(
                hidden_dim=dlb_hidden_dim,
                depth=dlb_depth,
                device=device,
            )
            self.weight_net_p = ResidualWeightNet(
                hidden_dim=dlb_hidden_dim,
                depth=dlb_depth,
                device=device,
            )
            print(f"[NavierStokes2D] Using learnable per-point data weights "
                  f"with MLP(hidden_dim={dlb_hidden_dim}, depth={dlb_depth}).")
        
        print(f"[NavierStokes2D] use_ebm = {self.use_ebm}, use_nll = {self.use_nll}")
        print(f"[NavierStokes2D] use_phase = {self.use_phase}")
        print(f"[NavierStokes2D] data_loss_kind = {self.data_loss_kind}")
        print(f"[NavierStokes2D] use_data_loss_balancer = {self.use_data_loss_balancer}")
        print(f"[NavierStokes2D] data_loss_balancer_kind = {self.data_loss_balancer_kind}")

        # ----- Optional trainable offset θ0 for non-zero mean noise (PINN-off style) -----
        offset_cfg = cfg.get("offset", {}) or {}
        self.use_offset = bool(offset_cfg.get("enabled", False))
        if self.use_offset:
            init = float(offset_cfg.get("init", 0.0))
            # scalar parameter θ0 that will only be used in the DATA term
            self.offset = torch.nn.Parameter(
                torch.tensor(init, dtype=torch.float32, device=device)
            )
            print(f"[NavierStokes2D] Using trainable data offset θ0, init={init}")
        else:
            self.offset = None

    def _init_noisy_dataset(self):
        """
        Generate a fixed set of noisy measurements:
            X_data ~ Uniform(domain x time)
            u_clean = u_star(X_data)
            v_clean = v_star(X_data)
            p_clean = p_star(X_data)
            u_data  = u_clean + epsilon,     epsilon ~ noise distribution (from PINN-EBM)
            v_data  = v_clean + epsilon
            p_data  = p_clean + epsilon
        """
        kind = self.noise_cfg.get("kind", "G")     # 'G', 'u', '3G', ...
        n = self.n_data_total
            
        base_dtype = self.u_data.dtype

        # Determine noise scale "f" as in PINN-EBM: a factor times average magnitude of u*
        base_scale = float(self.noise_cfg.get("scale", 0.1))  # relative to mean |u|
        mean_level = float(self.u_data.abs().mean().detach().cpu())
        f = base_scale * (mean_level if mean_level > 0 else 1.0)

        # Build noise distribution; this uses the PINN-EBM function
        self.noise_model = get_noise(kind, f, pars=0)

        # In PINN-EBM, sample() often expects a shape list (e.g. [N])
        eps_u = self.noise_model.sample(n).to(self.device, dtype=base_dtype).view(-1, 1)  # [n, 1]
        eps_v = self.noise_model.sample(n).to(self.device, dtype=base_dtype).view(-1, 1)  # [n, 1]
        eps_p = self.noise_model.sample(n).to(self.device, dtype=base_dtype).view(-1, 1)  # [n, 1]

        self.extra_noise_mask_u = torch.zeros(n, dtype=torch.bool)
        self.extra_noise_mask_v = torch.zeros(n, dtype=torch.bool)
        self.extra_noise_mask_p = torch.zeros(n, dtype=torch.bool)
        if self.use_extra_noise:
            n_extra = int(self.extra_noise_cfg.get("n_points", 0))
            print(f"Adding extra noise to {n_extra} points.")
            idx_u = torch.randperm(n, device=self.device)[:n_extra]
            idx_v = torch.randperm(n, device=self.device)[:n_extra]
            idx_p = torch.randperm(n, device=self.device)[:n_extra]
            self.extra_noise_mask_u[idx_u.cpu()] = True
            self.extra_noise_mask_v[idx_v.cpu()] = True
            self.extra_noise_mask_p[idx_p.cpu()] = True

            scale_min = float(self.extra_noise_cfg.get("scale_min", 5.0))
            scale_max = float(self.extra_noise_cfg.get("scale_max", 10.0))

            # scale factor sampling for each noise point: scale factor ~ Uniform(scale_min, scale_max)
            factors_u = torch.empty(n_extra, 1, device=self.device, dtype=base_dtype).uniform_(scale_min, scale_max)
            factors_v = torch.empty(n_extra, 1, device=self.device, dtype=base_dtype).uniform_(scale_min, scale_max)
            factors_p = torch.empty(n_extra, 1, device=self.device, dtype=base_dtype).uniform_(scale_min, scale_max)

            # amplitude_i in [scale_min * f, scale_max * f]
            amp_u = factors_u * f
            amp_v = factors_v * f
            amp_p = factors_p * f

            signs_u = torch.randint(0, 2, amp_u.shape, device=self.device, dtype=amp_u.dtype) * 2 - 1
            signs_v = torch.randint(0, 2, amp_v.shape, device=self.device, dtype=amp_v.dtype) * 2 - 1
            signs_p = torch.randint(0, 2, amp_p.shape, device=self.device, dtype=amp_p.dtype) * 2 - 1
            extra_eps_u = signs_u * amp_u
            extra_eps_v = signs_v * amp_v
            extra_eps_p = signs_p * amp_p

            # overwrite outliers to the base noise
            eps_u[idx_u] = extra_eps_u
            eps_v[idx_v] = extra_eps_v
            eps_p[idx_p] = extra_eps_p

        self.u_noisy, self.v_noisy, self.p_noisy = self.u_data + eps_u, self.v_data + eps_v, self.p_data + eps_p

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
        else:
            raise ValueError(f"Unknown data_loss_kind: {kind}")

    # -------- Sampling --------
    def sample_batch(self, n_f, n_b, n_0):
        M, T = self.X_res.size(0), self.T_res.size(0)
        
        # interior
        idx_xy = torch.randint(0, M, (n_f,), device=self.device)
        idx_t  = torch.randint(0, T, (n_f,), device=self.device)
        X_f = torch.cat([self.X_res[idx_xy], self.T_res[idx_t].unsqueeze(1)], dim=1)

        batch = {"X_f": X_f}
            
        # attach noisy data mini-batch for data loss
        if self.use_data and self.X_data is not None and self.n_data_batch > 0:
            n = self.X_data.size(0)
            k = min(self.n_data_batch, n)
            idx = torch.randint(0, n, (k,), device=self.device)
            batch["X_d"] = self.X_data[idx]
            batch["u_d"], batch["v_d"], batch["p_d"] = self.u_noisy[idx], self.v_noisy[idx], self.p_noisy[idx]

        return batch

    # -------- Losses --------
    def pde_residual_loss(self, model, batch):
        X = make_leaf(batch["X_f"])        # [N,3] (x,y,t)
        out = model(X)                     # [N,3] -> (u,v,p)
        u, v, p = out[:,0:1], out[:,1:2], out[:,2:3]

        du = grad_sum(u, X)                # [N,3] -> (u_x,u_y,u_t)
        dv = grad_sum(v, X)
        dp = grad_sum(p, X)

        u_x, u_y, u_t = du[:,0:1], du[:,1:2], du[:,2:3]
        v_x, v_y, v_t = dv[:,0:1], dv[:,1:2], dv[:,2:3]
        p_x, p_y, _   = dp[:,0:1], dp[:,1:2], dp[:,2:3]

        d2ux = grad_sum(u_x, X); d2uy = grad_sum(u_y, X)
        d2vx = grad_sum(v_x, X); d2vy = grad_sum(v_y, X)
        u_xx, u_yy = d2ux[:,0:1], d2uy[:,1:2]
        v_xx, v_yy = d2vx[:,0:1], d2vy[:,1:2]

        # Momentum residuals (bring all to left-hand side)
        res_u = u_t + u * u_x + v * u_y + (1.0/self.rho) * p_x - self.nu * (u_xx + u_yy)
        res_v = v_t + u * v_x + v * v_y + (1.0/self.rho) * p_y - self.nu * (v_xx + v_yy)

        # Continuity residual
        div = u_x + v_y
        
        # Weighted squared residuals
        return self.w_mom * (res_u**2 + res_v**2) + self.w_div * (div**2)
    
    def data_loss(self, model, batch, phase=1):
        if "X_d" not in batch:
            return torch.tensor(0.0, device=self.device)
        
        Xd = batch["X_d"]
        out = model(Xd) # [Nd,3]
        u_pred, v_pred, p_pred = out[:,0:1], out[:,1:2], out[:,2:3]

        residual_u = u_pred - batch["u_d"]
        residual_v = v_pred - batch["v_d"]
        residual_p = p_pred - batch["p_d"]

        if self.use_offset and phase == 1:
            residual_u = residual_u - self.offset
            residual_v = residual_v - self.offset
            residual_p = residual_p - self.offset

        data_loss_value_u = self._data_loss(residual_u)
        data_loss_value_v = self._data_loss(residual_v)
        data_loss_value_p = self._data_loss(residual_p)

        if phase == 0:
            if self.ebm_u is not None:
                nll_ebm_u, nll_ebm_mean_u = self.ebm_u.train_step(residual_u.detach())
                nll_ebm_v, nll_ebm_mean_v = self.ebm_v.train_step(residual_v.detach())
                nll_ebm_p, nll_ebm_mean_p = self.ebm_p.train_step(residual_p.detach())
                batch["ebm_nll_u"], batch["ebm_nll_v"], batch["ebm_nll_p"] = nll_ebm_mean_u, nll_ebm_mean_v, nll_ebm_mean_p
                
                if self.use_data_loss_balancer:
                    if self.data_loss_balancer_kind == "mlp" and self.weight_net is not None:
                        w_u = self.weight_net_u(residual_u.detach())
                        w_v = self.weight_net_v(residual_v.detach())
                        w_p = self.weight_net_p(residual_p.detach())
                    else:
                        w_u = self.ebm_u.data_weight(residual_u.detach(), kind=self.data_loss_balancer_kind) # [N,1]
                        w_v = self.ebm_v.data_weight(residual_v.detach(), kind=self.data_loss_balancer_kind)
                        w_p = self.ebm_p.data_weight(residual_p.detach(), kind=self.data_loss_balancer_kind)

                    loss_u = (w_u * data_loss_value_u).view(-1)
                    loss_v = (w_v * data_loss_value_v).view(-1)
                    loss_p = (w_p * data_loss_value_p).view(-1)
                else:
                    loss_u = data_loss_value_u.view(-1)
                    loss_v = data_loss_value_v.view(-1)
                    loss_p = data_loss_value_p.view(-1)
                total_data_loss = torch.cat([loss_u, loss_v, loss_p], dim=0)
                return total_data_loss

        elif phase == 1:
            total_data_loss = torch.cat([data_loss_value_u.view(-1), data_loss_value_v.view(-1), data_loss_value_p.view(-1)], dim=0)
            return total_data_loss
        
        elif phase == 2:
            if self.ebm_u is not None:
                nll_ebm_u, nll_ebm_mean_u = self.ebm_u.train_step(residual_u.detach())
                nll_ebm_v, nll_ebm_mean_v = self.ebm_v.train_step(residual_v.detach())
                nll_ebm_p, nll_ebm_mean_p = self.ebm_p.train_step(residual_p.detach())
                batch["ebm_nll_u"], batch["ebm_nll_v"], batch["ebm_nll_p"] = nll_ebm_mean_u, nll_ebm_mean_v, nll_ebm_mean_p
                
                if self.use_nll:
                    data_loss_u, data_loss_v, data_loss_p = nll_ebm_u, nll_ebm_v, nll_ebm_p
                else:
                    data_loss_u = data_loss_value_u
                    data_loss_v = data_loss_value_v
                    data_loss_p = data_loss_value_p
                if self.use_data_loss_balancer:
                    if self.data_loss_balancer_kind == "mlp" and self.weight_net is not None:
                        w_u = self.weight_net_u(Xd)  # [Nd, 1]
                        w_v = self.weight_net_v(Xd)  # [Nd, 1]
                        w_p = self.weight_net_p(Xd)  # [Nd, 1]
                    else:
                        w_u = self.ebm_u.data_weight(residual_u.detach(), kind=self.data_loss_balancer_kind) # [N,1]
                        w_v = self.ebm_v.data_weight(residual_v.detach(), kind=self.data_loss_balancer_kind)
                        w_p = self.ebm_p.data_weight(residual_p.detach(), kind=self.data_loss_balancer_kind)
                    
                    loss_u = (w_u * data_loss_u).view(-1)
                    loss_v = (w_v * data_loss_v).view(-1)
                    loss_p = (w_p * data_loss_p).view(-1)
                else:
                    loss_u = data_loss_u.view(-1)
                    loss_v = data_loss_v.view(-1)
                    loss_p = data_loss_p.view(-1)
                total_data_loss = torch.cat([loss_u, loss_v, loss_p], dim=0)
                return total_data_loss
        
        else:
            raise ValueError(f"Invalid phase for data_loss: {phase}")
        
    def extra_params(self):
        """Experiment-specific trainable parameters (e.g., θ0, weight_net)."""
        params = []
        if isinstance(getattr(self, "offset", None), torch.nn.Parameter):
            params.append(self.offset)
        if getattr(self, "weight_net", None) is not None:
            params.extend(list(self.weight_net.parameters()))
        return params

    # -------- Evaluation --------
    def relative_l2_on_grid(self, model, grid_cfg):
        """
        Compute relative L2 error on the *same grid* as the simulation:

            rel = ||[u,v,p]_pred - [u,v,p]_true||_2 / ||[u,v,p]_true||_2

        evaluated on a few time slices (t0, tmid, t1) using u_grid, v_grid, p_grid
        from the FEniCS simulation. Points inside the cylinder are NaN in the
        stored fields and are excluded from the norm.
        """
        import numpy as np  # ensure np is imported at top of file too

        # Simulation grid shapes
        Nt, Ny, Nx = self.u_res.shape  # [Nt_snap, Ny, Nx]

        # Time indices to evaluate: t0, mid, t1 (like other experiments)
        if Nt >= 3:
            idxs = [0, Nt // 2, Nt - 1]
        else:
            idxs = list(range(Nt))

        # Build (x, y) grid to match [Ny, Nx] layout used in u_res/v_res/p_res
        # y = rows (Ny), x = columns (Nx)
        y = self.y_grid  # [Ny]
        x = self.x_grid  # [Nx]
        if isinstance(x, np.ndarray):
            x = torch.tensor(x, dtype=torch.float32, device=self.device)
            self.x_grid = x
        if isinstance(y, np.ndarray):
            y = torch.tensor(y, dtype=torch.float32, device=self.device)
            self.y_grid = y
        Yg, Xg = torch.meshgrid(y, x, indexing="ij")  # each [Ny, Nx]

        rels = []
        with torch.no_grad():
            for k in idxs:
                t_val = self.T_res[k]  # scalar
                T = torch.full_like(Xg, t_val)  # [Ny, Nx]

                # Build input (x, y, t) for the model: [Ny*Nx, 3]
                XYT = torch.stack(
                    [Xg.reshape(-1), Yg.reshape(-1), T.reshape(-1)],
                    dim=1,
                )  # [Ny*Nx, 3]

                # Model prediction: (u, v, p)
                out = model(XYT).reshape(Ny, Nx, 3)
                u_pred = out[..., 0]
                v_pred = out[..., 1]
                p_pred = out[..., 2]

                # Ground truth from FEniCS
                u_true = self.u_res[k]  # [Ny, Nx]
                v_true = self.v_res[k]
                p_true = self.p_res[k]

                # Mask out NaNs (inside cylinder)
                mask = (
                    torch.isfinite(u_true)
                    & torch.isfinite(v_true)
                    & torch.isfinite(p_true)
                )

                if mask.sum() == 0:
                    continue  # skip if something went wrong

                du = (u_pred - u_true)[mask]
                dv = (v_pred - v_true)[mask]
                dp = (p_pred - p_true)[mask]

                num = (du**2).sum() + (dv**2).sum() + (dp**2).sum()
                den = (u_true[mask]**2).sum() + (v_true[mask]**2).sum() + (p_true[mask]**2).sum()

                rel = torch.sqrt(num / den)
                rels.append(rel.item())

        # Average across time slices
        if len(rels) == 0:
            return float("nan")
        return float(np.mean(rels))

    # -------- Plots --------
    def plot_final(self, model, grid_cfg, out_dir):
        from pinnlab.utils.plotting import save_plots_2d
        nx, ny, nt = grid_cfg["nx"], grid_cfg["ny"], grid_cfg["nt"]
        Xg, Yg = linspace_2d(self.rect.xa, self.rect.xb, self.rect.ya, self.rect.yb, nx, ny, self.rect.device)
        ts = torch.linspace(self.t0, self.t1, nt, device=self.rect.device)

        figs = {}
        with torch.no_grad():
            for label, ti in zip(["t0","tmid","t1"], [0, nt//2, nt-1]):
                T = torch.full_like(Xg, ts[ti])
                XYT = torch.stack([Xg.reshape(-1), Yg.reshape(-1), T.reshape(-1)], dim=1)
                out = model(XYT).reshape(nx, ny, 3).cpu().numpy()

                U_pred = out[:,:,0]; V_pred = out[:,:,1]; P_pred = out[:,:,2]
                U_true = self.u_star(Xg, Yg, T).cpu().numpy()
                V_true = self.v_star(Xg, Yg, T).cpu().numpy()
                P_true = self.p_star(Xg, Yg, T).cpu().numpy()

                figs.update(save_plots_2d(Xg.cpu().numpy(), Yg.cpu().numpy(), U_true, U_pred, out_dir, f"ns2d_u_{label}"))
                figs.update(save_plots_2d(Xg.cpu().numpy(), Yg.cpu().numpy(), V_true, V_pred, out_dir, f"ns2d_v_{label}"))
                figs.update(save_plots_2d(Xg.cpu().numpy(), Yg.cpu().numpy(), P_true, P_pred, out_dir, f"ns2d_p_{label}"))
        return figs

    def make_video(self, model, grid, out_dir, fps=10, filename="evolution.mp4"):
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

        extent = [
            float(self.rect.xa), float(self.rect.xb),
            float(self.rect.ya), float(self.rect.yb),
        ]

        # ---------- First pass: determine global color ranges ----------
        u_min, u_max = None, None
        v_min, v_max = None, None
        p_min, p_max = None, None
        err_max = 0.0

        model.eval()
        with torch.no_grad():
            for tval in ts:
                T = torch.full_like(Xg, tval)
                XYT = torch.stack(
                    [Xg.reshape(-1), Yg.reshape(-1), T.reshape(-1)], dim=1
                )  # [nx*ny, 3]

                out = model(XYT).reshape(nx, ny, 3)
                U_pred = out[:, :, 0]
                V_pred = out[:, :, 1]
                P_pred = out[:, :, 2]

                U_true = self.u_star(Xg, Yg, T)
                V_true = self.v_star(Xg, Yg, T)
                P_true = self.p_star(Xg, Yg, T)

                # For each variable, include both true and predicted in global range
                u_min_t = float(torch.min(torch.stack([U_true.min(), U_pred.min()])).detach().cpu())
                u_max_t = float(torch.max(torch.stack([U_true.max(), U_pred.max()])).detach().cpu())
                v_min_t = float(torch.min(torch.stack([V_true.min(), V_pred.min()])).detach().cpu())
                v_max_t = float(torch.max(torch.stack([V_true.max(), V_pred.max()])).detach().cpu())
                p_min_t = float(torch.min(torch.stack([P_true.min(), P_pred.min()])).detach().cpu())
                p_max_t = float(torch.max(torch.stack([P_true.max(), P_pred.max()])).detach().cpu())

                u_min = u_min_t if u_min is None else min(u_min, u_min_t)
                u_max = u_max_t if u_max is None else max(u_max, u_max_t)
                v_min = v_min_t if v_min is None else min(v_min, v_min_t)
                v_max = v_max_t if v_max is None else max(v_max, v_max_t)
                p_min = p_min_t if p_min is None else min(p_min, p_min_t)
                p_max = p_max_t if p_max is None else max(p_max, p_max_t)
                
                err_max = max(err_max,
                              float(torch.max(torch.abs(U_true - U_pred)).detach().cpu()),
                              float(torch.max(torch.abs(V_true - V_pred)).detach().cpu()),
                              float(torch.max(torch.abs(P_true - P_pred)).detach().cpu()))

        # Safety
        if err_max <= 0:
            err_max = 1e-6

        if u_min is None:
            # No frames (e.g. nt == 0)
            return None

        # ---------- Second pass: generate frames ----------
        frames = []
        with torch.no_grad():
            for tval in ts:
                T = torch.full_like(Xg, tval)
                XYT = torch.stack(
                    [Xg.reshape(-1), Yg.reshape(-1), T.reshape(-1)], dim=1
                )  # [nx*ny, 3]

                out = model(XYT).reshape(nx, ny, 3)
                U_pred = out[:, :, 0]
                V_pred = out[:, :, 1]
                P_pred = out[:, :, 2]

                U_true = self.u_star(Xg, Yg, T)
                V_true = self.v_star(Xg, Yg, T)
                P_true = self.p_star(Xg, Yg, T)
                
                U_err = np.abs((U_true - U_pred).cpu().numpy())
                V_err = np.abs((V_true - V_pred).cpu().numpy())
                P_err = np.abs((P_true - P_pred).cpu().numpy())

                # Relative L2 errors per component
                def rel_l2(pred, true):
                    num = torch.linalg.norm((pred - true).reshape(-1))
                    den = torch.linalg.norm(true.reshape(-1)) + 1e-12
                    return float((num / den).detach().cpu())

                rel_u = rel_l2(U_pred, U_true)
                rel_v = rel_l2(V_pred, V_true)
                rel_p = rel_l2(P_pred, P_true)

                # Convert to numpy for plotting
                U_true_np = U_true.cpu().numpy().T
                V_true_np = V_true.cpu().numpy().T
                P_true_np = P_true.cpu().numpy().T
                U_pred_np = U_pred.cpu().numpy().T
                V_pred_np = V_pred.cpu().numpy().T
                P_pred_np = P_pred.cpu().numpy().T

                fig, axes = plt.subplots(3, 3, figsize=(18, 18), dpi=120)
                fig.suptitle(
                    f"2D Navier–Stokes (Taylor–Green) at t = {float(tval):.4f}\n"
                    f"rel L2: u={rel_u:.2e}, v={rel_v:.2e}, p={rel_p:.2e}"
                )

                # --- Top row: true fields ---
                im_u_true = axes[0, 0].imshow(
                    U_true_np, origin="lower", extent=extent,
                    vmin=u_min, vmax=u_max, aspect="auto"
                )
                axes[0, 0].set_title("u* (true)")
                axes[0, 0].set_xlabel("x"); axes[0, 0].set_ylabel("y")
                fig.colorbar(im_u_true, ax=axes[0, 0], fraction=0.046, pad=0.04)

                im_v_true = axes[0, 1].imshow(
                    V_true_np, origin="lower", extent=extent,
                    vmin=v_min, vmax=v_max, aspect="auto"
                )
                axes[0, 1].set_title("v* (true)")
                axes[0, 1].set_xlabel("x"); axes[0, 1].set_ylabel("y")
                fig.colorbar(im_v_true, ax=axes[0, 1], fraction=0.046, pad=0.04)

                im_p_true = axes[0, 2].imshow(
                    P_true_np, origin="lower", extent=extent,
                    vmin=p_min, vmax=p_max, aspect="auto"
                )
                axes[0, 2].set_title("p* (true)")
                axes[0, 2].set_xlabel("x"); axes[0, 2].set_ylabel("y")
                fig.colorbar(im_p_true, ax=axes[0, 2], fraction=0.046, pad=0.04)

                # --- Middle row: predicted fields ---
                im_u_pred = axes[1, 0].imshow(
                    U_pred_np, origin="lower", extent=extent,
                    vmin=u_min, vmax=u_max, aspect="auto"
                )
                axes[1, 0].set_title("û (pred)")
                axes[1, 0].set_xlabel("x"); axes[1, 0].set_ylabel("y")
                fig.colorbar(im_u_pred, ax=axes[1, 0], fraction=0.046, pad=0.04)

                im_v_pred = axes[1, 1].imshow(
                    V_pred_np, origin="lower", extent=extent,
                    vmin=v_min, vmax=v_max, aspect="auto"
                )
                axes[1, 1].set_title("v̂ (pred)")
                axes[1, 1].set_xlabel("x"); axes[1, 1].set_ylabel("y")
                fig.colorbar(im_v_pred, ax=axes[1, 1], fraction=0.046, pad=0.04)

                im_p_pred = axes[1, 2].imshow(
                    P_pred_np, origin="lower", extent=extent,
                    vmin=p_min, vmax=p_max, aspect="auto"
                )
                axes[1, 2].set_title("p̂ (pred)")
                axes[1, 2].set_xlabel("x"); axes[1, 2].set_ylabel("y")
                fig.colorbar(im_p_pred, ax=axes[1, 2], fraction=0.046, pad=0.04)
                
                # --- Bottom row: absolute error ---
                im_u_err = axes[2, 0].imshow(
                    U_err.T, origin="lower", extent=extent,
                    vmin=0.0, vmax=err_max, aspect="auto"
                )
                axes[2, 0].set_title("|û - u*| (abs error)")
                axes[2, 0].set_xlabel("x"); axes[2, 0].set_ylabel("y")
                fig.colorbar(im_u_err, ax=axes[2, 0], fraction=0.046, pad=0.04)
                
                im_v_err = axes[2, 1].imshow(
                    V_err.T, origin="lower", extent=extent,
                    vmin=0.0, vmax=err_max, aspect="auto"
                )
                axes[2, 1].set_title("|v̂ - v*| (abs error)")
                axes[2, 1].set_xlabel("x"); axes[2, 1].set_ylabel("y")
                fig.colorbar(im_v_err, ax=axes[2, 1], fraction=0.046, pad=0.04)
                
                im_p_err = axes[2, 2].imshow(
                    P_err.T, origin="lower", extent=extent,
                    vmin=0.0, vmax=err_max, aspect="auto"
                )
                axes[2, 2].set_title("|p̂ - p*| (abs error)")
                axes[2, 2].set_xlabel("x"); axes[2, 2].set_ylabel("y")
                fig.colorbar(im_p_err, ax=axes[2, 2], fraction=0.046, pad=0.04)

                fig.tight_layout(rect=[0, 0.03, 1, 0.95])

                # Convert figure to frame
                fig.canvas.draw()
                w, h = fig.canvas.get_width_height()
                buf = np.frombuffer(fig.canvas.tostring_rgb(), dtype=np.uint8)
                frame = buf.reshape(h, w, 3)
                frames.append(frame.copy())
                plt.close(fig)

        if not frames:
            return None

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
        
        try:
            self._make_noise_videos(model, grid, out_dir, fps, filename)
        except Exception as e:
            print(f"[make_video] Warning: noise video failed with error: {e}")

        return path
    
    def _make_noise_videos(self, model, grid, out_dir, fps, filename):
        # Need both true noise model and EBM_u to make this meaningful
        if getattr(self, "noise_model", None) is None:
            return None, None
        if getattr(self, "ebm_u", None) is None:
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

        extent = [
            float(self.rect.xa), float(self.rect.xb),
            float(self.rect.ya), float(self.rect.yb),
        ]

        frames = []

        model.eval()
        with torch.no_grad():
            for tval in ts:
                # Full space-time grid
                T = torch.full_like(Xg, tval)
                XYT = torch.stack(
                    [Xg.reshape(-1), Yg.reshape(-1), T.reshape(-1)], dim=1
                )  # [nx*ny, 3]

                # True solution and prediction on full grid (u-component only)
                U_true = self.u_star(Xg, Yg, T)                 # [nx, ny]
                V_true = self.v_star(Xg, Yg, T)
                P_true = self.p_star(Xg, Yg, T)
                out = model(XYT).reshape(nx, ny, 3)
                U_pred, V_pred, P_pred = out[:, :, 0], out[:, :, 1], out[:, :, 2] # [nx, ny]

                # Sample true noise on full grid (whole domain)
                eps_true_flat = self.noise_model.sample(nx * ny).to(self.rect.device)
                eps_true = eps_true_flat.view(nx, ny)           # [nx, ny]

                # Noisy observations and residuals
                U_noisy = U_true + eps_true                     # [nx, ny]
                V_noisy = V_true + eps_true
                P_noisy = P_true + eps_true
                R_u = (U_noisy - U_pred)                    # [nx, ny]
                R_v = (V_noisy - V_pred)
                R_p = (P_noisy - P_pred)

                # Flatten for 1D distributions
                eps_flat = eps_true.detach().cpu().numpy().reshape(-1)
                res_flat_u = R_u.detach().cpu().numpy().reshape(-1)
                res_flat_v = R_v.detach().cpu().numpy().reshape(-1)
                res_flat_p = R_p.detach().cpu().numpy().reshape(-1)

                # Range for plots / pdf grid
                max_val = max(
                    float(np.max(np.abs(eps_flat))) if eps_flat.size > 0 else 0.0,
                    float(np.max(np.abs(res_flat_u))) if res_flat_u.size > 0 else 0.0,
                    float(np.max(np.abs(res_flat_v))) if res_flat_v.size > 0 else 0.0,
                    float(np.max(np.abs(res_flat_p))) if res_flat_p.size > 0 else 0.0,
                    1e-3,
                )
                R = 1.5 * max_val
                r_grid = np.linspace(-R, R, 200, dtype=np.float32)
                r_torch = torch.from_numpy(r_grid).float().to(self.rect.device)

                # ---- True noise pdf (from analytic noise model, if available) ----
                pdf_true = None
                if hasattr(self.noise_model, "pdf"):
                    # noise.py is written assuming CPU tensor input
                    r_cpu = torch.from_numpy(r_grid).float()  # CPU tensor
                    pdf_true_tensor = self.noise_model.pdf(r_cpu)
                    if isinstance(pdf_true_tensor, torch.Tensor):
                        pdf_true = pdf_true_tensor.detach().cpu().numpy()
                    else:
                        pdf_true = np.asarray(pdf_true_tensor)

                # ---- EBM_u pdf (from log q_theta) ----
                with torch.no_grad():
                    log_q = self.ebm_u(r_torch.unsqueeze(-1)).squeeze(-1)  # [200]
                    log_q = log_q - log_q.max()  # shift for numerical stability
                    pdf_unn = torch.exp(log_q)   # unnormalized density
                    Z = torch.trapezoid(pdf_unn, r_torch)
                    pdf_ebm = (pdf_unn / (Z + 1e-12)).cpu().numpy()

                # ---- Build figure ----
                fig, axes = plt.subplots(3, 3, figsize=(18, 18), dpi=120)
                fig.suptitle(
                    f"Noise distributions (u-component) at t = {float(tval):.4f}"
                )

                # [0,0] True noise field ε_u(x,y,t)
                im0 = axes[0, 0].imshow(
                    eps_true.cpu().numpy().T,
                    origin="lower",
                    extent=extent,
                    vmin=-R,
                    vmax=R,
                    aspect="auto",
                    cmap="coolwarm",
                )
                axes[0, 0].set_title("True noise field ε_u*(x,y,t)")
                axes[0, 0].set_xlabel("x")
                axes[0, 0].set_ylabel("y")
                fig.colorbar(im0, ax=axes[0, 0], fraction=0.046, pad=0.04)

                # [0,1] Residual field r_u(x,y,t)
                im1 = axes[0, 1].imshow(
                    R_u.cpu().numpy().T,
                    origin="lower",
                    extent=extent,
                    vmin=-R,
                    vmax=R,
                    aspect="auto",
                    cmap="coolwarm",
                )
                axes[0, 1].set_title("Residual r_u(x,y,t) = u_noisy - û")
                axes[0, 1].set_xlabel("x")
                axes[0, 1].set_ylabel("y")
                fig.colorbar(im1, ax=axes[0, 1], fraction=0.046, pad=0.04)

                # [0,2] Residual field r_v(x,y,t)
                im2 = axes[0, 2].imshow(
                    R_v.cpu().numpy().T,
                    origin="lower",
                    extent=extent,
                    vmin=-R,
                    vmax=R,
                    aspect="auto",
                    cmap="coolwarm",
                )
                axes[0, 2].set_title("Residual r_v(x,y,t) = v_noisy - v̂")
                axes[0, 2].set_xlabel("x")
                axes[0, 2].set_ylabel("y")
                fig.colorbar(im2, ax=axes[0, 2], fraction=0.046, pad=0.04)

                # [1,0] Residual field r_p(x,y,t)
                im3 = axes[1, 0].imshow(
                    R_p.cpu().numpy().T,
                    origin="lower",
                    extent=extent,
                    vmin=-R,
                    vmax=R,
                    aspect="auto",
                    cmap="coolwarm",
                )
                axes[1, 0].set_title("Residual r_p(x,y,t) = p_noisy - p̂")
                axes[1, 0].set_xlabel("x")
                axes[1, 0].set_ylabel("y")
                fig.colorbar(im3, ax=axes[1, 0], fraction=0.046, pad=0.04)
                
                # [1,1] Histograms of ε* and r_u on same axes
                axes[1, 1].hist(
                    eps_flat, bins=40, density=True,
                    alpha=0.5, label="true noise sample ε_u*",
                )
                axes[1, 1].hist(
                    res_flat_u, bins=40, density=True,
                    alpha=0.5, label="residual sample r_u",
                )
                axes[1, 1].set_xlim(-R, R)
                axes[1, 1].set_xlabel("value")
                axes[1, 1].set_ylabel("density")
                axes[1, 1].set_title("Empirical distributions over whole domain")
                axes[1, 1].legend(loc="upper right", fontsize=8)
                
                # [1,2] Histograms of ε* and r_v on same axes
                axes[1, 2].hist(
                    eps_flat, bins=40, density=True,
                    alpha=0.5, label="true noise sample ε_v*",
                )
                axes[1, 2].hist(
                    res_flat_v, bins=40, density=True,
                    alpha=0.5, label="residual sample r_v",
                )
                axes[1, 2].set_xlim(-R, R)
                axes[1, 2].set_xlabel("value")
                axes[1, 2].set_ylabel("density")
                axes[1, 2].set_title("Empirical distributions over whole domain")
                axes[1, 2].legend(loc="upper right", fontsize=8)
                
                # [2,0] Histograms of ε* and r_p on same axes
                axes[2, 0].hist(
                    eps_flat, bins=40, density=True,
                    alpha=0.5, label="true noise sample ε_p*",
                )
                axes[2, 0].hist(
                    res_flat_p, bins=40, density=True,
                    alpha=0.5, label="residual sample r_p",
                )
                axes[2, 0].set_xlim(-R, R)
                axes[2, 0].set_xlabel("value")
                axes[2, 0].set_ylabel("density")
                axes[2, 0].set_title("Empirical distributions over whole domain")
                axes[2, 0].legend(loc="upper right", fontsize=8)

                # [2,1] True vs EBM pdf
                if pdf_true is not None:
                    axes[2, 1].plot(
                        r_grid, pdf_true,
                        label="true noise pdf",
                        linewidth=1.5,
                    )
                axes[2, 1].plot(
                    r_grid, pdf_ebm,
                    label="EBM_u pdf",
                    linestyle="--",
                    linewidth=1.5,
                )
                axes[2, 1].set_xlim(-R, R)
                axes[2, 1].set_xlabel("value")
                axes[2, 1].set_ylabel("density")
                axes[2, 1].set_title("True vs EBM (u) noise pdf")
                axes[2, 1].legend(loc="upper right", fontsize=8)

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