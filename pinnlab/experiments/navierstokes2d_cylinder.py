# navierstokes2d.py

import os
import numpy as np
import torch
import matplotlib.pyplot as plt
import matplotlib
import imageio.v2 as imageio
import sys
from tqdm import trange

from pinnlab.experiments.base import BaseExperiment, make_leaf, grad_sum
from pinnlab.data.noise import get_noise
from pinnlab.utils.ebm import EBM, ResidualWeightNet, EBM2D, TrainableLikelihoodGate
from pinnlab.utils.data_loss import (
    data_loss_mse,
    data_loss_l1,
    data_loss_q_gaussian,
)

from concurrent.futures import ProcessPoolExecutor
import seaborn as sns
from sklearn.metrics import confusion_matrix
import traceback

matplotlib.use('Agg')

# Helper to safely import multiprocessing context
def import_multiprocessing():
    import multiprocessing as mp
    return mp

def create_obstacle_mask(X, Y, obstacles):
    """
    Generates a boolean mask where True indicates point is INSIDE an obstacle.
    """
    # Initialize mask as all False (no obstacles)
    combined_mask = np.zeros(X.shape, dtype=bool)
    
    for obs in obstacles:
        # Support dict keys from config
        cx = obs.get('x', 0.5)
        cy = obs.get('y', 0.5)
        r  = obs.get('r', 0.1)
        
        dist = np.sqrt((X - cx)**2 + (Y - cy)**2)
        # Logical OR to combine masks
        combined_mask = np.logical_or(combined_mask, dist < r)
        
    return combined_mask

def render_frame_worker(args):
    """
    Worker function to render a single 2x2 frame.
    Updated to accept a list of 'obstacles' instead of single cylinder params.
    """
    # Unpack arguments
    (t_val, u_true, v_true, u_pred, v_pred, X_grid, Y_grid, 
     X_meas_slice, mag_meas_slice, vmin, vmax, error_max, obstacles) = args

    # Use 2x2 layout
    fig, ax = plt.subplots(2, 2, figsize=(12, 10))
    plt.suptitle(f"Time t={t_val:.2f}", y=0.95)

    # --- Pre-processing ---
    # Generate Composite Mask
    mask_cyl = create_obstacle_mask(X_grid, Y_grid, obstacles)

    # Magnitudes
    mag_pred = np.sqrt(u_pred**2 + v_pred**2)
    mag_true = np.sqrt(u_true**2 + v_true**2)

    # Mask obstacle area on grid plots (Set to NaN or 0)
    mag_pred[mask_cyl] = 0
    mag_true[mask_cyl] = 0
    
    error = np.abs(mag_true - mag_pred)
    # Ensure error inside cylinders is 0 so it doesn't mess up the plot colorbar
    error[mask_cyl] = 0
    
    extent = [0, 2, 0, 1] # You might want to pass DOMAIN bounds in args if they change

    # --- PLOT 1 (Top-Left): True Magnitude ---
    im0 = ax[0, 0].imshow(mag_true, origin='lower', extent=extent, vmin=vmin, vmax=vmax, cmap='jet')
    ax[0, 0].set_title("True Velocity Magnitude |V|")
    ax[0, 0].set_ylabel("Y")
    plt.colorbar(im0, ax=ax[0, 0], fraction=0.046, pad=0.04)

    # --- PLOT 2 (Top-Right): Predicted Magnitude ---
    im1 = ax[0, 1].imshow(mag_pred, origin='lower', extent=extent, vmin=vmin, vmax=vmax, cmap='jet')
    ax[0, 1].set_title("Predicted Magnitude |V|")
    plt.colorbar(im1, ax=ax[0, 1], fraction=0.046, pad=0.04)

    # --- PLOT 3 (Bottom-Left): Noisy Measurement Data ---
    # 1. Plot faint gray background of true flow
    ax[1, 0].imshow(mag_true, origin='lower', extent=extent, vmin=vmin, vmax=vmax, cmap='gray', alpha=0.2)
    
    # 2. Scatter plot noisy measurements
    if X_meas_slice is not None and len(X_meas_slice) > 0:
        sc = ax[1, 0].scatter(X_meas_slice[:, 0], X_meas_slice[:, 1], c=mag_meas_slice, 
                              vmin=vmin, vmax=vmax, cmap='jet', s=30, edgecolors='k', linewidth=0.5)
        plt.colorbar(sc, ax=ax[1, 0], fraction=0.046, pad=0.04)
        ax[1, 0].set_title(f"Noisy Measurements (N={len(X_meas_slice)})")
    else:
        ax[1, 0].set_title("No Measurements in this time slice")
        
    ax[1, 0].set_xlim(0, 2)
    ax[1, 0].set_ylim(0, 1)
    ax[1, 0].set_ylabel("Y")
    ax[1, 0].set_xlabel("X")

    # --- PLOT 4 (Bottom-Right): Absolute Error ---
    im2 = ax[1, 1].imshow(error, origin='lower', extent=extent, vmin=0, vmax=error_max, cmap='inferno')
    ax[1, 1].set_title(f"Absolute Error (Max over video: {error_max:.4f})")
    ax[1, 1].set_xlabel("X")
    plt.colorbar(im2, ax=ax[1, 1], fraction=0.046, pad=0.04)

    # --- Finalize ---
    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    
    fig.canvas.draw()
    w, h = fig.canvas.get_width_height()
    buf = np.frombuffer(fig.canvas.tostring_rgb(), dtype=np.uint8)
    frame = buf.reshape(h, w, 3)
    plt.close(fig)
    return frame

def render_noise_worker(args):
    """
    Worker to render 4-panel Noise Analysis frame.
    
    Layout:
    [0,0] True Noise Field (u-component)
    [0,1] Residual Field (u-component)
    [1,0] Histograms (True Noise vs Residual) - Combined u & v
    [1,1] PDF Curves (True Model vs EBM)
    """
    (t_val, eps_u_grid, res_u_grid, 
     eps_flat_all, res_flat_all, 
     r_grid, pdf_true, pdf_ebm, 
     R_range, extent) = args

    fig, ax = plt.subplots(2, 2, figsize=(12, 10))
    plt.suptitle(f"Noise Analysis t={t_val:.2f}", y=0.95)

    # Common visual settings
    cmap = 'coolwarm' 
    # vmin/vmax symmetric around 0 for noise
    vm = R_range 

    # --- [0,0] True Noise Field (U component) ---
    im0 = ax[0, 0].imshow(eps_u_grid, origin='lower', extent=extent, 
                          vmin=-vm, vmax=vm, cmap=cmap)
    ax[0, 0].set_title("True Noise Field $\epsilon_u(x,y)$")
    plt.colorbar(im0, ax=ax[0, 0], fraction=0.046, pad=0.04)

    # --- [0,1] Residual Field (U component) ---
    # Residual = (u_noisy - u_pred)
    im1 = ax[0, 1].imshow(res_u_grid, origin='lower', extent=extent, 
                          vmin=-vm, vmax=vm, cmap=cmap)
    ax[0, 1].set_title("Residual Field $r_u = y_u - \hat{u}$")
    plt.colorbar(im1, ax=ax[0, 1], fraction=0.046, pad=0.04)

    # --- [1,0] Histograms (Combined U and V) ---
    # We use density=True to compare with PDFs
    ax[1, 0].hist(eps_flat_all, bins=50, density=True, alpha=0.5, color='gray', label='True Noise $\epsilon$')
    ax[1, 0].hist(res_flat_all, bins=50, density=True, alpha=0.5, color='red', label='Residual $r$')
    ax[1, 0].set_title("Empirical Distributions (u & v combined)")
    ax[1, 0].legend(loc='upper right', fontsize=8)
    ax[1, 0].set_xlim(-vm, vm)
    ax[1, 0].grid(True, alpha=0.3)

    # --- [1,1] PDF Curves ---
    if pdf_true is not None:
        ax[1, 1].plot(r_grid, pdf_true, 'k-', lw=2, label='True Noise Model')
    
    if pdf_ebm is not None:
        ax[1, 1].plot(r_grid, pdf_ebm, 'b--', lw=2, label='EBM Learned')
        
    ax[1, 1].set_title("Probability Density Functions")
    ax[1, 1].legend(loc='upper right', fontsize=8)
    ax[1, 1].set_xlim(-vm, vm)
    ax[1, 1].set_ylim(bottom=0)
    ax[1, 1].grid(True, alpha=0.3)

    # Finalize
    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    
    # Rasterize
    fig.canvas.draw()
    w, h = fig.canvas.get_width_height()
    buf = np.frombuffer(fig.canvas.tostring_rgb(), dtype=np.uint8)
    frame = buf.reshape(h, w, 3)
    plt.close(fig)
    
    return frame

class NavierStokesCylinder(BaseExperiment):
    """
    Incompressible Navier-Stokes Equation on a 2D domain with a Cylinder.
    
    Variables: u, v (velocity), p (pressure)
    Equations:
        u_t + u*u_x + v*u_y = -p_x + nu*(u_xx + u_yy)
        v_t + u*v_x + v*v_y = -p_y + nu*(v_xx + v_yy)
        u_x + v_y = 0  (Continuity)
        
    Data:
        Loaded from 'ns_data.npz' generated by the CFD script.
    """

    def __init__(self, cfg, device):
        super().__init__(cfg, device)
        self.device = device
        
        # Load Simulation Data
        dir_path = cfg["dir_path"]
        simulation_tag = cfg["simulation_tag"]
        data_path = os.path.join(dir_path, simulation_tag, "data.npz")
        if not os.path.exists(data_path):
            raise FileNotFoundError(f"Could not find generated data: {data_path}")
        
        print(f"[NavierStokes2D] Loading data from {data_path}")
        raw_data = np.load(data_path)
        
        # PDE Constants from data
        self.nu = float(raw_data['viscosity'])
        
        # 1. Collocation Points (Fixed Grid from file)
        self.X_f_all = torch.from_numpy(raw_data['X_f']).float().to(device) # [N_f, 3] (x, y, t)
        
        # 2. Measurement Data (Clean - to be corrupted)
        self.X_u_clean = torch.from_numpy(raw_data['X_u']).float().to(device) # [N_u, 3]
        self.Y_u_clean = torch.from_numpy(raw_data['Y_u']).float().to(device) # [N_u, 2] (u, v)
        
        # 3. Validation Grid (For video/eval)
        self.val_t = raw_data['t_grid']
        self.val_x = raw_data['x_grid']
        self.val_y = raw_data['y_grid']
        self.val_u = raw_data['u_full']
        self.val_v = raw_data['v_full']
        
        if "obstacles" in cfg:
            self.obstacles = cfg["obstacles"]
        elif "cylinder" in cfg:
            # Backward compatibility: Convert single cylinder dict to list
            self.obstacles = [cfg["cylinder"]]
        else:
            self.obstacles = [] # Empty domain
            
        print(f"[NavierStokes2D] Loaded {len(self.obstacles)} obstacles.")

        self.cylinder_x = cfg["cylinder"]["x"]
        self.cylinder_y = cfg["cylinder"]["y"]
        self.cylinder_r = cfg["cylinder"]["r"]

        # Noise Configuration
        noise_cfg = cfg.get("noise", None)
        self.use_data = bool(noise_cfg["enabled"])
        self.noise_cfg = noise_cfg
        self.n_data_batch = int(noise_cfg["batch_size"])

        self.extra_noise_cfg = noise_cfg.get("extra_noise", {})
        self.use_extra_noise = bool(self.extra_noise_cfg["enabled"])
        
        self.X_data = None
        self.y_data = None
        self.y_clean = None
        self.noise_model = None
        
        # Apply Noise / Outliers to loaded data
        if self.use_data:
            self._init_noisy_dataset()

        # EBM and Loss Balancer Setup
        ebm_cfg = cfg.get("ebm", {}) or {}
        self.use_ebm = bool(ebm_cfg.get("enabled", False))
        self.use_nll = bool(ebm_cfg.get("use_nll", False))
        self.ebm_kind = ebm_cfg.get("kind", "2D") # 1D / 2D
        self.ebm_init_train_epochs = int(ebm_cfg["init_train_epochs"])
        
        if self.use_ebm:
            if self.ebm_kind == "2D":
                self.ebm = EBM2D(
                    hidden_dim=ebm_cfg.get("hidden_dim", 32),
                    depth=ebm_cfg.get("depth", 3),
                    num_grid=ebm_cfg.get("num_grid", 256),
                    max_range_factor=ebm_cfg.get("max_range_factor", 2.5),
                    lr=ebm_cfg.get("lr", 1e-3),
                    input_dim=2,
                    device=device,
                )
            else: # 1D EBM
                self.ebm = EBM(
                    hidden_dim=ebm_cfg.get("hidden_dim", 32),
                    depth=ebm_cfg.get("depth", 3),
                    num_grid=ebm_cfg.get("num_grid", 256),
                    max_range_factor=ebm_cfg.get("max_range_factor", 2.5),
                    lr=ebm_cfg.get("lr", 1e-3),
                    input_dim=1,
                    device=device,
                )
        else:
            self.ebm = None

        data_loss_cfg = cfg.get("data_loss", {}) or {}
        self.data_loss_kind = data_loss_cfg.get("kind", "mse")
        self.q_gauss_q = float(data_loss_cfg.get("q", 1.2))
        beta_val = data_loss_cfg.get("beta", None)
        self.q_gauss_beta = float(beta_val) if beta_val is not None else None
        
        data_lb_cfg = cfg.get("data_loss_balancer", {})
        self.use_data_loss_balancer = bool(data_lb_cfg.get("use_loss_balancer", False))
        self.data_loss_balancer_kind = data_lb_cfg.get("kind", "pw")
        print(f"[NavierStokes2D] Data Loss Balancer: {self.data_loss_balancer_kind}, Enabled: {self.use_data_loss_balancer}")
        
        self.gate_module = None
        if self.data_loss_balancer_kind == "gated_trainable":
            self.gate_module = TrainableLikelihoodGate(
                init_cutoff_sigma=2.0, 
                init_steepness=5.0, 
                device=device
            )
        
        self.weight_net = None
        if self.use_data_loss_balancer and self.data_loss_balancer_kind == "mlp":
            wn_cfg = data_lb_cfg.get("weight_net", {}) or {}
            self.weight_net = ResidualWeightNet(
                hidden_dim=int(wn_cfg.get("hidden_dim", 32)),
                depth=int(wn_cfg.get("depth", 2)),
                device=device,
            )

        # Optional Offset
        offset_cfg = cfg.get("offset", {}) or {}
        self.use_offset = bool(offset_cfg.get("enabled", False))
        if self.use_offset:
            # We have 2 variables (u,v), so offset should probably be size 2 or scalar
            init = float(offset_cfg.get("init", 0.0))
            self.offset = torch.nn.Parameter(torch.tensor([init, init], dtype=torch.float32, device=device))
        else:
            self.offset = None

    def _init_noisy_dataset(self):
        """
        Takes the loaded CLEAN random measurements (self.Y_u_clean) and adds 
        SIGNAL-DEPENDENT synthetic noise + GLOBAL outliers.
        """
        y_clean = self.Y_u_clean # [N, 2] (u, v)
        n = y_clean.shape[0]
        
        # --- 1. CONFIGURATION ---
        # "relative_scale": fraction of the signal value (e.g., 0.05 = 5%)
        # "floor_scale": fraction of global mean to serve as noise floor (e.g., 0.01)
        # If these keys don't exist in config, we approximate your old behavior 
        # or set reasonable defaults.
        
        # Fallback to 'scale' if new keys aren't present to maintain backward compatibility
        legacy_scale = float(self.noise_cfg.get("scale", 0.1))
        alpha = float(self.noise_cfg.get("relative_scale", 0.0)) 
        beta = float(self.noise_cfg.get("floor_scale", legacy_scale))
        
        mean_level = float(y_clean.abs().mean().detach().cpu())
        if mean_level == 0: mean_level = 1.0

        # --- 2. CALCULATE LOCAL NOISE SCALE (Heteroscedastic) ---
        # shape: [N, 2]
        # Noise Sigma_i = alpha * |y_i| + beta * mean_global
        sigma_local = alpha * y_clean.abs() + beta * mean_level
        
        # --- 3. GENERATE BASE NOISE (Standard Distribution) ---
        kind = self.noise_cfg.get("kind", "G")
        
        # Initialize noise model with scale=1.0 to get standard distribution (Z-scores)
        self.noise_model = get_noise(kind, f=1.0, pars=0)
        
        # Sample standard noise (Z)
        if kind in ['MG2D']:
            z = self.noise_model.sample(n).float().to(self.device) 
        else:
            z_flat = self.noise_model.sample(n * 2).float().to(self.device)
            z = z_flat.view(n, 2)
            
        # Apply local scaling
        eps = z * sigma_local
        
        # Initialize indices list
        self.outlier_indices = []
        
        # --- 4. ADD OUTLIERS (Keep Global Scaling) ---
        # Outliers represent sensor failures, so they should NOT be scaled 
        # by local values (a glitch at the wall can still be huge).
        if self.use_extra_noise:
            n_extra = int(self.extra_noise_cfg.get("n_points", 0))
            if n_extra > 0:
                print(f"[NavierStokes2D] Injecting outliers into {n_extra} points.")
                
                # Outlier reference scale (Global)
                f_outlier = legacy_scale * mean_level
                
                idx = torch.randperm(n, device=self.device)[:n_extra]
                self.outlier_indices = idx.cpu().numpy()
                
                scale_min = float(self.extra_noise_cfg.get("scale_min", 5.0))
                scale_max = float(self.extra_noise_cfg.get("scale_max", 10.0))
                print("scale_min:", scale_min, "scale_max:", scale_max)
                
                factors = torch.empty(n_extra, 2, device=self.device).uniform_(scale_min, scale_max)
                
                # Outliers replace the noise at these points with massive errors
                amp = factors * f_outlier
                signs = torch.randint(0, 2, amp.shape, device=self.device).float() * 2 - 1
                eps[idx] = signs * amp
                
        y_noisy = y_clean + eps
        
        self.X_data = self.X_u_clean
        self.y_clean = y_clean
        self.y_data = y_noisy
        
        # Stats for debugging
        print(f"[Noise Init] Global Mean |u|: {mean_level:.4f}")
        print(f"[Noise Init] Local Sigma range: [{sigma_local.min():.4f}, {sigma_local.max():.4f}]")

    def sample_batch(self, n_f=None, n_b=None, n_0=None):
        # Note: n_b and n_0 are unused here as we use pre-generated file points 
        # which already contain domain info. We primarily use X_f.
        
        batch = {}
        
        # Sample Collocation Points from loaded fixed grid
        n_col = self.X_f_all.shape[0]
        idx_f = torch.randint(0, n_col, (n_f,), device=self.device)
        batch["X_f"] = self.X_f_all[idx_f]

        # Sample Measurement Data
        if self.use_data and self.X_data is not None:
            n_data = self.X_data.shape[0]
            k = min(self.n_data_batch, n_data)
            idx_d = torch.randint(0, n_data, (k,), device=self.device)
            batch["X_d"] = self.X_data[idx_d]
            batch["y_d"] = self.y_data[idx_d] # [k, 2]
            
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
            pred = model(X_d) # [N, 3]
            u_pred, v_pred = pred[:, 0:1], pred[:, 1:2]
            y_pred = torch.cat([u_pred, v_pred], dim=1) # [N, 2]
            
            if self.use_offset and self.offset is not None:
                y_pred = y_pred + self.offset

            residual = y_d - y_pred
            if self.ebm_kind == "1D":
                residual = residual.view(-1, 1)
            with torch.no_grad():
                batch_std = residual.std()
                scale_factor = torch.clamp(batch_std, min=1e-6)
            residual_scaled = residual / scale_factor
            nll_ebm, nll_ebm_mean = self.ebm.train_step(residual_scaled.detach())

    def pde_residual_loss(self, model, batch):
        X = make_leaf(batch["X_f"]) # [N, 3] (x, y, t)
        
        # Model output: [u, v, p]
        out = model(X)
        u, v, p = out[:, 0:1], out[:, 1:2], out[:, 2:3]
        
        # First derivatives
        du = grad_sum(u, X) # [u_x, u_y, u_t]
        dv = grad_sum(v, X) # [v_x, v_y, v_t]
        dp = grad_sum(p, X) # [p_x, p_y, p_t]
        
        u_x, u_y, u_t = du[:, 0:1], du[:, 1:2], du[:, 2:3]
        v_x, v_y, v_t = dv[:, 0:1], dv[:, 1:2], dv[:, 2:3]
        p_x, p_y      = dp[:, 0:1], dp[:, 1:2]
        
        # Second derivatives (Laplacian)
        d2ux = grad_sum(u_x, X)
        d2uy = grad_sum(u_y, X)
        u_xx, u_yy = d2ux[:, 0:1], d2uy[:, 1:2]
        
        d2vx = grad_sum(v_x, X)
        d2vy = grad_sum(v_y, X)
        v_xx, v_yy = d2vx[:, 0:1], d2vy[:, 1:2]
        
        # NS Residuals
        # 1. Momentum u
        res_u = u_t + (u * u_x + v * u_y) + p_x - self.nu * (u_xx + u_yy)
        # 2. Momentum v
        res_v = v_t + (u * v_x + v * v_y) + p_y - self.nu * (v_xx + v_yy)
        # 3. Continuity
        res_c = u_x + v_y
        
        loss = res_u.pow(2).mean() + res_v.pow(2).mean() + res_c.pow(2).mean()
        return loss

    def data_loss(self, model, batch, phase=1):
        if "X_d" not in batch or "y_d" not in batch:
            return torch.tensor(0.0, device=self.device)
            
        X_d = batch["X_d"]
        y_d = batch["y_d"] # [N, 2] (u_meas, v_meas)
        
        # Predict u, v (ignore p for data loss usually)
        pred = model(X_d)
        u_pred, v_pred = pred[:, 0:1], pred[:, 1:2]
        y_pred = torch.cat([u_pred, v_pred], dim=1) # [N, 2]
        
        if self.use_offset and self.offset is not None:
            y_pred = y_pred + self.offset
            
        # Flatten residuals: treat u and v errors as samples from the same noise distribution
        residual = y_d - y_pred # [N, 2]
        data_loss_value = self._data_loss(residual) # [2N, 1]
        
        if self.ebm_kind == "1D":
            residual = residual.view(-1, 1)
            
        with torch.no_grad():
            batch_std = residual.std()
            scale_factor = torch.clamp(batch_std, min=1e-6)
            residual_scaled = residual / scale_factor
            
        if self.ebm_kind == "1D":
            data_loss_value = data_loss_value.view(-1, 1) # [2*N, 1]

        # --- Phase Logic (EBM Training / Balancing) ---
        # Identical logic to Helmholtz, just operating on the combined u/v residuals
        if phase == 0: # Train EBM only
            if self.ebm is not None:
                nll_ebm, nll_ebm_mean = self.ebm.train_step(residual.detach())
                batch["ebm_nll"] = nll_ebm_mean
                
                if self.use_data_loss_balancer:
                    w, gate_reg_loss = self._get_weights(residual_scaled.detach())
                    weighted_loss = (w * data_loss_value).mean()
                    total_loss = weighted_loss + gate_reg_loss
                else:
                    total_loss = data_loss_value.mean()
                return total_loss
                
        elif phase == 1: # Standard PINN training
            return data_loss_value.view(-1)
            
        elif phase == 2: # PINN + EBM weighted
            if self.ebm is not None:
                nll_ebm, nll_ebm_mean = self.ebm.train_step(residual.detach())
                batch["ebm_nll"] = nll_ebm_mean
                
                loss_metric = nll_ebm if self.use_nll else data_loss_value
                
                if self.use_data_loss_balancer:
                    w, gate_reg_loss = self._get_weights(residual.detach())
                    weighted_loss = (w * loss_metric).mean()
                    total_loss = weighted_loss + gate_reg_loss
                else:
                    total_loss = loss_metric.mean()
                return total_loss
        return torch.tensor(0.0, device=self.device)

    def _data_loss(self, residual):
        if self.data_loss_kind == "mse":
            return data_loss_mse(residual)
        elif self.data_loss_kind == "l1":
            return data_loss_l1(residual)
        elif self.data_loss_kind == "q_gaussian":
            return data_loss_q_gaussian(residual, q=self.q_gauss_q, beta=self.q_gauss_beta)
        return residual.pow(2)

    def _get_weights(self, residual):
        # 1. MLP weighting
        if self.data_loss_balancer_kind == "mlp" and self.weight_net is not None:
            return self.weight_net(residual), torch.tensor(0.0, device=self.device)
        
        # 2. Trainable Gating
        elif self.data_loss_balancer_kind == "gated_trainable" and self.ebm is not None:
            # Get raw log-probabilities from EBM (don't need gradients for EBM here usually)
            # We detach residual because EBM training is separate, 
            # but we DO want gradients flowing back to gate_module parameters
            with torch.no_grad():
                log_q = self.ebm(residual.detach()) # [N, 1]
            
            # Pass through our trainable gate
            return self.gate_module(log_q)
        else:
            return self.ebm.data_weight(residual, kind=self.data_loss_balancer_kind), torch.tensor(0.0, device=self.device)
            
    def extra_params(self):
        params = []
        if isinstance(getattr(self, "offset", None), torch.nn.Parameter):
            params.append(self.offset)
        if getattr(self, "weight_net", None) is not None:
            params.extend(list(self.weight_net.parameters()))
        if getattr(self, "gate_module", None) is not None:
            params.extend(list(self.gate_module.parameters()))
        return params

    # --- Evaluation ---
    def relative_l2_on_grid(self, model, grid_cfg=None):
        # We use the loaded validation grid instead of regenerating
        model.eval()
        # Select middle time step for metric
        t_idx = len(self.val_t) // 2
        t_val = self.val_t[t_idx]
        
        # Create grid for this timestep
        X, Y = np.meshgrid(self.val_x, self.val_y)
        T = np.full_like(X, t_val)
        
        X_flat = X.flatten()
        Y_flat = Y.flatten()
        T_flat = T.flatten()
        
        mask_cyl = create_obstacle_mask(X_flat, Y_flat, self.obstacles)
        mask_valid = ~mask_cyl
        
        inputs = np.stack([X_flat, Y_flat, T_flat], axis=1)
        inputs_torch = torch.from_numpy(inputs).float().to(self.device)
        
        with torch.no_grad():
            pred = model(inputs_torch)
            u_pred = pred[:, 0].cpu().numpy()
            v_pred = pred[:, 1].cpu().numpy()
            
        u_true = self.val_u[t_idx].flatten()
        v_true = self.val_v[t_idx].flatten()
        
        # Calc Velocity Magnitude Error (only outside cylinder)
        mag_pred = np.sqrt(u_pred**2 + v_pred**2)[mask_valid]
        mag_true = np.sqrt(u_true**2 + v_true**2)[mask_valid]
        
        num = np.linalg.norm(mag_true - mag_pred)
        den = np.linalg.norm(mag_true)
        return float(num / den)

    def make_video(self, model, grid_cfg, out_dir, fps=10, filename="flow_evolution_2x2.mp4", phase=0):
        os.makedirs(out_dir, exist_ok=True)
        model.eval()
        
        # 1. Prepare Validation Grid
        X_grid, Y_grid = np.meshgrid(self.val_x, self.val_y)
        ny, nx = X_grid.shape
        
        # Pre-calculate cylinder mask for error checking on grid
        mask_cyl_grid = create_obstacle_mask(X_grid, Y_grid, self.obstacles)

        # 2. Precompute Global Limits for Velocity colorbars
        vmin = 0.0
        vmax = np.max(np.sqrt(self.val_u**2 + self.val_v**2))
        global_error_max = 0.0
        
        temp_inference_results = []
        
        if len(self.val_t) > 1:
            dt_window = (self.val_t[1] - self.val_t[0]) / 2.0
        else:
            dt_window = 0.05 # Fallback

        # 3. RUN INFERENCE (Gather data on GPU)
        print(f"[NavierStokes2D] Pre-calculating frames. Data slice window: +/-{dt_window:.3f}")
        
        with torch.no_grad():
            for i, t_val in enumerate(self.val_t):
                # Skip frames to speed up video generation if needed (e.g., every other frame)
                if i % 2 != 0: continue 
                
                # --- 3a. Slice Noisy Measurement Data for this timeframe ---
                X_meas_slice = None
                mag_meas_slice = None
                
                if self.use_data and self.X_data is not None:
                    X_d_cpu = self.X_data.cpu().numpy()
                    y_d_cpu = self.y_data.cpu().numpy()
                    mask_time = (X_d_cpu[:, 2] >= t_val - dt_window) & \
                                (X_d_cpu[:, 2] < t_val + dt_window)
                    
                    X_meas_slice = X_d_cpu[mask_time] # [k, 3]
                    y_meas_slice = y_d_cpu[mask_time] # [k, 2] (noisy u, v)
                    
                    if len(y_meas_slice) > 0:
                        # Calculate magnitude of noisy measurements for plotting
                        mag_meas_slice = np.sqrt(y_meas_slice[:, 0]**2 + y_meas_slice[:, 1]**2)

                # --- 3b. Grid Inference ---
                T_grid = np.full_like(X_grid, t_val)
                inputs = np.stack([X_grid.flatten(), Y_grid.flatten(), T_grid.flatten()], axis=1)
                inputs_torch = torch.from_numpy(inputs).float().to(self.device)
                
                out = model(inputs_torch)
                
                # Move grid predictions to CPU
                u_pred_grid = out[:, 0].reshape(ny, nx).cpu().numpy()
                v_pred_grid = out[:, 1].reshape(ny, nx).cpu().numpy()
                u_true_grid = self.val_u[i]
                v_true_grid = self.val_v[i]
                
                # --- 3c. Calculate Max Error for Global Scaling ---
                mag_pred_tmp = np.sqrt(u_pred_grid**2 + v_pred_grid**2)
                mag_true_tmp = np.sqrt(u_true_grid**2 + v_true_grid**2)
                
                # Zero out cylinder on grid to avoid fake errors at boundary
                mag_pred_tmp[mask_cyl_grid] = 0
                mag_true_tmp[mask_cyl_grid] = 0
                
                curr_max = np.max(np.abs(mag_true_tmp - mag_pred_tmp))
                if curr_max > global_error_max:
                    global_error_max = curr_max
                
                # Store all necessary data for this frame
                temp_inference_results.append({
                    't_val': t_val,
                    'u_true': u_true_grid, 'v_true': v_true_grid,
                    'u_pred': u_pred_grid, 'v_pred': v_pred_grid,
                    'X_meas_slice': X_meas_slice, 'mag_meas_slice': mag_meas_slice
                })

        print(f"[NavierStokes2D] Global Max Error calculated: {global_error_max:.5f}")
        
        # Handle case where model is perfect or untrained and error is 0
        if global_error_max == 0: global_error_max = 1.0 # avoid divide by zero in plots

        # 4. PREPARE WORKER ARGUMENTS
        render_args_list = []
        for res in temp_inference_results:
            # Pack arguments into tuple for the worker function
            # Pass copies of grid arrays to ensure process safety
            args = (
                res['t_val'], 
                res['u_true'], res['v_true'], 
                res['u_pred'], res['v_pred'], 
                X_grid.copy(), Y_grid.copy(), 
                res['X_meas_slice'], res['mag_meas_slice'],
                vmin, vmax, global_error_max,
                self.obstacles
            )
            render_args_list.append(args)

        # 5. RENDER IN PARALLEL (CPU part)
        # Use slightly fewer workers than cores to leave room for system processes
        n_workers = max(1, os.cpu_count() - 2)
        print(f"[NavierStokes2D] Rendering {len(render_args_list)} frames using {n_workers} workers...")
        
        frames = []
        ctx = import_multiprocessing().get_context("fork") if os.name != 'nt' else None
        with ProcessPoolExecutor(max_workers=n_workers, mp_context=ctx) as executor:
            results = executor.map(render_frame_worker, render_args_list)
            for i, frame in enumerate(results):
                frames.append(frame)
                if (i+1) % 10 == 0: print(f"Rendered frame {i+1}/{len(render_args_list)}")

        # 6. Save Video
        path = os.path.join(out_dir, filename)
        imageio.mimsave(path, frames, fps=fps, macro_block_size=None)
        print(f"[NavierStokes2D] Video saved to {path}")
        
        if phase == 2:
            try:
                self._make_noise_videos(model, out_dir, fps, filename)
            except Exception as e:
                print(f"Warning: Failed to create noise analysis video: {e}")
                traceback.print_exc()
        return path
    
    def plot_final(self, model, grid_cfg, out_dir):
        return None
    
    def _make_noise_videos(self, model, out_dir, fps, filename):
        if self.noise_model is None or self.ebm is None:
            print("[NavierStokes2D] Skipping noise video (missing noise_model or ebm).")
            return

        print("[NavierStokes2D] Generating Noise Analysis video...")
        base, ext = os.path.splitext(filename)
        vid_filename = f"{base}_noise_analysis{ext}"
        
        # 1. Setup Grid
        X, Y = np.meshgrid(self.val_x, self.val_y)
        ny, nx = X.shape
        extent = [0, 2, 0, 1] # Fixed domain size for cylinder
        
        # 2. Determine Plotting Range (R)
        # We need a fixed X-axis range for histograms/PDFs. 
        # A safe bet is 3-4 standard deviations of the noise scale.
        # We can look at the noise config or sample a bit.
        dummy_sample = self.noise_model.sample(1000)
        max_val = float(dummy_sample.abs().max().cpu())
        R_range = max_val * 1.5
        
        # 3. Pre-calculate EBM PDF (Static curve)
        # The EBM itself is fixed after training.
        r_grid_np = np.linspace(-R_range, R_range, 200).astype(np.float32)
        r_grid_torch = torch.from_numpy(r_grid_np).to(self.device).view(-1, 1)
        
        pdf_ebm = None
        with torch.no_grad():
            log_q = self.ebm(r_grid_torch).flatten()
            log_q = log_q - log_q.max() # Stability
            q_unn = torch.exp(log_q)
            Z = torch.trapezoid(q_unn, r_grid_torch.flatten())
            pdf_ebm = (q_unn / Z).cpu().numpy()

        # 4. Pre-calculate True PDF (Static curve)
        pdf_true = None
        if hasattr(self.noise_model, "pdf"):
            # Assuming noise_model.pdf accepts cpu tensor
            r_cpu = torch.from_numpy(r_grid_np)
            pdf_true = self.noise_model.pdf(r_cpu)
            if isinstance(pdf_true, torch.Tensor):
                pdf_true = pdf_true.detach().cpu().numpy()

        # 5. GPU Inference Phase
        render_args_list = []
        
        with torch.no_grad():
            for i, t_val in enumerate(self.val_t):
                if i % 2 != 0: continue 

                # Grid inputs
                T = np.full_like(X, t_val)
                inputs = np.stack([X.flatten(), Y.flatten(), T.flatten()], axis=1)
                inputs_torch = torch.from_numpy(inputs).float().to(self.device)
                
                # Model Prediction
                out = model(inputs_torch)
                u_pred, v_pred = out[:, 0], out[:, 1]
                
                # True Solution
                u_true = torch.from_numpy(self.val_u[i].flatten()).to(self.device)
                v_true = torch.from_numpy(self.val_v[i].flatten()).to(self.device)
                
                # Sample "True" Noise for this specific grid/time
                # We sample for u and v separately
                n_points = u_true.shape[0]
                kind = self.noise_cfg.get("kind", "G")
                if kind in ['MG2D']:
                    eps = self.noise_model.sample(n_points).float().to(self.device)
                    eps_u, eps_v = eps[:, 0], eps[:, 1]
                else:
                    eps_flat = self.noise_model.sample(n_points * 2).float().to(self.device)
                    eps_u, eps_v = eps_flat[:n_points], eps_flat[n_points:]
                
                # Add noise to get "Noisy Observation"
                u_noisy = u_true + eps_u
                v_noisy = v_true + eps_v
                
                # Calculate Residuals
                res_u = u_noisy - u_pred
                res_v = v_noisy - v_pred
                
                # --- Prepare Data for Worker ---
                
                # Field Plots (We only plot U component for visual clarity)
                eps_u_grid = eps_u.view(ny, nx).cpu().numpy()
                res_u_grid = res_u.view(ny, nx).cpu().numpy()
                
                # Histograms (We combine U and V for robust stats)
                # Downsample slightly if grid is huge to save IPC overhead
                eps_combined = torch.cat([eps_u, eps_v]).cpu().numpy()
                res_combined = torch.cat([res_u, res_v]).cpu().numpy()
                
                # Subsample for histograms to keep pickling fast (optional, e.g. 10k points)
                if eps_combined.shape[0] > 10000:
                    idx = np.random.choice(eps_combined.shape[0], 10000, replace=False)
                    eps_combined = eps_combined[idx]
                    res_combined = res_combined[idx]

                args = (
                    t_val, 
                    eps_u_grid, res_u_grid, 
                    eps_combined, res_combined,
                    r_grid_np, pdf_true, pdf_ebm,
                    R_range, extent
                )
                render_args_list.append(args)

        # 6. Render Phase
        print(f"[NavierStokes2D] Rendering noise analysis ({len(render_args_list)} frames)...")
        n_workers = max(1, os.cpu_count() - 2)
        frames = []
        
        # Use 'fork' context on Linux for speed, default on others
        ctx = __import__("multiprocessing").get_context("fork") if os.name != 'nt' else None
        
        with ProcessPoolExecutor(max_workers=n_workers, mp_context=ctx) as executor:
            results = executor.map(render_noise_worker, render_args_list)
            for frame in results:
                frames.append(frame)

        # 7. Save
        path = os.path.join(out_dir, vid_filename)
        imageio.mimsave(path, frames, fps=fps, macro_block_size=None)
        print(f"[NavierStokes2D] Noise video saved to {path}")
        
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
            u_pred, v_pred = pred[:, 0:1], pred[:, 1:2]
            y_pred = torch.cat([u_pred, v_pred], dim=1) # [N, 2]
            if self.use_offset and self.offset is not None:
                y_pred = y_pred + self.offset
            
            # Raw residuals [N, 2]
            residual = self.y_data - y_pred
            
            # Flatten to 1D for EBM [2N, 1]
            # NOTE: We must track which indices are outliers in the FLATTENED array.
            # Original outliers are indices `idx` in range [0, N).
            # In flattened array [u0, v0, u1, v1...], outlier i affects 2*i and 2*i+1.
            res_flat = residual.view(-1, 1)
            
            # 2. Adaptive Standardization (Global Stats)
            batch_std = res_flat.std()
            scale_factor = torch.clamp(batch_std, min=1e-6)
            res_scaled = res_flat / scale_factor
            
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