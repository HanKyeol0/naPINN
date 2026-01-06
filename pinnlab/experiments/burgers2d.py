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
from pinnlab.utils.ebm import EBM, EBM2D, ResidualWeightNet, TrainableLikelihoodGate
from pinnlab.utils.data_loss import (
    data_loss_mse,
    data_loss_l1,
    data_loss_q_gaussian,
)

from concurrent.futures import ProcessPoolExecutor
import matplotlib.pyplot as plt
import seaborn as sns
import scipy.stats as stats
from sklearn.metrics import confusion_matrix
import traceback

# Use Agg backend for headless video generation
matplotlib.use('Agg')

def import_multiprocessing():
    import multiprocessing as mp
    return mp

# --- WORKER: Flow Evolution (Physical Solution) ---
def render_frame_worker(args):
    """
    Render physical solution: True Magnitude, Predicted Magnitude, Noisy Data, Error.
    """
    (t_val, u_true, v_true, u_pred, v_pred, 
     X_meas_slice, mag_meas_slice, vmin, vmax, error_max, extent) = args

    fig, ax = plt.subplots(2, 2, figsize=(10, 10), dpi=100)
    plt.suptitle(f"Burgers 2D Flow | t={t_val:.3f}", y=0.95, fontsize=14)

    # Calculate Magnitudes
    mag_pred = np.sqrt(u_pred**2 + v_pred**2)
    mag_true = np.sqrt(u_true**2 + v_true**2)
    error = np.abs(mag_true - mag_pred)

    # [0,0] True Magnitude
    im0 = ax[0, 0].imshow(mag_true, origin='lower', extent=extent, vmin=vmin, vmax=vmax, cmap='jet')
    ax[0, 0].set_title("True Magnitude |V*|")
    ax[0, 0].set_ylabel("y")
    plt.colorbar(im0, ax=ax[0, 0], fraction=0.046, pad=0.04)

    # [0,1] Predicted Magnitude
    im1 = ax[0, 1].imshow(mag_pred, origin='lower', extent=extent, vmin=vmin, vmax=vmax, cmap='jet')
    ax[0, 1].set_title("Predicted Magnitude |V_hat|")
    plt.colorbar(im1, ax=ax[0, 1], fraction=0.046, pad=0.04)

    # [1,0] Noisy Measurement Data
    # Background: Faint gray true flow to see context
    ax[1, 0].imshow(mag_true, origin='lower', extent=extent, vmin=vmin, vmax=vmax, cmap='gray', alpha=0.15)
    
    if X_meas_slice is not None and len(X_meas_slice) > 0:
        sc = ax[1, 0].scatter(X_meas_slice[:, 0], X_meas_slice[:, 1], c=mag_meas_slice, 
                              vmin=vmin, vmax=vmax, cmap='jet', s=15, edgecolors='none', alpha=0.9)
        plt.colorbar(sc, ax=ax[1, 0], fraction=0.046, pad=0.04)
        ax[1, 0].set_title(f"Noisy Measurements (N={len(X_meas_slice)})")
    else:
        ax[1, 0].set_title("No Measurements")
        
    ax[1, 0].set_xlim(extent[0], extent[1])
    ax[1, 0].set_ylim(extent[2], extent[3])
    ax[1, 0].set_ylabel("y"); ax[1, 0].set_xlabel("x")

    # [1,1] Absolute Error
    im2 = ax[1, 1].imshow(error, origin='lower', extent=extent, vmin=0, vmax=error_max, cmap='inferno')
    ax[1, 1].set_title(f"Absolute Error |V* - V_hat|")
    ax[1, 1].set_xlabel("x")
    plt.colorbar(im2, ax=ax[1, 1], fraction=0.046, pad=0.04)

    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    
    fig.canvas.draw()
    w, h = fig.canvas.get_width_height()
    buf = np.frombuffer(fig.canvas.tostring_rgb(), dtype=np.uint8)
    frame = buf.reshape(h, w, 3)
    plt.close(fig)
    return frame


# --- WORKER: Noise Distribution Analysis ---
def render_noise_worker(args):
    """
    Render noise analysis:
    [0,0] True Noise Field (u-component)
    [0,1] Residual Field (u-component)
    [1,0] Histograms (u and v combined)
    [1,1] PDF Comparison (True vs EBM slice)
    """
    (t_val, eps_u_grid, res_u_grid, 
     eps_flat_all, res_flat_all, 
     r_grid, pdf_true, pdf_ebm,
     R_range, extent) = args

    fig, axes = plt.subplots(2, 2, figsize=(12, 10), dpi=100)
    plt.suptitle(f"Noise Distribution Analysis hahaha.. | t={t_val:.3f}", y=0.96, fontsize=14)

    cmap = 'coolwarm' 
    vm = R_range

    # --- [0,0] True Noise Field (u-component) ---
    im0 = axes[0, 0].imshow(eps_u_grid, origin='lower', extent=extent, 
                          vmin=-vm, vmax=vm, cmap=cmap)
    axes[0, 0].set_title("True Noise Field $\epsilon_u(x,y)$")
    axes[0, 0].set_ylabel("y")
    fig.colorbar(im0, ax=axes[0, 0], fraction=0.046, pad=0.04)

    # --- [0,1] Residual Field (u-component) ---
    im1 = axes[0, 1].imshow(res_u_grid, origin='lower', extent=extent, 
                          vmin=-vm, vmax=vm, cmap=cmap)
    axes[0, 1].set_title("Residual Field $r_u = y_u - \hat{u}$")
    fig.colorbar(im1, ax=axes[0, 1], fraction=0.046, pad=0.04)

    # --- [1,0] Histograms (Combined u and v) ---
    # We combine u and v to see the aggregate noise statistics
    axes[1, 0].hist(eps_flat_all, bins=60, density=True, alpha=0.5, color='gray', label='True Noise $\epsilon$')
    axes[1, 0].hist(res_flat_all, bins=60, density=True, alpha=0.5, color='red', label='Residual $r$')
    axes[1, 0].set_title("Empirical Distributions (u & v pooled)")
    axes[1, 0].legend(loc='upper right', fontsize=9)
    axes[1, 0].set_xlim(-vm, vm)
    axes[1, 0].set_xlabel("Value")
    axes[1, 0].grid(True, alpha=0.3, linestyle='--')

    # --- [1,1] PDF Curves ---
    if pdf_true is not None:
        axes[1, 1].plot(r_grid, pdf_true, 'k-', lw=2, label='True Noise Model')
    
    if pdf_ebm is not None:
        axes[1, 1].plot(r_grid, pdf_ebm, 'b--', lw=2, label='EBM Learned $p(r,0)$')
        
    axes[1, 1].set_title("Probability Density Functions")
    axes[1, 1].legend(loc='upper right', fontsize=9)
    axes[1, 1].set_xlim(-vm, vm)
    axes[1, 1].set_ylim(bottom=0)
    axes[1, 1].set_xlabel("Value")
    axes[1, 1].grid(True, alpha=0.3, linestyle='--')

    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    
    fig.canvas.draw()
    w, h = fig.canvas.get_width_height()
    buf = np.frombuffer(fig.canvas.tostring_rgb(), dtype=np.uint8)
    frame = buf.reshape(h, w, 3)
    plt.close(fig)
    return frame


class Burgers2D(BaseExperiment):
    """
    2D Viscous Burgers' Equation.
    Variables: u, v
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
        
        print(f"[Burgers2D] Loading data from {data_path}")
        raw_data = np.load(data_path)
        
        pde_cfg = cfg.get("pde", {}) or {}
        self.learn_nu = pde_cfg.get("learn_nu", True)
        self.true_nu = float(raw_data['viscosity'])
        if self.learn_nu:
            init_nu = float(pde_cfg.get("init_nu", 0.0))
            print(f"[Burgers2D] PDE Parameter nu is TRAINABLE. Init: {init_nu}, True: {self.true_nu}")
            self.nu = torch.nn.Parameter(torch.tensor(init_nu, dtype=torch.float32, device=device))
        else:
            print(f"[Burgers2D] PDE Parameter nu is FIXED. Value: {self.true_nu}")
            self.nu = self.true_nu # Float
        
        # 1. Collocation Points
        self.X_f_all = torch.from_numpy(raw_data['X_f']).float().to(device) # [N_f, 3] (x, y, t)
        
        # 2. Measurement Data (Clean)
        self.X_u_clean = torch.from_numpy(raw_data['X_u']).float().to(device) # [N_u, 3]
        self.Y_u_clean = torch.from_numpy(raw_data['Y_u']).float().to(device) # [N_u, 2] (u, v)
        
        # 3. Validation Grid
        self.val_t = raw_data['t_grid']
        self.val_x = raw_data['x_grid']
        self.val_y = raw_data['y_grid']
        self.val_u = raw_data['u_full']
        self.val_v = raw_data['v_full']
        
        self.extent = [self.val_x.min(), self.val_x.max(), self.val_y.min(), self.val_y.max()]
        
        # Noise Config
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
        
        if self.use_data:
            self._init_noisy_dataset()

        # EBM and Loss Balancer Setup
        ebm_cfg = cfg.get("ebm", {}) or {}
        self.use_ebm = bool(ebm_cfg.get("enabled", False))
        self.use_nll = bool(ebm_cfg.get("use_nll", False))
        self.ebm_kind = ebm_cfg.get("kind", "1D") # 1D / 2D
        self.ebm_init_train_epochs = int(ebm_cfg["init_train_epochs"])
        
        if self.use_ebm:
            if self.ebm_kind == "2D":
                # Note: input_dim=2 because Burgers has (u, v)
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
            self.running_std = torch.tensor(1.0, device=device)
            self.momentum = 0.05 # Update rate (alpha)
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
        
        self.gate_module = None
        if self.data_loss_balancer_kind == "gated_trainable":
            self.rejection_cost = float(data_lb_cfg.get("rejection_cost", 0.5))
            self.gate_module = TrainableLikelihoodGate(device=device, rejection_cost=self.rejection_cost)
        
        self.weight_net = None
        if self.use_data_loss_balancer and self.data_loss_balancer_kind == "mlp":
            wn_cfg = data_lb_cfg.get("weight_net", {}) or {}
            self.weight_net = ResidualWeightNet(
                hidden_dim=int(wn_cfg.get("hidden_dim", 32)),
                depth=int(wn_cfg.get("depth", 2)),
                device=device,
            )

        offset_cfg = cfg.get("offset", {}) or {}
        self.use_offset = bool(offset_cfg.get("enabled", False))
        if self.use_offset:
            init = float(offset_cfg.get("init", 0.0))
            self.offset = torch.nn.Parameter(torch.tensor([init, init], dtype=torch.float32, device=device))
        else:
            self.offset = None

    def state_dict(self):
        """
        Returns a dictionary containing the experiment's optimization state.
        This includes the running_std, offset, and trainable gate parameters.
        """
        state = {
            'running_std': self.running_std,
            'offset': self.offset,
        }
        if self.learn_nu:
            state['nu'] = self.nu
            
        if self.use_offset and self.offset is not None:
            state['offset'] = self.offset
        
        # Save EBM state if it exists
        if self.ebm is not None:
            state['ebm'] = self.ebm.state_dict()
            state['ebm_optimizer'] = self.ebm.optimizer.state_dict()
            
        # Save Gate state if it exists
        if self.gate_module is not None:
            state['gate_module'] = self.gate_module.state_dict()
            
        # Save WeightNet state if it exists
        if self.weight_net is not None:
            state['weight_net'] = self.weight_net.state_dict()
            
        return state

    def load_state_dict(self, state_dict):
        """
        Loads the experiment's optimization state.
        """
        if 'running_std' in state_dict:
            self.running_std.copy_(state_dict['running_std'].to(self.device))
            print(f"[Burgers2D] Loaded running_std: {self.running_std.item():.4f}")
            
        if 'offset' in state_dict and self.offset is not None:
            with torch.no_grad():
                self.offset.copy_(state_dict['offset'].to(self.device))
                
        if 'nu' in state_dict and self.learn_nu:
            with torch.no_grad():
                self.nu.copy_(state_dict['nu'].to(self.device))
                print(f"[Burgers2D] Loaded learned nu: {self.nu.item():.6f}")
                
        if 'ebm' in state_dict and self.ebm is not None:
            self.ebm.load_state_dict(state_dict['ebm'])
            if 'ebm_optimizer' in state_dict:
                self.ebm.optimizer.load_state_dict(state_dict['ebm_optimizer'])
            
        if 'gate_module' in state_dict and self.gate_module is not None:
            self.gate_module.load_state_dict(state_dict['gate_module'])
            
        if 'weight_net' in state_dict and self.weight_net is not None:
            self.weight_net.load_state_dict(state_dict['weight_net'])

    def _init_noisy_dataset(self):
        y_clean = self.Y_u_clean # [N, 2]
        n = y_clean.shape[0]
        
        legacy_scale = float(self.noise_cfg.get("scale", 0.1))
        alpha = float(self.noise_cfg.get("relative_scale", 0.0)) 
        beta = float(self.noise_cfg.get("floor_scale", legacy_scale))
        
        mean_level = float(y_clean.abs().mean().detach().cpu())
        if mean_level == 0: mean_level = 1.0

        # Heteroscedastic scaling
        self.sigma_local = alpha * y_clean.abs() + beta * mean_level
        
        # Base Noise
        kind = self.noise_cfg.get("kind", "G")
        self.noise_model = get_noise(kind, f=1.0, pars=0)
        
        if kind in ['MG2D']:
            z = self.noise_model.sample(n).float().to(self.device) 
        else:
            z_flat = self.noise_model.sample(n * 2).float().to(self.device)
            z = z_flat.view(n, 2)
            
        eps = z * self.sigma_local
        
        # Initialize indices list
        self.outlier_indices = []
        
        # Outliers
        if self.use_extra_noise:
            n_extra = int(self.extra_noise_cfg.get("n_points", 0))
            if n_extra > 0:
                print(f"[Burgers2D] Injecting outliers into {n_extra} points.")
                f_outlier = legacy_scale * mean_level
                
                idx = torch.randperm(n, device=self.device)[:n_extra]
                self.outlier_indices = idx.cpu().numpy()
                scale_min = float(self.extra_noise_cfg.get("scale_min", 5.0))
                scale_max = float(self.extra_noise_cfg.get("scale_max", 10.0))
                
                factors = torch.empty(n_extra, 2, device=self.device).uniform_(scale_min, scale_max)
                amp = factors * f_outlier
                signs = torch.randint(0, 2, amp.shape, device=self.device).float() * 2 - 1
                eps[idx] = signs * amp
                
        y_noisy = y_clean + eps
        
        self.X_data = self.X_u_clean
        self.y_clean = y_clean
        self.y_data = y_noisy
        
        print(f"[Noise Init] Global Mean |u,v|: {mean_level:.4f}")

    def sample_batch(self, n_f=None, n_b=None, n_0=None):
        batch = {}
        n_col = self.X_f_all.shape[0]
        idx_f = torch.randint(0, n_col, (n_f,), device=self.device)
        batch["X_f"] = self.X_f_all[idx_f]

        if self.use_data and self.X_data is not None:
            n_data = self.X_data.shape[0]
            k = min(self.n_data_batch, n_data)
            idx_d = torch.randint(0, n_data, (k,), device=self.device)
            batch["X_d"] = self.X_data[idx_d]
            batch["y_d"] = self.y_data[idx_d] 
            
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

    def pde_residual_loss(self, model, batch):
        X = make_leaf(batch["X_f"]) # [N, 3]
        out = model(X)
        u, v = out[:, 0:1], out[:, 1:2]
        
        du = grad_sum(u, X)
        dv = grad_sum(v, X)
        u_x, u_y, u_t = du[:, 0:1], du[:, 1:2], du[:, 2:3]
        v_x, v_y, v_t = dv[:, 0:1], dv[:, 1:2], dv[:, 2:3]
        
        d2ux = grad_sum(u_x, X); d2uy = grad_sum(u_y, X)
        u_xx, u_yy = d2ux[:, 0:1], d2uy[:, 1:2]
        
        d2vx = grad_sum(v_x, X); d2vy = grad_sum(v_y, X)
        v_xx, v_yy = d2vx[:, 0:1], d2vy[:, 1:2]
        
        res_u = u_t + (u * u_x + v * u_y) - self.nu * (u_xx + u_yy)
        res_v = v_t + (u * v_x + v * v_y) - self.nu * (v_xx + v_yy)
        
        loss = res_u.pow(2).mean() + res_v.pow(2).mean()
        
        # X = make_leaf(batch["X_d"])
        # out = model(X)
        
        # u, v = out[:, 0:1], out[:, 1:2]
        
        # du = grad_sum(u, X)
        # dv = grad_sum(v, X)
        # u_x, u_y, u_t = du[:, 0:1], du[:, 1:2], du[:, 2:3]
        # v_x, v_y, v_t = dv[:, 0:1], dv[:, 1:2], dv[:, 2:3]
        
        # d2ux = grad_sum(u_x, X); d2uy = grad_sum(u_y, X)
        # u_xx, u_yy = d2ux[:, 0:1], d2uy[:, 1:2]
        
        # d2vx = grad_sum(v_x, X); d2vy = grad_sum(v_y, X)
        # v_xx, v_yy = d2vx[:, 0:1], d2vy[:, 1:2]
        
        # res_u = u_t + (u * u_x + v * u_y) - self.nu * (u_xx + u_yy)
        # res_v = v_t + (u * v_x + v * v_y) - self.nu * (v_xx + v_yy)
        
        # PDE_loss_on_data = res_u.pow(2).mean() + res_v.pow(2).mean()
        
        # loss = loss + PDE_loss_on_data
        
        return loss

    def data_loss(self, model, batch, phase=1):
        if "X_d" not in batch or "y_d" not in batch:
            return torch.tensor(0.0, device=self.device)
            
        X_d = batch["X_d"]
        y_d = batch["y_d"] 
        
        pred = model(X_d)
        
        if self.use_offset and self.offset is not None:
            pred = pred + self.offset
        
        residual = y_d - pred # [N, 2]
        data_loss_value = self._data_loss(residual)
        
        if self.ebm_kind == "1D":
            residual = residual.view(-1, 1)

        # --- FIX: ADAPTIVE STANDARDIZATION ---
        # We calculate the standard deviation of the current batch of residuals.
        # This keeps the input to the EBM roughly N(0, 1), preventing collapse.
        with torch.no_grad():
            # Calculate scale per component (u and v might have different scales)
            # or global scale. Global is usually safer to preserve relative structure.
            batch_std = residual.std()
            
            if model.training and phase!=1:
                # Robust update: Clamp batch_std to avoid explosions from single bad batches
                # If running_std is 1.0, don't let batch_std force it to 100.0 instantly.
                current_std_clamped = torch.clamp(batch_std, min=1e-6, max=self.running_std * 10)
                self.running_std.mul_(1 - self.momentum).add_(current_std_clamped * self.momentum)
            
        # Standardized residuals for the EBM
        residual_scaled = residual / self.running_std
        
        if self.ebm_kind == "1D":
            data_loss_value = data_loss_value.view(-1, 1) # [2*N, 1]

        # --- Phase Logic ---
        if phase == 0: # Train EBM only
            if self.ebm is not None:
                # TRAIN on SCALED residuals
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
                
        elif phase == 1: # Standard PINN training
            return data_loss_value.view(-1)
            
        elif phase == 2: # PINN + EBM Weighted
            if self.ebm is not None:
                # TRAIN on SCALED residuals
                nll_ebm, nll_ebm_mean = self.ebm.train_step(residual_scaled.detach())
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

    def _data_loss(self, residual):
        if self.data_loss_kind == "mse":
            return data_loss_mse(residual)
        elif self.data_loss_kind == "L1":
            return data_loss_l1(residual)
        elif self.data_loss_kind == "q_gaussian":
            return data_loss_q_gaussian(residual, q=self.q_gauss_q, beta=self.q_gauss_beta)
        return residual.pow(2)

    def _get_weights(self, residual): 
        # Note: The 'residual' passed here is ALREADY scaled by the code above
        if self.data_loss_balancer_kind == "mlp" and self.weight_net is not None:
            return self.weight_net(residual), torch.tensor(0.0, device=self.device)
        elif self.data_loss_balancer_kind == "gated_trainable" and self.ebm is not None:
            with torch.no_grad():
                log_q = self.ebm(residual.detach())
            return self.gate_module(log_q)
        else:
            return self.ebm.data_weight(residual, kind=self.data_loss_balancer_kind), torch.tensor(0.0, device=self.device)

    def extra_params(self):
        params = []
        if isinstance(getattr(self, "offset", None), torch.nn.Parameter):
            params.append(self.offset)
        if isinstance(self.nu, torch.nn.Parameter):
            params.append(self.nu)
        if getattr(self, "weight_net", None) is not None:
            params.extend(list(self.weight_net.parameters()))
        if getattr(self, "gate_module", None) is not None:
            params.extend(list(self.gate_module.parameters()))
        return params

    def relative_l2_on_grid(self, model, grid_cfg=None):
        model.eval()
        t_idx = len(self.val_t) // 2
        t_val = self.val_t[t_idx]
        
        X, Y = np.meshgrid(self.val_x, self.val_y)
        T = np.full_like(X, t_val)
        
        inputs = np.stack([X.flatten(), Y.flatten(), T.flatten()], axis=1)
        inputs_torch = torch.from_numpy(inputs).float().to(self.device)
        
        with torch.no_grad():
            pred = model(inputs_torch)
            u_pred = pred[:, 0].cpu().numpy()
            v_pred = pred[:, 1].cpu().numpy()
            
        u_true = self.val_u[t_idx].flatten()
        v_true = self.val_v[t_idx].flatten()
        
        mag_pred = np.sqrt(u_pred**2 + v_pred**2)
        mag_true = np.sqrt(u_true**2 + v_true**2)
        
        num = np.linalg.norm(mag_true - mag_pred)
        den = np.linalg.norm(mag_true)
        return float(num / den)

    def make_video(self, model, grid_cfg, out_dir, fps=10, filename="flow_evolution_burgers.mp4", phase=0):
        os.makedirs(out_dir, exist_ok=True)
        model.eval()
        
        X_grid, Y_grid = np.meshgrid(self.val_x, self.val_y)
        ny, nx = X_grid.shape
        
        vmin = 0.0
        vmax = np.max(np.sqrt(self.val_u**2 + self.val_v**2))
        global_error_max = 0.0
        
        temp_inference_results = []
        
        if len(self.val_t) > 1:
            dt_window = (self.val_t[1] - self.val_t[0]) / 2.0
        else:
            dt_window = 0.05

        print(f"[Burgers2D] Pre-calculating frames. Extent: {self.extent}")
        
        with torch.no_grad():
            for i, t_val in enumerate(self.val_t):
                # Sample frames to save time (every 2nd frame)
                if i % 2 != 0: continue 
                
                X_meas_slice = None
                mag_meas_slice = None
                
                if self.use_data and self.X_data is not None:
                    X_d_cpu = self.X_data.cpu().numpy()
                    y_d_cpu = self.y_data.cpu().numpy()
                    mask_time = (X_d_cpu[:, 2] >= t_val - dt_window) & \
                                (X_d_cpu[:, 2] < t_val + dt_window)
                    
                    X_meas_slice = X_d_cpu[mask_time]
                    y_meas_slice = y_d_cpu[mask_time]
                    
                    if len(y_meas_slice) > 0:
                        mag_meas_slice = np.sqrt(y_meas_slice[:, 0]**2 + y_meas_slice[:, 1]**2)

                T_grid = np.full_like(X_grid, t_val)
                inputs = np.stack([X_grid.flatten(), Y_grid.flatten(), T_grid.flatten()], axis=1)
                inputs_torch = torch.from_numpy(inputs).float().to(self.device)
                
                out = model(inputs_torch)
                
                u_pred_grid = out[:, 0].reshape(ny, nx).cpu().numpy()
                v_pred_grid = out[:, 1].reshape(ny, nx).cpu().numpy()
                u_true_grid = self.val_u[i]
                v_true_grid = self.val_v[i]
                
                mag_pred_tmp = np.sqrt(u_pred_grid**2 + v_pred_grid**2)
                mag_true_tmp = np.sqrt(u_true_grid**2 + v_true_grid**2)
                
                curr_max = np.max(np.abs(mag_true_tmp - mag_pred_tmp))
                if curr_max > global_error_max:
                    global_error_max = curr_max
                
                temp_inference_results.append({
                    't_val': t_val,
                    'u_true': u_true_grid, 'v_true': v_true_grid,
                    'u_pred': u_pred_grid, 'v_pred': v_pred_grid,
                    'X_meas_slice': X_meas_slice, 'mag_meas_slice': mag_meas_slice
                })

        if global_error_max == 0: global_error_max = 1.0

        render_args_list = []
        for res in temp_inference_results:
            args = (
                res['t_val'], 
                res['u_true'], res['v_true'], 
                res['u_pred'], res['v_pred'], 
                res['X_meas_slice'], res['mag_meas_slice'],
                vmin, vmax, global_error_max,
                self.extent
            )
            render_args_list.append(args)

        n_workers = max(1, os.cpu_count() - 2) 
        print(f"[Burgers2D] Rendering {len(render_args_list)} frames using {n_workers} workers...")
        
        frames = []
        ctx = import_multiprocessing().get_context("fork") if os.name != 'nt' else None
        with ProcessPoolExecutor(max_workers=n_workers, mp_context=ctx) as executor:
            results = executor.map(render_frame_worker, render_args_list)
            for i, frame in enumerate(results):
                frames.append(frame)

        path = os.path.join(out_dir, filename)
        imageio.mimsave(path, frames, fps=fps, macro_block_size=None)
        print(f"[Burgers2D] Video saved to {path}")

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
            return

        print("[Burgers2D] Generating Noise Analysis video...")
        base, ext = os.path.splitext(filename)
        vid_filename = f"{base}_noise_analysis{ext}"
        
        X, Y = np.meshgrid(self.val_x, self.val_y)
        ny, nx = X.shape
        
        # 1. Get the "Lens" the EBM uses (EMA Std)
        # This guarantees the plot matches the training logic exactly.
        ref_std = float(self.running_std.item())
        
        # 2. Determine Plotting Range (Visuals Only)
        R_range = 1 # ref_std * 5.0
        
        # r = torch.linspace(-R_range, R_range, 200, device=self.device).view(-1, 1)

        # with torch.no_grad():
        #     s = r / ref_std                      # EBM이 학습한 입력공간(scaled residual)
        #     log_q = self.ebm(s).squeeze(-1)      # log q_theta(s)
        #     m = log_q.max()
        #     q_unn = torch.exp(log_q - m)         # unnormalized in r-grid (up to constant)
        #     Z = torch.trapezoid(q_unn, r.squeeze()) + 1e-12
        #     pdf_ebm = (q_unn / Z).cpu().numpy()

        # r_grid_np = r.squeeze().cpu().numpy()
        
        # --- EBM PDF Generation ---
        # Grid in ORIGINAL (raw) residual space
        r_grid_np = np.linspace(-R_range, R_range, 200).astype(np.float32)
        r_grid_torch = torch.from_numpy(r_grid_np).to(self.device).view(-1, 1)
        
        pdf_ebm = None
        with torch.no_grad():
            # SCALE using the EMA running_std
            r_input_scaled = r_grid_torch / ref_std

            # grid = self.ebm.make_grid(r_input_scaled, num_grid=200)
            # grid_input = grid.unsqueeze(-1)
            # Get unnormalized log-density
            log_q = self.ebm(r_input_scaled).squeeze(-1) # [200]
            m = log_q.max()
            q_unn = torch.exp(log_q - m)
            
            # Normalize PDF over the ORIGINAL grid range (r_grid_np)
            # This automatically accounts for the Jacobian 1/sigma scaling
            Z = torch.trapezoid(q_unn, r_grid_torch.squeeze())
            pdf_ebm = (q_unn / Z).cpu().numpy()

        # 3. Pre-calculate True PDF (1D)
        r_cpu = torch.from_numpy(r_grid_np)
        noise_scale = self.sigma_local.mean().item()
        print("Average noise scale:", noise_scale)
        pdf_true = (self.noise_model.pdf(r_cpu / noise_scale) / noise_scale).numpy()

        render_args_list = []
        
        with torch.no_grad():
            for i, t_val in enumerate(self.val_t):
                if i % 2 != 0: continue 

                # A. Grid Inference
                T = np.full_like(X, t_val)
                inputs = np.stack([X.flatten(), Y.flatten(), T.flatten()], axis=1)
                inputs_torch = torch.from_numpy(inputs).float().to(self.device)
                
                out = model(inputs_torch)
                u_pred, v_pred = out[:, 0], out[:, 1]
                
                u_true = torch.from_numpy(self.val_u[i].flatten()).to(self.device)
                v_true = torch.from_numpy(self.val_v[i].flatten()).to(self.device)
                
                # B. Sample Noise & Create Residuals
                n_points = u_true.shape[0]
                
                # Check noise kind
                kind = self.noise_cfg.get("kind", "G")
                if kind == 'MG2D':
                    eps_vec = self.noise_model.sample(n_points).float().to(self.device)
                    eps_u, eps_v = eps_vec[:, 0], eps_vec[:, 1]
                else:
                    eps_flat = self.noise_model.sample(n_points * 2).float().to(self.device)
                    eps_u, eps_v = eps_flat[:n_points], eps_flat[n_points:]
                
                # Create noisy observations
                u_noisy = u_true + eps_u
                v_noisy = v_true + eps_v
                
                # Calculate Raw Residuals
                res_u = u_noisy - u_pred
                res_v = v_noisy - v_pred
                
                # C. Prepare Data for Worker
                eps_u_grid = eps_u.view(ny, nx).cpu().numpy()
                res_u_grid = res_u.view(ny, nx).cpu().numpy()
                
                # Combine U and V for robust 1D Histogram
                eps_combined = torch.cat([eps_u, eps_v]).cpu().numpy()
                res_combined = torch.cat([res_u, res_v]).cpu().numpy()
                
                # Downsample for speed
                if eps_combined.shape[0] > 20000:
                    idx = np.random.choice(eps_combined.shape[0], 20000, replace=False)
                    eps_combined = eps_combined[idx]
                    res_combined = res_combined[idx]

                args = (
                    t_val, 
                    eps_u_grid, res_u_grid, 
                    eps_combined, res_combined,
                    r_grid_np, pdf_true, pdf_ebm,
                    R_range, self.extent
                )
                render_args_list.append(args)

        # 4. Render
        n_workers = max(1, os.cpu_count() - 2)
        frames = []
        ctx = import_multiprocessing().get_context("fork") if os.name != 'nt' else None
        
        with ProcessPoolExecutor(max_workers=n_workers, mp_context=ctx) as executor:
            results = executor.map(render_noise_worker, render_args_list)
            for frame in results:
                frames.append(frame)

        path = os.path.join(out_dir, vid_filename)
        imageio.mimsave(path, frames, fps=fps, macro_block_size=None)
        print(f"[Burgers2D] Noise video saved to {path}")

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