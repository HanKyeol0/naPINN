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
from pinnlab.utils.ebm import (
    LearnableThresholdGate,
    QuantileThresholdGate,
    TrainableLikelihoodGate,
)
from pinnlab.utils.density import create_density_estimator
from pinnlab.utils.data_loss import (
    data_loss_mse,
    data_loss_l1,
    data_loss_q_gaussian,
)

from concurrent.futures import ProcessPoolExecutor
import seaborn as sns
from sklearn.metrics import confusion_matrix
import traceback

# Use Agg backend for headless video generation
matplotlib.use('Agg')

def import_multiprocessing():
    import multiprocessing as mp
    return mp

# --- WORKER: Flow Evolution (Physical Solution) ---
def render_frame_worker_u(args):
    """
    Render physical solution: True State u, Predicted State u, Noisy Data, Error.
    """
    (t_val, u_true, v_true, u_pred, v_pred, 
     X_meas_slice, mag_meas_slice, vmin, vmax, error_max, extent, frames_dir) = args

    fig, ax = plt.subplots(2, 2, figsize=(10, 10), dpi=100)
    plt.suptitle(f"LambdaOmega 2D Flow | t={t_val:.3f}", y=0.95, fontsize=14)

    # --- CHANGE: Visualize Component 'u' instead of Magnitude ---
    # Magnitude is static for spiral waves; 'u' shows the rotation.
    
    # [0,0] True State u
    # Use a diverging colormap (e.g., 'twilight' or 'seismic') for waves
    im0 = ax[0, 0].imshow(u_true, origin='lower', extent=extent, vmin=vmin, vmax=vmax, cmap='twilight')
    ax[0, 0].set_title("True State $u(x,y)$")
    ax[0, 0].set_ylabel("y")
    plt.colorbar(im0, ax=ax[0, 0], fraction=0.046, pad=0.04)

    # [0,1] Predicted State u
    im1 = ax[0, 1].imshow(u_pred, origin='lower', extent=extent, vmin=vmin, vmax=vmax, cmap='twilight')
    ax[0, 1].set_title(r"Predicted State $\hat{u}(x,y)$")
    plt.colorbar(im1, ax=ax[0, 1], fraction=0.046, pad=0.04)

    # [1,0] Noisy Measurement Data (u component)
    # Background: Faint true flow
    ax[1, 0].imshow(u_true, origin='lower', extent=extent, vmin=vmin, vmax=vmax, cmap='gray', alpha=0.15)
    
    if X_meas_slice is not None and len(X_meas_slice) > 0:
        sc = ax[1, 0].scatter(X_meas_slice[:, 0], X_meas_slice[:, 1], c=mag_meas_slice, vmin=vmin, vmax=vmax,
                              cmap='twilight', s=15, edgecolors='none') # label='Sensors'
        # ax[1, 0].legend(loc='upper right')
        plt.colorbar(sc, ax=ax[1, 0], fraction=0.046, pad=0.04)
        ax[1, 0].set_title(f"Sensor Locations (N={len(X_meas_slice)})")
    else:
        ax[1, 0].set_title("No Measurements")
        
    ax[1, 0].set_xlim(extent[0], extent[1])
    ax[1, 0].set_ylim(extent[2], extent[3])
    ax[1, 0].set_ylabel("y"); ax[1, 0].set_xlabel("x")

    # [1,1] Absolute Error (Magnitude error is still a good metric)
    # Or use error in u: |u_true - u_pred|
    error = np.abs(u_true - u_pred)
    im2 = ax[1, 1].imshow(error, origin='lower', extent=extent, vmin=0, vmax=error_max, cmap='inferno')
    print(f"u error_max: {error_max}")
    ax[1, 1].set_title(r"Absolute Error $|u - \hat{u}|$")
    ax[1, 1].set_xlabel("x")
    plt.colorbar(im2, ax=ax[1, 1], fraction=0.046, pad=0.04)

    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    
    # save PDF frame
    frame_path = os.path.join(frames_dir, f"u_t{float(t_val):.3f}.pdf")
    fig.savefig(frame_path, format='pdf', bbox_inches='tight')
    
    fig.canvas.draw()
    w, h = fig.canvas.get_width_height()
    buf = np.frombuffer(fig.canvas.tostring_rgb(), dtype=np.uint8)
    frame = buf.reshape(h, w, 3)
    plt.close(fig)
    return frame

def render_frame_worker_v(args):
    """
    Render physical solution: True State u, Predicted State u, Noisy Data, Error.
    """
    (t_val, u_true, v_true, u_pred, v_pred, 
     X_meas_slice, mag_meas_slice, vmin, vmax, error_max, extent, frames_dir) = args

    fig, ax = plt.subplots(2, 2, figsize=(10, 10), dpi=100)
    plt.suptitle(f"LambdaOmega 2D Flow | t={t_val:.3f}", y=0.95, fontsize=14)
    
    # [0,0] True State v
    im0 = ax[0, 0].imshow(v_true, origin='lower', extent=extent, vmin=vmin, vmax=vmax, cmap='twilight')
    ax[0, 0].set_title("True State $v(x,y)$")
    ax[0, 0].set_ylabel("y")
    plt.colorbar(im0, ax=ax[0, 0], fraction=0.046, pad=0.04)

    # [0,1] Predicted State v
    im1 = ax[0, 1].imshow(v_pred, origin='lower', extent=extent, vmin=vmin, vmax=vmax, cmap='twilight')
    ax[0, 1].set_title(r"Predicted State $\hat{v}(x,y)$")
    plt.colorbar(im1, ax=ax[0, 1], fraction=0.046, pad=0.04)

    # [1,0] Noisy Measurement Data (v component)
    # Background: Faint true flow
    ax[1, 0].imshow(v_true, origin='lower', extent=extent, vmin=vmin, vmax=vmax, cmap='gray', alpha=0.15)
    
    if X_meas_slice is not None and len(X_meas_slice) > 0:
        # Visualize the measured 'v' value, not magnitude
        # We assume X_meas_slice contains [x, y, t] and we need the value 'v' corresponding to it.
        # However, the previous logic passed 'mag_meas_slice'. 
        # Ideally, pass 'v_meas_slice' in args. 
        # Fallback: Plot position only if values aren't passed, or use mag if that's all we have.
        # For now, let's keep plotting the scatter magnitude or just positions.
        sc = ax[1, 0].scatter(X_meas_slice[:, 0], X_meas_slice[:, 1], c='k', 
                              s=5, alpha=0.5, label='Sensors')
        ax[1, 0].legend(loc='upper right')
        ax[1, 0].set_title(f"Sensor Locations (N={len(X_meas_slice)})")
    else:
        ax[1, 0].set_title("No Measurements")
        
    ax[1, 0].set_xlim(extent[0], extent[1])
    ax[1, 0].set_ylim(extent[2], extent[3])
    ax[1, 0].set_ylabel("y"); ax[1, 0].set_xlabel("x")

    # [1,1] Absolute Error (Magnitude error is still a good metric)
    # Or use error in u: |v_true - v_pred|
    error = np.abs(v_true - v_pred)
    im2 = ax[1, 1].imshow(error, origin='lower', extent=extent, vmin=0, vmax=error_max, cmap='inferno')
    print(f"v error_max: {error_max}")
    ax[1, 1].set_title(r"Absolute Error $|v - \hat{v}|$")
    ax[1, 1].set_xlabel("x")
    plt.colorbar(im2, ax=ax[1, 1], fraction=0.046, pad=0.04)

    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    
    # save PDF frame
    frame_path = os.path.join(frames_dir, f"v_t{float(t_val):.3f}.pdf")
    fig.savefig(frame_path, format='pdf', bbox_inches='tight')
    
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
     R_range, extent, frames_dir) = args

    fig, axes = plt.subplots(2, 2, figsize=(12, 10), dpi=100)
    plt.suptitle(f"Noise Distribution Analysis hahaha.. | t={t_val:.3f}", y=0.96, fontsize=14)

    cmap = 'coolwarm' 
    vm = R_range

    # --- [0,0] True Noise Field (u-component) ---
    im0 = axes[0, 0].imshow(eps_u_grid, origin='lower', extent=extent, 
                          vmin=-vm, vmax=vm, cmap=cmap)
    axes[0, 0].set_title(r"True Noise Field $\epsilon_u(x,y)$")
    axes[0, 0].set_ylabel("y")
    fig.colorbar(im0, ax=axes[0, 0], fraction=0.046, pad=0.04)

    # --- [0,1] Residual Field (u-component) ---
    im1 = axes[0, 1].imshow(res_u_grid, origin='lower', extent=extent, 
                          vmin=-vm, vmax=vm, cmap=cmap)
    axes[0, 1].set_title(r"Residual Field $r_u = y_u - \hat{u}$")
    fig.colorbar(im1, ax=axes[0, 1], fraction=0.046, pad=0.04)

    # --- [1,0] Histograms (Combined u and v) ---
    # We combine u and v to see the aggregate noise statistics
    axes[1, 0].hist(eps_flat_all, bins=60, density=True, alpha=0.5, color='gray', label=r'True Noise $\epsilon$')
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
    
    # save PDF frame
    frame_path = os.path.join(frames_dir, f"t_{float(t_val):.3f}.pdf")
    fig.savefig(frame_path, format='pdf', bbox_inches='tight')
    
    fig.canvas.draw()
    w, h = fig.canvas.get_width_height()
    buf = np.frombuffer(fig.canvas.tostring_rgb(), dtype=np.uint8)
    frame = buf.reshape(h, w, 3)
    plt.close(fig)
    return frame


class LambdaOmega2D(BaseExperiment):
    def __init__(self, cfg, device):
        super().__init__(cfg, device)
        self.device = device
        
        # Load Simulation Data
        dir_path = cfg["dir_path"]
        simulation_tag = cfg["simulation_tag"]
        data_path = os.path.join(dir_path, simulation_tag, "data.npz")
        if not os.path.exists(data_path):
            raise FileNotFoundError(f"Could not find generated data: {data_path}")
        
        print(f"[LambdaOmega2D] Loading data from {data_path}")
        raw_data = np.load(data_path)
        
        pde_cfg = cfg.get("pde", {}) or {}
        self.learn_params = cfg.get("pde", {}).get("learn_params", False)
        true_beta = float(raw_data['beta'])
        if self.learn_params:
            print(f"[LambdaOmega2D] Learning beta parameter (true value: {true_beta})")
            self.beta = torch.nn.Parameter(torch.tensor(0.0, device=device)) # Init guess
        else:
            print(f"[LambdaOmega2D] Using fixed beta parameter: {true_beta}")
            self.beta = true_beta
        
        self.d_u = float(raw_data['d_u'])
        self.d_v = float(raw_data['d_v'])
        
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
        self.noise_pars = noise_cfg.get("pars", 0)
        self.n_data_batch = int(noise_cfg["batch_size"])
        self.par_list = noise_cfg.get("par_list", None)

        self.extra_noise_cfg = noise_cfg.get("extra_noise", {})
        self.use_extra_noise = bool(self.extra_noise_cfg["enabled"])
        self.outlier_kind = self.extra_noise_cfg.get("kind", "std")  # 'std' or 'mean_level'
        # 'additive' reproduces every completed run; 'replacement' matches the
        # manuscript's wording. See _init_noisy_dataset.
        self.outlier_mode = self.extra_noise_cfg.get("outlier_mode", "additive")
        
        self.X_data = None
        self.y_data = None
        self.y_clean = None
        self.noise_model = None
        
        if self.use_data:
            self._init_noisy_dataset()

        # EBM and Loss Balancer Setup
        ebm_cfg = cfg.get("ebm", {}) or {}
        self.use_ebm = bool(ebm_cfg.get("enabled", False))
        self.ebm_init_train_epochs = int(ebm_cfg["init_train_epochs"])
        
        self.running_std = torch.tensor(1.0, device=device)
        self.std_mode = ebm_cfg.get("std_mode", "ema")  # "ema" | "batch"
        self.momentum = float(ebm_cfg.get("momentum", 0.05))

        if self.use_ebm:
            self.ebm = create_density_estimator(ebm_cfg, input_dim=1, device=device)
        else:
            self.ebm = None

        data_loss_cfg = cfg.get("data_loss", {}) or {}
        self.data_loss_kind = data_loss_cfg.get("kind", "mse")
        self.q_gauss_q = float(data_loss_cfg.get("q", 1.2))
        beta_val = data_loss_cfg.get("beta", None)
        self.q_gauss_beta = float(beta_val) if beta_val is not None else None

        data_lb_cfg = cfg.get("data_loss_balancer", {})
        self.use_data_loss_balancer = bool(data_lb_cfg.get("use_loss_balancer", False))
        self.data_loss_balancer_kind = data_lb_cfg.get("kind", "gated_trainable")

        self.gate_module = None
        if self.use_data_loss_balancer and self.data_loss_balancer_kind == "gated_trainable":
            self.rejection_cost = float(data_lb_cfg.get("rejection_cost", 0.5))
            # Distinct key names: the pre-existing ``init_steepness`` key
            # belongs to LearnableThresholdGate, and reusing it here would
            # silently change every completed gated_trainable run. Absent
            # keys fall back to the constructor values used so far.
            self.gate_module = TrainableLikelihoodGate(
                device=device,
                rejection_cost=self.rejection_cost,
                init_cutoff_sigma=float(
                    data_lb_cfg.get("gate_init_cutoff_sigma", 2.0)
                ),
                init_steepness=float(
                    data_lb_cfg.get("gate_init_steepness", 30.0)
                ),
            )

        self.quantile_gate = None
        if self.use_data_loss_balancer and self.data_loss_balancer_kind == "quantile":
            self.quantile_gate = QuantileThresholdGate(
                quantile=float(data_lb_cfg.get("quantile", 0.95)),
                steepness=float(data_lb_cfg.get("steepness", 10.0)),
                device=device,
            )

        self.threshold_gate = None
        if self.use_data_loss_balancer and self.data_loss_balancer_kind == "threshold":
            self.threshold_gate = LearnableThresholdGate(
                init_threshold=float(data_lb_cfg.get("init_threshold", 1.0)),
                init_steepness=float(data_lb_cfg.get("init_steepness", 10.0)),
                rejection_cost=float(data_lb_cfg.get("rejection_cost", 0.5)),
                device=device,
            )

    def state_dict(self):
        state = {'running_std': self.running_std}
        if self.learn_params:
            state['beta'] = self.beta
        
        # Save EBM state if it exists
        if self.ebm is not None:
            state['ebm'] = self.ebm.state_dict()
            state['ebm_optimizer'] = self.ebm.optimizer.state_dict()
            
        # Save Gate state if it exists
        if self.gate_module is not None:
            state['gate_module'] = self.gate_module.state_dict()
            
        if self.threshold_gate is not None:
            state['threshold_gate'] = self.threshold_gate.state_dict()

        return state

    def load_state_dict(self, state_dict):
        if 'running_std' in state_dict:
            self.running_std.copy_(state_dict['running_std'].to(self.device))
            print(f"[LambdaOmega2D] Loaded running_std: {self.running_std.item():.4f}")

        if 'beta' in state_dict and self.learn_params:
            with torch.no_grad():
                self.beta.copy_(state_dict['beta'].to(self.device))
                print(f"[LambdaOmega2D] Loaded learned beta: {self.beta.item():.6f}")

        if 'ebm' in state_dict and self.ebm is not None:
            self.ebm.load_state_dict(state_dict['ebm'])
            if 'ebm_optimizer' in state_dict:
                self.ebm.optimizer.load_state_dict(state_dict['ebm_optimizer'])

        if 'gate_module' in state_dict and self.gate_module is not None:
            self.gate_module.load_state_dict(state_dict['gate_module'])

        if 'threshold_gate' in state_dict and self.threshold_gate is not None:
            self.threshold_gate.load_state_dict(state_dict['threshold_gate'])

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
        if kind == "4G":
            self.noise_model = get_noise(kind, f=1.0, pars=self.noise_pars, par_list=self.par_list)
        else:
            self.noise_model = get_noise(kind, f=1.0, pars=self.noise_pars)
        
        z = self.noise_model.sample(n * 2).float().to(self.device).view(n, 2)
            
        eps = z * self.sigma_local
        noise_std = eps.std(unbiased=True).item()
        
        # Initialize indices list
        self.outlier_indices = []
        replacement_rows = torch.zeros(
            eps.shape[0], dtype=torch.bool, device=self.device
        )
        replacement_values = torch.zeros_like(eps)
        
        # Outliers
        if self.use_extra_noise:
            n_extra = int(self.extra_noise_cfg.get("n_points", 0))
            if n_extra > 0:
                print(f"[LambdaOmega2D] Injecting outliers into {n_extra} points.")
                
                idx = torch.randperm(n, device=self.device)[:n_extra]
                self.outlier_indices = idx.cpu().numpy()
                scale_min = float(self.extra_noise_cfg.get("scale_min", 5.0))
                scale_max = float(self.extra_noise_cfg.get("scale_max", 10.0))
                
                factors = torch.empty(n_extra, 2, device=self.device).uniform_(scale_min, scale_max)
                
                if self.outlier_kind == "std":
                    amp = factors * noise_std
                else:  # 'mean_level'
                    f_outlier = legacy_scale * mean_level
                    amp = factors * f_outlier
                # The manuscript describes gross outliers as replacing an
                # observation, while this code has always added a positive
                # offset to it. ``outlier_mode`` makes the difference explicit;
                # the default reproduces every completed run bit-for-bit.
                if self.outlier_mode == "replacement":
                    replacement_rows[idx] = True
                    replacement_values[idx] = amp
                else:
                    eps[idx] += amp
                
        y_noisy = y_clean + eps
        if self.outlier_mode == "replacement":
            # A replaced reading carries no information about the local true
            # value, so the observation becomes the spurious magnitude itself
            # rather than the true value plus an offset. The corrupted rows,
            # their draw and the background noise are otherwise identical to
            # the additive protocol, which makes the two paired.
            y_noisy = torch.where(
                replacement_rows.view(-1, 1), replacement_values, y_noisy
            )
        
        self.X_data = self.X_u_clean
        self.y_clean = y_clean
        self.y_data = y_noisy
        
        print(f"[Noise Init] {n} measurements; mean |u,v|={mean_level:.4f}")

    def sample_batch(self, n_f):
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
        if self.ebm is None:
            print("[EBM Init] Skipped — no EBM configured.")
            return
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
            residual = (y_d - pred).view(-1, 1)
            with torch.no_grad():
                batch_std = residual.std()
                if self.std_mode == "ema":
                    currend_std_clamped = torch.clamp(batch_std, min=1e-6, max=self.running_std * 10)
                    self.running_std.mul(1 - self.momentum).add_(currend_std_clamped * self.momentum)
                    std_for_scale = self.running_std
                else:  # "batch"
                    std_for_scale = torch.clamp(batch_std, min=1e-6)
                    self.running_std.fill_(std_for_scale)
            residual_scaled = residual / std_for_scale
            self.ebm.train_step(residual_scaled.detach())

    def pde_residual_loss(self, model, batch):
        X = make_leaf(batch["X_f"]) # [N, 3] (x,y,t)
        out = model(X)
        u, v = out[:, 0:1], out[:, 1:2]
        
        du = grad_sum(u, X); dv = grad_sum(v, X)
        u_x, u_y, u_t = du[:, 0:1], du[:, 1:2], du[:, 2:3]
        v_x, v_y, v_t = dv[:, 0:1], dv[:, 1:2], dv[:, 2:3]
        
        d2ux = grad_sum(u_x, X); d2uy = grad_sum(u_y, X)
        u_xx, u_yy = d2ux[:, 0:1], d2uy[:, 1:2]
        
        d2vx = grad_sum(v_x, X); d2vy = grad_sum(v_y, X)
        v_xx, v_yy = d2vx[:, 0:1], d2vy[:, 1:2]
        
        # Reaction terms
        r2 = u**2 + v**2
        lam = 1.0 - r2
        # Note: omega(r) = -beta * r^2
        omega = -self.beta * r2
        
        res_u = u_t - self.d_u*(u_xx + u_yy) - (lam*u - omega*v)
        res_v = v_t - self.d_v*(v_xx + v_yy) - (omega*u + lam*v)
        
        return res_u.pow(2).mean() + res_v.pow(2).mean()

    def data_loss(self, model, batch, phase=1):
        if "X_d" not in batch or "y_d" not in batch:
            return torch.tensor(0.0, device=self.device)
            
        X_d = batch["X_d"]
        y_d = batch["y_d"] 
        
        pred = model(X_d)

        residual = y_d - pred # [N, 2]
        data_loss_value = self._data_loss(residual).view(-1, 1)
        residual = residual.view(-1, 1)

        with torch.no_grad():
            batch_std = residual.std()

            if model.training and phase!=1:
                if self.std_mode == "ema":
                    current_std_clamped = torch.clamp(batch_std, min=1e-6, max=self.running_std * 10)
                    self.running_std.mul_(1 - self.momentum).add_(current_std_clamped * self.momentum)
                    std_for_scale = self.running_std
                else:  # "batch"
                    std_for_scale = torch.clamp(batch_std, min=1e-6)
                    self.running_std.fill_(std_for_scale)
            else:
                std_for_scale = self.running_std
        residual_scaled = residual / std_for_scale

        if phase == 0:
            if self.ebm is not None:
                _, nll_ebm_mean = self.ebm.train_step(residual_scaled.detach())
                batch["ebm_nll"] = nll_ebm_mean

            if self.use_data_loss_balancer:
                w, gate_reg_loss = self._get_weights(residual_scaled.detach())
                weighted_loss = (w * data_loss_value).mean()
                total_loss = weighted_loss + gate_reg_loss
            else:
                total_loss = data_loss_value.mean()
            return total_loss

        elif phase == 1: # Standard PINN training
            return data_loss_value.mean()
            
        elif phase == 2: # PINN + EBM Weighted
            if self.ebm is not None:
                _, nll_ebm_mean = self.ebm.train_step(residual_scaled.detach())
                batch["ebm_nll"] = nll_ebm_mean

            if self.use_data_loss_balancer:
                w, gate_reg_loss = self._get_weights(residual_scaled.detach())
                self._last_n_filtered = int((w < 0.5).sum().item())
                self._last_n_total = w.numel()
                weighted_loss = (w * data_loss_value).mean()
                total_loss = weighted_loss + gate_reg_loss
            else:
                total_loss = data_loss_value.mean()
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
        if self.data_loss_balancer_kind == "quantile" and self.quantile_gate is not None:
            return self.quantile_gate(residual)
        elif self.data_loss_balancer_kind == "threshold" and self.threshold_gate is not None:
            return self.threshold_gate(residual)
        elif self.data_loss_balancer_kind == "gated_trainable" and self.ebm is not None:
            with torch.no_grad():
                log_q = self.ebm(residual.detach())
            return self.gate_module(log_q)
        raise ValueError(
            f"Unsupported data_loss_balancer kind: {self.data_loss_balancer_kind}"
        )

    def extra_params(self):
        params = []
        if isinstance(self.beta, torch.nn.Parameter):
            params.append(self.beta)
        if getattr(self, "gate_module", None) is not None:
            params.extend(list(self.gate_module.parameters()))
        if getattr(self, "threshold_gate", None) is not None:
            params.extend(list(self.threshold_gate.parameters()))
        return params

    def eval_on_grid(self, model, grid_cfg=None, batch_size=10000):
        model.eval()
        
        T, Y, X = np.meshgrid(self.val_t, self.val_y, self.val_x, indexing='ij')
        flat_x = X.flatten()
        flat_y = Y.flatten()
        flat_t = T.flatten()
        inputs = np.stack([flat_x, flat_y, flat_t], axis=1)
        
        # Ground Truth (flattened to match)
        u_true = self.val_u.flatten()
        v_true = self.val_v.flatten()
        
        # 2. Batched Prediction (Safe for Memory)
        u_pred_list, v_pred_list = [], []
        num_samples = inputs.shape[0]
        
        with torch.no_grad():
            for i in range(0, num_samples, batch_size):
                # Prepare batch
                batch_inputs = inputs[i : i+batch_size]
                batch_tensor = torch.from_numpy(batch_inputs).float().to(self.device)
                pred = model(batch_tensor)
                
                u_pred_list.append(pred[:, 0].cpu().numpy())
                v_pred_list.append(pred[:, 1].cpu().numpy())
        
        # Concatenate all batches
        u_pred = np.concatenate(u_pred_list)
        v_pred = np.concatenate(v_pred_list)
        
        err_sq = (u_true - u_pred)**2 + (v_true - v_pred)**2
        true_sq = u_true**2 + v_true**2
        
        # rMAE
        err_mag = np.sqrt(err_sq)
        true_mag = np.sqrt(true_sq)
        mae_vec = np.mean(err_mag)
        mean_true_mag = np.mean(true_mag)
        rmae = float(mae_vec / (mean_true_mag))
        
        # rMSE
        rmse_vec = np.sqrt(np.mean(err_sq))
        rms_true = np.sqrt(np.mean(true_sq))
        rmse = float(rmse_vec / (rms_true))

        print(f"[Eval] rMAE: {rmae:.6f} | rMSE: {rmse:.6f}")
        return {"rMAE": rmae, "rMSE": rmse}

    def make_video(self, model, grid_cfg, out_dir, fps=10, filename="flow_evolution_lambdaomega.mp4", phase=0): 
        model.eval()
        os.makedirs(out_dir, exist_ok=True)
        frame_dir_name = filename.replace(".mp4", "_frames")
        frames_dir = os.path.join(out_dir, frame_dir_name)
        os.makedirs(frames_dir, exist_ok=True)

        X_grid, Y_grid = np.meshgrid(self.val_x, self.val_y)
        ny, nx = X_grid.shape

        vmin = -1.5
        vmax = 1.5
        global_error_max = 0.0
        
        temp_inference_results = []
        if len(self.val_t) > 1:
            dt_window = (self.val_t[1] - self.val_t[0]) / 2.0
        else:
            dt_window = 0.05

        print(f"[LambdaOmega2D] Pre-calculating frames. Extent: {self.extent}")
        
        with torch.no_grad():
            for i, t_val in enumerate(self.val_t):
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

                curr_max = np.max(np.abs(u_true_grid - u_pred_grid))
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
                self.extent, frames_dir
            )
            render_args_list.append(args)

        n_workers = max(1, os.cpu_count() - 2) 
        print(f"[LambdaOmega2D] Rendering {len(render_args_list)} frames using {n_workers} workers...")
        
        frames_u = []
        frames_v = []
        ctx = import_multiprocessing().get_context("fork") if os.name != 'nt' else None
        with ProcessPoolExecutor(max_workers=n_workers, mp_context=ctx) as executor:
            results_u = executor.map(render_frame_worker_u, render_args_list)
            for i, frame in enumerate(results_u):
                frames_u.append(frame)
            results_v = executor.map(render_frame_worker_v, render_args_list)
            for i, frame in enumerate(results_v):
                frames_v.append(frame)

        filename_u = filename.replace(".mp4", "_u.mp4")
        filename_v = filename.replace(".mp4", "_v.mp4")
        path_u = os.path.join(out_dir, filename_u)
        path_v = os.path.join(out_dir, filename_v)
        imageio.mimsave(path_u, frames_u, fps=fps, macro_block_size=None)
        imageio.mimsave(path_v, frames_v, fps=fps, macro_block_size=None)
        print(f"[LambdaOmega2D] Video saved to {path_u} and {path_v}")

        if phase == 2:
            try:
                self._make_noise_videos(model, out_dir, fps, filename)
            except Exception as e:
                print(f"Warning: Failed to create noise analysis video: {e}")
                traceback.print_exc()

        return path_u
    
    def plot_final(self, model, grid_cfg, out_dir):
        return None

    def _make_noise_videos(self, model, out_dir, fps, filename):
        if self.noise_model is None or self.ebm is None:
            return

        print("[LambdaOmega2D] Generating Noise Analysis video...")
        base, ext = os.path.splitext(filename)
        vid_filename = f"{base}_noise_analysis{ext}"
                
        frame_dir_name = filename.replace(".mp4", "_noise_frames")
        frames_dir = os.path.join(out_dir, frame_dir_name)
        os.makedirs(frames_dir, exist_ok=True)
        
        X, Y = np.meshgrid(self.val_x, self.val_y)
        ny, nx = X.shape
        
        # 1. Get the "Lens" the EBM uses (EMA Std)
        # This guarantees the plot matches the training logic exactly.
        ref_std = float(self.running_std.item())
        
        # 2. Determine Plotting Range (Visuals Only)
        R_range = 1 # ref_std * 5.0
        
        # --- EBM PDF Generation ---
        r_grid_np = np.linspace(-R_range, R_range, 200).astype(np.float32)
        r_grid_torch = torch.from_numpy(r_grid_np).to(self.device).view(-1, 1)
        
        pdf_ebm = None
        with torch.no_grad():
            # SCALE using the EMA running_std
            r_input_scaled = r_grid_torch / ref_std

            log_q = self.ebm(r_input_scaled).squeeze(-1) # [200]
            m = log_q.max()
            q_unn = torch.exp(log_q - m)
            
            # Normalize PDF over the ORIGINAL grid range (r_grid_np)
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
                    R_range, self.extent, frames_dir
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
        print(f"[LambdaOmega2D] Noise video saved to {path}")

    def evaluate_gate_performance(self, model, out_dir, filename_prefix=None):
        if not self.use_extra_noise:
            print("[Evaluate] Extra noise not used in this experiment. Skipping gate evaluation.")
            return

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
        # Prediction: 1 if w < 0.5 (Rejected), 0 if w >= 0.5 (Accepted)
        preds = (w_cpu < 0.5).astype(int)
        
        cm = confusion_matrix(labels, preds) 

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
