import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from mpl_toolkits.axes_grid1 import make_axes_locatable
import os

# ==========================================
# 1. Configuration & PDE Definition
# ==========================================
class Config:
    seed = 42
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # Domain limits
    x_min, x_max = 0.0, 1.0
    y_min, y_max = 0.0, 1.0
    t_min, t_max = 0.0, 1.0
    
    # Grid Resolution (Fixed Grid)
    Nx = 50
    Ny = 50
    Nt = 20
    
    # Patch Parameters
    patch_size_x = 10
    patch_size_y = 10
    patch_size_t = 5
    batch_size = 16  # How many patches per iteration
    epochs = 2000
    lr = 1e-3

    # PDE Constants
    alpha = 0.01  # Diffusion coefficient

# Set seeds
torch.manual_seed(Config.seed)
np.random.seed(Config.seed)

# Analytical Solution: u(x,y,t) = sin(pi*x) * sin(pi*y) * exp(-t)
def analytical_u(x, y, t):
    return torch.sin(np.pi * x) * torch.sin(np.pi * y) * torch.exp(-t)

# Source term forcing the analytical solution
# u_t = -u
# u_xx = -pi^2 * u
# u_yy = -pi^2 * u
# PDE: u_t - alpha * (u_xx + u_yy) = f
# f = -u - alpha * (-2 * pi^2 * u) = u * (2 * alpha * pi^2 - 1)
def source_term(u, alpha):
    return u * (2 * alpha * (np.pi**2) - 1)

# ==========================================
# 2. Data Preparation (Patch Sampling)
# ==========================================
class PatchDataset(Dataset):
    def __init__(self, cfg):
        # 1. Create Global Grid
        x = torch.linspace(cfg.x_min, cfg.x_max, cfg.Nx)
        y = torch.linspace(cfg.y_min, cfg.y_max, cfg.Ny)
        t = torch.linspace(cfg.t_min, cfg.t_max, cfg.Nt)
        
        # Meshgrid (t, y, x) order
        T, Y, X = torch.meshgrid(t, y, x, indexing='ij')
        
        # Coordinate Tensor: Shape [Nt, Ny, Nx, 3]
        self.coords = torch.stack([X, Y, T], dim=-1)
        
        # 2. Create Type Map (Masks)
        # 0: Interior (PDE), 1: IC, 2: BC
        self.type_map = torch.zeros((cfg.Nt, cfg.Ny, cfg.Nx), dtype=torch.long)
        
        # Initial Condition (t=0)
        self.type_map[0, :, :] = 1
        
        # Boundary Conditions (x=0, x=1, y=0, y=1)
        # Note: BC overwrites IC at corner points if t=0, which is acceptable
        self.type_map[:, 0, :] = 2 # y_min
        self.type_map[:, -1, :] = 2 # y_max
        self.type_map[:, :, 0] = 2 # x_min
        self.type_map[:, :, -1] = 2 # x_max
        
        # Pre-calculate analytical values for BC/IC enforcement
        self.exact_values = analytical_u(X, Y, T)
        
        self.cfg = cfg
        
        # Calculate valid starting indices for patches
        self.valid_t = cfg.Nt - cfg.patch_size_t + 1
        self.valid_y = cfg.Ny - cfg.patch_size_y + 1
        self.valid_x = cfg.Nx - cfg.patch_size_x + 1
        
        self.num_possible_patches = self.valid_t * self.valid_y * self.valid_x

    def __len__(self):
        # We can define epoch length arbitrarily or cover all patches
        return self.num_possible_patches // 2 # Sampling half the possible patches per epoch

    def __getitem__(self, idx):
        # Randomly sample a patch top-left corner
        t_start = np.random.randint(0, self.valid_t)
        y_start = np.random.randint(0, self.valid_y)
        x_start = np.random.randint(0, self.valid_x)
        
        # Slice the patch
        t_end = t_start + self.cfg.patch_size_t
        y_end = y_start + self.cfg.patch_size_y
        x_end = x_start + self.cfg.patch_size_x
        
        # Extract data
        patch_coords = self.coords[t_start:t_end, y_start:y_end, x_start:x_end, :]
        patch_types = self.type_map[t_start:t_end, y_start:y_end, x_start:x_end]
        patch_exact = self.exact_values[t_start:t_end, y_start:y_end, x_start:x_end]
        
        # Flatten the spatial dims of the patch for the MLP [N_points, features]
        # But we keep batch dimension in DataLoader
        return {
            'coords': patch_coords.reshape(-1, 3), # (Patch_Points, 3)
            'types': patch_types.reshape(-1),      # (Patch_Points,)
            'exact': patch_exact.reshape(-1)       # (Patch_Points,)
        }

# ==========================================
# 3. Model Architecture
# ==========================================
class PINN(nn.Module):
    def __init__(self):
        super(PINN, self).__init__()
        # Input: x, y, t
        self.net = nn.Sequential(
            nn.Linear(3, 64),
            nn.Tanh(),
            nn.Linear(64, 64),
            nn.Tanh(),
            nn.Linear(64, 64),
            nn.Tanh(),
            nn.Linear(64, 1) # Output: u
        )
        
    def forward(self, x):
        return self.net(x)

# ==========================================
# 4. Training Engine
# ==========================================
def compute_gradients(u, x):
    # Enable grad computation for x
    grads = torch.autograd.grad(
        u, x, 
        grad_outputs=torch.ones_like(u), 
        create_graph=True, 
        retain_graph=True
    )[0]
    
    u_x = grads[:, 0].unsqueeze(1)
    u_y = grads[:, 1].unsqueeze(1)
    u_t = grads[:, 2].unsqueeze(1)
    
    u_xx = torch.autograd.grad(u_x, x, grad_outputs=torch.ones_like(u_x), create_graph=True)[0][:, 0].unsqueeze(1)
    u_yy = torch.autograd.grad(u_y, x, grad_outputs=torch.ones_like(u_y), create_graph=True)[0][:, 1].unsqueeze(1)
    
    return u_t, u_xx, u_yy

def train():
    cfg = Config()
    dataset = PatchDataset(cfg)
    dataloader = DataLoader(dataset, batch_size=cfg.batch_size, shuffle=True)
    
    model = PINN().to(cfg.device)
    optimizer = optim.Adam(model.parameters(), lr=cfg.lr)
    
    print(f"Starting Training on {cfg.device}...")
    print(f"Patch Size: {cfg.patch_size_t}x{cfg.patch_size_y}x{cfg.patch_size_x}")
    
    history = []

    for epoch in range(cfg.epochs):
        total_loss = 0
        
        for batch in dataloader:
            # batch['coords'] shape: [Batch_Size, Patch_Points, 3]
            # Flatten batch and patch dims for parallel processing: [Batch*Points, 3]
            inputs = batch['coords'].to(cfg.device).view(-1, 3)
            types = batch['types'].to(cfg.device).view(-1)
            exact = batch['exact'].to(cfg.device).view(-1, 1)
            
            inputs.requires_grad = True
            
            # Forward Pass
            u_pred = model(inputs)
            
            # --- Loss Calculation Masking ---
            # Mask 0: Interior (PDE Loss)
            # Mask 1: IC (Data Loss)
            # Mask 2: BC (Data Loss)
            
            # 1. IC and BC Loss
            mask_bc_ic = (types == 1) | (types == 2)
            if mask_bc_ic.sum() > 0:
                loss_bc_ic = torch.mean((u_pred[mask_bc_ic] - exact[mask_bc_ic])**2)
            else:
                loss_bc_ic = torch.tensor(0.0).to(cfg.device)
                
            # 2. PDE Residual Loss
            mask_pde = (types == 0)
            if mask_pde.sum() > 0:
                # Only compute derivatives where necessary (optimization)
                # However, for batch efficiency in autograd, we usually compute all then mask
                u_t, u_xx, u_yy = compute_gradients(u_pred, inputs)
                
                # PDE: u_t - alpha(u_xx + u_yy) - source = 0
                f = source_term(u_pred, cfg.alpha)
                res = u_t - cfg.alpha * (u_xx + u_yy) - f
                
                loss_pde = torch.mean(res[mask_pde]**2)
            else:
                loss_pde = torch.tensor(0.0).to(cfg.device)
            
            # Total Loss
            loss = loss_pde + loss_bc_ic
            
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            total_loss += loss.item()
            
        if epoch % 100 == 0:
            print(f"Epoch {epoch}/{cfg.epochs} | Loss: {total_loss:.5f}")
            history.append(total_loss)

    return model, cfg, dataset

# ==========================================
# 5. Visualization (Video)
# ==========================================
def make_video(model, cfg, dataset):
    print("Generating Video...")
    model.eval()
    
    # Grid for visualization
    # We reuse the dataset coordinates but reshape them to [Nt, Ny, Nx, 3]
    full_coords = dataset.coords.to(cfg.device) # [Nt, Ny, Nx, 3]
    full_exact = dataset.exact_values.numpy()
    
    Nt, Ny, Nx, _ = full_coords.shape
    
    # Predict in time-slices to save memory if needed
    u_pred_full = []
    
    with torch.no_grad():
        for t in range(Nt):
            # coords at time t: [Ny, Nx, 3] -> flatten -> predict -> reshape
            slice_coords = full_coords[t, :, :, :].reshape(-1, 3)
            slice_pred = model(slice_coords)
            u_pred_full.append(slice_pred.cpu().numpy().reshape(Ny, Nx))
            
    u_pred_full = np.array(u_pred_full) # [Nt, Ny, Nx]
    abs_error = np.abs(full_exact - u_pred_full)
    
    # Setup Plots
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    
    # Initial frame limits
    vmin, vmax = np.min(full_exact), np.max(full_exact)
    
    # Plot objects
    im1 = axes[0].imshow(full_exact[0], vmin=vmin, vmax=vmax, origin='lower', extent=[0,1,0,1], cmap='jet')
    axes[0].set_title("True System")
    fig.colorbar(im1, ax=axes[0])

    im2 = axes[1].imshow(u_pred_full[0], vmin=vmin, vmax=vmax, origin='lower', extent=[0,1,0,1], cmap='jet')
    axes[1].set_title("Predicted System")
    fig.colorbar(im2, ax=axes[1])
    
    im3 = axes[2].imshow(abs_error[0], vmin=0, vmax=np.max(abs_error), origin='lower', extent=[0,1,0,1], cmap='magma')
    axes[2].set_title("Absolute Error")
    fig.colorbar(im3, ax=axes[2])
    
    tx = fig.suptitle(f"Time: 0.00")

    def animate(i):
        # Update data
        im1.set_data(full_exact[i])
        im2.set_data(u_pred_full[i])
        im3.set_data(abs_error[i])
        tx.set_text(f"Time: {dataset.coords[i,0,0,2].item():.2f}")
        return im1, im2, im3, tx

    ani = animation.FuncAnimation(fig, animate, frames=Nt, interval=200, blit=False)
    
    save_path = 'patch_pinn_result.mp4'
    ani.save(save_path, writer='ffmpeg', fps=10)
    print(f"Video saved to {save_path}")
    plt.close()

if __name__ == "__main__":
    trained_model, config, dset = train()
    make_video(trained_model, config, dset)