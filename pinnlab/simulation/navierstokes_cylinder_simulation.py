# generate_ns_data.py
import numpy as np
from tqdm import tqdm
from scipy.interpolate import RegularGridInterpolator
import os

# --- 1. Simulation Configuration ---
DOMAIN_SIZE = (2.0, 1.0)
N_POINTS = (200, 100)
CYLINDER_CENTER = (0.5, 0.5)
CYLINDER_RADIUS = 0.1
VISCOSITY = 0.005
DENSITY = 1.0

# --- CHANGE THESE LINES ---
DT = 0.001                # Reduced from 0.01 to 0.001 for stability
T_MAX = 0.4               
N_STEPS = int(T_MAX / DT)
# --------------------------

# Output configuration
N_COLLOCATION = 20000 
N_MEASUREMENT = 5000 
OUTPUT_FILE = "pinnlab/simulation/simulation_result/ns_data.npz"

def define_domain(nx, ny):
    x = np.linspace(0, DOMAIN_SIZE[0], nx)
    y = np.linspace(0, DOMAIN_SIZE[1], ny)
    X, Y = np.meshgrid(x, y)
    return x, y, X, Y

def create_cylinder_mask(X, Y, center, radius):
    dist = np.sqrt((X - center[0])**2 + (Y - center[1])**2)
    return (dist < radius).astype(float)

# --- 2. CFD Solver (Chorin's Projection) ---
def solve_navier_stokes():
    nx, ny = N_POINTS
    dx = DOMAIN_SIZE[0] / (nx - 1)
    dy = DOMAIN_SIZE[1] / (ny - 1)
    
    x, y, X, Y = define_domain(nx, ny)
    cylinder_mask = create_cylinder_mask(X, Y, CYLINDER_CENTER, CYLINDER_RADIUS)
    
    u = np.zeros((ny, nx))
    v = np.zeros((ny, nx))
    p = np.zeros((ny, nx))
    
    # Storage
    u_hist, v_hist, p_hist, t_hist = [], [], [], []

    print(f"Starting CFD Simulation ({N_STEPS} steps)...")
    
    for n in tqdm(range(N_STEPS)):
        # Laplacian
        lap_u = (np.roll(u, 1, axis=0) + np.roll(u, -1, axis=0) - 2*u) / dy**2 + \
                (np.roll(u, 1, axis=1) + np.roll(u, -1, axis=1) - 2*u) / dx**2
        lap_v = (np.roll(v, 1, axis=0) + np.roll(v, -1, axis=0) - 2*v) / dy**2 + \
                (np.roll(v, 1, axis=1) + np.roll(v, -1, axis=1) - 2*v) / dx**2

        # Advection
        du_dx = (np.roll(u, -1, axis=1) - np.roll(u, 1, axis=1)) / (2*dx)
        du_dy = (np.roll(u, -1, axis=0) - np.roll(u, 1, axis=0)) / (2*dy)
        dv_dx = (np.roll(v, -1, axis=1) - np.roll(v, 1, axis=1)) / (2*dx)
        dv_dy = (np.roll(v, -1, axis=0) - np.roll(v, 1, axis=0)) / (2*dy)

        # Tentative step
        u_star = u + DT * (VISCOSITY * lap_u - (u * du_dx + v * du_dy))
        v_star = v + DT * (VISCOSITY * lap_v - (u * dv_dx + v * dv_dy))

        # BCs
        u_star[:, 0] = 4 * 1.0 * y * (1.0 - y); v_star[:, 0] = 0 # Inlet
        u_star[:, -1] = u_star[:, -2]; v_star[:, -1] = v_star[:, -2] # Outlet
        u_star[0, :] = 0; u_star[-1, :] = 0; v_star[0, :] = 0; v_star[-1, :] = 0 # Walls
        u_star[cylinder_mask == 1] = 0; v_star[cylinder_mask == 1] = 0

        # Pressure Poisson
        div_u_star = (np.roll(u_star, -1, axis=1) - np.roll(u_star, 1, axis=1)) / (2*dx) + \
                     (np.roll(v_star, -1, axis=0) - np.roll(v_star, 1, axis=0)) / (2*dy)
        b = (DENSITY / DT) * div_u_star

        for _ in range(50): 
            p = ((np.roll(p, 1, axis=1) + np.roll(p, -1, axis=1)) * dy**2 + 
                 (np.roll(p, 1, axis=0) + np.roll(p, -1, axis=0)) * dx**2 - 
                 b * dx**2 * dy**2) / (2 * (dx**2 + dy**2))
            p[:, -1] = 0; p[:, 0] = p[:, 1]; p[0, :] = p[1, :]; p[-1, :] = p[-2, :]

        # Correction
        dp_dx = (np.roll(p, -1, axis=1) - np.roll(p, 1, axis=1)) / (2*dx)
        dp_dy = (np.roll(p, -1, axis=0) - np.roll(p, 1, axis=0)) / (2*dy)

        u = u_star - (DT / DENSITY) * dp_dx
        v = v_star - (DT / DENSITY) * dp_dy

        # BCs & Mask
        u[:, 0] = 4 * 1.0 * y * (1.0 - y); v[:, 0] = 0
        u[:, -1] = u[:, -2]; v[:, -1] = v[:, -2]
        u[0, :] = 0; u[-1, :] = 0; v[0, :] = 0; v[-1, :] = 0
        u[cylinder_mask == 1] = 0; v[cylinder_mask == 1] = 0

        if n % 5 == 0: 
            u_hist.append(u.copy())
            v_hist.append(v.copy())
            p_hist.append(p.copy())
            t_hist.append(n * DT)

    return np.array(u_hist), np.array(v_hist), np.array(p_hist), np.array(t_hist), x, y

# --- 3. Sampling Logic ---
def process_and_save_data(u_sol, v_sol, p_sol, t_sol, x_grid, y_grid):
    print("Processing datasets for PINN...")
    
    interp_u = RegularGridInterpolator((t_sol, y_grid, x_grid), u_sol, method='linear', bounds_error=False, fill_value=None)
    interp_v = RegularGridInterpolator((t_sol, y_grid, x_grid), v_sol, method='linear', bounds_error=False, fill_value=None)
    interp_p = RegularGridInterpolator((t_sol, y_grid, x_grid), p_sol, method='linear', bounds_error=False, fill_value=None)
    
    # A. Collocation (Fixed Grid)
    t_idx = np.linspace(0, len(t_sol)-1, int(N_COLLOCATION**(1/3))).astype(int)
    y_idx = np.linspace(0, len(y_grid)-1, int(N_COLLOCATION**(1/3))).astype(int)
    x_idx = np.linspace(0, len(x_grid)-1, int(N_COLLOCATION**(1/3))).astype(int)
    
    T_mesh, Y_mesh, X_mesh = np.meshgrid(t_sol[t_idx], y_grid[y_idx], x_grid[x_idx], indexing='ij')
    X_col = X_mesh.flatten()[:, None]
    Y_col = Y_mesh.flatten()[:, None]
    T_col = T_mesh.flatten()[:, None]
    
    # Mask cylinder
    dist = np.sqrt((X_col - CYLINDER_CENTER[0])**2 + (Y_col - CYLINDER_CENTER[1])**2)
    valid_mask = (dist >= CYLINDER_RADIUS).flatten()
    X_f = np.hstack((X_col[valid_mask], Y_col[valid_mask], T_col[valid_mask])) # [x, y, t]
    
    # B. Measurements (Random positions)
    t_rand = np.random.uniform(t_sol[0], t_sol[-1], N_MEASUREMENT)
    x_rand = np.random.uniform(x_grid[0], x_grid[-1], N_MEASUREMENT)
    y_rand = np.random.uniform(y_grid[0], y_grid[-1], N_MEASUREMENT)
    
    dist_rand = np.sqrt((x_rand - CYLINDER_CENTER[0])**2 + (y_rand - CYLINDER_CENTER[1])**2)
    mask_outside = dist_rand >= CYLINDER_RADIUS
    
    t_meas = t_rand[mask_outside]
    x_meas = x_rand[mask_outside]
    y_meas = y_rand[mask_outside]
    
    query_points = np.stack([t_meas, y_meas, x_meas], axis=1)
    u_meas = interp_u(query_points)
    v_meas = interp_v(query_points)
    
    X_u = np.stack([x_meas, y_meas, t_meas], axis=1) # Inputs
    Y_u = np.stack([u_meas, v_meas], axis=1)         # Targets (u, v)

    # Save everything
    print(f"Saving to {OUTPUT_FILE}...")
    np.savez(OUTPUT_FILE, 
             X_f=X_f,       # Collocation inputs
             X_u=X_u,       # Measurement inputs
             Y_u=Y_u,       # Measurement targets (clean)
             # Full grid data for video/validation
             t_grid=t_sol, x_grid=x_grid, y_grid=y_grid,
             u_full=u_sol, v_full=v_sol, p_full=p_sol,
             viscosity=VISCOSITY)
    print("Done.")

if __name__ == "__main__":
    u_h, v_h, p_h, t_h, x_g, y_g = solve_navier_stokes()
    process_and_save_data(u_h, v_h, p_h, t_h, x_g, y_g)