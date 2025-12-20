# burgers_simulation_2.py

import numpy as np
from tqdm import tqdm
from scipy.interpolate import RegularGridInterpolator
import os, yaml, argparse

def load_yaml(path):
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)
    
def _save_yaml(path, obj):
    with open(path, "w", encoding="utf-8") as f:
        yaml.safe_dump(obj, f)

# --- NEW: Random Field Generator ---
def generate_random_field(nx, ny, domain_size, alpha=3.0, seed=None):
    """
    Generates a Gaussian Random Field (GRF) to use as a chaotic initial condition.
    alpha: Power law decay rate of the spectrum (controls smoothness). 
           Lower alpha = rougher, Higher alpha = smoother.
    """
    if seed is not None:
        np.random.seed(seed)
        
    # Wave numbers
    kx = np.fft.fftfreq(nx) * nx
    ky = np.fft.fftfreq(ny) * ny
    KX, KY = np.meshgrid(kx, ky)
    
    # K magnitude (avoid division by zero at 0,0)
    K = np.sqrt(KX**2 + KY**2)
    K[0, 0] = 1.0 
    
    # Random phase and amplitude
    noise = np.random.normal(size=(ny, nx)) + 1j * np.random.normal(size=(ny, nx))
    
    # Apply power law spectrum: E(k) ~ k^(-alpha)
    amplitude = K ** (-alpha)
    amplitude[0, 0] = 0.0 # Remove DC component (mean = 0)
    
    field_k = noise * amplitude
    
    # Inverse FFT to get real space field
    field = np.fft.ifft2(field_k).real
    
    # Normalize to range [-1, 1] approximately, then scale
    field = field / np.std(field) 
    return field

def main(args):
    cfg = load_yaml(args.config)
    
    # --- 1. Simulation Configuration ---
    XA, XB = cfg["domain"]["x"]
    YA, YB = cfg["domain"]["y"]
    DOMAIN_SIZE = (XB - XA, YB - YA)
    nx = cfg["simulation_points"]["nx"] 
    ny = cfg["simulation_points"]["ny"] 
    # NOTE: For Periodic BCs, we don't necessarily need the +1 point if we treat the last point as wrapping to first.
    # But to keep code consistent with grid logic, we can keep (nx, ny) or (nx+1, ny+1).
    # Here we stick to the config size.
    
    VISCOSITY = cfg["nu"]
    DT = cfg["simulation_points"]["dt"]

    BURN_IN_TIME = cfg["domain"]["burn_in_time"]
    RECORD_TIME = cfg["domain"]["record_time"]
    N_STEPS_BURN = int(BURN_IN_TIME / DT)
    N_STEPS_RECORD = int(RECORD_TIME / DT)
    RECORD_EVERY = cfg["simulation_points"].get("every", 1)

    # Output configuration
    N_MEASUREMENT = cfg["n_measurement"]
    DIR_PATH = cfg["dir_path"]
    SIMULATION_TAG = cfg["simulation_tag"]

    os.makedirs(os.path.join(DIR_PATH, SIMULATION_TAG), exist_ok=True)
    DATA_PATH = os.path.join(DIR_PATH, SIMULATION_TAG, "data.npz")
    CONFIG_PATH = os.path.join(DIR_PATH, SIMULATION_TAG, "config.yaml")

    def define_domain(nx, ny):
        x = np.linspace(XA, XB, nx)
        y = np.linspace(YA, YB, ny)
        X, Y = np.meshgrid(x, y)
        return x, y, X, Y

    # --- 2. 2D Burgers Solver (Upwind Scheme) ---
    def solve_burgers():
        x, y, X, Y = define_domain(nx, ny)
        dx = DOMAIN_SIZE[0] / nx
        dy = DOMAIN_SIZE[1] / ny

        # --- Initial Condition ---
        print("Generating Random Initial Conditions...")
        # Using alpha=3.0 is a safe balance for Upwind
        u = generate_random_field(nx, ny, DOMAIN_SIZE, alpha=3.0, seed=42) 
        v = generate_random_field(nx, ny, DOMAIN_SIZE, alpha=3.0, seed=123)
        u = u * 1.5
        v = v * 1.5

        u_hist, v_hist, t_hist = [], [], []

        print(f"1. Burn-in Phase ({N_STEPS_BURN} steps)...")
        print(f"2. Recording Phase ({N_STEPS_RECORD} steps)...")
        
        total_steps = N_STEPS_BURN + N_STEPS_RECORD
        
        for n in tqdm(range(total_steps)):
            
            # --- 1. PRE-CALCULATE NEIGHBORS (Periodic) ---
            u_xp = np.roll(u, -1, axis=1); u_xm = np.roll(u, 1, axis=1)
            u_yp = np.roll(u, -1, axis=0); u_ym = np.roll(u, 1, axis=0)
            
            v_xp = np.roll(v, -1, axis=1); v_xm = np.roll(v, 1, axis=1)
            v_yp = np.roll(v, -1, axis=0); v_ym = np.roll(v, 1, axis=0)

            # --- 2. DIFFUSION (Central Difference is fine here) ---
            lap_u = (u_yp + u_ym - 2*u) / dy**2 + (u_xp + u_xm - 2*u) / dx**2
            lap_v = (v_yp + v_ym - 2*v) / dy**2 + (v_xp + v_xm - 2*v) / dx**2

            # --- 3. ADVECTION (UPWIND SCHEME - The Fix) ---
            # Backward Difference (use if velocity > 0)
            du_dx_b = (u - u_xm) / dx
            du_dy_b = (u - u_ym) / dy
            dv_dx_b = (v - v_xm) / dx
            dv_dy_b = (v - v_ym) / dy
            
            # Forward Difference (use if velocity < 0)
            du_dx_f = (u_xp - u) / dx
            du_dy_f = (u_yp - u) / dy
            dv_dx_f = (v_xp - v) / dx
            dv_dy_f = (v_yp - v) / dy

            # Apply Upwind Logic:
            # term: u * du/dx
            # if u > 0, use backward diff. if u < 0, use forward diff.
            adv_u_x = np.maximum(u, 0) * du_dx_b + np.minimum(u, 0) * du_dx_f
            adv_u_y = np.maximum(v, 0) * du_dy_b + np.minimum(v, 0) * du_dy_f
            
            adv_v_x = np.maximum(u, 0) * dv_dx_b + np.minimum(u, 0) * dv_dx_f
            adv_v_y = np.maximum(v, 0) * dv_dy_b + np.minimum(v, 0) * dv_dy_f

            # --- 4. UPDATE ---
            u_new = u + DT * (VISCOSITY * lap_u - (adv_u_x + adv_u_y))
            v_new = v + DT * (VISCOSITY * lap_v - (adv_v_x + adv_v_y))

            u = u_new
            v = v_new

            # Recording
            if n > N_STEPS_BURN:
                if (n - N_STEPS_BURN) % RECORD_EVERY == 0:
                    u_hist.append(u.copy())
                    v_hist.append(v.copy())
                    t_hist.append((n - N_STEPS_BURN) * DT)

        return np.array(u_hist), np.array(v_hist), np.array(t_hist), x, y

    # --- 3. Sampling Logic (Clean Data Only) ---
    def process_and_save_data(u_sol, v_sol, t_sol, x_grid, y_grid):
        print("Processing datasets for PINN (Clean Data)...")
        
        # NOTE: RegularGridInterpolator assumes a sorted grid. 
        # For periodic data, the edges are effectively connected, 
        # but the interpolator treats them as boundaries. 
        # Ideally, we pad the data for interpolation if we want perfect periodicity queries,
        # but standard interpolation is fine for training points inside the domain.

        interp_u = RegularGridInterpolator((t_sol, y_grid, x_grid), u_sol, method='linear', bounds_error=False, fill_value=None)
        interp_v = RegularGridInterpolator((t_sol, y_grid, x_grid), v_sol, method='linear', bounds_error=False, fill_value=None)

        # A. Collocation Points (Full Domain)
        T_mesh, Y_mesh, X_mesh = np.meshgrid(t_sol, y_grid, x_grid, indexing='ij')
        X_col = X_mesh.flatten()[:, None]
        Y_col = Y_mesh.flatten()[:, None]
        T_col = T_mesh.flatten()[:, None]
        
        X_f = np.hstack((X_col, Y_col, T_col)) # [x, y, t]

        # B. Measurements (Random positions)
        t_meas = np.random.uniform(t_sol[0], t_sol[-1], N_MEASUREMENT)
        x_meas = np.random.uniform(x_grid[0], x_grid[-1], N_MEASUREMENT)
        y_meas = np.random.uniform(y_grid[0], y_grid[-1], N_MEASUREMENT)
        
        query_points = np.stack([t_meas, y_meas, x_meas], axis=1)
        u_meas = interp_u(query_points)
        v_meas = interp_v(query_points)
        
        X_u = np.stack([x_meas, y_meas, t_meas], axis=1) 
        Y_u = np.stack([u_meas, v_meas], axis=1)

        print(f"Saving to {DATA_PATH}...")
        np.savez(DATA_PATH, 
                 X_f=X_f,       
                 X_u=X_u,      
                 Y_u=Y_u,       
                 t_grid=t_sol, x_grid=x_grid, y_grid=y_grid,
                 u_full=u_sol, v_full=v_sol,
                 viscosity=VISCOSITY)
        
        _save_yaml(CONFIG_PATH, cfg)
        print("Done.")

    u_h, v_h, t_h, x_g, y_g = solve_burgers()
    process_and_save_data(u_h, v_h, t_h, x_g, y_g)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', required=True)
    args = parser.parse_args()
    main(args)