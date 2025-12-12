# navierstokes_cylinder_simulation_2.py
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

def main(args):
    cfg = load_yaml(args.config)
    
    # --- 1. Simulation Configuration ---
    XA, XB = cfg["domain"]["x"]
    YA, YB = cfg["domain"]["y"]
    DOMAIN_SIZE = XB - XA, YB - YA
    N_POINTS = (cfg["simulation_points"]["nx"]+1, cfg["simulation_points"]["ny"]+1)
    
    # These single cylinder vars are kept for backward compatibility if needed, 
    # but the loop below relies on the 'obstacles' list.
    CYLINDER_CENTER = (cfg["cylinder"]["x"], cfg["cylinder"]["y"])
    CYLINDER_RADIUS = cfg["cylinder"]["r"]
    
    VISCOSITY = cfg["nu"]
    DENSITY = cfg["rho"]

    DT = cfg["simulation_points"]["dt"]
    # --------------------------
    
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
        x = np.linspace(0, DOMAIN_SIZE[0], nx)
        y = np.linspace(0, DOMAIN_SIZE[1], ny)
        X, Y = np.meshgrid(x, y)
        return x, y, X, Y

    def create_obstacle_mask(X, Y, obstacles):
        # Initialize mask with all False (0.0)
        combined_mask = np.zeros_like(X, dtype=float)
        
        for obs in obstacles:
            # Check config keys to support different shapes if needed
            cx, cy, r = obs['x'], obs['y'], obs['r']
            dist = np.sqrt((X - cx)**2 + (Y - cy)**2)
            # Logical OR: if inside any cylinder, set to 1.0
            combined_mask = np.maximum(combined_mask, (dist < r).astype(float))
            
        return combined_mask

    # --- 2. CFD Solver (Chorin's Projection) ---
    def solve_navier_stokes():
        nx, ny = N_POINTS
        dx = DOMAIN_SIZE[0] / (nx - 1)
        dy = DOMAIN_SIZE[1] / (ny - 1)
        
        x, y, X, Y = define_domain(nx, ny)
        
        # Load obstacles list from config
        obstacles = cfg.get("obstacles", [{"x": CYLINDER_CENTER[0], "y": CYLINDER_CENTER[1], "r": CYLINDER_RADIUS}])
        obstacle_mask = create_obstacle_mask(X, Y, obstacles)

        u = np.zeros((ny, nx))
        v = np.zeros((ny, nx))
        p = np.zeros((ny, nx))
        
        # Storage
        u_hist, v_hist, p_hist, t_hist = [], [], [], []

        print(f"1. Burn-in Phase ({N_STEPS_BURN} steps)... Developing instability...")
        print(f"2. Recording Phase ({N_STEPS_RECORD} steps)...")
        
        total_steps = N_STEPS_BURN + N_STEPS_RECORD
        
        for n in tqdm(range(total_steps)):
            # --- PERTURBATION TO BREAK SYMMETRY ---
            if n == 10: 
                # Add small noise to v-velocity to trigger Karman Vortex Street
                noise = np.random.normal(0, 0.1, u.shape) * np.exp(-((X-0.5)**2 + (Y-0.5)**2)/0.1)
                v += noise

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
            # Slow down inlet (multiplier 2.0 instead of 4.0)
            u_star[:, 0] = 2.0 * y * (1.0 - y); v_star[:, 0] = 0 
            u_star[:, -1] = u_star[:, -2]; v_star[:, -1] = v_star[:, -2]
            u_star[0, :] = 0; u_star[-1, :] = 0; v_star[0, :] = 0; v_star[-1, :] = 0
            u_star[obstacle_mask == 1] = 0; v_star[obstacle_mask == 1] = 0

            # Pressure Poisson
            div_u_star = (np.roll(u_star, -1, axis=1) - np.roll(u_star, 1, axis=1)) / (2*dx) + \
                        (np.roll(v_star, -1, axis=0) - np.roll(v_star, 1, axis=0)) / (2*dy)
            b = (DENSITY / DT) * div_u_star

            # Reduced iterations for speed (since DT is small, pressure doesn't change much)
            for _ in range(10): 
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
            u[:, 0] = 2.0 * y * (1.0 - y); v[:, 0] = 0
            u[:, -1] = u[:, -2]; v[:, -1] = v[:, -2]
            u[0, :] = 0; u[-1, :] = 0; v[0, :] = 0; v[-1, :] = 0
            u[obstacle_mask == 1] = 0; v[obstacle_mask == 1] = 0

            # Recording logic
            if n > N_STEPS_BURN:
                # Save every RECORD_EVERY step (approx every 0.02s simulation time)
                if n % RECORD_EVERY == 0: 
                    u_hist.append(u.copy())
                    v_hist.append(v.copy())
                    p_hist.append(p.copy())
                    t_hist.append((n - N_STEPS_BURN) * DT)

        return np.array(u_hist), np.array(v_hist), np.array(p_hist), np.array(t_hist), x, y

    # --- 3. Sampling Logic (UPDATED) ---
    def process_and_save_data(u_sol, v_sol, p_sol, t_sol, x_grid, y_grid):
        print("Processing datasets for PINN...")
        
        # Load obstacles list for masking
        obstacles = cfg.get("obstacles", [{"x": CYLINDER_CENTER[0], "y": CYLINDER_CENTER[1], "r": CYLINDER_RADIUS}])

        interp_u = RegularGridInterpolator((t_sol, y_grid, x_grid), u_sol, method='linear', bounds_error=False, fill_value=None)
        interp_v = RegularGridInterpolator((t_sol, y_grid, x_grid), v_sol, method='linear', bounds_error=False, fill_value=None)
        interp_p = RegularGridInterpolator((t_sol, y_grid, x_grid), p_sol, method='linear', bounds_error=False, fill_value=None)

        T_mesh, Y_mesh, X_mesh = np.meshgrid(t_sol, y_grid, x_grid, indexing='ij')

        X_col = X_mesh.flatten()[:, None]
        Y_col = Y_mesh.flatten()[:, None]
        T_col = T_mesh.flatten()[:, None]
        
        # --- FIX: Use helper function to mask multiple cylinders ---
        # Note: X_col/Y_col are flattened arrays. We pass them to create_obstacle_mask directly.
        # The helper returns 1.0 (True) inside obstacle, 0.0 outside.
        
        # We need to reshape for the helper if it expects a specific shape, 
        # but numpy broadcasting works on flat arrays too.
        flat_mask = create_obstacle_mask(X_col.flatten(), Y_col.flatten(), obstacles)
        
        # Keep points where mask == 0 (Outside obstacles)
        valid_mask = (flat_mask == 0)
        
        X_f = np.hstack((X_col[valid_mask], Y_col[valid_mask], T_col[valid_mask])) # [x, y, t]
        
        # B. Measurements (Random positions)
        t_rand = np.random.uniform(t_sol[0], t_sol[-1], N_MEASUREMENT)
        x_rand = np.random.uniform(x_grid[0], x_grid[-1], N_MEASUREMENT)
        y_rand = np.random.uniform(y_grid[0], y_grid[-1], N_MEASUREMENT)
        
        # --- FIX: Mask random points too ---
        rand_mask = create_obstacle_mask(x_rand, y_rand, obstacles)
        mask_outside = (rand_mask == 0)
        
        t_meas = t_rand[mask_outside]
        x_meas = x_rand[mask_outside]
        y_meas = y_rand[mask_outside]
        
        query_points = np.stack([t_meas, y_meas, x_meas], axis=1)
        u_meas = interp_u(query_points)
        v_meas = interp_v(query_points)
        
        X_u = np.stack([x_meas, y_meas, t_meas], axis=1) # Inputs
        Y_u = np.stack([u_meas, v_meas], axis=1)         # Targets (u, v)

        # Save everything
        print(f"Saving to {DATA_PATH}...")
        np.savez(DATA_PATH, 
                X_f=X_f,       # Collocation inputs
                X_u=X_u,       # Measurement inputs
                Y_u=Y_u,       # Measurement targets (clean)
                # Full grid data for video/validation
                t_grid=t_sol, x_grid=x_grid, y_grid=y_grid,
                u_full=u_sol, v_full=v_sol, p_full=p_sol,
                viscosity=VISCOSITY)
        # Save config
        _save_yaml(CONFIG_PATH, cfg)
        print("Done.")
        
    u_h, v_h, p_h, t_h, x_g, y_g = solve_navier_stokes()
    process_and_save_data(u_h, v_h, p_h, t_h, x_g, y_g)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', required=True)
    args = parser.parse_args()
    main(args)