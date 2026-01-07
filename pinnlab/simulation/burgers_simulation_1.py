# burgers_simulation.py

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
    DOMAIN_SIZE = (XB - XA, YB - YA)
    nx = cfg["simulation_points"]["nx"] + 1
    ny = cfg["simulation_points"]["ny"] + 1
    N_POINTS = (nx, ny)
    
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

    # --- 2. 2D Burgers Solver ---
    def solve_burgers():
        x, y, X, Y = define_domain(nx, ny)
        dx = DOMAIN_SIZE[0] / (nx - 1)
        dy = DOMAIN_SIZE[1] / (ny - 1)

        # Initial Condition: Asymmetric & Dynamic
        # Combining sinusoidal waves with an off-center Gaussian blob to break symmetry
        u = -np.sin(np.pi * X) * np.cos(2 * np.pi * Y) + \
            0.5 * np.exp(-((X - 0.5)**2 + (Y - 0.5)**2) / 0.1)
        
        v = 0.5 * np.sin(2 * np.pi * X) * np.cos(np.pi * Y) - \
            0.5 * np.exp(-((X - 1.2)**2 + (Y - 0.8)**2) / 0.1) 

        # Storage
        u_hist, v_hist, t_hist = [], [], []

        print(f"1. Burn-in Phase ({N_STEPS_BURN} steps)...")
        print(f"2. Recording Phase ({N_STEPS_RECORD} steps)...")
        
        total_steps = N_STEPS_BURN + N_STEPS_RECORD
        
        for n in tqdm(range(total_steps)):
            
            # Derivatives (Central Difference with Periodic Wrapping)
            u_xp = np.roll(u, -1, axis=1)
            u_xm = np.roll(u, 1, axis=1)
            u_yp = np.roll(u, -1, axis=0)
            u_ym = np.roll(u, 1, axis=0)
            
            v_xp = np.roll(v, -1, axis=1)
            v_xm = np.roll(v, 1, axis=1)
            v_yp = np.roll(v, -1, axis=0)
            v_ym = np.roll(v, 1, axis=0)

            # Laplacians
            lap_u = (u_yp + u_ym - 2*u) / dy**2 + (u_xp + u_xm - 2*u) / dx**2
            lap_v = (v_yp + v_ym - 2*v) / dy**2 + (v_xp + v_xm - 2*v) / dx**2

            # Advection terms
            du_dx = (u_xp - u_xm) / (2*dx)
            du_dy = (u_yp - u_ym) / (2*dy)
            dv_dx = (v_xp - v_xm) / (2*dx)
            dv_dy = (v_yp - v_ym) / (2*dy)

            # Update Step (Explicit Euler)
            u_new = u + DT * (VISCOSITY * lap_u - (u * du_dx + v * du_dy))
            v_new = v + DT * (VISCOSITY * lap_v - (u * dv_dx + v * dv_dy))

            # Apply Dirichlet BCs (Wall = 0)
            u_new[0, :] = 0; u_new[-1, :] = 0; u_new[:, 0] = 0; u_new[:, -1] = 0
            v_new[0, :] = 0; v_new[-1, :] = 0; v_new[:, 0] = 0; v_new[:, -1] = 0
            
            u = u_new
            v = v_new

            # Recording logic
            if n > N_STEPS_BURN:
                if (n - N_STEPS_BURN) % RECORD_EVERY == 0:
                    u_hist.append(u.copy())
                    v_hist.append(v.copy())
                    t_hist.append((n - N_STEPS_BURN) * DT)

        return np.array(u_hist), np.array(v_hist), np.array(t_hist), x, y

    # --- 3. Sampling Logic (Clean Data Only) ---
    def process_and_save_data(u_sol, v_sol, t_sol, x_grid, y_grid):
        print("Processing datasets for PINN (Clean Data)...")
        
        measure_cfg = cfg.get("measurement", {})
        measure_kind = measure_cfg.get("measure_kind", "random")

        # Interpolators for off-grid sampling
        interp_u = RegularGridInterpolator((t_sol, y_grid, x_grid), u_sol, method='linear', bounds_error=False, fill_value=None)
        interp_v = RegularGridInterpolator((t_sol, y_grid, x_grid), v_sol, method='linear', bounds_error=False, fill_value=None)

        t_meas, x_meas, y_meas = [], [], []
        
        if measure_kind == "fixed_grid":
            print("sampling strategy: fixed grid positions")
            sens_xn = measure_cfg.get("sensor_nx", 10)
            sens_yn = measure_cfg.get("sensor_ny", 10)
            xs_sensor = np.linspace(x_grid[0], x_grid[-1], sens_xn)
            ys_sensor = np.linspace(y_grid[0], y_grid[-1], sens_yn)
            X_s, Y_s = np.meshgrid(xs_sensor, ys_sensor)
            
            sensor_x = X_s.flatten()
            sensor_y = Y_s.flatten()
        
        elif measure_kind == "fixed_random":
            print("Sampling Strategy: Fixed Random Spatial Points")
            n_sensors = measure_cfg.get("n_sensors", 100)
            
            # Randomly place sensors once
            sensor_x = np.random.uniform(x_grid[0], x_grid[-1], n_sensors)
            sensor_y = np.random.uniform(y_grid[0], y_grid[-1], n_sensors)
        
        elif measure_kind == "random":
            print("Sampling Strategy: Fully Random (Spatiotemporal)")
            # This mimics your original code
            N_MEASUREMENT = cfg["n_measurement"]
            t_meas = np.random.uniform(t_sol[0], t_sol[-1], N_MEASUREMENT)
            x_meas = np.random.uniform(x_grid[0], x_grid[-1], N_MEASUREMENT)
            y_meas = np.random.uniform(y_grid[0], y_grid[-1], N_MEASUREMENT)
            
            # Skip the time-loop broadcasting below
            sensor_x, sensor_y = None, None
        
        else:
            raise ValueError(f"Unknown measurement kind: {measure_kind}")
        
        
        if measure_kind in ["fixed_grid", "fixed_random"]:
            # For fixed sensors, we record at EVERY time step in t_sol
            # Or you can subsample time if needed: t_sol[::skip]
            
            all_t, all_x, all_y = [], [], []
            
            for t_val in t_sol:
                # Repeat sensor locations for this time step
                all_t.append(np.full_like(sensor_x, t_val))
                all_x.append(sensor_x)
                all_y.append(sensor_y)
            
            t_meas = np.concatenate(all_t)
            x_meas = np.concatenate(all_x)
            y_meas = np.concatenate(all_y)

        # A. Measurements (Random positions)
        query_points = np.stack([t_meas, y_meas, x_meas], axis=1)
        print(f"number of query points: {query_points.shape[0]}")
        u_meas = interp_u(query_points)
        v_meas = interp_v(query_points)

        # B. Collocation Points (Full Domain)
        T_mesh, Y_mesh, X_mesh = np.meshgrid(t_sol, y_grid, x_grid, indexing='ij')
        X_col = X_mesh.flatten()[:, None]
        Y_col = Y_mesh.flatten()[:, None]
        T_col = T_mesh.flatten()[:, None]
        
        # Flatten and stack
        X_f = np.hstack((X_col, Y_col, T_col)) # [x, y, t]
        
        # Prepare Tensors
        X_u = np.stack([x_meas, y_meas, t_meas], axis=1) # Inputs
        Y_u = np.stack([u_meas, v_meas], axis=1)         # Targets (Clean)

        # Save everything
        print(f"Saving to {DATA_PATH}...")
        np.savez(DATA_PATH, 
                 X_f=X_f,       # Collocation inputs
                 X_u=X_u,       # Measurement inputs
                 Y_u=Y_u,       # Measurement targets (CLEAN)
                 # Full grid data for video/visualization
                 t_grid=t_sol, x_grid=x_grid, y_grid=y_grid,
                 u_full=u_sol, v_full=v_sol,
                 viscosity=VISCOSITY)
        
        # Save config
        _save_yaml(CONFIG_PATH, cfg)
        print("Done.")

    # Execution
    u_h, v_h, t_h, x_g, y_g = solve_burgers()
    process_and_save_data(u_h, v_h, t_h, x_g, y_g)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', required=True)
    args = parser.parse_args()
    main(args)