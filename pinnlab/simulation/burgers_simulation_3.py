# run: scripts/simulation/burgers_3.sh

import numpy as np
from tqdm import tqdm
from scipy.interpolate import RegularGridInterpolator
import os, yaml, argparse
import matplotlib.pyplot as plt
import matplotlib.animation as animation

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
    cx = cfg["collocation_points"]["nx"] + 1
    cy = cfg["collocation_points"]["ny"] + 1
    N_POINTS = (nx, ny)
    
    VISCOSITY = cfg["pde"]["nu"]
    DT = cfg["simulation_points"]["dt"]
    CT = cfg["collocation_points"]["dt"]

    BURN_IN_TIME = cfg["domain"]["burn_in_time"]
    RECORD_TIME = cfg["domain"]["record_time"]
    N_STEPS_BURN = int(BURN_IN_TIME / DT)
    N_STEPS_RECORD = int(RECORD_TIME / DT)
    RECORD_EVERY = cfg["simulation_points"].get("every", 1)

    # Output configuration
    DIR_PATH = cfg["dir_path"]
    SIMULATION_TAG = cfg["simulation_tag"]

    os.makedirs(os.path.join(DIR_PATH, SIMULATION_TAG), exist_ok=True)
    DATA_PATH = os.path.join(DIR_PATH, SIMULATION_TAG, "data.npz")
    CONFIG_PATH = os.path.join(DIR_PATH, SIMULATION_TAG, "config.yaml")
    VIDEO_PATH = os.path.join(DIR_PATH, SIMULATION_TAG, "dynamics.mp4")

    def define_domain(nx, ny):
        x = np.linspace(XA, XB, nx)
        y = np.linspace(YA, YB, ny)
        X, Y = np.meshgrid(x, y)
        return x, y, X, Y
    
    def generate_grf(X, Y, alpha=4.0, seed=None):
        if seed: np.random.seed(seed)
        
        # Wavenumbers
        kx = np.fft.fftfreq(X.shape[1]) * X.shape[1]
        ky = np.fft.fftfreq(X.shape[0]) * X.shape[0]
        KX, KY = np.meshgrid(kx, ky)
        
        # Random noise
        noise = np.random.normal(0, 1, X.shape) + 1j * np.random.normal(0, 1, X.shape)
        
        # Energy spectrum
        K_mag = np.sqrt(KX**2 + KY**2)
        K_mag[0,0] = 1.0
        scale = K_mag ** (-alpha / 2.0)
        scale[0,0] = 0.0 
        
        # Inverse FFT
        field_f = noise * scale
        field = np.real(np.fft.ifft2(field_f))
        
        # Normalize so max absolute value is exactly 1.0 (Prevents overflow)
        max_val = np.max(np.abs(field))
        if max_val > 1e-8:
            field = field / max_val
            
        return field

    # --- 2. 2D Burgers Solver ---
    def solve_burgers():
        x, y, X, Y = define_domain(nx, ny)
        x_col, y_col, X_col, Y_col = define_domain(cx, cy)
        dx = DOMAIN_SIZE[0] / (nx - 1)
        dy = DOMAIN_SIZE[1] / (ny - 1)

        # Check CFL Condition for stability
        cfl_num = (DT / dx**2) * VISCOSITY
        print(f"Stability Check - Viscous CFL: {cfl_num:.5f} (Should be < 0.5 for explicit Euler)")

        print("Initializing with Gaussian Random Field (Burgulence)...")
        # u and v initialized with different random seeds
        u = generate_grf(X, Y, alpha=5.0, seed=1) 
        v = generate_grf(X, Y, alpha=5.0, seed=2) 

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
            
            u = u_new
            v = v_new

            # Recording logic
            if n > N_STEPS_BURN:
                if (n - N_STEPS_BURN) % RECORD_EVERY == 0:
                    u_hist.append(u.copy())
                    v_hist.append(v.copy())
                    t_hist.append((n - N_STEPS_BURN) * DT)

        return np.array(u_hist), np.array(v_hist), np.array(t_hist), x, y, x_col, y_col

    # --- 3. Sampling Logic ---
    def process_and_save_data(u_sol, v_sol, t_sol, x_grid, y_grid, x_col, y_col):
        print("Processing datasets for PINN (Clean Data)...")
        
        measure_cfg = cfg.get("measurement", {})
        measure_kind = measure_cfg.get("measure_kind", "random")

        interp_u = RegularGridInterpolator((t_sol, y_grid, x_grid), u_sol, method='linear', bounds_error=False, fill_value=None)
        interp_v = RegularGridInterpolator((t_sol, y_grid, x_grid), v_sol, method='linear', bounds_error=False, fill_value=None)

        t_meas, x_meas, y_meas = [], [], []
        
        # ... [Sampling logic remains the same] ...
        if measure_kind == "fixed_grid":
            print("sampling strategy: fixed grid positions")
            sens_xn = measure_cfg.get("sensor_nx", 10)
            sens_yn = measure_cfg.get("sensor_ny", 10)
            xs_sensor = np.linspace(x_grid[0], x_grid[-1], sens_xn)
            ys_sensor = np.linspace(y_grid[0], y_grid[-1], sens_yn)
            X_s, Y_s = np.meshgrid(xs_sensor, ys_sensor)
            sensor_x, sensor_y = X_s.flatten(), Y_s.flatten()
        elif measure_kind == "fixed_random":
            print("Sampling Strategy: Fixed Random Spatial Points")
            n_sensors = measure_cfg.get("n_sensors", 100)
            sensor_x = np.random.uniform(x_grid[0], x_grid[-1], n_sensors)
            sensor_y = np.random.uniform(y_grid[0], y_grid[-1], n_sensors)
        elif measure_kind == "random":
            print("Sampling Strategy: Fully Random (Spatiotemporal)")
            N_MEASUREMENT = cfg["n_measurement"]
            t_meas = np.random.uniform(t_sol[0], t_sol[-1], N_MEASUREMENT)
            x_meas = np.random.uniform(x_grid[0], x_grid[-1], N_MEASUREMENT)
            y_meas = np.random.uniform(y_grid[0], y_grid[-1], N_MEASUREMENT)
            sensor_x, sensor_y = None, None
        else:
            raise ValueError(f"Unknown measurement kind: {measure_kind}")
        
        if measure_kind in ["fixed_grid", "fixed_random"]:
            all_t, all_x, all_y = [], [], []
            for t_val in t_sol:
                all_t.append(np.full_like(sensor_x, t_val))
                all_x.append(sensor_x)
                all_y.append(sensor_y)
            t_meas, x_meas, y_meas = np.concatenate(all_t), np.concatenate(all_x), np.concatenate(all_y)

        # Query and Save
        query_points = np.stack([t_meas, y_meas, x_meas], axis=1)
        print(f"number of query points: {query_points.shape[0]}")
        u_meas = interp_u(query_points)
        v_meas = interp_v(query_points)

        T_mesh, Y_mesh, X_mesh = np.meshgrid(t_sol, y_col, x_col, indexing='ij')
        X_f = np.hstack((X_mesh.flatten()[:, None], Y_mesh.flatten()[:, None], T_mesh.flatten()[:, None]))
        X_u = np.stack([x_meas, y_meas, t_meas], axis=1)
        Y_u = np.stack([u_meas, v_meas], axis=1)

        print(f"Saving to {DATA_PATH}...")
        np.savez(DATA_PATH, X_f=X_f, X_u=X_u, Y_u=Y_u, t_grid=t_sol, x_grid=x_grid, y_grid=y_grid, u_full=u_sol, v_full=v_sol, viscosity=VISCOSITY)
        _save_yaml(CONFIG_PATH, cfg)
        print("Done.")

        return X_u # Return X_u for visualization

    # --- 4. Video Generation ---
    def save_visualizations(u_full, v_full, t_grid, x_grid, y_grid, X_u):
        print(f"Generating video to {VIDEO_PATH}...")
        
        # Prepare data
        mag = np.sqrt(u_full**2 + v_full**2)
        X, Y = np.meshgrid(x_grid, y_grid)
        
        # Sensor data columns: [x, y, t]
        meas_x, meas_y, meas_t = X_u[:, 0], X_u[:, 1], X_u[:, 2]
        dt_frame = t_grid[1] - t_grid[0] if len(t_grid) > 1 else 0.01

        # Plot setup
        fig, ax = plt.subplots(figsize=(6, 5))
        vmin, vmax = 0, np.max(mag)
        cax = ax.pcolormesh(X, Y, mag[0], shading='auto', cmap='inferno', vmin=vmin, vmax=vmax)
        fig.colorbar(cax, ax=ax, label='Velocity Magnitude |V|')
        
        # Sensor points (cyan dots)
        sensor_scat = ax.scatter([], [], c='cyan', s=15, edgecolors='white', linewidth=0.5, label='Sensors')
        ax.legend(loc='upper right')
        title = ax.set_title(f"Burgulence t={t_grid[0]:.3f}")

        def update(frame_idx):
            t_current = t_grid[frame_idx]
            
            # Update background field
            cax.set_array(mag[frame_idx].ravel())
            title.set_text(f"Burgulence t={t_current:.3f}")
            
            # Update sensors: find measurements close to current time (within half dt)
            mask = np.abs(meas_t - t_current) < (dt_frame / 2.0)
            if np.any(mask):
                sensor_scat.set_offsets(np.c_[meas_x[mask], meas_y[mask]])
            else:
                sensor_scat.set_offsets(np.empty((0, 2)))
            return cax, sensor_scat, title

        ani = animation.FuncAnimation(fig, update, frames=len(t_grid), interval=50, blit=False)

        if animation.writers.is_available("ffmpeg"):
            ani.save(VIDEO_PATH, writer="ffmpeg", fps=20)
            print("Video saved.")
        else:
            fallback_path = os.path.splitext(VIDEO_PATH)[0] + ".gif"
            print("MovieWriter ffmpeg unavailable; saving GIF instead.")
            ani.save(fallback_path, writer=animation.PillowWriter(fps=20))
            print(f"Video saved to {fallback_path}.")
        plt.close()

    # Execution
    u_h, v_h, t_h, x_g, y_g, x_c, y_c = solve_burgers()
    X_u_data = process_and_save_data(u_h, v_h, t_h, x_g, y_g, x_c, y_c)
    save_visualizations(u_h, v_h, t_h, x_g, y_g, X_u_data)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', required=True)
    args = parser.parse_args()
    main(args)
