# navierstokes_cylinder_simulation_2.py
import numpy as np
from tqdm import tqdm
from scipy.interpolate import RegularGridInterpolator
import os, yaml, argparse
import matplotlib.pyplot as plt
import matplotlib.animation as animation

# Configure matplotlib for headless environments if needed
import matplotlib
matplotlib.use('Agg')

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
    
    # Obstacles
    CYLINDER_CENTER = (cfg["cylinder"]["x"], cfg["cylinder"]["y"])
    CYLINDER_RADIUS = cfg["cylinder"]["r"]
    
    pde_cfg = cfg.get("pde", {}) or {}
    VISCOSITY = pde_cfg["nu"]
    DENSITY = pde_cfg["rho"]
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
    VIDEO_PATH = os.path.join(DIR_PATH, SIMULATION_TAG, "dynamics.mp4")

    def define_domain(nx, ny):
        x = np.linspace(0, DOMAIN_SIZE[0], nx)
        y = np.linspace(0, DOMAIN_SIZE[1], ny)
        X, Y = np.meshgrid(x, y)
        return x, y, X, Y

    def create_obstacle_mask(X, Y, obstacles):
        # Initialize mask with all False (0.0)
        combined_mask = np.zeros_like(X, dtype=float)
        
        for obs in obstacles:
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
            u_star[:, 0] = 2.0 * y * (1.0 - y); v_star[:, 0] = 0 
            u_star[:, -1] = u_star[:, -2]; v_star[:, -1] = v_star[:, -2]
            u_star[0, :] = 0; u_star[-1, :] = 0; v_star[0, :] = 0; v_star[-1, :] = 0
            u_star[obstacle_mask == 1] = 0; v_star[obstacle_mask == 1] = 0

            # Pressure Poisson
            div_u_star = (np.roll(u_star, -1, axis=1) - np.roll(u_star, 1, axis=1)) / (2*dx) + \
                        (np.roll(v_star, -1, axis=0) - np.roll(v_star, 1, axis=0)) / (2*dy)
            b = (DENSITY / DT) * div_u_star

            # Reduced iterations
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
                if (n - N_STEPS_BURN) % RECORD_EVERY == 0:
                    u_hist.append(u.copy())
                    v_hist.append(v.copy())
                    p_hist.append(p.copy())
                    t_hist.append((n - N_STEPS_BURN) * DT)

        return np.array(u_hist), np.array(v_hist), np.array(p_hist), np.array(t_hist), x, y

    # --- 3. Sampling Logic (UPDATED) ---
    def process_and_save_data(u_sol, v_sol, p_sol, t_sol, x_grid, y_grid):
        print("Processing datasets for PINN...")
        
        # Measurement Config
        measure_cfg = cfg.get("measurement", {})
        measure_kind = measure_cfg.get("measure_kind", "random") # random, fixed_grid, fixed_random
        without_boundary = measure_cfg.get("without_boundary", False)
        print(f"Measurement without boundary: {without_boundary}")
        
        obstacles = cfg.get("obstacles", [{"x": CYLINDER_CENTER[0], "y": CYLINDER_CENTER[1], "r": CYLINDER_RADIUS}])

        interp_u = RegularGridInterpolator((t_sol, y_grid, x_grid), u_sol, method='linear', bounds_error=False, fill_value=None)
        interp_v = RegularGridInterpolator((t_sol, y_grid, x_grid), v_sol, method='linear', bounds_error=False, fill_value=None)
        interp_p = RegularGridInterpolator((t_sol, y_grid, x_grid), p_sol, method='linear', bounds_error=False, fill_value=None)

        # A. Collocation Points (Grid Based)
        T_mesh, Y_mesh, X_mesh = np.meshgrid(t_sol, y_grid, x_grid, indexing='ij')
        X_col = X_mesh.flatten()[:, None]
        Y_col = Y_mesh.flatten()[:, None]
        T_col = T_mesh.flatten()[:, None]
        
        flat_mask = create_obstacle_mask(X_col.flatten(), Y_col.flatten(), obstacles)
        valid_mask = (flat_mask == 0)
        
        X_f = np.hstack((X_col[valid_mask], Y_col[valid_mask], T_col[valid_mask])) # [x, y, t]
        
        # B. Measurement Points
        t_meas, x_meas, y_meas = [], [], []
        
        # Helper to filter points inside cylinder
        def filter_sensors(xs, ys, obs_list):
            mask = create_obstacle_mask(xs, ys, obs_list)
            return xs[mask == 0], ys[mask == 0]
        
        dx_sim = x_grid[1] - x_grid[0]
        dy_sim = y_grid[1] - y_grid[0]
        
        x_min_inner = x_grid[0] + dx_sim * 5
        x_max_inner = x_grid[-1] - dx_sim * 5
        y_min_inner = y_grid[0] + dy_sim * 5
        y_max_inner = y_grid[-1] - dy_sim * 5

        if measure_kind == "fixed_grid":
            print("Sampling Strategy: Fixed Grid Positions")
            sens_xn = measure_cfg.get("sensor_nx", 20)
            sens_yn = measure_cfg.get("sensor_ny", 10)
            
            # Create grid
            if without_boundary:
                xs = np.linspace(x_min_inner, x_max_inner, sens_xn)
                ys = np.linspace(y_min_inner, y_max_inner, sens_yn)
            else:
                xs = np.linspace(x_grid[0], x_grid[-1], sens_xn)
                ys = np.linspace(y_grid[0], y_grid[-1], sens_yn)
            X_s, Y_s = np.meshgrid(xs, ys)
            sensor_x, sensor_y = X_s.flatten(), Y_s.flatten()
            
            # Remove sensors inside cylinder
            sensor_x, sensor_y = filter_sensors(sensor_x, sensor_y, obstacles)
            
        elif measure_kind == "fixed_random":
            print("Sampling Strategy: Fixed Random Spatial Points")
            n_sensors_target = measure_cfg.get("n_sensors", 200)
            
            # Oversample initially to account for rejection by cylinder
            oversample = int(n_sensors_target * 1.5)
            sensor_x = np.random.uniform(x_grid[0], x_grid[-1], oversample)
            sensor_y = np.random.uniform(y_grid[0], y_grid[-1], oversample)
            
            # Filter
            sensor_x, sensor_y = filter_sensors(sensor_x, sensor_y, obstacles)
            
            # Trim to desired count
            if len(sensor_x) > n_sensors_target:
                sensor_x = sensor_x[:n_sensors_target]
                sensor_y = sensor_y[:n_sensors_target]
                
        elif measure_kind == "random":
            print("Sampling Strategy: Fully Random (Spatiotemporal)")
            N_MEAS = cfg["n_measurement"]
            # Sample random points
            tr = np.random.uniform(t_sol[0], t_sol[-1], N_MEAS)
            xr = np.random.uniform(x_grid[0], x_grid[-1], N_MEAS)
            yr = np.random.uniform(y_grid[0], y_grid[-1], N_MEAS)
            
            # Filter points inside cylinder
            mask = create_obstacle_mask(xr, yr, obstacles)
            mask_out = (mask == 0)
            
            t_meas, x_meas, y_meas = tr[mask_out], xr[mask_out], yr[mask_out]
            sensor_x, sensor_y = None, None # No fixed sensors
            
        else:
            raise ValueError(f"Unknown measurement kind: {measure_kind}")

        # Expand fixed sensors across time
        if measure_kind in ["fixed_grid", "fixed_random"]:
            all_t, all_x, all_y = [], [], []
            for t_val in t_sol:
                all_t.append(np.full_like(sensor_x, t_val))
                all_x.append(sensor_x)
                all_y.append(sensor_y)
            t_meas = np.concatenate(all_t)
            x_meas = np.concatenate(all_x)
            y_meas = np.concatenate(all_y)

        # Query values
        query_points = np.stack([t_meas, y_meas, x_meas], axis=1)
        print(f"Number of measurement points: {query_points.shape[0]}")
        
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
        
        return X_u # Return inputs for visualization

    # --- 4. Visualization (GIF) ---
    def save_visualizations(u_full, v_full, t_grid, x_grid, y_grid, X_u):
        print(f"Generating video to {VIDEO_PATH}...")
        
        # Calculate magnitude
        mag = np.sqrt(u_full**2 + v_full**2)
        X, Y = np.meshgrid(x_grid, y_grid)
        
        # Measurement data: [x, y, t]
        meas_x, meas_y, meas_t = X_u[:, 0], X_u[:, 1], X_u[:, 2]
        
        # Setup Figure
        fig, ax = plt.subplots(figsize=(8, 4))
        vmin, vmax = 0, np.max(mag)
        
        # Plot initial frame
        cax = ax.pcolormesh(X, Y, mag[0], shading='auto', cmap='jet', vmin=vmin, vmax=vmax)
        fig.colorbar(cax, ax=ax, label='Velocity Magnitude |V|')
        
        # Add cylinder patch(es) for visualization context
        obstacles = cfg.get("obstacles", [{"x": CYLINDER_CENTER[0], "y": CYLINDER_CENTER[1], "r": CYLINDER_RADIUS}])
        for obs in obstacles:
            circle = plt.Circle((obs['x'], obs['y']), obs['r'], color='gray', zorder=10)
            ax.add_patch(circle)
            
        # Sensor scatter plot
        sensor_scat = ax.scatter([], [], c='black', s=5, label='Sensors', zorder=20)
        ax.legend(loc='upper right')
        
        title = ax.set_title(f"Navier-Stokes t={t_grid[0]:.3f}")
        ax.set_aspect('equal')
        ax.set_xlim(x_grid[0], x_grid[-1])
        ax.set_ylim(y_grid[0], y_grid[-1])

        # Animation update function
        dt_frame = t_grid[1] - t_grid[0] if len(t_grid) > 1 else 0.01
        
        def update(frame_idx):
            t_current = t_grid[frame_idx]
            
            # Update field
            cax.set_array(mag[frame_idx].ravel())
            title.set_text(f"Navier-Stokes t={t_current:.3f}")
            
            # Update sensors active at this time slice
            # We look for points within +/- half a timestep
            mask = np.abs(meas_t - t_current) < (dt_frame / 2.0)
            
            if np.any(mask):
                sensor_scat.set_offsets(np.c_[meas_x[mask], meas_y[mask]])
            else:
                sensor_scat.set_offsets(np.empty((0, 2)))
                
            return cax, sensor_scat, title

        # Create animation
        ani = animation.FuncAnimation(fig, update, frames=len(t_grid), interval=50, blit=False)

        # Save
        if animation.writers.is_available("ffmpeg"):
            ani.save(VIDEO_PATH, writer="ffmpeg", fps=20)
            print("Video saved (ffmpeg).")
        else:
            fallback_path = os.path.splitext(VIDEO_PATH)[0] + ".gif"
            print("MovieWriter ffmpeg unavailable; saving GIF instead.")
            ani.save(fallback_path, writer=animation.PillowWriter(fps=20))
            print(f"Video saved to {fallback_path}.")
        plt.close()

    u_h, v_h, p_h, t_h, x_g, y_g = solve_navier_stokes()
    X_u_data = process_and_save_data(u_h, v_h, p_h, t_h, x_g, y_g)
    save_visualizations(u_h, v_h, t_h, x_g, y_g, X_u_data)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', required=True)
    args = parser.parse_args()
    main(args)