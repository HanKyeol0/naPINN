import numpy as np
from tqdm import tqdm
from scipy.interpolate import RegularGridInterpolator
import os, yaml, argparse
import matplotlib.pyplot as plt
import matplotlib.animation as animation
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
    DOMAIN_SIZE = (XB - XA, YB - YA)
    nx = cfg["simulation_points"]["nx"]
    ny = cfg["simulation_points"]["ny"]
    
    # PDE Parameters
    beta = cfg["pde"]["beta"]
    d_u = cfg["pde"]["d_u"]
    d_v = cfg["pde"]["d_v"] # Usually d_u = d_v for lambda-omega

    DT = cfg["simulation_points"]["dt"]
    RECORD_TIME = cfg["domain"]["record_time"]
    N_STEPS = int(RECORD_TIME / DT)
    RECORD_EVERY = cfg["simulation_points"].get("every", 1)

    DIR_PATH = cfg["dir_path"]
    SIMULATION_TAG = cfg["simulation_tag"]
    os.makedirs(os.path.join(DIR_PATH, SIMULATION_TAG), exist_ok=True)
    DATA_PATH = os.path.join(DIR_PATH, SIMULATION_TAG, "data.npz")
    CONFIG_PATH = os.path.join(DIR_PATH, SIMULATION_TAG, "config.yaml")
    VIDEO_PATH_U = os.path.join(DIR_PATH, SIMULATION_TAG, "dynamics_u.mp4")
    VIDEO_PATH_V = os.path.join(DIR_PATH, SIMULATION_TAG, "dynamics_v.mp4")

    def define_domain(nx, ny):
        x = np.linspace(XA, XB, nx)
        y = np.linspace(YA, YB, ny)
        X, Y = np.meshgrid(x, y)
        return x, y, X, Y

    # --- 2. Solver (Spectral / Finite Difference) ---
    # Using Finite Difference for consistency with Burgers code structure
    def solve_lambda_omega():
        x, y, X, Y = define_domain(nx, ny)
        dx = DOMAIN_SIZE[0] / (nx - 1)
        dy = DOMAIN_SIZE[1] / (ny - 1)

        # Initial Condition: Spiral Wave
        # u = r(x,y) * cos(theta(x,y)), v = r(x,y) * sin(theta(x,y))
        # Usually initialized with simple gradients or random noise to form spirals
        print("Initializing Spiral Wave...")
        u = np.tanh(np.sqrt(X**2 + Y**2)) * np.cos(np.angle(X + 1j*Y) - np.sqrt(X**2 + Y**2))
        v = np.tanh(np.sqrt(X**2 + Y**2)) * np.sin(np.angle(X + 1j*Y) - np.sqrt(X**2 + Y**2))

        u_hist, v_hist, t_hist = [], [], []
        
        print(f"Running Simulation ({N_STEPS} steps)...")
        
        for n in tqdm(range(N_STEPS)):
            # Laplacians (Periodic or Neumann? Usually Periodic for spiral patterns on torus, 
            # or Zero-Flux for box. Let's assume Periodic for rich dynamics)
            lap_u = (np.roll(u, 1, axis=0) + np.roll(u, -1, axis=0) - 2*u) / dy**2 + \
                    (np.roll(u, 1, axis=1) + np.roll(u, -1, axis=1) - 2*u) / dx**2
            lap_v = (np.roll(v, 1, axis=0) + np.roll(v, -1, axis=0) - 2*v) / dy**2 + \
                    (np.roll(v, 1, axis=1) + np.roll(v, -1, axis=1) - 2*v) / dx**2

            # Reaction Terms
            r2 = u**2 + v**2
            lam = 1.0 - r2
            omega = -beta * r2
            
            # Update (Explicit Euler)
            # u_t = d_u*lap_u + lam*u - omega*v
            # v_t = d_v*lap_v + omega*u + lam*v
            
            u_new = u + DT * (d_u * lap_u + lam * u - omega * v)
            v_new = v + DT * (d_v * lap_v + omega * u + lam * v)
            
            u = u_new
            v = v_new

            if n % RECORD_EVERY == 0:
                u_hist.append(u.copy())
                v_hist.append(v.copy())
                t_hist.append(n * DT)

        return np.array(u_hist), np.array(v_hist), np.array(t_hist), x, y

    # --- 3. Sampling Logic ---
    def process_and_save_data(u_sol, v_sol, t_sol, x_grid, y_grid):
        print("Processing datasets for PINN...")
        measure_cfg = cfg.get("measurement", {})
        measure_kind = measure_cfg.get("measure_kind", "random")

        interp_u = RegularGridInterpolator((t_sol, y_grid, x_grid), u_sol, method='linear', bounds_error=False, fill_value=None)
        interp_v = RegularGridInterpolator((t_sol, y_grid, x_grid), v_sol, method='linear', bounds_error=False, fill_value=None)

        if measure_kind == "fixed_grid":
            sens_xn = measure_cfg.get("sensor_nx", 20)
            sens_yn = measure_cfg.get("sensor_ny", 20)
            xs = np.linspace(x_grid[0], x_grid[-1], sens_xn)
            ys = np.linspace(y_grid[0], y_grid[-1], sens_yn)
            X_s, Y_s = np.meshgrid(xs, ys)
            sensor_x, sensor_y = X_s.flatten(), Y_s.flatten()
        elif measure_kind == "fixed_random":
            n_sensors = measure_cfg.get("n_sensors", 100)
            sensor_x = np.random.uniform(x_grid[0], x_grid[-1], n_sensors)
            sensor_y = np.random.uniform(y_grid[0], y_grid[-1], n_sensors)
        else: # random
            N_MEAS = cfg["n_measurement"]
            t_meas = np.random.uniform(t_sol[0], t_sol[-1], N_MEAS)
            x_meas = np.random.uniform(x_grid[0], x_grid[-1], N_MEAS)
            y_meas = np.random.uniform(y_grid[0], y_grid[-1], N_MEAS)
            
        if measure_kind in ["fixed_grid", "fixed_random"]:
            all_t, all_x, all_y = [], [], []
            for t_val in t_sol:
                all_t.append(np.full_like(sensor_x, t_val))
                all_x.append(sensor_x)
                all_y.append(sensor_y)
            t_meas, x_meas, y_meas = np.concatenate(all_t), np.concatenate(all_x), np.concatenate(all_y)

        query_points = np.stack([t_meas, y_meas, x_meas], axis=1)
        u_meas = interp_u(query_points)
        v_meas = interp_v(query_points)

        # Collocation
        cx = cfg["collocation_points"]["nx"]
        cy = cfg["collocation_points"]["ny"]
        xc_grid = np.linspace(XA, XB, cx)
        yc_grid = np.linspace(YA, YB, cy)
        T_mesh, Y_mesh, X_mesh = np.meshgrid(t_sol, yc_grid, xc_grid, indexing='ij')
        X_f = np.hstack((X_mesh.flatten()[:, None], Y_mesh.flatten()[:, None], T_mesh.flatten()[:, None]))
        
        X_u = np.stack([x_meas, y_meas, t_meas], axis=1)
        Y_u = np.stack([u_meas, v_meas], axis=1)

        print(f"Saving to {DATA_PATH}...")
        np.savez(DATA_PATH, 
                 X_f=X_f, X_u=X_u, Y_u=Y_u, 
                 t_grid=t_sol, x_grid=x_grid, y_grid=y_grid, 
                 u_full=u_sol, v_full=v_sol, 
                 beta=beta, d_u=d_u, d_v=d_v)
        _save_yaml(CONFIG_PATH, cfg)
        return X_u

    # --- 4. Video ---
    def save_visualizations(u_full, v_full, t_grid, x_grid, y_grid, X_u):
        print(f"Generating video to {VIDEO_PATH_U} and {VIDEO_PATH_V}...")
        X, Y = np.meshgrid(x_grid, y_grid)
        meas_x, meas_y, meas_t = X_u[:, 0], X_u[:, 1], X_u[:, 2]
        dt_frame = t_grid[1] - t_grid[0] if len(t_grid) > 1 else 0.01

        fig_u, ax_u = plt.subplots(figsize=(6, 5))
        cax_u = ax_u.pcolormesh(X, Y, u_full[0], shading='auto', cmap='twilight', vmin=-1.5, vmax=1.5)
        fig_u.colorbar(cax_u, ax=ax_u, label='State u')
        sensor_scat_u = ax_u.scatter([], [], c='k', s=10, alpha=0.5, label='Sensors')
        title_u = ax_u.set_title(f"Lambda-Omega t={t_grid[0]:.3f}")
        
        fig_v, ax_v = plt.subplots(figsize=(6, 5))
        cax_v = ax_v.pcolormesh(X, Y, v_full[0], shading='auto', cmap='twilight', vmin=-1.5, vmax=1.5)
        fig_v.colorbar(cax_v, ax=ax_v, label='State v')
        sensor_scat_v = ax_v.scatter([], [], c='k', s=10, alpha=0.5, label='Sensors')
        title_v = ax_v.set_title(f"Lambda-Omega t={t_grid[0]:.3f}")

        def update_u(frame_idx):
            t_current = t_grid[frame_idx]
            cax_u.set_array(u_full[frame_idx].ravel())
            title_u.set_text(f"Lambda-Omega t={t_current:.3f}")
            mask = np.abs(meas_t - t_current) < (dt_frame / 2.0)
            if np.any(mask):
                sensor_scat_u.set_offsets(np.c_[meas_x[mask], meas_y[mask]])
            else:
                sensor_scat_u.set_offsets(np.empty((0, 2)))
            return cax_u, sensor_scat_u, title_u
        
        def update_v(frame_idx):
            t_current = t_grid[frame_idx]
            cax_v.set_array(v_full[frame_idx].ravel())
            title_v.set_text(f"Lambda-Omega t={t_current:.3f}")
            mask = np.abs(meas_t - t_current) < (dt_frame / 2.0)
            if np.any(mask):
                sensor_scat_v.set_offsets(np.c_[meas_x[mask], meas_y[mask]])
            else:
                sensor_scat_v.set_offsets(np.empty((0, 2)))
            return cax_v, sensor_scat_v, title_v

        ani_u = animation.FuncAnimation(fig_u, update_u, frames=len(t_grid), interval=50, blit=False)
        ani_v = animation.FuncAnimation(fig_v, update_v, frames=len(t_grid), interval=50, blit=False)
        if animation.writers.is_available("ffmpeg"):
            ani_u.save(VIDEO_PATH_U, writer="ffmpeg", fps=20)
            ani_v.save(VIDEO_PATH_V, writer="ffmpeg", fps=20)
        else:
            ani_u.save(os.path.splitext(VIDEO_PATH_U)[0]+".gif", writer="pillow", fps=20)
            ani_v.save(os.path.splitext(VIDEO_PATH_V)[0]+".gif", writer="pillow", fps=20)
        plt.close()

    u_h, v_h, t_h, x_g, y_g = solve_lambda_omega()
    X_u_data = process_and_save_data(u_h, v_h, t_h, x_g, y_g)
    if not args.skip_video:
        save_visualizations(u_h, v_h, t_h, x_g, y_g, X_u_data)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', required=True)
    parser.add_argument(
        "--skip-video",
        action="store_true",
        help="Generate the numerical dataset without rendering MP4/GIF files.",
    )
    args = parser.parse_args()
    main(args)
