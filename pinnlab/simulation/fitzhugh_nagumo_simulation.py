# run: scripts/simulation/fitzhugh_nagumo_rd.sh

import argparse
import os

import numpy as np
from scipy.interpolate import RegularGridInterpolator
from tqdm import tqdm
import yaml
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

    pde_cfg = cfg.get("pde", {}) or {}
    DU = float(pde_cfg.get("Du", 1.0))
    DV = float(pde_cfg.get("Dv", 1.0))
    A = float(pde_cfg.get("a", 0.7))
    B = float(pde_cfg.get("b", 0.8))
    EPS = float(pde_cfg.get("epsilon", 0.08))
    I_EXT = float(pde_cfg.get("I", 0.5))
    I_AMP = float(pde_cfg.get("I_amp", 0.0))
    I_PERIOD = float(pde_cfg.get("I_period", 0.0))

    DT = cfg["simulation_points"]["dt"]

    stability_cfg = cfg.get("stability", {}) or {}
    max_abs = stability_cfg.get("max_abs", 5.0)
    max_abs = float(max_abs) if max_abs is not None else None

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
        if seed is not None:
            np.random.seed(seed)

        kx = np.fft.fftfreq(X.shape[1]) * X.shape[1]
        ky = np.fft.fftfreq(X.shape[0]) * X.shape[0]
        KX, KY = np.meshgrid(kx, ky)

        noise = np.random.normal(0, 1, X.shape) + 1j * np.random.normal(0, 1, X.shape)

        K_mag = np.sqrt(KX**2 + KY**2)
        K_mag[0, 0] = 1.0
        scale = K_mag ** (-alpha / 2.0)
        scale[0, 0] = 0.0

        field_f = noise * scale
        field = np.real(np.fft.ifft2(field_f))

        max_val = np.max(np.abs(field))
        if max_val > 1e-8:
            field = field / max_val

        return field

    def init_conditions(X, Y):
        init_cfg = cfg.get("initial_condition", {}) or {}
        init_kind = init_cfg.get("kind", "grf")
        init_scale = float(init_cfg.get("scale", 0.2))
        init_noise = float(init_cfg.get("noise_scale", 0.0))
        grf_alpha = float(init_cfg.get("grf_alpha", 5.0))
        bump_sigma = float(init_cfg.get("bump_sigma", 0.8))
        bump_center = init_cfg.get("bump_center", None)

        if bump_center is None:
            bump_center = [(XA + XB) * 0.5, (YA + YB) * 0.5]
        cx0, cy0 = float(bump_center[0]), float(bump_center[1])

        if init_kind == "gaussian_bump":
            r2 = (X - cx0) ** 2 + (Y - cy0) ** 2
            bump = np.exp(-0.5 * r2 / (bump_sigma**2))
            u0 = init_scale * bump
            v0 = 0.5 * init_scale * bump
        elif init_kind == "sine":
            kx = float(init_cfg.get("kx", 1.0))
            ky = float(init_cfg.get("ky", 1.0))
            phase = float(init_cfg.get("phase", 0.0))
            Lx, Ly = DOMAIN_SIZE
            u0 = init_scale * np.sin(2.0 * np.pi * kx * (X - XA) / Lx + phase)
            v0 = init_scale * np.sin(2.0 * np.pi * ky * (Y - YA) / Ly + phase)
        elif init_kind == "grf":
            u0 = init_scale * generate_grf(X, Y, alpha=grf_alpha, seed=1)
            v0 = init_scale * generate_grf(X, Y, alpha=grf_alpha, seed=2)
        else:
            raise ValueError(f"Unknown initial condition kind: {init_kind}")

        if init_noise > 0.0:
            u0 = u0 + init_noise * np.random.normal(0.0, 1.0, size=u0.shape)
            v0 = v0 + init_noise * np.random.normal(0.0, 1.0, size=v0.shape)

        return u0.astype(np.float64), v0.astype(np.float64)

    def external_current(t):
        if I_AMP == 0.0:
            return I_EXT
        if I_PERIOD > 0.0:
            return I_EXT + I_AMP * np.sin(2.0 * np.pi * t / I_PERIOD)
        return I_EXT

    # --- 2. FitzHugh-Nagumo RD Solver ---
    def solve_fitzhugh_nagumo():
        x, y, X, Y = define_domain(nx, ny)
        x_col, y_col, _, _ = define_domain(cx, cy)
        dx = DOMAIN_SIZE[0] / (nx - 1)
        dy = DOMAIN_SIZE[1] / (ny - 1)

        cfl_u = DT * DU * (1.0 / dx**2 + 1.0 / dy**2)
        cfl_v = DT * DV * (1.0 / dx**2 + 1.0 / dy**2)
        print(f"Stability Check - CFL (u): {cfl_u:.5f}, CFL (v): {cfl_v:.5f}")
        if cfl_u > 0.5 or cfl_v > 0.5:
            print("[Warning] CFL is high for explicit diffusion; consider reducing dt.")

        print("Initializing state...")
        u, v = init_conditions(X, Y)

        u_hist, v_hist, t_hist = [], [], []

        print(f"1. Burn-in Phase ({N_STEPS_BURN} steps)...")
        print(f"2. Recording Phase ({N_STEPS_RECORD} steps)...")
        total_steps = N_STEPS_BURN + N_STEPS_RECORD

        for n in tqdm(range(total_steps)):
            t = n * DT
            I_t = external_current(t)
            if max_abs is not None:
                u = np.clip(u, -max_abs, max_abs)
                v = np.clip(v, -max_abs, max_abs)

            u_xp = np.roll(u, -1, axis=1)
            u_xm = np.roll(u, 1, axis=1)
            u_yp = np.roll(u, -1, axis=0)
            u_ym = np.roll(u, 1, axis=0)

            v_xp = np.roll(v, -1, axis=1)
            v_xm = np.roll(v, 1, axis=1)
            v_yp = np.roll(v, -1, axis=0)
            v_ym = np.roll(v, 1, axis=0)

            lap_u = (u_yp + u_ym - 2 * u) / dy**2 + (u_xp + u_xm - 2 * u) / dx**2
            lap_v = (v_yp + v_ym - 2 * v) / dy**2 + (v_xp + v_xm - 2 * v) / dx**2

            u_safe = np.clip(u, -max_abs, max_abs) if max_abs is not None else u
            v_safe = np.clip(v, -max_abs, max_abs) if max_abs is not None else v

            du_dt = DU * lap_u + u_safe - u_safe**3 - v_safe + I_t
            dv_dt = DV * lap_v + EPS * (u_safe - A * v_safe - B)

            u = u + DT * du_dt
            v = v + DT * dv_dt
            if not (np.isfinite(u).all() and np.isfinite(v).all()):
                raise FloatingPointError(
                    "NaN/Inf detected in simulation. Reduce dt or lower max_abs."
                )

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

        interp_u = RegularGridInterpolator((t_sol, y_grid, x_grid), u_sol, method="linear", bounds_error=False, fill_value=None)
        interp_v = RegularGridInterpolator((t_sol, y_grid, x_grid), v_sol, method="linear", bounds_error=False, fill_value=None)

        t_meas, x_meas, y_meas = [], [], []

        if measure_kind == "fixed_grid":
            print("Sampling Strategy: Fixed grid positions")
            sens_xn = measure_cfg.get("sensor_nx", 10)
            sens_yn = measure_cfg.get("sensor_ny", 10)
            xs_sensor = np.linspace(x_grid[0], x_grid[-1], sens_xn)
            ys_sensor = np.linspace(y_grid[0], y_grid[-1], sens_yn)
            X_s, Y_s = np.meshgrid(xs_sensor, ys_sensor)
            sensor_x, sensor_y = X_s.flatten(), Y_s.flatten()
        elif measure_kind == "fixed_random":
            print("Sampling Strategy: Fixed random spatial points")
            n_sensors = measure_cfg.get("n_sensors", 100)
            sensor_x = np.random.uniform(x_grid[0], x_grid[-1], n_sensors)
            sensor_y = np.random.uniform(y_grid[0], y_grid[-1], n_sensors)
        elif measure_kind == "random":
            print("Sampling Strategy: Fully random (spatiotemporal)")
            n_measure = cfg["n_measurement"]
            t_meas = np.random.uniform(t_sol[0], t_sol[-1], n_measure)
            x_meas = np.random.uniform(x_grid[0], x_grid[-1], n_measure)
            y_meas = np.random.uniform(y_grid[0], y_grid[-1], n_measure)
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

        query_points = np.stack([t_meas, y_meas, x_meas], axis=1)
        print(f"number of query points: {query_points.shape[0]}")
        u_meas = interp_u(query_points)
        v_meas = interp_v(query_points)

        T_mesh, Y_mesh, X_mesh = np.meshgrid(t_sol, y_col, x_col, indexing="ij")
        X_f = np.hstack((X_mesh.flatten()[:, None], Y_mesh.flatten()[:, None], T_mesh.flatten()[:, None]))
        X_u = np.stack([x_meas, y_meas, t_meas], axis=1)
        Y_u = np.stack([u_meas, v_meas], axis=1)

        print(f"Saving to {DATA_PATH}...")
        np.savez(
            DATA_PATH,
            X_f=X_f,
            X_u=X_u,
            Y_u=Y_u,
            t_grid=t_sol,
            x_grid=x_grid,
            y_grid=y_grid,
            u_full=u_sol,
            v_full=v_sol,
            Du=DU,
            Dv=DV,
            a=A,
            b=B,
            epsilon=EPS,
            I=I_EXT,
        )
        _save_yaml(CONFIG_PATH, cfg)
        print("Done.")

        return X_u

    # --- 4. Video Generation ---
    def save_visualizations(u_full, v_full, t_grid, x_grid, y_grid, X_u):
        print(f"Generating video to {VIDEO_PATH}...")

        mag = np.sqrt(u_full**2 + v_full**2)
        X, Y = np.meshgrid(x_grid, y_grid)

        meas_x, meas_y, meas_t = X_u[:, 0], X_u[:, 1], X_u[:, 2]
        dt_frame = t_grid[1] - t_grid[0] if len(t_grid) > 1 else 0.01

        fig, ax = plt.subplots(figsize=(6, 5))
        vmin, vmax = 0, np.max(mag)
        cax = ax.pcolormesh(X, Y, mag[0], shading="auto", cmap="magma", vmin=vmin, vmax=vmax)
        fig.colorbar(cax, ax=ax, label="Magnitude |U|")

        sensor_scat = ax.scatter([], [], c="cyan", s=15, edgecolors="white", linewidth=0.5, label="Sensors")
        ax.legend(loc="upper right")
        title = ax.set_title(f"FitzHugh-Nagumo RD t={t_grid[0]:.3f}")

        def update(frame_idx):
            t_current = t_grid[frame_idx]
            cax.set_array(mag[frame_idx].ravel())
            title.set_text(f"FitzHugh-Nagumo RD t={t_current:.3f}")

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

    u_h, v_h, t_h, x_g, y_g, x_c, y_c = solve_fitzhugh_nagumo()
    X_u_data = process_and_save_data(u_h, v_h, t_h, x_g, y_g, x_c, y_c)
    save_visualizations(u_h, v_h, t_h, x_g, y_g, X_u_data)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    args = parser.parse_args()
    main(args)
