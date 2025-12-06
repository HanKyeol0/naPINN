from dolfin import *
import numpy as np
import os

# ----------------------------
# Parameters (must match sim)
# ----------------------------
mu_value  = 0.001
rho_value = 1.0

T = 5.0
num_steps = 5000
dt = T / num_steps

save_every = 50          # snapshot interval (adjust as you like)
Nx, Ny      = 128, 32    # grid resolution
N_data      = 2000       # random measurement points

out_file    = "cylinder_random_data.npz"

# ----------------------------
# Load mesh and spaces
# ----------------------------
mesh = Mesh("navier_stokes_cylinder/cylinder.xml.gz")

V = VectorFunctionSpace(mesh, "P", 2)
Q = FunctionSpace(mesh, "P", 1)

u = Function(V)
p = Function(Q)

# ----------------------------
# TimeSeries from simulation
# ----------------------------
ts_u = TimeSeries("navier_stokes_cylinder/velocity_series")
ts_p = TimeSeries("navier_stokes_cylinder/pressure_series")

# ----------------------------
# Uniform grid (your PINN grid)
# ----------------------------
x_vals = np.linspace(0.0, 2.2, Nx)
y_vals = np.linspace(0.0, 0.41, Ny)
Xg, Yg = np.meshgrid(x_vals, y_vals, indexing="xy")

num_snapshots = num_steps // save_every + 1
U_grid = np.zeros((num_snapshots, Ny, Nx), dtype=float)
V_grid = np.zeros_like(U_grid)
P_grid = np.zeros_like(U_grid)
T_grid = np.zeros(num_snapshots, dtype=float)

cx, cy, R = 0.2, 0.2, 0.05

def sample_grid(u_fenics, p_fenics, snap_idx, t_val):
    for j in range(Ny):
        for i in range(Nx):
            x = float(Xg[j, i])
            y = float(Yg[j, i])
            # inside cylinder -> NaN
            if (x - cx)**2 + (y - cy)**2 <= (R + 1e-6)**2:
                U_grid[snap_idx, j, i] = np.nan
                V_grid[snap_idx, j, i] = np.nan
                P_grid[snap_idx, j, i] = np.nan
            else:
                pt = Point(x, y)
                vel  = u_fenics(pt)
                pres = p_fenics(pt)
                U_grid[snap_idx, j, i] = vel[0]
                V_grid[snap_idx, j, i] = vel[1]
                P_grid[snap_idx, j, i] = pres
    T_grid[snap_idx] = t_val

# ----------------------------
# Random measurement points
# ----------------------------
rng = np.random.default_rng(0)
Lx = 2.2
H  = 0.41

def sample_point_in_domain():
    while True:
        x = rng.uniform(0.0, Lx)
        y = rng.uniform(0.0, H)
        if (x - cx)**2 + (y - cy)**2 > (R + 1e-6)**2:
            return x, y

x_data = np.empty(N_data, dtype=float)
y_data = np.empty(N_data, dtype=float)

# random time indices: 0..num_steps (inclusive)
t_index = rng.integers(0, num_steps+1, size=N_data)

indices_per_t = [[] for _ in range(num_steps+1)]
for j in range(N_data):
    xj, yj = sample_point_in_domain()
    x_data[j] = xj
    y_data[j] = yj
    k = int(t_index[j])
    indices_per_t[k].append(j)

u_data = np.empty(N_data, dtype=float)
v_data = np.empty(N_data, dtype=float)
p_data = np.empty(N_data, dtype=float)
t_data = t_index * dt

# ----------------------------
# Time loop: retrieve from TimeSeries and sample
# ----------------------------

# t = 0: IC is zero flow, zero pressure
for j in indices_per_t[0]:
    u_data[j] = 0.0
    v_data[j] = 0.0
    p_data[j] = 0.0

# also store grid snapshot at t=0 (all zeros outside cylinder)
sample_grid(lambda pt: (0.0, 0.0), lambda pt: 0.0, 0, 0.0)
snap_idx = 1

# loop over steps 1..num_steps
for n in range(1, num_steps+1):
    t = n * dt

    # retrieve fields from TimeSeries at time t
    ts_u.retrieve(u.vector(), t)
    ts_p.retrieve(p.vector(), t)

    # random data at this time
    for j in indices_per_t[n]:
        pt = Point(float(x_data[j]), float(y_data[j]))
        vel  = u(pt)
        pres = p(pt)
        u_data[j] = vel[0]
        v_data[j] = vel[1]
        p_data[j] = pres

    # grid snapshot every save_every
    if n % save_every == 0:
        print(f"Sampling grid snapshot {snap_idx} at t = {t:.3f}")
        sample_grid(u, p, snap_idx, t)
        snap_idx += 1

# ----------------------------
# Final sanity check
# ----------------------------
print("Any NaN in u_data?", np.isnan(u_data).any())
print("Any NaN in v_data?", np.isnan(v_data).any())
print("Any NaN in p_data?", np.isnan(p_data).any())

# ----------------------------
# Save dataset
# ----------------------------
np.savez_compressed(
    out_file,
    x_grid=x_vals,
    y_grid=y_vals,
    t_grid=T_grid,
    u_grid=U_grid,
    v_grid=V_grid,
    p_grid=P_grid,
    x_data=x_data,
    y_data=y_data,
    t_data=t_data,
    u_data=u_data,
    v_data=v_data,
    p_data=p_data,
    # parameters (for completeness)
    T=np.array([T]),
    num_steps=np.array([num_steps]),
    dt=np.array([dt]),
    mu=np.array([mu_value]),
    rho=np.array([rho_value]),
)

print("Saved dataset to:", os.path.abspath(out_file))
