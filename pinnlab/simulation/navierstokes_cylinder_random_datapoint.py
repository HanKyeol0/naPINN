from dolfin import *
from mshr import Rectangle, Circle, generate_mesh
import numpy as np
import os

# ----------------------------
# Parameters
# ----------------------------
Re          = 30.0
U_mean      = 0.3
H           = 0.41
D           = 0.1         # cylinder diameter
nu          = U_mean * D / Re   # from Re = U*D/nu

T_final   = 1.0
num_steps = 1000
dt        = T_final / num_steps    # 0.001

mu_value  = 0.001
rho_value = 1.0

save_every  = 10           # grid snapshot interval
Nx, Ny      = 128, 32     # grid resolution for visualization

N_data      = 2000       # number of random measurement points

out_file    = "cylinder_Re100_random_data.npz"

# ----------------------------
# Mesh and boundaries
# ----------------------------
channel  = Rectangle(Point(0.0, 0.0), Point(2.2, 0.41))
cylinder = Circle(Point(0.2, 0.2), 0.05)
domain   = channel - cylinder
mesh     = generate_mesh(domain, 64)  # increase for higher accuracy

class Inlet(SubDomain):
    def inside(self, x, on_boundary):
        return on_boundary and near(x[0], 0.0)

class Outlet(SubDomain):
    def inside(self, x, on_boundary):
        return on_boundary and near(x[0], 2.2)

class Walls(SubDomain):
    def inside(self, x, on_boundary):
        return on_boundary and (near(x[1], 0.0) or near(x[1], 0.41))

class Cylinder(SubDomain):
    def inside(self, x, on_boundary):
        return on_boundary and (x[0]-0.2)**2 + (x[1]-0.2)**2 < (0.05 + 1e-3)**2

inlet  = Inlet()
outlet = Outlet()
walls  = Walls()
cyl    = Cylinder()

boundaries = MeshFunction("size_t", mesh, mesh.topology().dim()-1, 0)
boundaries.set_all(0)
inlet.mark(boundaries, 1)
outlet.mark(boundaries, 2)
walls.mark(boundaries, 3)
cyl.mark(boundaries, 4)

ds = Measure("ds", domain=mesh, subdomain_data=boundaries)

# ----------------------------
# Function spaces
# ----------------------------
V = VectorFunctionSpace(mesh, "P", 2)  # velocity
Q = FunctionSpace(mesh, "P", 1)        # pressure

u_trial = TrialFunction(V)
v_test  = TestFunction(V)
p_trial = TrialFunction(Q)
q_test  = TestFunction(Q)

u_n = Function(V)   # velocity at previous step
u_  = Function(V)   # tentative velocity
p_n = Function(Q)   # pressure at previous step
p_  = Function(Q)   # pressure at new step

# Reasonable initial condition: zero velocity everywhere, zero pressure
# Or, I can choose a more physical initial condition (e.g. steady parabolic inflow)
u_n.assign(Constant((0.0, 0.0)))
u_.assign(Constant((0.0, 0.0)))
p_n.assign(Constant(0.0))
p_.assign(Constant(0.0))

# ----------------------------
# Boundary conditions
# ----------------------------
U_max = 1.5 * U_mean
inflow_profile = ("4.0*U_max*x[1]*(H - x[1])/(H*H)", "0.0")
u_inlet = Expression(inflow_profile, degree=2, U_max=U_max, H=H)

bc_inlet = DirichletBC(V, u_inlet, boundaries, 1)
bc_walls = DirichletBC(V, Constant((0.0, 0.0)), boundaries, 3)
bc_cyl   = DirichletBC(V, Constant((0.0, 0.0)), boundaries, 4)
bcs_u    = [bc_inlet, bc_walls, bc_cyl]

bc_p_out = DirichletBC(Q, Constant(0.0), boundaries, 2)
bcs_p    = [bc_p_out]

# ----------------------------
# Variational forms (IPCS)
# ----------------------------
dt_const = Constant(dt)
nu_const = Constant(nu)

rho = Constant(1.0)
mu  = Constant(0.001)   # dynamic viscosity ~ nu, gives Re ~ 30 with U_mean=0.3

k = Constant(dt)
n = FacetNormal(mesh)
f = Constant((0.0, 0.0))

def epsilon(u):
    return sym(nabla_grad(u))

def sigma(u, p):
    return 2*mu*epsilon(u) - p*Identity(len(u))

U_mid = 0.5 * (u_trial + u_n)

F1 = rho*dot((u_trial - u_n) / k, v_test)*dx \
     + inner(sigma(U_mid, p_n), epsilon(v_test))*dx \
     + dot(p_n*n, v_test)*ds \
     - dot(mu*nabla_grad(U_mid)*n, v_test)*ds \
     - dot(f, v_test)*dx
a1, L1 = lhs(F1), rhs(F1)

# Step 2: pressure correction
a2 = dot(nabla_grad(p_trial), nabla_grad(q_test))*dx
L2 = dot(nabla_grad(p_n), nabla_grad(q_test))*dx - (1.0/k)*div(u_)*q_test*dx

# Step 3: velocity correction
a3 = dot(u_trial, v_test)*dx
L3 = dot(u_, v_test)*dx - k*dot(nabla_grad(p_ - p_n), v_test)*dx

A1 = assemble(a1)
A2 = assemble(a2)
A3 = assemble(a3)
for bc in bcs_u:
    bc.apply(A1)
for bc in bcs_p:
    bc.apply(A2)

# ----------------------------
# Uniform grid for visualization / PDE residuals
# ----------------------------
x_vals = np.linspace(0.0, 2.2, Nx)
y_vals = np.linspace(0.0, 0.41, Ny)
Xg, Yg = np.meshgrid(x_vals, y_vals, indexing="xy")

num_snapshots = num_steps // save_every + 1
U_grid = np.zeros((num_snapshots, Ny, Nx), dtype=float)
V_grid = np.zeros_like(U_grid)
P_grid = np.zeros_like(U_grid)
T_grid = np.zeros(num_snapshots, dtype=float)

def sample_grid(u_fenics, p_fenics, snap_idx, t_val):
    for j in range(Ny):
        for i in range(Nx):
            pt = Point(float(Xg[j, i]), float(Yg[j, i]))
            # inside cylinder -> NaN
            if (pt[0]-0.2)**2 + (pt[1]-0.2)**2 <= (0.05 + 1e-6)**2:
                U_grid[snap_idx, j, i] = np.nan
                V_grid[snap_idx, j, i] = np.nan
                P_grid[snap_idx, j, i] = np.nan
            else:
                vel  = u_fenics(pt)
                pres = p_fenics(pt)
                U_grid[snap_idx, j, i] = vel[0]
                V_grid[snap_idx, j, i] = vel[1]
                P_grid[snap_idx, j, i] = pres
    T_grid[snap_idx] = t_val

# ----------------------------
# Random measurement points (for data loss)
# ----------------------------
rng = np.random.default_rng(0)
Lx = 2.2
R  = 0.05
cx, cy = 0.2, 0.2

def sample_point_in_domain():
    """Uniform in rectangle, reject points inside cylinder."""
    while True:
        x = rng.uniform(0.0, Lx)
        y = rng.uniform(0.0, H)
        if (x - cx)**2 + (y - cy)**2 > (R + 1e-6)**2:
            return x, y

x_data = np.empty(N_data, dtype=float)
y_data = np.empty(N_data, dtype=float)

# random time indices 0..num_steps (inclusive)
t_index = rng.integers(0, num_steps, size=N_data, endpoint=True)

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

# evaluate initial condition (t = 0)
for j in indices_per_t[0]:
    pt = Point(float(x_data[j]), float(y_data[j]))
    vel  = u_n(pt)
    pres = p_n(pt)
    u_data[j] = vel[0]
    v_data[j] = vel[1]
    p_data[j] = pres

# initial grid snapshot
sample_grid(u_n, p_n, 0, 0.0)
snap_idx = 1

# ----------------------------
# Time-stepping
# ----------------------------
t = 0.0
for n in range(1, num_steps+1):
    t += dt

    # Step 1
    b1 = assemble(L1)
    for bc in bcs_u:
        bc.apply(b1)
    solve(A1, u_.vector(), b1)

    # Step 2
    b2 = assemble(L2)
    for bc in bcs_p:
        bc.apply(b2)
    solve(A2, p_.vector(), b2)

    # Step 3
    b3 = assemble(L3)
    for bc in bcs_u:
        bc.apply(b3)
    solve(A3, u_n.vector(), b3)
    p_n.assign(p_)
    
    u_max = u_n.vector().norm("linf")
    p_max = p_n.vector().norm("linf")

    if not np.isfinite(u_max) or u_max > 1e3:
        print(f"*** Blow-up at step {n}, t={t:.4f}, u_max={u_max:e}")

    # Evaluate random data points that belong to this time index
    for j in indices_per_t[n]:
        pt = Point(float(x_data[j]), float(y_data[j]))
        vel  = u_n(pt)
        pres = p_n(pt)
        u_data[j] = vel[0]
        v_data[j] = vel[1]
        p_data[j] = pres

    # Save grid snapshot
    if n % save_every == 0:
        print(f"Sampling grid snapshot {snap_idx} at t = {t:.3f}")
        sample_grid(u_n, p_n, snap_idx, t)
        snap_idx += 1

print("Any NaN in u_data?", np.isnan(u_data).any())
print("Any NaN in v_data?", np.isnan(v_data).any())
print("Any NaN in p_data?", np.isnan(p_data).any())

# ----------------------------
# Save dataset
# ----------------------------
np.savez_compressed(
    out_file,
    # grid for PDE residuals / visualization
    x_grid=x_vals,
    y_grid=y_vals,
    t_grid=T_grid,
    u_grid=U_grid,
    v_grid=V_grid,
    p_grid=P_grid,
    # random data points for measurement loss
    x_data=x_data,
    y_data=y_data,
    t_data=t_data,
    u_data=u_data,
    v_data=v_data,
    p_data=p_data,
    # parameters
    Re=np.array([Re]),
    nu=np.array([nu]),
    U_mean=np.array([U_mean]),
)

print("Saved dataset to:", os.path.abspath(out_file))
