# cylinder_flow_fenics.py
#
# Time-dependent 2D incompressible Navier–Stokes:
#   u_t + (u·∇)u - ν Δu + ∇p = 0
#                 ∇·u        = 0
#
# Domain: 2.2 x 0.41 channel with a circular cylinder.
# Output: uniform grid samples (x, y, t, u, v, p) -> cylinder_Re100.npz

from dolfin import *
from mshr import Rectangle, Circle, generate_mesh
import numpy as np
import os

# ----------------------------
# Parameters
# ----------------------------
Re          = 100.0        # target Reynolds number
U_mean      = 0.3          # average inflow velocity
H           = 0.41         # channel height
nu          = U_mean * 0.1 / Re   # kinematic viscosity (via Re = U * D / nu, D=0.1)

T_final     = 8.0          # final time
num_steps   = 800          # time steps
dt          = T_final / num_steps

save_every  = 4            # sample solution every N time steps
Nx, Ny      = 256, 64      # uniform grid resolution for output

# Cylinder geometry (must match the mesh)
CYL_XC = 0.2
CYL_YC = 0.2
CYL_R  = 0.05

cyl_mask = np.zeros((Ny, Nx), dtype=bool)

out_file = "cylinder_Re100.npz"

# ----------------------------
# Mesh and boundaries
# ----------------------------
channel = Rectangle(Point(0.0, 0.0), Point(2.2, 0.41))
cylinder = Circle(Point(0.2, 0.2), 0.05)
domain = channel - cylinder
mesh = generate_mesh(domain, 96)   # increase for higher accuracy

# Boundary markers
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

inlet   = Inlet()
outlet  = Outlet()
walls   = Walls()
cyl     = Cylinder()

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

u_trial  = TrialFunction(V)
v_test   = TestFunction(V)
p_trial  = TrialFunction(Q)
q_test   = TestFunction(Q)

u_n = Function(V)   # velocity at previous step
u_  = Function(V)   # tentative velocity
p_n = Function(Q)   # pressure at previous step
p_  = Function(Q)   # pressure at new step

# ----------------------------
# Boundary conditions
# ----------------------------
# Parabolic inflow profile (max ~ 1.5*U_mean)
U_max = 1.5 * U_mean
inflow_profile = ("4.0*U_max*x[1]*(H - x[1])/(H*H)", "0.0")

# Pass parameters as keywords to Expression
u_inlet = Expression(inflow_profile, degree=2, U_max=U_max, H=H)

bc_inlet = DirichletBC(V, u_inlet, boundaries, 1)
bc_walls = DirichletBC(V, Constant((0.0, 0.0)), boundaries, 3)
bc_cyl   = DirichletBC(V, Constant((0.0, 0.0)), boundaries, 4)
bcs_u    = [bc_inlet, bc_walls, bc_cyl]

bc_p_out = DirichletBC(Q, Constant(0.0), boundaries, 2)
bcs_p    = [bc_p_out]

# ----------------------------
# Variational forms (IPCS scheme)
# ----------------------------
dt_const = Constant(dt)
nu_const = Constant(nu)

# Step 1: tentative velocity
U_mid = 0.5*(u_trial + u_n)
F1 = (1.0/dt_const)*inner(u_trial - u_n, v_test)*dx \
     + inner(dot(u_n, nabla_grad(U_mid)), v_test)*dx \
     + nu_const*inner(grad(U_mid), grad(v_test))*dx
a1, L1 = lhs(F1), rhs(F1)

# Step 2: pressure correction
F2 = inner(grad(p_trial), grad(q_test))*dx - (1.0/dt_const)*div(u_)*q_test*dx
a2, L2 = lhs(F2), rhs(F2)

# Step 3: velocity update
F3 = inner(u_trial, v_test)*dx - inner(u_, v_test)*dx \
     + dt_const*inner(grad(p_ - p_n), v_test)*dx
a3, L3 = lhs(F3), rhs(F3)

A1 = assemble(a1)
A2 = assemble(a2)
A3 = assemble(a3)

for bc in bcs_u:
    bc.apply(A1)
for bc in bcs_p:
    bc.apply(A2)

# ----------------------------
# Uniform grid for sampling
# ----------------------------
x_vals = np.linspace(0.0, 2.2, Nx)
y_vals = np.linspace(0.0, 0.41, Ny)
Xg, Yg = np.meshgrid(x_vals, y_vals, indexing="xy")

num_snapshots = num_steps // save_every + 1
U = np.zeros((num_snapshots, Ny, Nx), dtype=float)
V = np.zeros_like(U)
P = np.zeros_like(U)
T = np.zeros(num_snapshots, dtype=float)

# Helper to sample FE fields on uniform grid
def sample_field(u_fenics, p_fenics, t_idx, t_val):
    """
    Sample FE fields on the uniform grid.
    Points inside the cylinder are marked as NaN and flagged in cyl_mask.
    """
    for j in range(Ny):
        for i in range(Nx):
            x_ = float(Xg[j, i])
            y_ = float(Yg[j, i])

            # Check if inside solid cylinder
            if (x_ - CYL_XC)**2 + (y_ - CYL_YC)**2 <= CYL_R**2:
                cyl_mask[j, i] = True
                U[t_idx, j, i] = np.nan
                V[t_idx, j, i] = np.nan
                P[t_idx, j, i] = np.nan
            else:
                pt = Point(x_, y_)
                vel = u_fenics(pt)
                pres = p_fenics(pt)
                U[t_idx, j, i] = vel[0]
                V[t_idx, j, i] = vel[1]
                P[t_idx, j, i] = pres

    T[t_idx] = t_val


# store initial condition (t = 0)
sample_field(u_n, p_n, 0, 0.0)
snap_idx = 1

# ----------------------------
# Time-stepping loop
# ----------------------------
t = 0.0
for n in range(1, num_steps + 1):
    t += dt

    # Step 1: tentative velocity
    b1 = assemble(L1)
    for bc in bcs_u:
        bc.apply(b1)
    solve(A1, u_.vector(), b1)

    # Step 2: pressure
    b2 = assemble(L2)
    for bc in bcs_p:
        bc.apply(b2)
    solve(A2, p_.vector(), b2)

    # Step 3: velocity update
    b3 = assemble(L3)
    for bc in bcs_u:
        bc.apply(b3)
    solve(A3, u_n.vector(), b3)

    # Update pressure
    p_n.assign(p_)

    # Save snapshot
    if n % save_every == 0:
        print(f"Sampling snapshot {snap_idx} at t = {t:.3f}")
        sample_field(u_n, p_n, snap_idx, t)
        snap_idx += 1

# ----------------------------
# Save dataset
# ----------------------------
np.savez_compressed(
    out_file,
    x=x_vals,
    y=y_vals,
    t=T,
    u=U,
    v=V,
    p=P,
    obstacle_mask=cyl_mask,
    Re=np.array([Re]),
    nu=np.array([nu]),
    U_mean=np.array([U_mean]),
)
print(f"Saved dataset to {os.path.abspath(out_file)}")
