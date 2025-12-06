from __future__ import print_function
from dolfin import *
from mshr import *
import numpy as np

# ----------------------------
# Parameters (exactly ft08)
# ----------------------------
T = 5.0            # final time
num_steps = 1000   # number of time steps
dt = T / num_steps # time step size
mu_val = 0.001     # dynamic viscosity
rho_val = 1.0      # density

# ----------------------------
# Mesh
# ----------------------------
channel = Rectangle(Point(0.0, 0.0), Point(2.2, 0.41))
cylinder = Circle(Point(0.2, 0.2), 0.05)
domain = channel - cylinder
mesh = generate_mesh(domain, 64)

# ----------------------------
# Function spaces
# ----------------------------
V = VectorFunctionSpace(mesh, "P", 2)
Q = FunctionSpace(mesh, "P", 1)

# ----------------------------
# Boundaries (string form, like tutorial)
# ----------------------------
inflow   = "near(x[0], 0.0)"
outflow  = "near(x[0], 2.2)"
walls    = "near(x[1], 0.0) || near(x[1], 0.41)"
cyl_expr = "on_boundary && x[0]>0.1 && x[0]<0.3 && x[1]>0.1 && x[1]<0.3"

# ----------------------------
# Inflow profile
# ----------------------------
U_mean = 0.3
U_max  = 1.5 * U_mean
inflow_profile = ("4.0*U_max*x[1]*(0.41 - x[1]) / pow(0.41, 2)", "0.0")

bcu_inflow   = DirichletBC(V, Expression(inflow_profile, degree=2, U_max=U_max), inflow)
bcu_walls    = DirichletBC(V, Constant((0.0, 0.0)), walls)
bcu_cylinder = DirichletBC(V, Constant((0.0, 0.0)), cyl_expr)
bcp_outflow  = DirichletBC(Q, Constant(0.0), outflow)

bcu = [bcu_inflow, bcu_walls, bcu_cylinder]
bcp = [bcp_outflow]

# ----------------------------
# Trial / test / solution functions
# ----------------------------
u = TrialFunction(V)
v = TestFunction(V)
p = TrialFunction(Q)
q = TestFunction(Q)

u_n = Function(V)   # velocity at previous step
u_  = Function(V)   # velocity at current step
p_n = Function(Q)   # pressure at previous step
p_  = Function(Q)   # pressure at current step

# ----------------------------
# Variational forms (original IPCS)
# ----------------------------
U  = 0.5 * (u_n + u)
n  = FacetNormal(mesh)
f  = Constant((0.0, 0.0))
k  = Constant(dt)
mu  = Constant(mu_val)
rho = Constant(rho_val)

def epsilon(u):
    return sym(nabla_grad(u))

def sigma(u, p):
    return 2*mu*epsilon(u) - p*Identity(len(u))

# Step 1: tentative velocity
F1 = rho*dot((u - u_n)/k, v)*dx \
   + rho*dot(dot(u_n, nabla_grad(u_n)), v)*dx \
   + inner(sigma(U, p_n), epsilon(v))*dx \
   + dot(p_n*n, v)*ds \
   - dot(mu*nabla_grad(U)*n, v)*ds \
   - dot(f, v)*dx
a1 = lhs(F1)
L1 = rhs(F1)

# Step 2: pressure correction
a2 = dot(nabla_grad(p), nabla_grad(q))*dx
L2 = dot(nabla_grad(p_n), nabla_grad(q))*dx - (1.0/k)*div(u_)*q*dx

# Step 3: velocity correction
a3 = dot(u, v)*dx
L3 = dot(u_, v)*dx - k*dot(nabla_grad(p_ - p_n), v)*dx

# Assemble matrices once (standard trick from tutorial)
A1 = assemble(a1)
A2 = assemble(a2)
A3 = assemble(a3)
for bc in bcu:
    bc.apply(A1)
for bc in bcp:
    bc.apply(A2)

# ----------------------------
# Output files (same as tutorial)
# ----------------------------
xdmffile_u = XDMFFile("navier_stokes_cylinder/velocity.xdmf")
xdmffile_p = XDMFFile("navier_stokes_cylinder/pressure.xdmf")

timeseries_u = TimeSeries("navier_stokes_cylinder/velocity_series")
timeseries_p = TimeSeries("navier_stokes_cylinder/pressure_series")

File("navier_stokes_cylinder/cylinder.xml.gz") << mesh

# ----------------------------
# Time-stepping
# ----------------------------
t = 0.0
for n_step in range(num_steps):
    t += dt

    # Step 1
    b1 = assemble(L1)
    for bc in bcu:
        bc.apply(b1)
    solve(A1, u_.vector(), b1, "bicgstab", "hypre_amg")

    # Step 2
    b2 = assemble(L2)
    for bc in bcp:
        bc.apply(b2)
    solve(A2, p_.vector(), b2, "bicgstab", "hypre_amg")

    # Step 3
    b3 = assemble(L3)
    solve(A3, u_.vector(), b3, "cg", "sor")

    # Save
    xdmffile_u.write(u_, t)
    xdmffile_p.write(p_, t)
    timeseries_u.store(u_.vector(), t)
    timeseries_p.store(p_.vector(), t)

    # Update previous solution
    u_n.assign(u_)
    p_n.assign(p_)

    # Simple monitor
    u_max = u_.vector().norm("linf")
    print(f"Step {n_step+1}/{num_steps}, t={t:.3f}, u_max={u_max:.4f}")

print("Simulation finished.")
