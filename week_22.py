#%%
import numpy as np
import solvers
from math import pi
#%%
alpha = 0
beta = 0
N = 1000
#
m = int(2)
a = 0
b = 1
t0 = 0
t_f = 2
D = 0.1
p = np.nan
q = lambda u,x,t,p: -m**2*pi**2*D/(b-a)**2 * np.sin(m*pi * (x-a)/(b-a)) 
f = lambda x,t: np.zeros(len(x))

#analytic derivative
du_dx = lambda x,t: (np.exp(-m**2*D*pi**2 * t/(b-a)**2)-1) * m*pi/(b-a) * np.cos(m*pi * (x-a)/(b-a))

bc_left = solvers.Boundary_Condition("Neumann",a, lambda t: du_dx(a,t))
bc_right = solvers.Boundary_Condition("Dirichlet",b,0)

u,x,t = solvers.diffusion_solve(bc_left, bc_right, f,t0,t_f, q, p, N,D = D,explicit_solver=False, implicit_solver = 'thomas',dt = 0.01)

def anal_u(x,t):
    x_vec = np.sin(m*pi * (x-a)/(b-a))
    x_array = np.tile(x_vec[:,None],(1,len(t)))
    t_vec = np.exp(-m**2*D*pi**2 * t/(b-a)**2)-1
    u = x_array*t_vec
    return u
u_anal = anal_u(x,t)

# %%
