#%%
import numpy as np
import solvers
import matplotlib.pyplot as plt
import timeit
import cProfile
from math import pi
#%%
a,b = 0,1
alpha,beta = 0,0
D = 0.1
N = 1000
dt = 0.001
t0 = 0
t_f = 2
bc_left = solvers.Boundary_Condition("Dirichlet",a,alpha)
bc_right = solvers.Boundary_Condition("Dirichlet",b,beta)
grid = solvers.Grid(N,bc_left.x,bc_right.x)
dx = grid.dx
f = lambda x,t: np.sin(pi*x)
q = lambda u,x,t,p: np.zeros(len(x))
p = np.nan

#%%
pr = cProfile.Profile()
pr.enable()
u,x,t = solvers.diffusion_solve(bc_left, bc_right, f,t0,t_f,q , p, N,D = D, explicit_solver = False,implicit_solver='thomas', dt = dt)
pr.disable()
pr.print_stats(sort='time')
# %%
