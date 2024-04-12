#%%Method of lines
import numpy as np
import solvers
from math import pi 
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

#%% Example 1
# Variables that can't change for analytical solution
alpha = 0
beta = 0

#
m = int(2)
a = 0
b = 1
t0 = 0
t_f = 1
D = 0.1
p = np.nan

#
q = lambda u,x,t,p: np.zeros(len(x))
f = lambda x,t: np.sin(m*pi * (x-a)/(b-a))


N = 100
bc_left = solvers.Boundary_Condition("Dirichlet",a,alpha)
bc_right = solvers.Boundary_Condition("Dirichlet",b,beta)

#analytic derivative
du_dx = lambda x,t: np.exp(-m**2*D*pi**2 * t/(b-a)**2) * m*pi/(b-a) * np.cos(m*pi * (x-a)/(b-a))

u,x,t = solvers.diffusion_solve(bc_left, bc_right, f,t0,t_f,q , p, N,D = D)

def anal_u(x,t):
    x_vec = np.sin(m*pi * (x-a)/(b-a))
    x_array = np.tile(x_vec[:,None],(1,len(t)))
    t_vec = np.exp(-m**2*D*pi**2 * t/(b-a)**2)
    u = x_array*t_vec
    return u
u_anal = anal_u(x,t)



#%% Example 2
alpha = 0
beta = 0

#
m = int(2)
a = 0
b = 1
t0 = 0
t_f = 1
D = 0.1
p = np.nan
q = lambda u,x,t,p: -m**2*pi**2*D/(b-a)**2 * np.sin(m*pi * (x-a)/(b-a)) 
f = lambda x,t: np.zeros(len(x))

#analytic derivative
du_dx = lambda x,t: (np.exp(-m**2*D*pi**2 * t/(b-a)**2)-1) * m*pi/(b-a) * np.cos(m*pi * (x-a)/(b-a))

bc_left = solvers.Boundary_Condition("Neumann",a, lambda t: du_dx(a,t))
bc_right = solvers.Boundary_Condition("Dirichlet",b,0)

u,x,t = solvers.diffusion_solve(bc_left, bc_right, f,t0,t_f, q, p, N,D = D)

def anal_u(x,t):
    x_vec = np.sin(m*pi * (x-a)/(b-a))
    x_array = np.tile(x_vec[:,None],(1,len(t)))
    t_vec = np.exp(-m**2*D*pi**2 * t/(b-a)**2)-1
    u = x_array*t_vec
    return u
u_anal = anal_u(x,t)



# %% 3D plot of numerical solution
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
X, T = np.meshgrid(x, t)
ax.plot_surface(X, T, u.T, cmap='viridis')
ax.set_xlabel('x')
ax.set_ylabel('t')
ax.set_zlabel('u')
plt.show()
# %% 3D plot of analytical solution
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
X, T = np.meshgrid(x, t)
ax.plot_surface(X, T, u_anal.T, cmap='viridis')
ax.set_xlabel('x')
ax.set_ylabel('t')
ax.set_zlabel('u')
plt.show()
