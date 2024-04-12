#%%Method of lines
import numpy as np
import solvers
from math import sin,pi 
import matplotlib.pyplot as plt
#%%
a = 0
b = 1
t0 = 0
t_f = 1
N = 100
D = 1
p = np.nan
bc_left = solvers.Boundary_Condition("Dirichlet",a,0)
bc_right = solvers.Boundary_Condition("Dirichlet",b,0)
f = lambda x,t: np.sin(pi * (x-a)/(b-a))
u,x,t = solvers.diffusion_solve(bc_left, bc_right, f,t0,t_f, np.nan, p, N)
#%%
def anal_u(x,t):
    x_vec = np.sin(pi * (x-a)/(b-a))
    x_array = np.tile(x_vec[:,None],(1,len(t)))
    t_vec = np.exp(-D*pi**2 * t/(b-a)**2)
    u = x_array*t_vec
    return u
u_anal = anal_u(x,t)
# %% plot in space domain for t = 0, 0.5, 1
plt.figure()
plt.subplot(3,1,1)
plt.plot(x,u[:,0],label = 'numerical')
plt.plot(x,u_anal[:,0],label = 'analytical',color = 'black', linestyle = '--')
plt.title('t = 0')
plt.legend()
plt.subplot(3,1,2)
plt.plot(x,u[:,int(N/2)],label = 'numerical')
plt.plot(x,u_anal[:,int(N/2)],label = 'analytical',color = 'black', linestyle = '--')
plt.title('t = 0.5')
plt.legend()
plt.subplot(3,1,3)
plt.plot(x,u[:,-1],label = 'numerical')
plt.plot(x,u_anal[:,-1],label = 'analytical',color = 'black', linestyle = '--')
plt.title('t = 1')
plt.legend()
plt.show()
# %% plot in time domain for x = 0, 0.5, 1
plt.figure()
plt.subplot(3,1,1)
plt.plot(t, u[0, :], label='numerical')
plt.plot(t, u_anal[0, :], label='analytical', color='black', linestyle='--')
plt.title('x = 0')
plt.legend()
plt.subplot(3, 1, 2)
plt.plot(t, u[int(N/2), :], label='numerical')
plt.plot(t, u_anal[int(N/2), :], label='analytical', color='black', linestyle='--')
plt.title('x = 0.5')
plt.legend()
plt.subplot(3, 1, 3)
plt.plot(t, u[-1, :], label='numerical')
plt.plot(t, u_anal[-1, :], label='analytical', color='black', linestyle='--')
plt.title('x = 1')
plt.legend()
plt.show()
# %%
