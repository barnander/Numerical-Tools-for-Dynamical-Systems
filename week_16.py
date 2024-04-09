#%%week 16
import numpy as np
import matplotlib.pyplot as plt
from scipy import optimize as opt
from math import nan
import solvers
import ODEs
from ipywidgets import interact
#%%
# set up ODE

def Hopf_normal(x,t,p):
    u1,u2 = x
    beta = p[0]
    du1 = beta*u1 - u2 - u1 * (u1**2 + u2**2)
    du2 = u1 + beta*u2 - u2*(u1**2 + u2**2)
    return np.array([du1,du2])

#%% natural parameter continuation
#set up beta params
"""
p0,pend,n = 5,0,100 
x0 = np.array([2,2])
betas = np.linspace(p0,pend,n)
x = np.tile(np.nan,(np.size(x0),np.size(betas)))
for (i,beta) in enumerate(betas):
    sol = opt.fsolve(lambda x: Hopf_normal(x,np.nan,beta),x0)
    x[:,i] = sol
    x0 = sol
plt.plot(betas,x[0],betas,x[1])

# %%
x0 = np.array([2,2])
betas,x = solvers.natural_p_cont(Hopf_normal,p0,pend,x0)
plt.plot(betas,x[0],betas,x[1])
# %%
def fold_normal(x,t,r):
    return r + x^2
p0 = -1
pend = 1
x0 = 0.5
betas,x = solvers.natural_p_cont(fold_normal,p0,pend,x0)

# %%
ode = ODEs.Hopf_Super
x0 = np.array([0.7,0.7,1])
p0 = -1.
pend = 1.
betas,x = solvers.natural_p_cont(ode,p0,pend,x0)
plt.plot(betas[0],x[0])
# %%
ode(x0,np.nan,p0)
"""
# %%
def LK_model(x,t,p):
    dx = x[0]*(1-x[0])-(p[0]*x[0]*x[1])/(p[2]+x[0])
    dy = p[1]*x[1]*(1-(x[1]/x[0]))
    return np.array([dx,dy])
def Alg_Cubic(x,t,p):
    c = p[0]
    return x**3 - x + c
a = 1
d=0.1
bend=0.15
b0 =0.35

#p0 = np.array([a,b0,d])
#pend = np.array([a,bend,d])

p0 = np.array([3.])
pend = np.array([-1.])

x_T0 = np.array([0.2])
delta_max = 1e-3

#x,betas = solvers.natural_p_cont(Alg_Cubic,p0,pend,x_T0,delta_max,LC = False)
x,ps = solvers.pseudo_arc(ODEs.Pitchfork_Super,x_T0,p0,pend,0,innit_h=10**-6)

# %% plot distance from equilibrium against p
eqm =np.tile(0.27015621,(2,25))

#compute distance from eq to x
r = np.sqrt(((x-eqm)**2).sum(0))
plt.plot(betas[1][:],r)
# %% 3d plot of x,y on LC/eqm against beta
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
ax.scatter(x[0, :], x[1, :], betas[1,:])
plt.show()
# %% do full fixed point/limit cycle surface
n = 25
LC_points = x[:,0][None,:]
betas_augmented = np.array([betas[:,0]])
for i,x0 in enumerate([x[:,i] for i in range(n)]):
    sol,_ = solvers.solve_to(LK_model,betas[:,i],x0,0,25,1e-3)
    _,len_sol = sol.shape
    LC_points = np.concatenate((LC_points,np.transpose(sol)))
    betas_augmented = np.concatenate((betas_augmented,np.tile(betas[:,i],(len_sol,1))))
    print(i)
#%%
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
ax.scatter(LC_points[:, 0], LC_points[:, 1], betas_augmented[:,1])
plt.show()



# %% Surface plot
fig = plt.figure(figsize=(10, 7))
ax = fig.add_subplot(111, projection='3d')
surf = ax.plot_surface(LC_points[:, 0], LC_points[:, 1], betas_augmented[:,1], cmap='viridis', edgecolor='none')
ax.set_title('Surface Plot')
ax.set_xlabel('X axis')
ax.set_ylabel('Y axis')
ax.set_zlabel('Z axis')

# Show the plot
plt.colorbar(surf)
plt.show()
# %%
