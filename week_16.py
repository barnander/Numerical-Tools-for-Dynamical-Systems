#%%week 16
import numpy as np
import matplotlib.pyplot as plt
from scipy import optimize as opt
from math import nan
import solvers
import ODEs
#%%
# set up ODE

def Hopf_normal(x,t,p):
    u1,u2 = x
    beta = p
    du1 = beta*u1 - u2 - u1 * (u1**2 + u2**2)
    du2 = u1 + beta*u2 - u2*(u1**2 + u2**2)
    return np.array([du1,du2])

#%% natural parameter continuation
#set up beta params
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

# %%
def LK_model(x,t,p):
    dx = x[0]*(1-x[0])-(p[0]*x[0]*x[1])/(p[2]+x[0])
    dy = p[1]*x[1]*(1-(x[1]/x[0]))
    return np.array([dx,dy])
a = 1
d=0.1
b0=0.15
bend =0.35

p0 = np.array([a,b0,d])
pend = np.array([a,bend,d])
x_T0 = np.array([0.3,0.25,20])
delta_max = 1e-3

betas,x = solvers.natural_p_cont(LK_model,p0,pend,x_T0,delta_max)
# %% plot distance from equilibrium against p
eqm =np.tile(0.27015621,(2,25))

#compute distance from eq to x
r = np.sqrt(((x-eqm)**2).sum(0))
plt.plot(betas[1][:],r)
# %% 3d plot of x,y on LC/eqm against beta
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')

ax.scatter(x[0, :], x[1, :], betas[1,:])
# %% do full fixed point/limit cycle surface
n = 25
LC_points = np.array([])
for i,x0 in enumerate([x[:,i] for i in range(n)]):
    sol,_ = solvers.solve_to(LK_model,betas[:,i],x0,0,25,1e-3)
    LC_points = np.append(LC_points,np.array([sol]))
    print(i)
ax.scatter(LC_points[0, :], LC_points[1, :], betas[1,:])


# %%
