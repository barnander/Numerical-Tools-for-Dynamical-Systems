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
ode = ODEs.Transcritical
x0 = -0.5
pend = np.array([0.6])
p0 = np.array([-1])
betas,x = solvers.natural_p_cont(ode,p0,pend,x0)
# %%
