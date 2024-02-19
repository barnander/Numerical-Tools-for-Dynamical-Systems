#%%week 16
import numpy as np
import matplotlib.pyplot as plt
from scipy import optimize as opt
from math import nan
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
p0,pend,n = 0,5,100 
x0 = np.array([2,2])
betas = np.linspace(pend,p0,n)
x = np.tile(np.nan,(np.size(x0),np.size(betas)))
for (i,beta) in enumerate(betas):
    sol = opt.fsolve(lambda x: Hopf_normal(x,np.nan,beta),x0)
    x[:,i] = sol
    x0 = sol
plt.plot(betas,x[0],betas,x[1])

# %%
