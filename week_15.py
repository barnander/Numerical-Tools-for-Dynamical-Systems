#%%Packages
import numpy as np
import solvers
import matplotlib.pyplot as plt
#%% define function
def LK_model(x,t,p):
    dx = x[0]*(1-x[0])-(p[0]*x[0]*x[1])/(p[2]+x[0])
    dy = p[1]*x[1]*(1-(x[1]/x[0]))
    return np.array([dx,dy])

#%% 
a = 1
d=0.1
# %%
b1=0.2
b2 = 0.3
x0 = 0.2
y0 = 0.5
t0 = 0
t_final = 100
initial_conds = {'t':t0,'x':np.array([x0,y0])} 
ode_params = np.array([a,b1,d])
delta_max = 1e-3

x,t = solvers.solve_to(LK_model,ode_params,initial_conds,t_final,delta_max,solver = 'RK4')


# %%
