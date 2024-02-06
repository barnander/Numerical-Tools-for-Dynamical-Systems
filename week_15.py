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
y0 = 0.1
t0 = 0
t_final = 500
initial_conds = {'t':t0,'x':np.array([x0,y0])} 
ode_params = np.array([a,b1,d])
delta_max = 1e-3

x,t = solvers.solve_to(LK_model,ode_params,initial_conds,t_final,delta_max,solver = 'RK4')



# %% plot results
#plot time series
plt.plot(t,x[0],t,x[1])

#%% 
#plot phase portrait
plt.plot(x[0],x[1])

# %% pointcarre map
#take cross-section at x = 0.3 with a tolerance of +- 1e-3 (same as max_delta)
y = np.array([])
delta_t = np.array([])
t_last = 0
tol = 1e-4
for i in range(len(t)):
    if (x[0,i]> 0.3-tol) and (x[0,i]<0.3+tol):
        y = np.append(y,x[1,i])
        delta_t = np.append(delta_t,t[i]-t_last)
        t_last = t[i]

        
plt.scatter(np.arange(0,len(delta_t)),delta_t)
plt.figure()
plt.scatter(np.arange(0,len(delta_t)),y)

#compute period





# %%
