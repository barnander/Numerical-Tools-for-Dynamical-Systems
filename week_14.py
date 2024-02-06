#%%


# %% Import packages a solvers module
import time
import numpy as np
import math
import matplotlib.pyplot as plt
import solvers
#%% define ode and parameters
def f(t,x):
    return np.array([x[0]])

x_1 = np.array([1])
t_1 = 0
h = 0.01
t_final = 1
initial_conds = {'x' : x_1, 't' : t_1}

t,x = solvers.solve_to(f,initial_conds,t_final,h)

# %% plot error against step size & time methods
hs = np.logspace(-5,-1)
x_2 = np.array([])
time_euler = np.array([])

for h in hs:
    start = time.time()
    t,x_solved = solvers.solve_to(f,initial_conds,t_final,h,solver = 'Euler')
    end = time.time()
    x_2 = np.append(x_2, x_solved[0,-1])
    time_euler = np.append(time_euler,end-start)
#compute error (true solution is e)
error = np.abs(math.e - x_2)
plt.loglog(hs,error)
plt.xlabel('log(h)')
plt.ylabel('log(error)')
plt.grid(True)
plt.show()
# %% Runge Kuta, Euler comparison
x_2_rk4 = np.array([])
time_rk4 = np.array([])

for h in hs:
    start = time.time()
    t,x_rk4 = solvers.solve_to(f,initial_conds,t_final,h,solver = 'RK4')
    end = time.time()

    x_2_rk4 = np.append(x_2_rk4, x_rk4[0,-1])
    time_rk4 = np.append(time_rk4,end-start)

#compute error (true solution is e)
eul = plt.loglog(hs,error)
rk4 = plt.loglog(hs,np.abs(math.e - x_2_rk4))
plt.xlabel('h')
plt.ylabel('Error')
plt.legend(['Euler Method','Runge-Kutta'])
plt.grid(True)
plt.show()


#%% plot computation time
plt.figure()
eul = plt.loglog(hs,time_euler)
rk4 = plt.loglog(hs,time_rk4)
plt.legend(['Euler Method','Runge-Kutta'])
plt.xlabel('h')
plt.ylabel('Computational Time')
plt.show()

# %%
