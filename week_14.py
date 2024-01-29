# %% packages
import numpy as np
import matplotlib.pyplot as plt
import math
#%% ODE solvers
def solve_to(f,initial_conds, t_final, delta_max, solver = 'Euler'):
    x_1 = initial_conds['x']
    t_1 = initial_conds['t']   
    ode_state = {'x_n':x_1,'t_n':t_1}
    #uncomment if you want to track xs over time-period
    t = np.array([t_1])
    x = np.array([x_1])
    if solver == 'Euler':
        solve_step = euler_step
    elif solver == 'RK4':
        solve_step = rk4_step

    while ode_state['t_n'] < t_final:
        #t_n = t[-1]
        #x_n = x[-1]
        #ode_state = {'x_n':x_n,'t_n':t_n}
        ode_state = solve_step(f,ode_state,delta_max)
        #t = np.append(t,ode_state['t_n'])
        #x = np.append(x,ode_state['x_n'])
    return ode_state

def euler_step(f,ode_state,h):
    """
    ode_state: type: dictionary
    """
    x_n = ode_state['x_n']
    t_n = ode_state['t_n']
    x_n_plus_1 = x_n + h*f(t_n,x_n)
    return {'x_n': x_n_plus_1, 't_n' : t_n + h}

def rk4_step(f,ode_state,h):
    x_n = ode_state['x_n']
    t_n = ode_state['t_n']
    k1 = f(t_n,x_n)
    k2 = f(t_n + h/2,x_n + h*k1/2)
    k3 = f(t_n + h/2,x_n + h*k2/2)
    k4 = f(t_n + h, x_n + h*k3)
    x_n_plus_1 = x_n + h/6*(k1+2*k2+2*k3+k4)
    return {'x_n': x_n_plus_1, 't_n' : t_n + h}

#%% specific Qs
def f(t,x):
    return x


# define ode and parameters
x_1 = 1
t_1 = 0
h = 0.01
t_final = 1
initial_conds = {'x' : x_1, 't' : t_1}

answer = solve_to(f,initial_conds,t_final,h)
# %% plot error against step size
hs = np.logspace(-7,-1)
x_2 = np.array([])

for h in hs:
    solved = solve_to(f,initial_conds,t_final,h)
    x_2 = np.append(x_2, solved['x_n'])

#compute error (true solution is e)
error = np.abs(math.e - x_2)
# plt.loglog(hs,error)
# plt.xlabel('log(h)')
# plt.ylabel('log(error)')
# plt.grid(True)
# plt.show()
# %% Runge Kuta, Euler comparison
x_2_euler = np.array([])
x_2_rk4 = np.array([])
for h in hs:
    solved_euler = solve_to(f,initial_conds,t_final,h)
    solved_rk4 = solve_to(f,initial_conds,t_final,h,solver = 'RK4')
    x_2_rk4 = np.append(x_2_rk4, solved_rk4['x_n'])
#compute error (true solution is e)
eul = plt.loglog(hs,error)
rk4 = plt.loglog(hs,np.abs(math.e - x_2_rk4))
plt.xlabel('log(h)')
plt.ylabel('log(error)')
plt.legend(['Euler Method','Runge-Kutta'])
plt.grid(True)
plt.show()
