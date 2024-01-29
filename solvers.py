#%% Packages
import numpy as np
#%% Functions
#one-step solvers wrapper
def solve_to(f,initial_conds, t_final, delta_max, solver = 'Euler'):
    """
    Applies one-step solvers to systems of odes from initial conditions to 
    to the specified endpoint (t_final) 

    Args:
        fs (list): list of functions [f(t,x1,x2,..),g(t,x1,x2,...),...] defining system
        initial_conds (dict): 't' points to initial time, 'x' points to a list of initial values in the state space
        t_final (float): end time
        delta_max (float): max step size
        solver (string): solver used
    """
    x_0 = initial_conds['x']
    t_0 = initial_conds['t']   
    ode_state = {'x_n':x_0,'t_n':t_0}

    #choose one-step solver  
    if solver == 'Euler':
        solve_step = euler_step
    elif solver == 'RK4':
        solve_step = rk4_step

    time = np.arange(t_0,t_final+ delta_max/2,delta_max)
    x = x_0
    for t in time[:-1]:
        ode_state = solve_step(f,ode_state,delta_max)
        x = np.column_stack((x,ode_state['x_n']))

    return time,x
    
"""     no_timesteps = int((t_final-t_0)/delta_max)+ 1
    time = np.zeros(no_timesteps)
    time[0] = t_0
    x = np.zeros((len(x_0),no_timesteps))
    x[:,0] = x_0
    counter = 1 """
"""     #while ode_state['t_n'] < t_final:
        #iterate through functions, computing the next value for each state variable
        ode_state = solve_step(f,ode_state,delta_max)
        time[counter] = ode_state['t_n']
        x[:,counter] = ode_state['x_n']
        counter+=1
 """
 

#one step solver functions
def euler_step(f,ode_state,h):
    """
    ode_state: type: dictionary
    """
    x_n = ode_state['x_n']
    t_n = ode_state['t_n']
    x_n_plus_1 = x_n + h*f(t_n,x_n)
    ode_state['x_n'] = x_n_plus_1
    ode_state['t_n'] = ode_state['t_n'] + h
    return ode_state

def rk4_step(f,ode_state,h):
    x_n = ode_state['x_n']
    t_n = ode_state['t_n']
    k1 = f(t_n,x_n)
    k2 = f(t_n + h/2,x_n + h*k1/2)
    k3 = f(t_n + h/2,x_n + h*k2/2)
    k4 = f(t_n + h, x_n + h*k3)
    x_n_plus_1 = x_n + h/6*(k1+2*k2+2*k3+k4)
    ode_state['x_n'] = x_n_plus_1
    ode_state['t_n'] = ode_state['t_n'] + h
    return ode_state



#%%
import numpy as np
import matplotlib.pyplot as plt
import solvers

def f_second_order(t,x):
    x_dot = x[1]
    y_dot = -x[0]
    return np.array([x_dot,y_dot])


x0 = 0
y0 = 1
initial_conds = {'t':0,'x':np.array([x0,y0])}
t_final = 1
h = 0.1
time,x = solvers.solve_to(f_second_order,initial_conds,t_final,h)

plt.plot(time,x[1])