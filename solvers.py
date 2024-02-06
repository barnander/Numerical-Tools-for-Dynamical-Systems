#%% Packages
import numpy as np
#%% Functions
#one-step solvers wrapper
def solve_to(f_gen,ode_params,initial_conds, t_final, delta_max, solver = 'Euler'):
    """
    Applies one-step solvers to systems of odes from initial conditions to 
    to the specified endpoint (t_final) 

    Args:
        f_gen (function): function describing system of odes with state variable and constant parameter input
        initial_conds (dict): 't' points to initial time, 'x' points to an np array of initial values in the state space
        t_final (float): end time
        delta_max (float): max step size
        solver (string): solver used
    """

    f = wrap_f(f_gen,ode_params)
    x_0 = initial_conds['x']
    t_0 = initial_conds['t']   
    ode_state = {'x_n':x_0,'t_n':t_0}
    #choose one-step solver  
    if solver == 'Euler':
        solve_step = euler_step
    elif solver == 'RK4':
        solve_step = rk4_step
    else:
        raise ValueError("Unsupported solver: {}".format(solver))
    
    total_time = t_final - t_0
    no_timesteps = int(np.ceil(total_time/delta_max))

    time = np.zeros(no_timesteps+1) #plus 1 so we include t0 and t_final
    time[0] = t_0
    x = np.zeros((len(x_0),no_timesteps+1))
    x[:,0] = x_0
    for i in range(1,no_timesteps+1):
        #iterate through functions, computing the next value for each state variable
        h = min(delta_max, t_final - ode_state['t_n']) #adapts the final h to give the solution at exactly t_final
        ode_state = solve_step(f,ode_state,h)
        time[i] = ode_state['t_n']
        x[:,i] = ode_state['x_n']
    return x,time
 

#hard encoding parameters to function
def wrap_f(f_gen,ode_params):
    def f(x,t):
        return f_gen(x,t,ode_params)
    return(f)

#one step solver functions
def euler_step(f,ode_state,h):
    """
    ode_state: type: dictionary
    """
    x_n = ode_state['x_n']
    t_n = ode_state['t_n']
    x_n_plus_1 = x_n + h*f(x_n,t_n)
    ode_state['x_n'] = x_n_plus_1
    ode_state['t_n'] = ode_state['t_n'] + h
    return ode_state

def rk4_step(f,ode_state,h):
    x_n = ode_state['x_n']
    t_n = ode_state['t_n']
    k1 = f(x_n,t_n)
    k2 = f(x_n + h*k1/2,t_n + h/2)
    k3 = f(x_n + h*k2/2,t_n + h/2)
    k4 = f(x_n + h*k3,t_n + h)
    x_n_plus_1 = x_n + h/6*(k1+2*k2+2*k3+k4)
    ode_state['x_n'] = x_n_plus_1
    ode_state['t_n'] = ode_state['t_n'] + h
    return ode_state




