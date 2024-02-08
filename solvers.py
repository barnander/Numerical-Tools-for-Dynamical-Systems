#%% Packages
import numpy as np
#%% Functions
#one-step solvers wrapper
def solve_to(f_gen,p,ICs,t_f, delta_max,solver = 'RK4'):
    """
    Applies one-step solvers to systems of odes from initial conditions to 
    to the specified endpoint (t_f) 

    Parameters:
        f_gen (function): function of x (np array), t (float), and p (np array) describing system of odes
        ICs (dict): initial conditions 't' key points to initial t, 'x' key points to an np array of initial values in the state space
        t_f (float): end time
        delta_max (float): max step size
        solver (string): solver used
    Returns:
        x (np array): 2D array where each row is a time series of the state variables over time period input
        t (np array): value of time at every computed x

    """
    f = wrap_f(f_gen,p)
    x0 = ICs['x']
    t0 = ICs['t']   
    ode_state = {'x_n':x0,'t_n':t0}
    #choose one-step solver

    if solver == 'Euler':
        solve_step = euler_step
    elif solver == 'RK4':
        solve_step = rk4_step
    else:
        raise ValueError("Unsupported solver: {}".format(solver))
    
    total_time = t_f - t0
    no_timesteps = int(np.ceil(total_time/delta_max))

    t = np.zeros(no_timesteps+1) #plus 1 so we include t0 and t_f
    t[0] = t0
    x = np.zeros((len(x0),no_timesteps+1))
    x[:,0] = x0
    for i in range(1,no_timesteps+1):
        #iterate through functions, computing the next value for each state variable
        h = min(delta_max, t_f - ode_state['t_n']) #adapts the final h to give the solution at exactly t_final
        ode_state = solve_step(f,ode_state,h)
        t[i] = ode_state['t_n']
        x[:,i] = ode_state['x_n']
    return x,t



def wrap_f(f_gen,p):
    """
    Hard encodes ode parameter constants to the function

    Parameters:
        f_gen (function): function determining system in terms of x, t, and p
        p (np.array): constant parameters of system used for numerical integration   
    Returns:
        f (function): function determining system in terms of x and t (p are hard encoded)
    """
    def f(x,t):
        return f_gen(x,t,p)
    return(f)

#one step solver functions
def euler_step(f,ode_state,h):
    """
    Computes one step of numerical integration using Euler method

    Parameters:
        f (function): function determining system in terms of x and t
        ode_state (dict): contains current values of space variables (key = 'x_n') and time (key = 't_n')
        h (float): size of euler step
    Returns:
        ode_state (dict): updated state of variables after euler step
    """
    x_n = ode_state['x_n']
    t_n = ode_state['t_n']
    x_n_plus_1 = x_n + h*f(x_n,t_n)
    ode_state['x_n'] = x_n_plus_1
    ode_state['t_n'] = ode_state['t_n'] + h
    return ode_state

def rk4_step(f,ode_state,h):
    """
    Computes one step of numerical integration using Runge Kutta 4th order method (RK4)

    Parameters:
        f (function): function determining system in terms of x and t
        ode_state (dict): contains current values of space variables (key = 'x_n') and time (key = 't_n')
        h (float): size of RK4 step
    Returns:
        ode_state (dict): updated state of variables after RK4 order step
    """
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

def finite_dif(f_x,f_x_plus_h,h):
    dif = (f_x_plus_h-f_x)/h
    return dif

def newton_meth(f,dif,x_n):
    x_n_plus_1 = x_n - f/dif
    return x_n_plus_1




def bvp_solve(f_gen,p,ICs,t_f, delta_max,BCs,solver = 'RK4',h=1e-7):
    f_x = np.ones(BCs.shape)+h
    while (np.abs(f_x)>h).all():
        x,t  = solve_to(f_gen,p,ICs,t_f, delta_max,solver)
        ICs_plus_h = ICs
        ICs_plus_h['x'] += h
        x_plus_h,_ = solve_to(f_gen,p,ICs_plus_h,t_f, delta_max,solver)
        f_x = x[:,-1]-BCs
        f_x_plus_h = x_plus_h[:,-1]- BCs
        dif = finite_dif(f_x,f_x_plus_h,h)
        ICs['x'] = newton_meth(f_x,dif,ICs['x'])
    return x,t

# %%
