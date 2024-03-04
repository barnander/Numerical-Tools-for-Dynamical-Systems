#%% Packages
import numpy as np
import scipy.optimize as opt
#%% Functions
#one-step solvers wrapper
def solve_to(f_gen,p,x0,t0,t_f, delta_max,solver = 'RK4'):
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
    #choose one-step solver

    if solver == 'Euler':
        solve_step = euler_step
    elif solver == 'RK4':
        solve_step = rk4_step
    else:
        raise ValueError("Unsupported solver: {}".format(solver))
    
    total_time = t_f - t0

    #make function robust to too small time scales
    if total_time < 2*delta_max:
        print("The duration of the integration is shorther than delta max")
        return np.array([x0]),t0
    no_timesteps = int(np.ceil(total_time/delta_max))

    #initialise t and x arrays
    t = np.zeros(no_timesteps+1) #plus 1 so we include t0 and t_f
    t[0] = t0
    x = np.zeros((len(x0),no_timesteps+1))
    x[:,0] = x0

    x_n,t_n=x0,t0
    for i in range(1,no_timesteps+1):
        #iterate through functions, computing the next value for each state variable
        h = min(delta_max, t_f - t_n) #adapts the final h to give the solution at exactly t_final
        x_n,t_n = solve_step(f,x_n,t_n,h)
        x[:,i] = x_n
        t[i] = t_n
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
def euler_step(f,x_n,t_n,h):
    """
    Computes one step of numerical integration using Euler method

    Parameters:
        f (function): function determining system in terms of x and t
        ode_state (dict): contains current values of space variables (key = 'x_n') and time (key = 't_n')
        h (float): size of euler step
    Returns:
        ode_state (dict): updated state of variables after euler step
    """
    x_n_plus_1 = x_n + h*f(x_n,t_n)
    return x_n_plus_1,t_n+h

def rk4_step(f,x_n,t_n,h):
    """
    Computes one step of numerical integration using Runge Kutta 4th order method (RK4)

    Parameters:
        f (function): function determining system in terms of x and t
        ode_state (dict): contains current values of space variables (key = 'x_n') and time (key = 't_n')
        h (float): size of RK4 step
    Returns:
        ode_state (dict): updated state of variables after RK4 order step
    """
    k1 = f(x_n,t_n)
    k2 = f(x_n + h*k1/2,t_n + h/2)
    k3 = f(x_n + h*k2/2,t_n + h/2)
    k4 = f(x_n + h*k3,t_n + h)
    x_n_plus_1 = x_n + h/6*(k1+2*k2+2*k3+k4)
    return x_n_plus_1,t_n + h

# def finite_dif(f_x,f_x_plus_h,h):
#     dif = (f_x_plus_h-f_x)/h
#     return dif

# def newton_meth(f,dif,x_n):
#     x_n_plus_1 = x_n - f/dif
#     return x_n_plus_1



def shoot_solve(f_gen,p,init_guess, delta_max,solver = 'RK4',phase_cond=False):
    """
    Finds limit cycles (LC) using  numerical shooting (potentially generalise to all BVPs later)
    Parameters:
        f_gen (function): function of x (np array), t (float), and p (np array) describing system of odes
        p (np array): parameters of system of equations
        init_guess (np array): initial guess for a point on the LC [0:-1], and for the period T [-1]
        conds (function): definition of phase condition when LC=True, or definition of BCs and phase condition when LC = False
        delta_max (float): max step size
        solver (string): solver used in integrator
        LC (boolean): if True, solving for a limit cycle
    """
    if not phase_cond:
        def pc(x,t,p):
            return f_gen(x,0,p)[0]
    else:
        pc = phase_cond
    
    #define function to root solve using  newton solver for limit cycles
    def g(x_T0):
        """
        Parameters:
            x_T0 (np array): array of initial conditions and time 
        """
        x0,T = x_T0[:-1],x_T0[-1]
        x,_ = solve_to(f_gen,p,x0,0,T,delta_max,solver)
        xf = x[:,-1]
        BC = xf-x0
        PC = pc(x0,T,p)
        return np.append(BC,PC)

    
    #run scipy newton root-finder on func_solve
    x_T0_solved = opt.fsolve(g,init_guess,xtol=delta_max*1.1) #match rootfinder tol to integrator tol

    return x_T0_solved[:-1], x_T0_solved[-1]



def natural_p_cont(ode,p0,pend,x_T0,delta_max = 1e-3,n=25,Eqm_only=False):
    """
    Performs natural parameter continuation on system of ODEs

    
    """
    #check that p0 and pend are the same type, length and that only one parameter changes:
    assert type(pend) == type(p0), "pend and p0 must be the same type"
    if type(p0) == np.ndarray:
        assert len(p0) == len(pend), "p0 and pend should have the same length"
        assert ((pend-p0)==0).sum() == len(p0)-1, "natural_p_cont only supports variation in one parameter"
    elif type(p0) == float or type(p0) == int:
        p0,pend = np.array([p0]),np.array([pend])
    else :
        raise ValueError("p0 and pend must be np.ndarray, float or int type")
    
    ps = np.linspace(p0,pend,n).transpose()
    if Eqm_only:
        x0 = x_T0
        x = np.tile(np.nan,(np.size(x_T0),n))
        for i,p in enumerate([ps[:,i] for i in range(n)]):
            sol = opt.fsolve(lambda x: ode(x,np.nan,p),x0)
            x[:,i] = sol
            x0=sol
    else:
        x = np.tile(np.nan,(np.size(x_T0)-1,n))
        for i,p in enumerate([ps[:,i] for i in range(n)]):
            sol,T0 = shoot_solve(ode,p,x_T0,delta_max)
            x[:,i] = sol
            x_T0 = np.append(sol,T0)      
    
    return ps,x


def secant_meth(ode,v0,v1,pend, max_it = 1e4):
    vi_minus_1 = v0
    vi = v1
    for i in range(max_it):
        secant = vi - vi_minus_1
        v_guess = vi + secant
        




