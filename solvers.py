#%% Packages
import numpy as np
import scipy.optimize as opt
#%% Functions
#one-step solvers wrapper
def solve_to(f_gen,p,x0,t0,t_f, delta_max,solver = 'RK4'):
    """
    Applies one-step solvers to systems of odes from initial conditions to 
    to the specified endpoint.
    Parameters:
        f_gen (function): function of x (np array), t (float), and p (np array) describing system of odes
        p (np array, float or int): parameters of system of equations
        x0 (np array): initial conditions for the system
        t0 (float): initial time
        t_f (float): final time
        delta_max (float): max step size
        solver (string): solver used
    Returns:
        x (np array): 2D array where each row is a time series of the state variables over time period input
        t (np array): value of time at every computed x

    """
    p,_ = param_assert(p)

    f = lambda x,t: f_gen(x,t,p)
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



def param_assert(p0, pend=None):
    """
    Asserts that the parameters are the same type as each other and are either floats, integers, or numpy arrays.
    Converts single float or int to a one-element numpy array for consistency.
    Parameters:
        p0 (float, int, or np.array): Initial parameter(s) of the system of equations.
        pend (float, int, or np.array, optional): End parameter(s) of the system of equations. Defaults to None.
    Returns:
        np.array: A numpy array of p0.
        np.array: A numpy array of pend.
    """
    # Check if p0 is a float or an int, and if so, convert to a numpy array with a single element
    if isinstance(p0, (float, int)):
        p0 = np.array([p0])
        pend = np.array([pend]) if pend is not None else None
    elif isinstance(p0, np.ndarray):
        # If p0 is an array, ensure pend is also an array and has the same shape
        assert pend is None or isinstance(pend, np.ndarray), "system parameters must be np.ndarray, float, or int type"
        assert pend is None or p0.shape == pend.shape, "p0 and pend should have the same shape"
    else:
        raise ValueError("system parameters must be np.ndarray, float, or int type")
    return p0, pend



#one step solver functions
def euler_step(f,x_n,t_n,h):
    """
    Computes one step of numerical integration using Euler method
    Parameters:
        f (function): function determining system in terms of x and t
        ode_state (dict): contains current values of space variables (key = 'x_n') and time (key = 't_n')
        x_n (np array): current values of space variables
        t_n (float): current value of time
        h (float): size of euler step
    Returns:
        x_n_plus_1 (np array): updated state of variables after euler step
        t_n_plus_1 (float): updated value of time after euler step
    """
    x_n_plus_1 = x_n + h*f(x_n,t_n)
    return x_n_plus_1,t_n+h

def rk4_step(f,x_n,t_n,h):
    """
    Computes one step of numerical integration using Runge Kutta 4th order method (RK4)

    Parameters:
        f (function): function determining system in terms of x and t
        x_n (np array): current values of space variables
        t_n (float): current value of time
        h (float): size of RK4 step
    Returns:
        x_n_plus_1 (np array): updated state of variables after RK4 step
        t_n_plus_1 (float): updated value of time after RK4 step
        """
    k1 = f(x_n,t_n)
    k2 = f(x_n + h*k1/2,t_n + h/2)
    k3 = f(x_n + h*k2/2,t_n + h/2)
    k4 = f(x_n + h*k3,t_n + h)
    x_n_plus_1 = x_n + h/6*(k1+2*k2+2*k3+k4)
    return x_n_plus_1,t_n + h


def shoot_solve(f_gen,p,init_guess, delta_max,solver = 'RK4',phase_cond=False):
    """
    Finds limit cycles (LC) using  numerical shooting (potentially generalise to all BVPs later)
    Parameters:
        f_gen (function): function of x (np array), t (float), and p (np array) describing system of odes
        p (np array): parameters of system of equations
        init_guess (np array): initial guess for a point on the LC [0:-1], and for the period T [-1]
        delta_max (float): max step size
        solver (string): solver used
        phase_cond (function): function of x and t that describes the phase condition of the LC
                                (if the value is set to False, the phase condition is set to zero velocity in the first state variable at time t = 0)
    Returns:
        x_T0_solved (np array): array of solved initial conditions to the Limit Cycle and period
    """
    p,_ = param_assert(p)
    if not phase_cond:
        #define default phase condition
        def pc(x,t,p):
            return f_gen(x,0,p)[0]
    else:
        pc = phase_cond
    
    #define function to root solve using  newton solver for limit cycles
    def g(x_T0):
        """
        Parameters:
            x_T0 (np array): array of initial conditions and time
        Returns:
            root_solve (np array): array of residuals of the root finding problem
        """
        x0,T = x_T0[:-1],x_T0[-1]
        print(x0)
        x,_ = solve_to(f_gen,p,x0,0,T,delta_max,solver)
        xf = x[:,-1]
        BC = xf-x0
        PC = pc(x0,T,p)
        root_solve = np.append(BC,PC)
        return root_solve

    
    #run scipy newton root-finder on g with initial guess
    x_T0_solved = opt.fsolve(g,init_guess,xtol=delta_max*1.1) #make sure rootfinder tol is higher than integrator tol to avoid numerical issues
    return x_T0_solved[:-1], x_T0_solved[-1]

def natural_p_cont(ode, p0, pend, x_T0, delta_max = 1e-3, n = 25, LC = True):
    """
    Performs natural parameter continuation on system of ODEs
    Parameters:
        ode (function): function of x (np array), t (float), and p (np array, integer or float) describing system of odes
        p0 (np array): initial parameter value(s)
        pend (np array): final parameter value(s)
        x_T0 (np array): initial guess for the system (include initial period guess for LCs as last element of the array)
        delta_max (float): max step size
        n (int): number of steps in the parameter continuation
        LC  (bool): if False, only equilibrium solutions are computed (not Limit Cycles)
    Returns: 
        ps (np array): array of parameter values
        x (np array): array of equilibrium points or points on the limit cycle (and period for LCs) for each parameter value
    
    """
    #check that p0 and pend are the same type, length and that only one parameter changes:
    p0,pend = param_assert(p0,pend)
   #initialise parameter array 
    ps = np.linspace(p0,pend,n).transpose()
    if LC:
        x = np.tile(np.nan,(np.size(x_T0+1),n))
        for i,p in enumerate([ps[:,i] for i in range(n)]):
            #use shooting method to find LCs and equilibrium points.
            sol,T0 = shoot_solve(ode,p,x_T0,delta_max)
            #update initial guess for next iteration
            x_T0 = np.append(sol,T0)    
            x[:,i] = sol     
    else:
        x0 = x_T0
        x = np.tile(np.nan,(np.size(x_T0),n))
        for i,p in enumerate([ps[:,i] for i in range(n)]):
            #use scipy root solver to find equilibrium points
            sol = opt.fsolve(lambda x: ode(x,np.nan,p),x0)
            x[:,i] = sol
            x0=sol
    return ps,x

def pseudo_arc_cond(vi_minus1, vi, vi_plus1): 
    delta = vi - vi_minus1
    v_pred = vi + delta
    return np.dot(vi_plus1 - v_pred, delta)

def pseudo_arc_step(ode,vi_minus1, vi, LC = False):
    #the first value in v arrays (augmented state vector) is the parameter value and the last value is the period
    
    #TODO make it work with multiple parameters and with LC finder

    #set up function with ode and pseudo-arclength condition
    ode_solve = lambda vi_plus1: np.append(ode(vi_plus1[1:],np.nan,vi_plus1[0]),pseudo_arc_cond(vi_minus1, vi, vi_plus1))
    
    #find vi_plus1 using fsolve
    v_sol = opt.fsolve(ode_solve, vi)
    return v_sol



def pseudo_arc(ode,x_T0,p0,pend,p_ind,max_it = 1e3 ,innit_h= 1e-3):
    p0,pend = param_assert(p0,pend)
    assert np.count_nonzero(pend - p0) == 1, "only one parameter should change"

    #find out wether we increase or decrease the parameter.
    if type(p0) == np.ndarray:
        direction = np.sign(pend[p_ind]-p0[p_ind])
    else:
        direction = np.sign(pend-p0)
    #initialise v0 array
    v0 = np.append(p0,x_T0)

    #do a step of natural parameter continuation to find v1
    p1 = p0 + direction * innit_h #making sure to take a step in the direction of pend
    x1 = opt.fsolve(lambda x: ode(x,np.nan,p1),x_T0)
    v1 = np.append(p1,x1)
    
    #initialise array of solutions
    vs = np.tile(np.nan,(np.size(v0),int(max_it)))
    vs[:,0] = v0
    vs[:,1] = v1
    for i in range(int(max_it)):
        #take a step of pseudo-arclength continuation and add to solutions array
        v2 = pseudo_arc_step(ode,v0,v1)
        vs[:,i+2] = v2
        #check if we have reached the end of the continuation
        if direction * v2[0] > direction * pend[p_ind]:
            print(f"Reached end of continuation after {i} iterations")
            return vs[:,:i+3]
        v0 = v1
        v1 = v2
    print("Max iterations reached")
    return vs



# %%
