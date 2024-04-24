#%% Packages
import numpy as np
import scipy
import scipy.optimize as opt
import inspect
import matplotlib.pyplot as plt
#%% Functions
def solve_to(ode, p, x0, t0, t_f, delta_max = 1e-3, solver = 'RK4'):
    """
    Applies one-step solvers to systems of odes from initial conditions to 
    to the specified endpoint.
    Parameters:
        ode (function): function of x (np array), t (float), and p (np array) describing system of odes
        p (np array, float or int): parameter(s) of system of equations
        x0 (np array): initial conditions for the system
        t0 (float): initial time
        t_f (float): final time
        delta_max (float): max step size
        solver (string): solver used
    Returns:
        x (np array): 2D array where each row is a time series of the state variables over discretisation of time
        t (np array): discretised time array

    """
    #assert that the parameters are of the right type
    p = param_assert(p)

    #hard encode the parameter to make the function of x and t only
    f = lambda x,t: ode(x,t,p)

    #choose one-step solver
    if solver == 'Euler':
        solve_step = euler_step
    elif solver == 'RK4':
        solve_step = rk4_step
    else:
        raise ValueError("Unsupported solver: {}".format(solver))
    

    #make function robust to too small time scales relative to delta_max
    total_time = t_f - t0
    if total_time <= 2*delta_max:
        print("The duration of the integration is shorther than delta max")
        #return initial conditions and time, adding a dimension to x0 to make it a 2D array for consistency
        return x0[x0[:],None],t0

    #initialise t and x arrays
    t = discr_t(t0,t_f,delta_max)
    x = np.zeros((len(x0),len(t)))

    #add initial conditions to x array
    x[:,0] = x0

    x_n = x0
    h = delta_max

    #iterate over time steps, solving for x_n+1 at each step
    for i,t_n in enumerate(t[:-2]):
        x_n = solve_step(f,x_n,t_n,h)
        x[:,i+1] = x_n

    #adapt size of final time-step to compute solution at t_f exactly
    h = t_f - t[-2] 
    x[:,-1] = solve_step(f,x_n,t[-2],h)
    return x,t

def discr_t(t0, t_f, delta_max):
    """
    Discretises time for numerical integration.
    Parameters:
        t0 (float): initial time
        t_f (float): final time
        delta_max (float): max step size
    Returns:
        t (np array): array of time values
    """
    t = np.arange(t0,t_f,delta_max) 
    t = np.append(t,t_f)
    return t


def param_assert(p):
    """
    Asserts that the parameters are a number or a numpy array. If the parameters are a number, converts them to a numpy array for consistency.
    Parameters:
        p (np array, float or int): parameter(s) of system of equations
    Returns:
        p (np array): parameter(s) of system of equations
    """
    # Check if p is a float or an int, and if so, convert to a numpy array with a single element
    if isinstance(p, (float, int)):
        p = np.array([p], dtype = float)
    # Check if p is a numpy array
    elif isinstance(p, np.ndarray):
        p = p
    # If p is not a float, int, or numpy array, raise an error
    else:
        raise ValueError("system parameters must be np.ndarray, float, or int type")
    return p



#one step solver functions
def euler_step(f, x_n, t_n, h):
    """
    Computes one step of numerical integration using Euler method
    Parameters:
        f (function): function determining ODE system in terms of x and t
        x_n (np array): current values of space variables
        t_n (float): current value of time
        h (float): size of Euler step
    Returns:
        x_n_plus_1 (np array): updated state of variables after euler step
    """
    x_n_plus_1 = x_n + h*f(x_n,t_n)
    return x_n_plus_1

def rk4_step(f, x_n, t_n, h):
    """
    Computes one step of numerical integration using Runge Kutta 4th order method (RK4)

    Parameters:
        f (function): function determining ODE system in terms of x and t
        x_n (np array): current values of space variables
        t_n (float): current value of time
        h (float): size of RK4 step
    Returns:
        x_n_plus_1 (np array): updated state of variables after RK4 step
        """
    k1 = f(x_n,t_n)
    k2 = f(x_n + h*k1/2,t_n + h/2)
    k3 = f(x_n + h*k2/2,t_n + h/2)
    k4 = f(x_n + h*k3,t_n + h)
    x_n_plus_1 = x_n + h/6*(k1+2*k2+2*k3+k4)
    return x_n_plus_1


#default boundary condition function for limit cycles
def LC_residual(ode, p, xs):
    """
    Computes the residual of limit cycles (LC) for a system of ODEs (x(T) - x(0))
    (ode definition and parameter(s) p are included so that Neumann BCs can be used and parameter-dependent BVPs can be solved by the shoot-solve function)
    Parameters:
        ode (function): function of x (np array), t (float), and p (np array) describing system of odes
        p (np array): parameter(s) of system of equations
        xs (np array): array of state variables at each time step (as solved by solve_to function)
    Returns:
        res (np array): array of residuals of the limit cycle problem
    """
    #compute residuals x(T) - x(0)
    res = xs[:,-1] - xs[:,0]
    return res

#default phase condition function for limit cycles
def default_pc(ode, p, xs):
    """
    Default phase condition for numerical shooting, zero velocity in the first state variable at time t = 0.
    Parameters:
        ode (function): function of x (np array), t (float), and p (np array) describing system of odes
        p (np array): parameters of system of equations
        xs (np array): array of state variables at each time step (as solved by solve_to function)
    Returns:
        res (np array): array of residuals of the phase condition
    """
    #extract the state variables at time t = 0
    x0 = xs[:,0]
    #determine the velocity in the first state variable at time t = 0
    return ode(x0,0,p)[0]

def choose_num_int(num_int_name):
    if num_int_name == "solve_to":
        return solve_to
    elif num_int_name == "solve_ivp":
        def solve_ivp(ode, p, x0, t0, t_f, delta_max = 1e-3, solver = 'RK45'):
            integration = scipy.integrate.solve_ivp(lambda t,x,p: ode(x,t,p),(t0,t_f),x0,args=(p,),max_step=delta_max,method=solver)
            if integration.success:
                return integration.y, integration.t
            else:
                raise RuntimeError("Integration failed")
        return solve_ivp

def shoot_solve(ode, p, x0, T0, delta_max = 1e-2, solver = 'RK4', boundary_cond = LC_residual, phase_cond=default_pc, num_int_name = "solve_to"):
    """
    Solves BVPs using the shooting method.
    The default setup aims to find a point x on a limit cycle, and the period T of the limit cycle, such that x(T) = x(0).
    However, the boundary condition and phase condition functions can be changed to solve other BVPs.
        e.g. 1: to solve a boundary value problem of the form dx/dt = f(x) and x(T) = x_f,
            set boundary_cond = lambda ode,p,xs: xs[:,-1] - x_f 
        e.g. 2: to solve a boundary value problem of the form dx/dt = f(x) and x'(T) = v_f,
            set boundary_cond = lambda ode,p,xs: ode(xs[:,-1],T,p) - v_f 
        In each of these cases, the output will be the initial value of x and the period of integration T required to satisfy the boundary condition.
    Parameters:
        ode (function): function of x (np array), t (float), and p (np array) describing system of odes
        p (np array, float or integer): parameter(s) of system of ODEs
        x0 (np array): initial guess for point on the LC (or other BVP)
        T0 (float): initial guess for the period of the LC (or other BVP)
        delta_max (float): max step size used in the numerical integration
        solver (string): solver used in the numerical integration
        boundary_cond (function): function of the system of ODEs, p, xs (2D array of state variables at each point in discretised time, as computed by numerical integration) and t that describes the boundary condition of the shooting method
        phase_cond (function): function of the system of ODEs, p, xs (nu) and t that describes the phase condition of the shooting method
    Returns:
        x (np array): point on the Limit Cycle or initial condition for the BVP
        T (float): period of the Limit Cycle or period of the BVP
    """
    p = param_assert(p)
    num_int = choose_num_int(num_int_name)
    #define function to root solve using fsolve
    def g(x_T):
        #split x and T from x_T
        x0,T = x_T[:-1],x_T[-1]
        #run numerical integration to x(T)
        xs,_ = num_int(ode,p,x0,0,T,delta_max = delta_max, solver= solver)
        #compute boundary condition residuals
        BC = boundary_cond(ode,p,xs)
        PC = phase_cond(ode,p,xs)
        res = np.append(BC,PC)
        return res
    
    #initialise initial space variable and period guess
    #these are concatenated into a single array for the root solver
    x_T0 = np.append(x0,T0)

    #run scipy newton root-finder on g with initial guess
    #for stability reasons, the tolerance of the root finder should be higher than that of the integrator
    tol = 1.01*delta_max
    x_T = opt.fsolve(g,x_T0,xtol=tol)

    #return the initial condition and period
    x = x_T[:-1]
    T = x_T[-1]
    return x, T

def natural_p_cont(residual_func, p0, p_ind, x0, h = 1e-2, N=50, tol = 1e-3):
    """
    Performs natural parameter continuation on a residual function.
    Parameters:
        residual_func (function): residual function of x (np array) and p (np array, integer or float)
        p0 (np array): initial parameter value(s)
        p_ind (int): index of the parameter that changes
        x0 (np array): initial guess for the residual function
        h (float): step size of the natural parameter continuation
        N (int): number of iterations
        tol (float): tolerance of the root solver
    Returns:
        x (np array): array of values satisfying the residual function for each parameter value
        ps (np array): array of parameter values
    """
    #check that p0 is of the right type, and convert to np array if it is a float or int
    p0 = param_assert(p0)
    #compute the end value of the parameter continuation
    p_end = p0[p_ind] + N*h
    #create array of varying parameter values
    p_vary = np.linspace(p0[p_ind], p_end, N)

    #initialise parameter array and solution array
    x = np.zeros((np.size(x0),N))
    ps = np.zeros((np.size(p0),N))
    p_n = p0.astype(float)
    for i,p in enumerate(p_vary):
        #update varying parameter value
        p_n[p_ind] = p
        #find root of residual function
        xi = opt.fsolve(residual_func, x0, args = (p_n), xtol=tol)
        #add result to results array
        x[:,i] = xi
        # add parameter value to parameter array
        ps[:,i] = p_n
        #update initial guess for next iteration
        x0 = xi
    return x,ps


def pseudo_arc(residual_func, p0, p_ind, x0, h= 1e-2, N = 50, tol = 1e-3):
    """
    Performs pseudo-arclength continuation on residual function.
    Parameters:
        residual_func (function): residual function of x (np array) and p (np array, integer or float)
        p0 (np array): initial parameter value(s)
        p_ind (int): index of the parameter that changes
        x0 (np array): initial guess for the system
        N (int): number of iterations
        innit_h (float): step size of initial natural parameter continuation step
        tol (float): tolerance of the root solver
    Returns:
        x (np array): array of values satisfying the residual function for each parameter value
        ps (np array): array of parameter values
    """
    #assert that the parameters are of the right type
    p0 = param_assert(p0)

    #find first value using initial guess
    x0 = opt.fsolve(residual_func,x0, args = (p0))

    #do a step of natural parameter continuation to find p1 and x1
    p1 = p0.copy()
    p1[p_ind] = p0[p_ind] + h 
    x1 = opt.fsolve(residual_func,x0, args = (p1))

    #initialise array of solutions
    x = np.zeros((len(x0),int(N)+2))
    ps = np.zeros((len(p0),int(N)+2))

    #and add solutions 0 and 1 to array 
    x[:,0] = x0
    x[:,1] = x1
    ps[:,0] = p0
    ps[:,1] = p1

    #form augmented state vectors v0 and v1
    v0 = np.append(p0[p_ind],x0)
    v1 = np.append(p1[p_ind],x1)

    #create variable p_sol to store the non-changing parameter values
    p_sol = p0.copy()
    for i in range(int(N)):
        #take a step of pseudo-arclength continuation and add to solutions array
        v2 = pseudo_arc_step(residual_func,v0,v1,p_sol,p_ind,tol)
        
        #add to solutions array
        x[:,i+2] = v2[1:]
        p_sol[p_ind] = v2[0] #update the varying parameter value
        ps[:,i+2] = p_sol

        #update values for v0 and v1 for next iteration
        v0 = v1.copy()
        v1 = v2
    return x,ps

def pseudo_arc_step(residual_func, v0, v1, p_sol, p_ind, tol):
    """
    Performs one step of pseudo-arclength continuation on residual function.
    Parameters:
        residual_func (function): residual function of x (np array) and p (np array, integer or float)
        v0 (np array): value of augmented state vector at step i-2
        v1 (np array): value of augmented state vector at step i-1
        p_sol (np array): parameter values at step i-1
        p_ind (int): index of the parameter that changes
        tol (float): tolerance of the root solver
    Returns:
        v2 (np array): value of augmented state vector at step i
    """

    #find the delta and predict v2
    delta = v1 - v0
    v_pred = v1 + delta

    #define function to solve
    def root_solve(v2):
        #update the varying parameter value
        p_sol[p_ind] = v2[0]
        #compute residual of pseudo-arclength condition
        pseudo_cond = np.dot(v2 - v_pred, delta)
        #compute residual of function condition
        func_cond = residual_func(v2[1:],p_sol)
        #combine residuals
        residuals = np.append(pseudo_cond,func_cond)
        return residuals
    #find v2 using fsolve
    v2 = opt.fsolve(root_solve,v_pred,xtol=tol)
    return v2

def parameter_continuation(cont_type, residual_func, p0, x0, p_ind = 0, h= 1e-1, N = 100, tol = 1e-3):
    """
    Wrapper function for parameter continuation methods.
    Parameters:
        cont_type (str): type of continuation method
        residual_func (function): function of x (np array) and p (np array, integer or float) we want to find the roots of
        p0 (np array): initial parameter value(s)
        x0 (np array): initial guess for the system
        p_ind (int): index of the parameter that changes (0 if only system only has one parameter)
        h (float): step size of natural parameter continuation or initial step size of pseudo-arclength continuation
        N (int): number of iterations
        tol (float): tolerance of the root solver
    Returns:
        x (np array): array of values satisfying the residual function for each parameter value
        ps (np array): array of parameter values
    """
    if cont_type == 'natural':
        cont = natural_p_cont
    elif cont_type == 'pseudo_arclength':
        cont = pseudo_arc
    else:
        raise ValueError("Unsupported continuation method: {}".format(cont_type))
    return cont(residual_func, p0, p_ind, x0, h = h, N = N, tol = tol)
    


def bifurcation_analysis(ode, p0, x0, p_ind = 0, T0 = 0, N = 50, LC=True, h= 1e-2, delta_max = 1e-2, phase_cond=default_pc, solver = 'RK4', cont_type = 'natural', num_int_name = "solve_to"):
    """
    Tracks equilibria or points on limit cycles of a system of ODEs as a parameter is varied.
    Parameters:
        ode (function): function of x (np array), t (float), and p (np array) describing system of odes
        p0 (np array): initial parameter value(s)
        x0 (np array): initial guess for the attractor (equilibrium or limit cycle)
        p_ind (int): index of the parameter that changes (0 if only system only has one parameter)
        T0 (float): initial guess for the period of the LC (0 if only equilibria are computed, i.e. LC = False)
        N (int): number of iterations of the continuation method
        LC (bool): if False, only equilibrium solutions are computed (not Limit Cycles)
        h (float): step size of natural parameter continuation or initial step size of pseudo-arclength continuation
        delta_max (float): max step size used in the numerical integration
        phase_cond (function): function of the system of ODEs, p and xs that describes the phase condition of the shooting method
        solver (str): solver used in the numerical integration
        cont_type (str): type of the continuation method
    """
    if LC:
        if T0 <= 0:
            raise ValueError("Positive initial guess for the period of the LC is required")
        num_int = choose_num_int(num_int_name)
        def fixed_point_func(x_T,p):
            x0,T = x_T[:-1],x_T[-1]
            xs,_ = num_int(ode,p,x0,0,T,delta_max=delta_max,solver=solver)
            res = np.append(LC_residual(ode,p,xs),phase_cond(ode,p,xs))
            return res      
    else:
        if T0:
            raise ValueError("Initial guess for the period of the LC must equal 0 when computing equilibria")
        def fixed_point_func(x_T,p):
            x = x_T[:-1]
            #we append a 0 to the end of the array to represent the "period of the LC" for equilibria
            return np.append(ode(x,0,p),0)
          
    #assert that the parameter is of the right type
    p0 = param_assert(p0)
    #set up initial guess for the continuation
    x_T0 = np.append(x0,T0)
    #for stability reasons, the tolerance of the root finder should be higher than that of the integrator
    tol = 1.01*delta_max
    #run continuation
    x_Ts,ps = parameter_continuation(cont_type, fixed_point_func, p0, x_T0, p_ind = p_ind, N = N, h = h, tol = tol)
    xs, Ts = x_Ts[:-1,:], x_Ts[-1,:]
    return xs, Ts, ps



class Boundary_Condition():
    """
    Class for boundary conditions in finite difference solvers.
        Provides a consistent framework for defining boundary conditions. To maximise generality
        the values at boundaries are defined as functions of time, so that time-dependent boundary conditions can be implemented effortlessly.
        However, this means a slight overhead in terms of computational cost.

    Attributes:
        type (str): type of boundary condition (Dirichlet, Neumann, Robin)
        x (float): position of the boundary condition
        value (function or tuple of functions): value of the boundary condition,
            for Dirichlet BCs, the value is a function of time, where u(x) = value(t) is the boundary condition
            for Robin BCs, the value is a tuple of functions of time, where u'(x) = value[0](t) - value[1](t) * u(x) is the boundary condition
            Neumann BCs are implemented as Robin BCs with value[1](t) = 0 

    Methods:
        add_left: adds left boundary condition values back to the grid for Dirichlet BCs
        add_right: adds right boundary condition values back to the grid for Dirichlet BCs
    """
    def __init__(self, BC_type, x, value):
        """
        Initialises the boundary condition object.
        Parameters:
            BC_type (str): type of boundary condition (Dirichlet, Neumann, Robin)
            x (float): position of the boundary condition
            value: value of the boundary condition:
                for Dirichlet BCs: input value is a number or a function of time, where u(x) = value(t) or u(x) = value is the boundary condition
                for Robin BCs: input value is a tuple of two numbers or of two functions of time, where u'(x) = value[0](t) - value[1](t) * u(x) is the boundary condition.
                for Neumann BCs: input value is a number or a function of time, where u'(x) = value(t) or u'(x) = value is the boundary condition.
        """
        self.type = BC_type
        self.x = x

        if self.type == "Dirichlet":
            if callable(value):
                self.value = value
            elif isinstance(value, (int, float)):
                #convert number to constant function
                self.value = lambda t: value
            else:
                raise ValueError("The value for Dirichlet boundary condition must be a function or a number.")


        elif self.type == "Robin":
            if isinstance(value, tuple) and len(value) == 2:
                if (callable(value[0]) and callable(value[1])):
                    self.value = value
                elif (isinstance(value[0], (int, float)) and isinstance(value[1], (int, float))):
                    #convert numbers to constant functions
                    self.value = (lambda t: value[0], lambda t: value[1])
                else:
                    raise ValueError("The values for Robin boundary condition must be either two functions or two numbers.")
            else:
                raise ValueError("The value for Robin boundary condition must be a tuple of two functions or two numbers.")

        elif self.type == "Neumann":
            if callable(value):
                #add a zero function to the tuple to represent the coefficient of u(x) to generalise to a Robin BC
                self.value = (value, lambda t: 0)
            elif isinstance(value, (int, float)):
                self.value = (lambda t: value, lambda t: 0)
            else:
                raise ValueError("The value for Neumann boundary condition must be a function or a number.")

        else:
            raise ValueError("Unsupported boundary condition type: {}".format(type(value)))

    def add_left(self,u,t=np.array([None])):
        """
        Adds left boundary condition value(s) back to the grid for Dirichlet BCs.
        If a time array is provided, the boundary condition value is evaluated at each time point.
        Otherwise, the problem is considered to be an ODE and the time array is not used.
        Parameters:
            u (np array): array of state variables at each point in discretised time
            t (np array): array of time values
        """
        #Adds left BC values back to grid for Dirichlet BCs
        if self.type == "Dirichlet":
            if t.any():
                u_left = np.zeros(len(t)) + self.value(t)
                u = np.vstack((u_left,u))  
            else:
                u = np.append(self.value(np.nan),u) 
        return u
    
    def add_right(self,u,t=np.array([None])):
        """
        Same as add_left, but for right boundary conditions.
        """
        #Adds left BC values back to grid for Dirichlet BCs
        if self.type == "Dirichlet":
            if t.any():
                u_right = np.zeros(len(t)) + self.value(t)
                u = np.vstack((u, u_right))  
            else:
                u = np.append(u, self.value(np.nan))  
        return u



class Grid():
    """
    Class for grid objects in finite difference solvers.
        Attributes:
            N (int): number of grid points
            a (float): left boundary of the grid
            b (float): right boundary of the grid
            dx (float): grid spacing
            x (np array): array of grid points
    """
    def __init__(self,N,a,b):
        self.N = N
        self.a = a
        self.b = b
        self.dx = (b-a)/N
        self.x = np.linspace(a,b,N+1)


def construct_A_diags_b(grid, bc_left, bc_right, D, P):
    """
    Constructs the diagonals of tridiagonal matrix A and the vector b that represent a second order ODE expressions of the form:
    Du'' + Pu' as a linear system Ax + b. This is done using the finite difference method.
    Parameters:
        grid (Grid): grid object defining space discretisation
        bc_left (Boundary_Condition): left boundary condition
        bc_right (Boundary_Condition): right boundary condition
    Returns:
        A_sub (np array): subdiagonal of the matrix A
        A_diag_func (func): diagonal of the matrix A as a function of t
        A_sup (np array): superdiagonal of the matrix A
        b_func (func): vector b as a function of time
        left_ind (int/None): index of the leftmost grid point used in the solver
        right_ind (int/None): index of the rightmost grid point used in the solver
    """

    N = grid.N
    dx = grid.dx

    #first seperate a and b into contributions from the first and second order terms
    #think of the system as Du'' + Pu' = D/(dx^2) (A_second + b_second) + P/(2dx) (A_first + b_first)


    #Make general tridiagonal matrices A_second and A_first

    #subdiagonals
    A_sub_first = -np.ones(N-2)
    A_sub_second = np.ones(N-2)

    #diagonals
    A_diag_first = np.zeros(N-1, dtype= object)
    A_diag_second = -2*np.ones(N-1, dtype = object)

    #superdiagonals
    A_sup_first = np.ones(N-2)
    A_sup_second = np.ones(N-2) 

    #make general vector b
    b_first = np.zeros(N-1, dtype= object)
    b_second = np.zeros(N-1, dtype= object)

    #initialise left_ind and right_ind for Dirichlet boundary conditions



    if bc_left.type == "Dirichlet":
        b_first[0] = lambda t : -bc_left.value(t)
        b_second[0] = bc_left.value

        left_ind = 1

    else:
        A_sub_first = np.append(-1,A_sub_first)
        A_sub_second = np.append(1,A_sub_second)

        A_diag_first = np.append(lambda t: -2*dx*bc_left.value[1](t),A_diag_first)
        A_diag_second = np.append(lambda t: -2 * (1 - dx * bc_left.value[1](t)),A_diag_second)

        A_sup_first = np.append(0,A_sup_first)
        A_sup_second = np.append(2,A_sup_second)

        b_first = np.append(lambda t: 2*dx*bc_left.value[0](t),b_first)
        b_second = np.append(lambda t: -2 * dx * bc_left.value[0](t),b_second)

        left_ind = None


    if bc_right.type == "Dirichlet":
        b_first[-1] = bc_right.value
        b_second[-1] = bc_right.value
        right_ind = -1
    else:
        A_sub_first = np.append(A_sub_first,0)
        A_sub_second = np.append(A_sub_second,2)

        A_diag_first = np.append(A_diag_first,lambda t: -2*dx*bc_right.value[1](t))
        A_diag_second = np.append(A_diag_second,lambda t: -2 * (1 + dx * bc_right.value[1](t)))

        A_sup_first = np.append(A_sup_first,1)
        A_sup_second = np.append(A_sup_second,1)
        
        b_first = np.append(b_first,lambda t: 2*dx* bc_right.value[0](t))
        b_second = np.append(b_second,lambda t: 2 * dx * bc_right.value[0](t))
        right_ind = None

    #combine the first and second order terms to get the full A and b according to D/(dx^2) (A_second + b_second) + P/(2dx) (A_first + b_first)
    #functions of time need to be defined for the main diagonal and the vector b as they depend on the values of the BCs
    def A_diag_func(t):
        A_diag_first_t = np.array([f(t) if callable(f) else f for f in A_diag_first],dtype = np.float64)
        A_diag_second_t = np.array([f(t) if callable(f) else f for f in A_diag_second],dtype = np.float64)
        A_diag_t = (D/dx**2) * A_diag_second_t + P/(2*dx) * A_diag_first_t
        return A_diag_t
    
    def b_func(t):
        b_first_t = np.array([f(t) if callable(f) else f for f in b_first],dtype = np.float64)
        b_second_t = np.array([f(t) if callable(f) else f for f in b_second],dtype = np.float64)
        b_t = (D/dx**2) * b_second_t + P/(2*dx) * b_first_t
        return b_t
    
    A_sub = (D/dx**2) * A_sub_second + P/(2*dx)*A_sub_first
    A_sup = (D/dx**2) * A_sup_second + P/(2*dx)*A_sup_first
    #TODO note that this is not the most computationaly efficient way to do this, but it is the most general
    return A_sub, A_diag_func, A_sup, b_func, left_ind, right_ind

def reform_A(A_sub,A_diag,A_sup):
    """
    Reforms a tridiagonal matrix from its diagonals.
    Parameters:
        A_sub (np array): subdiagonal of the matrix A
        A_diag (np array): diagonal of the matrix A
        A_sup (np array): superdiagonal of the matrix A
    Returns:
        A (np array): tridiagonal matrix A
    """
    return np.diag(A_sup,1) + np.diag(A_diag,0) + np.diag(A_sub,-1)

# Linear System solvers for Poisson equation using arrays of diagonals and b
def choose_lin_solve(solver_name):
    """
    Chooses the tridiagonal linear system solver based on the input string.
    Parameters:
        solver_name (str): name of the solver
    Returns:
        function: the linear solver function (function of the form lin_solve(A_sub, A_diag, A_sup, b))
    """
    if solver_name == 'np_solve':
        return lin_solve_numpy
    elif solver_name == 'sp_root':
        return lin_solve_scipy
    elif solver_name == 'thomas':
        return lin_solve_thomas
    elif solver_name == 'sparse':
        return lin_solve_sparse
    else:
        raise ValueError("Unsupported solver: {}".format(solver_name))
    
def lin_solve_numpy(A_sub,A_diag,A_sup,b):
    """
    Solves a linear system of equations of the form Ax = b, where A is a tridiagonal matrix, using the numpy linear solver
    Parameters:
        A_sub (np array): subdiagonal of the matrix A
        A_diag (np array): diagonal of the matrix A
        A_sup (np array): superdiagonal of the matrix A
        b (np array): vector b in the system of equations Ax = b
    Returns:
        u (np array): solution to the system of equations
    """
    #construct matrix A
    A = reform_A(A_sub,A_diag,A_sup)
    #solve for u
    u = np.linalg.solve(A,b)
    return u

def lin_solve_scipy(A_sub,A_diag,A_sup,b):
    """
    Solves a linear system of equations of the form Ax = b, where A is a tridiagonal matrix, using the root solver from scipy.optimize
    Parameters:
        A_sub (np array): subdiagonal of the matrix A
        A_diag (np array): diagonal of the matrix A
        A_sup (np array): superdiagonal of the matrix A
        b (np array): vector b in the system of equations Ax = b
    Returns:
        u (np array): solution to the system of equations
    """
    #construct matrix A
    A = reform_A(A_sub,A_diag,A_sup)
    #solve for u
    f = lambda u: A@u - b
    result = opt.root(f,np.zeros(len(b)))
    if result.success:
        u = result.x
    else:
        raise ValueError("Root solver failed")
    return u

def lin_solve_thomas(A_sub, A_diag, A_sup, b):
    """
    Solves a tridiagonal system of equations using the Thomas algorithm.
    Parameters:
        A_sub (np array): subdiagonal of the matrix A
        A_diag (np array): diagonal of the matrix A
        A_sup (np array): superdiagonal of the matrix A
        b (np array): vector b in the system of equations Ax = b
    Returns:
        u (np array): solution to the system of equations
    """
    N = len(b)
    #initialize arrays for the algorithm
    c = np.zeros(N-1)
    d = np.zeros(N)
    u = np.zeros(N)
    #forward sweep
    c[0] = A_sup[0]/A_diag[0]
    d[0] = b[0]/A_diag[0]
    for i in range(1, N-1):
        c[i] = A_sup[i]/(A_diag[i] - A_sub[i-1]*c[i-1])
        d[i] = (b[i] - A_sub[i-1]*d[i-1])/(A_diag[i] - A_sub[i-1]*c[i-1])
    #backward sweep
    d[-1] = (b[-1] - A_sub[-1]*d[-2])/(A_diag[-1] - A_sub[-1]*c[-1])
    u[-1] = d[-1]
    for i in range(N-2, -1, -1):
        u[i] = d[i] - c[i]*u[i+1]
    return u

def lin_solve_sparse(A_sub, A_diag, A_sup, b):
    """
    Solves a tridiagonal system of equations using sparse matrices.
    Parameters:
        A_sub (np array): subdiagonal of the matrix A
        A_diag (np array): diagonal of the matrix A
        A_sup (np array): superdiagonal of the matrix A
        b (np array): vector b in the system of equations Ax = b
    Returns:
        u (np array): solution to the system of equations
    """
    N = len(b)
    #construct matrix A
    A = scipy.sparse.diags([A_sub, A_diag, A_sup], [-1, 0, 1], format='csc')
    #solve for u
    u = scipy.sparse.linalg.spsolve(A, b)
    return u


def finite_diff(bc_left, bc_right, q, p, N, D=1,P = 0, u_innit = np.array(None), v = 1, dq_du = None, solver = 'thomas', max_iter = 100, tol = 1e-6, num_int_name = "solve_to"):
    """
    Solves 2nd order ODEs of the form Du'' + Pu' + q((u),x,p) = 0 using finite differences.
    If the source term is linear, the linear system that comes from finite differencing is solved directly.
    If the source term is non-linear, Newton's method is used to solve the system.
    The linearity of the system is determined by the number of arguments of the source term function.
    Parameters:
        bc_left (Boundary_Condition): left boundary condition
        bc_right (Boundary_Condition): right boundary condition
        q (function): source term, function of (u), x, and p
        p (np array): parameter(s) of the source term
        N (int): number of grid points
        D (float): diffusion coefficient
        P (float): convection coefficient
        u_innit (np array): initial guess for the solution
        v (float): damping factor for Newton's method (0<v<=1)
        dq_du (function): derivative of the source term with respect to u
        max_iter (int): maximum number of iterations for Newton's method
        tol (float): tolerance for Newton's method
        solver (string): solver used for linear systems
    Returns:
        u (np array): solution to the Poisson equation
        grid.x (np array): grid points
    """
    p = param_assert(p)
    if D == 0:
        raise ValueError("D must be non-zero, use solve_to if you want to solve a first order system of ODEs.")
    #form grid
    grid = Grid(N,bc_left.x,bc_right.x)
    dx = grid.dx

    # find diagonals of matrix A and vector b given boundary conditions
    #set up variables left_ind and right_ind that determine the first and last grid values used in the solver
    A_sub, A_diag_func, A_sup, b_func, left_ind, right_ind = construct_A_diags_b(grid, bc_left, bc_right,D,P)
    #There is no time dependency so compute constant vectors fo the system
    A_diag,b = A_diag_func(np.nan), b_func(np.nan)

    #choose solver
    lin_solve = choose_lin_solve(solver)
    
    #find number of arguments to q to determine if the source term is linear or non-linear
    n_args = len(inspect.signature(q).parameters)

    if n_args == 2:
        #solve linear system using chosen solver
        # define constant vector c
        c = -b - q(grid.x[left_ind:right_ind],p)
        #solve linear system of the shape Au = c using chosen solver
        u = lin_solve(A_sub,A_diag,A_sup,c)

    elif n_args == 3:
        num_int = choose_num_int(num_int_name)
        #solve non-linear system using Newton's method
        #initialise first guess for u
        if u_innit.any() == None:
            print("No initial guess provided, defaulting to zero vector. Results can be improved by providing an adequate initial guess.")
            #default to zero vector
            u = np.zeros(len(b))
        else:
            u = u_innit[left_ind:right_ind]

        if dq_du is None:
            #finite difference approximation for dq_du
            eps = 1e-6
            dq_du = lambda u, x, p: (q(u + eps, x, p) - q(u - eps, x, p)) / (2 * eps)
        
        A = reform_A(A_sub,A_diag,A_sup)


        #solve for u using Newton's method
        for i in range(max_iter):
            #compute discretised residual
            F = A@u + b + q(u,grid.x[left_ind:right_ind], p)
            #define Jacobian of the system
            J_F_diag = A_diag + dq_du(u,grid.x[left_ind:right_ind], p)
            #solve for correction V using linear solver
            V = lin_solve(A_sub,J_F_diag,A_sup,-F)
            #update u
            u += v*V
            #check for convergence
            if np.linalg.norm(V) < tol:
                print(f"Newton method converged within the tolerance after {i} iterations")
                break


    #add boundary conditions to solution for Dirichlet boundary conditions
    u = bc_left.add_left(u)
    u = bc_right.add_right(u)
    return u, grid.x


def meth_lines(bc_left, bc_right, f, t0, t_f, q, p, N, D = 1, P=0, dt = None, method = "imex", explicit_solver = 'RK4' ,linear_solver = 'thomas', tol = 1e-3, num_int_name = "solve_to"):
    """
    Solves PDEs of the form u_t = D*u_xx + P*u_x + q(u,x,t,p) using the method of lines.
    Parameters:
        bc_left (Boundary_Condition): left boundary condition
        bc_right (Boundary_Condition): right boundary condition
        f (function): initial condition, function of x and t0
        t0 (float): initial time
        t_f (float): final time
        q (function): source term, function of u, x, t, and p
        p (np array): parameter(s) of the source term
        N (int): number of grid points in space
        D (float): coefficient of the second space derivative term (default is 1)
        dt (float): time step size of the time integration (default is dx^2/(2*D))
        explicit_solver (string): solver used for explicit time integration
        linear_solver (string): solver used for to solve the linear system in implicit euler method
    Returns:
        u (np array): solution to the diffusion equation
        grid.x (np array): space grid points
        t (np array): time grid points 
    """
    p = param_assert(p)
    #discretise in space
    grid = Grid(N,bc_left.x,bc_right.x)
    dx = grid.dx

    #finite difference method to find diagonals of matrix A and vector b
    A_sub, A_diag_t, A_sup, b_t, left_ind, right_ind = construct_A_diags_b(grid, bc_left, bc_right,D,P)
    
    #set up initial condition u0
    u0 = f(grid.x[left_ind:right_ind],t0)

    #find number of arguments to q to determine if the ODE system resulting from 
    #the finite difference discretisation is linear or non-linear.
    n_args = len(inspect.signature(q).parameters)
    if n_args == 3:
        #define q as a function of u, x, t and p as explicit methods are the
        #same for linear and non-linear source terms.
        def q_u(u,x,t,p):
            return q(x,t,p)
    elif n_args == 4:
        q_u = q
    else:
        raise ValueError("Source term must be a function of u, x, t and p or x, t and p.")


    if method == "explicit":
        num_int = choose_num_int(num_int_name)
        #define du_dt as a function of u, t and p
        def du_dt(u,t,p):
            A = reform_A(A_sub,A_diag_t(t),A_sup)
            b = b_t(t)
            return (A @ u + b) + q_u(u,grid.x[left_ind:right_ind],t,p)
        #ensure stablilty of the time integration
        dt_stable = dx**2/(2*D)
        if not dt:
            #default value of dt to ensure stability
            dt = dt_stable
        elif dt > dt_stable:
            raise ValueError('dt must be smaller or equal to dx^2/(2*D) for explicit Euler method (where dx is granularity of the grid in space)')
        #solve using one-step solver
        u, t = num_int(du_dt,p,u0,t0,t_f,dt,solver = explicit_solver)

    elif method == "implicit" or method == "imex":
        #choose linear system solver
        lin_solve = choose_lin_solve(linear_solver)
        
        if not dt:
            raise ValueError("dt must be specified for implicit methods")

        #discretise in time
        t = discr_t(t0,t_f,dt)

        #initialise solution array
        u = np.zeros((len(u0),len(t)))
        u[:,0] = u0
        u_n = u0

        #solve linear systems at each time-step for implicit methods (imex or linear implicit euler)
        if n_args == 3 or method == "imex":
            for i,t_n in enumerate(t[:-1]):
            #construct diagonals of matrix M = I - dt * A at time t_n
                M_diag = 1 - dt * A_diag_t(t_n)
                M_sub = -dt * A_sub
                M_sup = -dt * A_sup
                #construct vector d = u + C*b + dt *q_u at time t_n
                d = u_n + dt*b_t(t_n)+ dt * q_u(u_n,grid.x[left_ind:right_ind],t_n,p)
                #solve for u_n+1 
                u_n_plus_1 = lin_solve(M_sub,M_diag,M_sup,d)
                u[:,i+1] = u_n_plus_1
                u_n = u_n_plus_1

        #solve non-linear system at each time-step for implicit euler using scipy's fsolve
        else:
            for i,t_n in enumerate(t[:-1]):
                A_diag = A_diag_t(t_n)
                A = reform_A(A_sub,A_diag,A_sup)
                b = b_t(t_n)
                #define residual function
                F = lambda u_n_plus_1 : A @ u_n_plus_1 + b + q_u(u_n_plus_1,grid.x[left_ind:right_ind],t_n,p) - (u_n_plus_1 - u_n)/dt
                #solve for u_n+1 using fsolve
                u_n_plus_1 = opt.fsolve(F,u_n,xtol=tol)
                u_n = u_n_plus_1
                u[:,i+1] = u_n
    else:
        raise ValueError("Unsupported method: {}".format(method))
    
    #add boundary conditions to solution for Dirichlet boundary conditions
    u = bc_left.add_left(u,t)
    u = bc_right.add_right(u,t)
    return u, grid.x, t

# %% Plotting Module
def plot_3D_sol(u,x,t):
    """
    Plots a 3D surface plot of a solution to a PDE.
    Parameters:
        u (np array): solution to the PDE
        x (np array): grid points in space
        t (np array): grid points in time
    """
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    X, T = np.meshgrid(x, t)
    ax.plot_surface(X, T, u.T, cmap='viridis')
    ax.set_xlabel('x')
    ax.set_ylabel('t')
    ax.set_zlabel('u')
    plt.show()
    return
