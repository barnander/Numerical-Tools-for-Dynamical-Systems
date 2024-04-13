#%% Packages
import numpy as np
import scipy.optimize as opt
import inspect
import matplotlib.pyplot as plt
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

    #initialise t and x arrays
    t = discr_t(t0,t_f,delta_max)

    x = np.zeros((len(x0),len(t)))
    x[:,0] = x0
    x_n = x0
    h = delta_max
    for i,t_n in enumerate(t[:-2]):
        #iterate through functions, computing the next value for each state variable
        x_n = solve_step(f,x_n,t_n,h)
        x[:,i+1] = x_n
    h = t_f - t[-2] #adapts the final h to give the solution at exactly t_final
    x[:,-1] = solve_step(f,x_n,t[-2],h)
    return x,t

def discr_t(t0,t_f,delta_max):
    t = np.arange(t0,t_f,delta_max) 
    t[0] = t0
    t = np.append(t,t_f)
    return t


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
        p0 = np.array([p0], dtype = float)
        pend = np.array([pend], dtype = float) if pend is not None else None
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
    return x_n_plus_1

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
    return x_n_plus_1



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
        x (np array): solved initial condition to the Limit Cycle
        T (float): period of the Limit Cycle
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
        x,_ = solve_to(f_gen,p,x0,0,T,delta_max,solver)
        xf = x[:,-1]
        BC = xf-x0
        PC = pc(x0,T,p)
        root_solve = np.append(BC,PC)
        return root_solve

    
    #run scipy newton root-finder on g with initial guess
    x_T_solved = opt.fsolve(g,init_guess,xtol=delta_max*1.1) #make sure rootfinder tol is higher than integrator tol to avoid numerical issues
    x = x_T_solved[:-1]
    T = x_T_solved[-1]
    return x, T



def natural_p_cont(ode, p0, pend, x_T0, delta_max = 1e-3, n = 100, LC = True):
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
        x_T (np array): array of equilibrium points or points on the limit cycle (and period for LCs) for each parameter value
    
    """
    #check that p0 and pend are the same type, length and that only one parameter changes:
    p0,pend = param_assert(p0,pend)
    #initialise parameter array and solution array
    ps = np.linspace(p0,pend,n)
    x_T = np.tile(np.nan,(np.size(x_T0),n))

    #define root finder (depending on wether we're looking for LCs or not)
    if LC:
        solve_func = lambda x_T,p: shoot_solve(ode,p,x_T,delta_max)
    else:
        solve_func = lambda x_T,p: opt.fsolve(ode,x_T,args =(np.nan,p))
    #iterate through parameters
    for i,p in enumerate(ps):
        #find equilibria/LCs
        x_Ti = solve_func(x_T0,p)
        #add result to results array
        x_T[:,i] = x_Ti
        #update initial guess for next iteration
        x_T0 = x_Ti
    #add finall value 
    x_T[:,-1] = x_Ti
    return x_T,ps.transpose()

def pseudo_arc_step(ode, x_T0, p0, x_T1, p1,p_ind, LC = False):
    """
    Performs one step of pseudo-arclength continuation on system of ODEs.
    Parameters:
        ode (function): function of x (np array), t (float), and p (np array, integer or float) describing system of odes
        x_T0 (np array): value of fixed point of the system (include initial period guess for LCs as last element of the array) at step i-1
        p0 (np array): parameter value(s) at step i-1
        x_T1 (np array): value of fixed point of the system (include initial period guess for LCs as last element of the array) at step i
        p1 (np array): parameter value(s) at step i
        p_ind (int): index of the parameter that changes
        LC (bool): if False, only equilibrium solutions are computed (not Limit Cycles)
    Returns:
        x_T2 (np array): value of fixed point of the system (includes initial period guess for LCs as last element of the array) at step i+1
        p2 (np array): parameter value(s) at step i+1
    """
    #TODO: include in same func as natural param cont    
    #TODO make it work with multiple parameters and with LC finder

    #form augmented state vectors
    v0 = np.append(p0[p_ind],x_T0)
    v1 = np.append(p1[p_ind],x_T1)

    #find the delta and predict v2
    delta = v1 - v0
    v_pred = v1 + delta
    #define function to solve
    if LC:
        #TODO: add LC solver
        basil = 5
    else:
        def root_solve(v2):
            p2 = p1.copy()
            p2[p_ind] = v2[0]
            pseudo_cond = np.dot(v2 - v_pred, delta)
            eqm_cond = ode(v2[1:], np.nan, p2)
            residuals = np.append(pseudo_cond,eqm_cond)
            return residuals
    
    #find v2 using fsolve
    v2 = opt.fsolve(root_solve,v1)

    #reform variable and parameter arrays
    x_T2 = v2[1:]
    p2 = p1.copy()
    p2[p_ind] = v2[0]
    return x_T2,p2



def pseudo_arc(ode,x_T0,p0,pend,p_ind,max_it = 1e3 ,innit_h= 1e-3):
    """
    Performs pseudo-arclength continuation on system of ODEs.
    Parameters:
        ode (function): function of x (np array), t (float), and p (np array, integer or float) describing system of odes
        x_T0 (np array): initial guess of value of fixed point of the system (include initial period guess for LCs as last element of the array)
        p0 (np array): initial parameter value(s)
        pend (np array): final parameter value(s)
        p_ind (int): index of the parameter that changes
        max_it (int): maximum number of iterations
        innit_h (float): initial step size
    """
    #TODO add LC solver
    #check that p0 and pend are the same type, length and that only one parameter changes:
    p0,pend = param_assert(p0,pend)
    assert np.count_nonzero(pend - p0) == 1, "only one parameter should change"

    #find X_T0
    x_T0 = opt.fsolve(lambda x: ode(x,np.nan,p0),x_T0)

    #find out wether we increase or decrease the parameter.
    direction = np.sign(pend[p_ind]-p0[p_ind])
    #do a step of natural parameter continuation to find v1
    p1 = p0.copy()
    p1[p_ind] = p0[p_ind] + direction * innit_h #making sure to take a step in the direction of pend
    x_T1 = opt.fsolve(lambda x: ode(x,np.nan,p1),x_T0)

    #initialise array of solutions
    x_T = np.tile(np.nan,(np.size(x_T0),int(max_it)+2))
    ps = np.tile(np.nan,(np.size(p0),int(max_it)+2))
    #and add values for v0 and v1
    x_T[:,0] = x_T0
    x_T[:,1] = x_T1
    ps[:,0] = p0
    ps[:,1] = p1
    for i in range(int(max_it)):
        #take a step of pseudo-arclength continuation and add to solutions array
        x_T2,p2 = pseudo_arc_step(ode,x_T0,p0,x_T1,p1,p_ind=p_ind)
        x_T[:,i+2] = x_T2
        ps[:,i+2] = p2

        #check if we have reached the end of the continuation
        if direction * p2[p_ind] > direction * pend[p_ind]:
            print(f"Reached end of continuation after {i} iterations")
            return x_T[:,:i+3], ps[:,:i+3]
        #update values for v0 and v1 for next iteration
        x_T0 = x_T1
        x_T1 = x_T2
        p0 = p1
        p1 = p2
    print("Max iterations reached")
    return x_T,ps

class Boundary_Condition():
    def __init__(self, BC_type, x, value):
        self.type = BC_type
        self.x = x

        if self.type == "Dirichlet":
            if callable(value):
                self.value = value
            elif isinstance(value, (int, float)):
                #TODO talk about how this isn't ideal but makes code more consice
                self.value = lambda t: value
            else:
                raise ValueError("The value for Dirichlet boundary condition must be a function or a number.")
            

        elif self.type == "Neumann":
            if callable(value):
                self.value = (value, lambda t: 0)
            elif isinstance(value, (int, float)):
                self.value = (lambda t: value, lambda t: 0)
            else:
                raise ValueError("The value for Neumann boundary condition must be a function or a number.")

        elif self.type == "Robin":
            if isinstance(value, tuple) and len(value) == 2:
                if (callable(value[0]) and callable(value[1])):

                    self.value = value
                elif (isinstance(value[0], (int, float)) and isinstance(value[1], (int, float))):

                    self.value = (lambda t: value[0], lambda t: value[1])
                else:
                    raise ValueError("The values for Robin boundary condition must be either two functions or two numbers.")
            else:
                raise ValueError("The value for Robin boundary condition must be a tuple of two functions or two numbers.")

        else:
            raise ValueError("Unsupported boundary condition type: {}".format(type(value)))


    #TODO try and actually implement this
    def add_left(self,u,t=np.array([None])):
        #Adds left BC values back to grid for Dirichlet BCs
        if self.type == "Dirichlet":
            if t.any():
                u_left = np.zeros(len(t)) + self.value(t)
                u = np.vstack((u_left,u))  
            else:
                u = np.append(self.value(np.nan),u) 
        return u
    
    def add_right(self,u,t=np.array([None])):
        #Adds left BC values back to grid for Dirichlet BCs
        if self.type == "Dirichlet":
            if t.any():
                u_right = np.zeros(len(t)) + self.value(t)
                u = np.vstack((u, u_right))  
            else:
                u = np.append(u, self.value(np.nan))  
        return u



class Grid():
    def __init__(self,N,a,b):
        self.N = N
        self.a = a
        self.b = b
        self.dx = (b-a)/N
        self.x = np.linspace(a,b,N+1)

def construct_A_diags_b(grid, bc_left, bc_right):
    """
    Constructs the diagonals of tridiagonal matrix A and the vector b for the Poisson equation given the grid and boundary conditions.
    Parameters:
        grid (Grid): grid object defining space discretisation
        bc_left (Boundary_Condition): left boundary condition
        bc_right (Boundary_Condition): right boundary condition
    Returns:
        A_sub (np array): subdiagonal of the matrix A
        A_diag_func (func): diagonal of the matrix A as a function of t
        A_sup (np array): superdiagonal of the matrix A
        b_func (func)): vector b in the system of equations Ax = b as a function of time
        left_ind (int/None): index of the leftmost grid point used in the solver
        right_ind (int/None): index of the rightmost grid point used in the solver
    """

    N = grid.N
    dx = grid.dx



    #Make general tridiagonal matrix A
    #superdiagonal
    A_sup = np.ones(N-2) 
    #subdiagonal
    A_sub = np.ones(N-2)
    #diagonal
    A_diag = -2*np.ones(N-1, dtype = object)

    #make general vector b
    b = np.zeros(N-1, dtype= object)

    #initialise left_ind and right_ind for Dirichlet boundary conditions
    left_ind = 1
    right_ind = -1


    if bc_left.type == "Dirichlet":
        b[0] = bc_left.value
    else:
        A_sup = np.append(2,A_sup)
        A_sub = np.append(1,A_sub)
        A_diag = np.append(lambda t: -2 * (1 - dx * bc_left.value[1](t)),A_diag)
        b = np.append(lambda t: -2 * dx * bc_left.value[0](t),b)
        left_ind = None

    if bc_right.type == "Dirichlet":
        b[-1] = bc_right.value
    else:
        A_sup = np.append(A_sup,1)
        A_sub = np.append(A_sub,2)
        A_diag = np.append(A_diag,lambda t: -2 * (1 + dx * bc_right.value[1](t)) )
        b = np.append(b,lambda t: 2 * dx * bc_right.value[0](t))
        right_ind = None

    def A_diag_func(t):
        A_diag_t = np.array([f(t) if callable(f) else f for f in A_diag],dtype = np.float64)
        return A_diag_t
    def b_func(t):
        b_t = np.array([f(t) if callable(f) else f for f in b],dtype = np.float64)
        return b_t
    

    #TODO note that this is not the most efficient way to do this, but it is the most general
    return A_sub, A_diag_func, A_sup, b_func, left_ind, right_ind

def reform_A(A_sub,A_diag,A_sup):
    return np.diag(A_sup,1) + np.diag(A_diag,0) + np.diag(A_sub,-1)

# Linear System solvers for Poisson equation using arrays of diagonals and b
def choose_lin_solve(solver_name):
    if solver_name == 'solve':
        return lin_solve_numpy
    elif solver_name == 'root':
        return lin_solve_scipy
    elif solver_name == 'thomas':
        return lin_solve_thomas
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




def poisson_solve(bc_left, bc_right,q, p, N, D=1, u_innit = np.array(None), v = 1, dq_du = None, max_iter = 100, tol = 1e-6, solver = 'solve'):
    """
    Solves the Poisson equation for a given grid, boundary conditions and source term.
    Parameters:
        bc_left (Boundary_Condition): left boundary condition
        bc_right (Boundary_Condition): right boundary condition
        N (int): number of grid points
        q (function): source term, function of x, (u) and p
        p (np array): parameter(s) of the source term
        D (float): diffusion coefficient
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
    #form grid
    grid = Grid(N,bc_left.x,bc_right.x)
    dx = grid.dx

    # find diagonals of matrix A and vector b given boundary conditions
    #set up variables left_ind and right_ind that determine the first and last grid values used in the solver
    A_sub, A_diag_func, A_sup, b_func, left_ind, right_ind = construct_A_diags_b(grid, bc_left, bc_right)
    A_diag,b = A_diag_func(np.nan), b_func(np.nan)
    #choose solver
    #TODO: add Newton's method
    lin_solve = choose_lin_solve(solver)
    
    #find number of arguments to q to determine if the source term is linear or non-linear
    n_args = len(inspect.signature(q).parameters)

    if n_args == 2:
        #solve linear system using chosen solver
        # define constant vector c
        c = -b - dx**2/D * q(grid.x[left_ind:right_ind],p)
        #solve linear system of the shape Au = c using chosen solver
        u = lin_solve(A_sub,A_diag,A_sup,c)

    elif n_args == 3:
        #solve non-linear system using Newton's method
        #initialise first guess for u
        if u_innit.any() == None:
            #default to zero vector
            u = np.zeros(len(b))
        else:
            u = u_innit[left_ind:right_ind]

        if dq_du is None:
            #finite difference approximation for dq_du
            eps = 1e-6
            dq_du = lambda u, x, p: (q(u + eps, x, p) - q(u - eps, x, p)) / (2 * eps)
        
        #define Jacobian of source term
        J_q = np.diag(dq_du(u,grid.x[left_ind:right_ind], p))
        #solve for u using Newton's method
        for i in range(max_iter):
            #form matrix A
            A = reform_A(A_sub,A_diag,A_sup)
            #compute discretised residual
            F = A@u + b + dx**2/D * q(u,grid.x[left_ind:right_ind], p)
            #define Jacobian of the system
            J_F = A + dx**2/D * J_q
            #extract diagonals of Jacobian
            J_sub, J_diag, J_sup = np.diag(J_F,-1), np.diag(J_F,0), np.diag(J_F,1)
            #solve for correction V using linear solver
            V = lin_solve(J_sub,J_diag,J_sup,-F)
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


def diffusion_solve(bc_left, bc_right, f,t0,t_f, q , p, N, D = 1, dt = None , explicit_solver = 'RK4' ,implicit_solver = False):
    #discretise in space
    grid = Grid(N,bc_left.x,bc_right.x)
    dx = grid.dx

    #find diagonals of matrix A and vector b given boundary conditions
    A_sub, A_diag_t, A_sup, b_t, left_ind, right_ind = construct_A_diags_b(grid, bc_left, bc_right)
    
    #set up u0
    u0 = f(grid.x[left_ind:right_ind],t0)

    if explicit_solver:
        #define du_dt as a function of u and t
        def du_dt(u,t,p):
            A = reform_A(A_sup,A_diag_t(t),A_sub)
            b = b_t(t)
            return D/dx**2 * (A @ u + b) + q(u,grid.x[left_ind:right_ind],t,p)
        #ensure stablilty of the time integration
        dt_stable = dx**2/(2*D)
        if not dt:
            #default value of delta_t to ensure stability
            dt = dt_stable
        elif dt > dt_stable:
            raise ValueError('dt must be smaller or equal to dx^2/(2*D) for explicit Euler method (where dx is granularity of the grid in space)')
        #solve using one-step solver
        u, t = solve_to(du_dt,p,u0,t0,t_f,dt,solver = explicit_solver)

    else:
        #choose solver
        lin_solve = choose_lin_solve(implicit_solver)
        if not dt:
            dt = dx**2/(2*D) #choose default dt value ()
        C = dt * D/dx**2
        t = discr_t(t0,t_f,dt)

        #initialise solution array
        u = np.zeros((len(u0),len(t)))
        u[:,0] = u0
        u_n = u0
        for i,t_n in enumerate(t[:-1]):
            #construct diagonals of matrix M = I-CA at time t_n
            M_diag = 1 - C * A_diag_t(t_n)
            M_sub = -C * A_sub
            M_sup = -C * A_sup

            #construct vector d = u + C*b + dt *q at time t_n
            d = u_n + C*b_t(t_n)+ dt *q(u,grid.x[left_ind:right_ind],t,p)

            #solve for u_n+1 
            u_n_plus_1 = lin_solve(M_sub,M_diag,M_sup,d)
            u[:,i+1] = u_n_plus_1
            u_n = u_n_plus_1


    #add boundary conditions to solution for Dirichlet boundary conditions
    u = bc_left.add_left(u,t)
    u = bc_right.add_right(u,t)
    return u, grid.x, t












# %% Plotting Modules
def plot_3D_sol(u,x,t):
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    X, T = np.meshgrid(x, t)
    ax.plot_surface(X, T, u.T, cmap='viridis')
    ax.set_xlabel('x')
    ax.set_ylabel('t')
    ax.set_zlabel('u')
    plt.show()
