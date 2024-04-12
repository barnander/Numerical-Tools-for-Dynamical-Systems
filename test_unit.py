#Unit Test 
import unittest
import solvers
import numpy as np
import math

class Test_Solve_To(unittest.TestCase):
    def test_1D(self):
        #test for single space variable and no ode params
        def f(x,t,p):
            return np.array([x[0]])
        x0 = np.array([1]) 
        t0 = 0
        t_f = 1
        h = 0.01
        x,t = solvers.solve_to(f,np.nan,x0,t0,t_f,h)
        self.assertAlmostEqual(x[0,-1],math.e)
        self.assertEqual(np.shape(x),(1,101))
        self.assertEqual(np.shape(t),(101,))
    def test_2D(self):
        #test for multiple space vars and ode params
        def f2(x,t,p):
            x_dot = p[0] * x[1]
            y_dot = -p[1] * x[0]
            return np.array([x_dot,y_dot])
        x0 = np.array([0,1])
        t0 = 0
        t_f = 100
        p = np.array([1,2])
        h = 0.01
        x,t = solvers.solve_to(f2,p,x0,t0,t_f,h)
        #find analytical sols
        anal_x = 1/math.sqrt(2) * math.sin(t_f * math.sqrt(2))
        anal_y = math.cos(math.sqrt(2) * t_f)
        self.assertAlmostEqual(x[0,-1],anal_x)
        self.assertAlmostEqual(x[1,-1],anal_y)

class Test_Poisson_Solve(unittest.TestCase):   
    def lin_DD(self):
        #test Poisson solver for linear source term, Dirichlet BCs and all solvers
        a = 0
        b = 1
        alpha = -1.5
        beta = 3
        D = 1
        p=1
        N = 100
        q = lambda x,p: np.zeros(len(x)) + p
        bc_left = solvers.Boundary_Condition("Dirichlet",a,alpha)
        bc_right = solvers.Boundary_Condition("Dirichlet",b,beta)
        solver = 'solve'
        analytical = lambda x: -1/(2*D) * (x-a)*(x-b) + (beta - alpha)/(b-a) * (x-a) + alpha
        u, x = solvers.poisson_solve(bc_left,bc_right,N,q,p,solver = solver)
        u_anal = analytical(x)
        self.assertAlmostEqual(np.linalg.norm(u-u_anal),0)
        #test Poisson solver for scipy root solver
        solver = 'root'
        u, x = solvers.poisson_solve(bc_left,bc_right,N,q,p,solver = solver)
        self.assertAlmostEqual(np.linalg.norm(u-u_anal),0)
        #test thomas solver
        solver = 'thomas'
        u, x = solvers.poisson_solve(bc_left,bc_right,N,q,p,solver = solver)
        self.assertAlmostEqual(np.linalg.norm(u-u_anal),0)
    def lin_Neumann(self):
        #test Poisson solver for linear source term, Neumann BCs and all solvers
        a = 0
        b = 1
        alpha = -1.5
        beta = 3
        delta = 4.999900000000057
        gamma = 4.000099999999392
        D = 1
        p=1
        N = 1000
        q = lambda x,p: np.zeros(len(x)) + p
        bc_left = solvers.Boundary_Condition("Dirichlet",a,alpha)
        bc_right = solvers.Boundary_Condition("Neumann",b,gamma)
        solver = 'thomas'
        analytical = lambda x: -1/(2*D) * (x-a)*(x-b) + (beta - alpha)/(b-a) * (x-a) + alpha
        u, x = solvers.poisson_solve(bc_left,bc_right,N,q,p,solver = solver) 
        u_anal = analytical(x)
        self.assertAlmostEqual(np.linalg.norm(u-u_anal),0)

        #test left Neumann BC
        bc_left = solvers.Boundary_Condition("Neumann",a,delta)
        bc_right = solvers.Boundary_Condition("Dirichlet",b,beta)
        u, x = solvers.poisson_solve(bc_left,bc_right,N,q,p,solver = solver)
        self.assertAlmostEqual(np.linalg.norm(u-u_anal),0)

    def test_Poisson_Solve_nonlin(self):
        #test Poisson solver for non-linear source term, Dirichlet BCs and numpy linalg solver
        q = lambda u,x,p : np.exp(p * u)
        a,b = 0,1
        alpha,beta = 0,0
        p = 0.1
        q = lambda u,x,p : np.exp(p * u)
        dq_du = lambda u,x,p : p * np.exp(p * u)
        innit_guess = np.ones(N+1)
        bc_left = solvers.Boundary_Condition("Dirichlet",a,alpha)
        bc_right = solvers.Boundary_Condition("Dirichlet",b,beta)
        solver = 'solve'
        u, x = solvers.poisson_solve(bc_left,bc_right,N,q, p,u_innit=innit_guess, dq_du=dq_du, tol = 1e-10, solver = solver)
        self.assertAlmostEqual(np.linalg.norm(u-u_anal),0)

        #test Poisson solver for non-linear source term, Neumann right BC and scipy root solver
        gamma = (u[-1]-u[-3])/(x[-1]-x[-3])
        bc_right = solvers.Boundary_Condition("Neumann",b,gamma)
        solver = 'root'
        u, x = solvers.poisson_solve(bc_left,bc_right,N,q, p,u_innit=innit_guess, dq_du=dq_du, tol = 1e-10, solver = solver)
        self.assertAlmostEqual(np.linalg.norm(u-u_anal),0)
        #test Poisson solver for non-linear source term, Neumann left BC and thomas solver
        bc_right = solvers.Boundary_Condition("Dirichlet",b,beta)
        gamma = (u[2]-u[0])/(x[2]-x[0])
        bc_left = solvers.Boundary_Condition("Neumann",a,gamma)
        solver = 'thomas'
        u, x = solvers.poisson_solve(bc_left,bc_right,N,q, p,u_innit=innit_guess, dq_du=dq_du, tol = 1e-10, solver = solver)

if __name__ == '__main__':
    unittest.main()