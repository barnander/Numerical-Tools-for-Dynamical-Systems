#Unit Test 
import unittest
import solvers
import numpy as np
import math

class TestAddFunction(unittest.TestCase):
    def test_solve_to(self):
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
    
    def test_finite_diff(self):
        #test Poisson solver for linear source term
        a = 1
        b = 10
        alpha = 48
        beta = 1
        bc_left = solvers.Boundary_Condition("Dirichlet",a,alpha)
        bc_right = solvers.Boundary_Condition("Dirichlet",b,beta)
        D = 2
        N = 1000
        f = lambda x: x
        analytical = lambda x: (-a**3*b + a*b**3 + 6*a*D - 6*b*D*alpha-x**3*(a-b) + x*(a**3 - b**3 + 6*D*alpha - 6*D)) / (6*D*(a-b))
        u, x = solvers.Poisson_Solve(bc_left,bc_right,N,f,D)
        self.assertAlmostEqual(np.linalg.norm(u-analytical(x)),0)
if __name__ == '__main__':
    unittest.main()