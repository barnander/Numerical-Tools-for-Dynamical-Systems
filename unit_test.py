#Unit Test 
import unittest
import solvers
import numpy as np
import math

class TestAddFunction(unittest.TestCase):
    def test_solve_to(self):
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
if __name__ == '__main__':
    unittest.main()