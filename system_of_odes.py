#week_14 system of odes
import numpy as np
import solvers
def f_second_order(t,x):
    x_dot = x[1]
    y_dot = -x[0]
    return np.array([x_dot,y_dot])


x0 = 0
y0 = 1
initial_conds = {'t':0,'x':np.array([x0,y0])}
sol = solvers.solve_to(f_second_order,initial_conds,1,0.01)
