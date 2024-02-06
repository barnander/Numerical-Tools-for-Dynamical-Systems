#week_14 system of odes
import numpy as np
import matplotlib.pyplot as plt
import solvers

def f_second_order(t,x):
    x_dot = x[1]
    y_dot = -x[0]
    return np.array([x_dot,y_dot])


x0 = 0
y0 = 1.5
initial_conds = {'t':0,'x':np.array([x0,y0])}
t_final = 100
h = 0.1
time,x = solvers.solve_to(f_second_order,initial_conds,t_final,h,'RK4')

plt.plot(time,x[0])