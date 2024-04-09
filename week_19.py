#%%
import numpy as np
import solvers
import matplotlib.pyplot as plt
from numpy import sin, cos, sqrt
#%%
a = 0
b = 10
alpha = 0
beta = -10
D = 5
N = 1000
f = lambda x: np.ones(len(x))
bc_left = solvers.Boundary_Condition("Dirichlet",a,alpha)
bc_right = solvers.Boundary_Condition("Dirichlet",b,beta)
analytical = lambda x: -1/(2*D) * (x-a)*(x-b) + (beta - alpha)/(b-a) * (x-a) + alpha
u, x = solvers.Poisson_Solve(bc_left,bc_right,N,f,D)
#%%
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


# %%
plt.scatter(x,u)
plt.plot(x,analytical(x))
plt.show()
# %%
