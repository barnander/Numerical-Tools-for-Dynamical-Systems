#%%
import numpy as np
import solvers
import matplotlib.pyplot as plt
from numpy import sin, cos, sqrt
#%%
a = 0
b = 1
alpha = -1.5
beta = 3
D = 1
N = 10000
f = lambda x,p: np.ones(len(x))
bc_left = solvers.Boundary_Condition("Dirichlet",a,alpha)
bc_right = solvers.Boundary_Condition("Neumann",b,-4.5)
solver = 'thomas'
analytical = lambda x,p: -1/(2*D) * (x-a)*(x-b) + (beta - alpha)/(b-a) * (x-a) + alpha
u, x = solvers.poisson_solve(bc_left,bc_right,N,f,np.nan,solver = solver)
u_anal = analytical(x,0)
#%%
a = 1
b = 10
alpha = 48
beta = 1
bc_left = solvers.Boundary_Condition("Dirichlet",a,alpha)
bc_right = solvers.Boundary_Condition("Dirichlet",b,beta)
D = 2
N = 1000
f = lambda x,p: x
analytical = lambda x: (-a**3*b + a*b**3 + 6*a*D - 6*b*D*alpha-x**3*(a-b) + x*(a**3 - b**3 + 6*D*alpha - 6*D)) / (6*D*(a-b))
u, x = solvers.poisson_solve(bc_left,bc_right,N,f,np.nan)
u_anal = analytical(x)

# %% Test non-linear solver
a,b = 0,1
alpha,beta = 0,0
D = 1
N = 100
p = 0.1
q = lambda u,x,p : np.exp(p * u)
dq_du = lambda u,x,p : p * np.exp(p * u)
innit_guess = np.ones(N+1)
bc_left = solvers.Boundary_Condition("Dirichlet",a,alpha)
bc_right = solvers.Boundary_Condition("Dirichlet",b,beta)
linear = False
solver = 'thomas'
u, x = solvers.poisson_solve(bc_left,bc_right,N,q, p, D=D,linear=linear,u_innit=innit_guess, dq_du=dq_du, tol = 10-10, solver = solver)

analytical = lambda x: -1/(2*D) * (x-a)*(x-b) + (beta - alpha)/(b-a) * (x-a) + alpha
u_anal = analytical(x)
# %%
plt.scatter(x,u)
plt.plot(x,u_anal)
plt.show()
# %%
