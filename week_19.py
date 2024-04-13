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
p = 1
N = 10000
f = lambda x,p: np.zeros(len(x)) + p
bc_left = solvers.Boundary_Condition("Dirichlet",a,alpha)
bc_right = solvers.Boundary_Condition("Dirichlet",b,beta)
solver = 'thomas'
analytical = lambda x,p: -1/(2*D) * (x-a)*(x-b) + (beta - alpha)/(b-a) * (x-a) + alpha
u, x = solvers.poisson_solve(bc_left,bc_right,f,p,N,solver = solver)
u_anal = analytical(x,0)
np.linalg.norm(u-u_anal)
delta = (u_anal[2]-u_anal[0])/ (x[2]-x[0])
gamma = (u_anal[-1]-u_anal[-3])/(x[-1]-x[-3])

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
u, x = solvers.poisson_solve(bc_left,bc_right,f,np.nan,N)
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
solver = 'thomas'
u, x = solvers.poisson_solve(bc_left,bc_right,q, p, N, D=D,u_innit=innit_guess, dq_du = dq_du,tol = 10-10, solver = solver)

analytical = lambda x: -1/(2*D) * (x-a)*(x-b) + (beta - alpha)/(b-a) * (x-a) + alpha
u_anal = analytical(x)
# %%
plt.scatter(x,u)
plt.plot(x,u_anal)
plt.show()

# %%
