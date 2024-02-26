import matplotlib.pyplot as plt
import numpy as np

# constant parameters
Q = 4e-7 #blood flow rate is 3.0~26 ml/min in arteries
r_0 = 1.8e-3 #unstressed tube radius
A_0 = np.pi * (r_0**2)
G_0 = 21.2e3 #tube wall elasticity coefficient
rho = 1050 #average blood density kg/m3
nu = 3.2e-6 #blood kinematic viscosity m2/s
alpha = 4/3
p_0 = 14665.5 #inlet blood pressure Pa
l1norm_target = 10e-20
l1norm = 1
# spatial discretization setting
x_0=0
x_N= 0.05 #The sample problem is a single 50mm vessel segment
number_of_nodes=5 # number of grid points
x_coord=np.linspace(x_0,x_N,number_of_nodes)
dx=(x_N-x_0)/(number_of_nodes-1)
#######################################################################################################################
A = x_coord=np.linspace(A_0,5e-6,number_of_nodes)
while l1norm > l1norm_target:
    An = A.copy()
    # unknown vector
    p=np.zeros(number_of_nodes)
    # stiffness matrix
    K=np.zeros([number_of_nodes,number_of_nodes])
    for i in range(2, number_of_nodes):
        K[i-1,i-1] = (An[i-1] - An[i]) / (2*rho)
    for i in range(0, number_of_nodes-1):
        K[i+1,i] = (- An[i]) / (2*rho)
    for i in range(1, number_of_nodes-1):
        K[i,i+1] =  An[i]
    K[0,0] = 1
    K[-1,-1] = An[-2] / (2*rho)
    # right hand side vector
    f=np.zeros(number_of_nodes)
    f[0] = p_0
    for i in range(1, number_of_nodes-1):
        f[i] = ((-4 * np.pi * nu * Q * dx) * ((1/An[i-1]) + (1/An[i]))) + ((alpha*(Q**2)) * ((1/An[i-1])-(1/An[i])))
    f[-1] = (-4 * np.pi * nu * Q * dx) / An[-2]
    p=np.linalg.solve(K,f)
    A = (A_0) * (1 + (p/G_0))
    l1norm = np.linalg.norm(A-An)
    #print(p)
    #print(A)
print(A)

plt.plot(x_coord, A)