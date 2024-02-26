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
number_of_nodes=11 # number of grid points
x_coord=np.linspace(x_0,x_N,number_of_nodes)
dx=(x_N-x_0)/(number_of_nodes-1)
###########################################################################################################################
A = x_coord=np.linspace(A_0,5e-6,number_of_nodes)
while l1norm > l1norm_target:
    An = A.copy()
    # unknown vector
    p=np.zeros(number_of_nodes)
    # stiffness matrix
    K=np.zeros([number_of_nodes,number_of_nodes])
    # Fill the diagonal
    K[np.arange(number_of_nodes)[1:-1],np.arange(number_of_nodes)[1:-1]]=(np.array(An[:-2]) - np.array(An[1:-1])) / (2*rho)
    # Fill the off-diagonals - lower
    K[np.arange(number_of_nodes)[1:],np.arange(number_of_nodes)[:-1]]=(-np.array(An[:-1])) / (2*rho)
    # Fill the off-diagonals - upper
    K[np.arange(number_of_nodes)[1:-1],np.arange(number_of_nodes)[2:]]=(np.array(An[2:])) / (2*rho)
    # clean boundary terms
    K[0,0] = 1
    K[-1,-1] = An[-2] / (2*rho)
    # right hand side vector
    f=np.zeros(number_of_nodes)
    f[1:-1] = ((-4 * np.pi * nu * Q * dx) * (np.reciprocal(An[:-2],dtype = float) + np.reciprocal(An[1:-1],dtype = float))) + ((alpha*(Q**2)) * (np.reciprocal(An[:-2],dtype = float) - np.reciprocal(An[1:-1],dtype = float)))
    # clean boundary terms
    f[0] = p_0
    f[-1] = (-4 * np.pi * nu * Q * dx) / An[-2]
    p=np.linalg.solve(K,f)
    A = (A_0) * (1 + (p/G_0))
    l1norm = np.linalg.norm(A-An)
    print(l1norm)
    #print(p)
    #print(A)
print(A)
plt.plot(x_coord, A)