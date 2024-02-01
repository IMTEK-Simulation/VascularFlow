import numpy as np
import matplotlib.pyplot as plt
import timeit
import time
# parameters
dynamic_viscosity = 1
density = 1000
l0 = - 1.5
l = 1.5
nx = 101
dx = (l - l0) / (nx - 1)
l1norm_target = 10e-4
l1norm = 1
l2norm_target = 10e-4
l2norm = 1
x_range = np.linspace(l0, l, nx)
u = np.zeros((nx))
p = np.zeros((nx))
#velocity calculation
while l1norm > l1norm_target:
    un = u.copy()
    a = ((2*np.pi)/3) * np.sin(((2*np.pi)/3) * x_range)
    b = 1 - 0.5 * np.cos(((2*np.pi)/3) * x_range)
    u[1:] = ((b/dx)/(a+(b/dx)))[1:] * un[0:-1]
    u[0] = 0.15
    l1norm = np.linalg.norm(u-un)
    #l1norm = (np.sum(np.abs(u[:]) - np.abs(un[:])) / np.sum(np.abs(un[:])))
#print(u[:])
plt.plot(x_range, u)
#%timeit u[:]

#pressure calculation
while l2norm > l2norm_target:
    pn = p.copy()
    c = (((4*np.pi)/9) * np.sin(((2*np.pi)/3) * x_range)) / (2 - np.cos(((2*np.pi)/3) * x_range))
    d = ((dynamic_viscosity/density)*8) / ((2 - np.cos(((2*np.pi)/3) * x_range))) **2
    p[:-1] = pn[1:] + (density * dx) * (((5/(3*dx)) * u[:-1] * (u[1:]-u[:-1])) + c[:-1] * u[:-1]**2 + d[:-1] * u[:-1])
    p[-1] = 0
    l2norm = np.linalg.norm(p-pn)
#print(p[:])
#plt.plot(x_range, p)
#%timeit p[:]