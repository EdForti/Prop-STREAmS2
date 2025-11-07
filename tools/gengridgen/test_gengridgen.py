## Test of gengridgen function

import numpy as np
import matplotlib.pyplot as plt
import gengridgen as ggg

# Grid for a jet of diameter D, with NpD points per diameter at Mach number M_j, where we want at least 4 points per wavelength for St_max in the far field. 

D          = 0.018
NpD        = 166
M_j        = 1.15294
St_max     = 1.55 
dxmax      = 1/(8*M_j*St_max*np.sqrt(2))

# Test 1: tanh: in a space equal to Delta, centered around x_b, the grid covers alpha times the grid range from deltax_0 to deltax_end

Lx         = 20*D
s_max      = (1+0.1)*0.2*0.6*38*D
alpha      = 0.9999
Delta      = 3*D
x_b        = (D+s_max+Delta)/2
deltax_0   = D/NpD
deltax_end = dxmax*D

deltax_func = lambda x: (deltax_0 + 0.5*(deltax_end - deltax_0)*(1+np.tanh(2*np.arctanh(alpha)*(x-x_b)/Delta)))

plt.rc('text', usetex=True)
plt.rc('font', family='serif')
plt.plot(np.linspace(0,Lx,1000), deltax_func(np.linspace(0,Lx,1000)), color='black', label='reference')
plt.xlabel(r'$x$')
plt.ylabel(r'$\Delta x$')

csi, x, nx, L = ggg.gengridgen(deltax_func, mode='L2N', L=Lx, edge1=100, edge2=10000)
print('Final nx = '+str(nx))
plt.plot(x[0:-1], np.diff(x), color='red', label=r'$N_x = {}$'.format(nx))

csi, x, nx, L = ggg.gengridgen(deltax_func, mode='N2L', nx=nx, edge1=0.8*L, edge2=1.2*L)
print('Final L = '+str(L))

plt.plot(x[0:-1], np.diff(x), color='blue', label=r'$L = {}$'.format(L))

plt.grid()
plt.axis('tight')
plt.legend()
plt.show()

# Test 2: softplus reverse + softplus
# The softplus function is a continuous approximation of the discontinuous activation function.
# deltax_e is the grid spacing at the activation point x_e, alpha is the increase wrt the -infty value of the spacing
# deltax_ratio is the ratio between the spacing Lx after the activation and the spacing before the activation point

x_e          = 14.81*D   # Activation point
Lx           = 40*D      # Length after activation point where deltax = deltax_ratio * deltax_e
alpha        = 0.01      # deltax increase from -inf to 0 of alpha*deltax_e
deltax_e     = (1.3*D/NpD)*(1+alpha) # deltax at activation point
deltax_ratio = 5         # deltax after Lx from activation / deltax at activation
expon        = np.log(2)*((1+alpha)*deltax_ratio-1)/alpha
b            = np.log(np.exp(expon)-1)/Lx
deltax_func  = lambda x: (deltax_e/(1+alpha) * (1 + alpha/np.log(2)*np.log(1+np.exp(b*(x-x_e))))) \
    + (deltax_e/(1+alpha) * (1 + alpha/np.log(2)*np.log(1+np.exp(b*(-x+x_e-2*D))))) - deltax_e; 

Lx  = x_e+Lx  # realmax = exp(709.7827128933840e+02) = Inf --> x could not exist!

plt.rc('text', usetex=True)
plt.rc('font', family='serif')
plt.plot(np.linspace(0,Lx,1000), deltax_func(np.linspace(0,Lx,1000)*1e6), color='black', label='reference')
plt.xlabel(r'$x$')
plt.ylabel(r'$\Delta x [\mu/s]$')

csi, x, nx, L = ggg.gengridgen(deltax_func, mode='L2N', L=Lx, edge1=100, edge2=10000)
print('Final nx = '+str(nx))
plt.plot(x[0:-1], np.diff(x), color='red', label=r'$N_x = {}$'.format(nx))

csi, x, nx, L = ggg.gengridgen(deltax_func, mode='N2L', nx=nx, edge1=0.8*L, edge2=1.2*L, write=True)
print('Final L = '+str(L))

plt.plot(x[0:-1], np.diff(x)*1e6, color='blue', label=r'$L = {}$'.format(L))

plt.grid()
plt.axis('tight')
plt.legend()
plt.show()
