## Test of gengridgen function

import numpy as np
import matplotlib.pyplot as plt
import gengridgen as ggg

# Grid for a jet of diameter D, with NpD points per diameter at Mach number M_j, where we want at least 4 points per wavelength for St_max in the far field. 

D          = 0.00458
NpD        = 100
M_j        = 0.186
St_max     = 0.3 
dxmax      = 1/(8*M_j*St_max*np.sqrt(2))

# The softplus function is a continuous approximation of the discontinuous activation function.
# deltax_e is the grid spacing at the activation point x_e, alpha is the increase wrt the -infty value of the spacing
# deltax_ratio is the ratio between the spacing Lx after the activation and the spacing before the activation point

x_e          = 28*D       # Activation point
Lx           = 12*D     # Length after activation point where deltax = deltax_ratio * deltax_e
alpha        = 0.3        # deltax increase from -inf to 0 of alpha*deltax_e
#deltax_e     = (1.3*D/NpD)*(1+alpha) # deltax at activation point
deltax_e     = 93e-6     # deltax at activation point
deltax_ratio = 30         # deltax after Lx from activation / deltax at activation
expon        = np.log(2)*((1+alpha)*deltax_ratio-1)/alpha
b            = (np.log(np.exp(expon)-1)/Lx)/2

deltax_func  = lambda x: (deltax_e/(1+alpha) * (1 + alpha/np.log(2)*np.log(1+np.exp(b*(x-x_e))))) \
    + (deltax_e/(1+alpha) * (1 + alpha/np.log(2)*np.log(1+np.exp(b*(-x+x_e-16*D))))) - deltax_e;

Lx  = x_e+Lx  # realmax = exp(709.7827128933840e+02) = Inf --> x could not exist!

print(min(deltax_func(np.linspace(0,Lx,1000)))*1e6,' micron')
print(max(deltax_func(np.linspace(0,Lx,1000)))*1e6,' micron')

plt.rc('text', usetex=True)
plt.rc('font', family='serif')
plt.plot(np.linspace(0,Lx,1000)/D, deltax_func(np.linspace(0,Lx,1000))*1e6, color='black', label='reference')
plt.axvline(12*D/D, color='blue')
plt.axvline(20*D/D, color='red')
plt.axvline(28*D/D, color='blue')
plt.xlabel(r'$x$')
plt.ylabel(r'$\Delta x [\mu m]$')
plt.show()

csi, x, nx, L = ggg.gengridgen(deltax_func, mode='L2N', L=Lx, edge1=100, edge2=10000)
print('Final nx = '+str(nx))
plt.plot(x[0:-1], np.diff(x), color='red', label=r'$N_x = {}$'.format(nx))

csi, x, nx, L = ggg.gengridgen(deltax_func, mode='N2L', nx=nx, edge1=0.9*L, edge2=1.3*L, write=True)
print('Final L = '+str(L))

plt.plot(x[0:-1], np.diff(x), color='blue', label=r'$L = {}$'.format(L))

plt.grid()
plt.axis('tight')
plt.legend()
plt.show()
