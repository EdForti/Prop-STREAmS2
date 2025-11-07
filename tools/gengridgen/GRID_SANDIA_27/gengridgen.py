def gengridgen(fx, mode='L2N', L=None, nx=None, edge1=None, edge2=None, write=False):
    """
    Evaluate the grid distribution on the basis of a given analytical function.
    If L2N: from a given length L, provide the necessary number of points to obtain the given distribution
    If N2L: from a given Nx, provide the minimum length with the given grid distribution.
           As bisection method is used, two starting points must be provided as the larger L2 value
           may induce an unfeasible number of points for exponentially growing grid distributions

    Parameters:
    fx     = (function) function for grid distribution (no default)
    mode   = (string)   'L2N'/'N2L' (default='L2N')
    L      = (float)    given length of the domain (default:None)
    nx     = (integer)  given number of pointsù (default: None)
    edge1, edge2 = (integer/float)    edge values defining range for bisec method to find nx (L2N, integer) or L ('N2L', float)
    write  = (logical)  write x.dat file flag (default: False)
    
    Returns:
    csi   = (float ndarray) non-uniform logical grid
    x     = (float ndarray) final target physical grid
    nx    = (integer)       final number of points
    L     = (flaot)         final length of the domain
    """
    
    import numpy as np
    from scipy.interpolate import pchip_interpolate

    deltax_0 = fx(0.)
    nx_fine = 1000001
    csi_fine = np.zeros(nx_fine)

    if (mode=='L2N'):
        x_fine = np.linspace(0,L,nx_fine)
        dx = x_fine[1] - x_fine[0]
        x_fine = np.append(x_fine, x_fine[-1]+dx)
        for i in range(1, nx_fine):
            xx = 0.5*(x_fine[i]+x_fine[i+1])
            fxx = fx(xx)
            csi_fine[i] = csi_fine[i-1] + dx/fxx #integral(intfun, x_fine(i), x_fine(i+1))
        csi_fine = csi_fine/csi_fine[-1]

        # Find zero of function = error(nx)
        nx1 = edge1 
        nx2 = edge2 

        csi = np.linspace(0,1,nx1) # ...here up to 1!
        x   = pchip_interpolate(csi_fine, x_fine[0:nx_fine], csi)/csi_fine[-1]
        dx0 = x[1]-x[0]
        err_nx1 = dx0 - deltax_0

        csi = np.linspace(0,1,nx2) # ...here up to 1!
        x   = pchip_interpolate(csi_fine, x_fine[0:nx_fine], csi)/csi_fine[-1]
        dx0 = x[1]-x[0]
        err_nx2 = dx0 - deltax_0

        if (err_nx1*err_nx2 > 0):
            print('Solution is not between nx1 and nx2.') 
            return None, None, None, None

        nx  = round((nx1 + nx2)/2)

        niter = 0

        while (niter < 100):

            print('nx = ' + str(nx))
            nx_old = nx 
            csi = np.linspace(0,1,nx) # ...here up to 1!
            x   = pchip_interpolate(csi_fine, x_fine[0:nx_fine], csi)/csi_fine[-1]

            dx0 = x[1]-x[0]

            err_nx = dx0 - deltax_0
            niter = niter + 1

            # Define nx new
            if (err_nx*err_nx1>0):
                nx1 = nx
                err_nx1 = err_nx 
            else:
                nx2 = nx
                err_nx2 = err_nx
                
            nx = round((nx1 + nx2)/2)

            if (abs(nx1-nx2) == 1):
                print('abs(nx1-nx2) = 1')
                if abs(err_nx1) > abs(err_nx2):
                    nx = nx2
                else:
                    nx = nx1
                break

    elif (mode=='N2L'):

        csi = np.linspace(0,1,nx)
        L1 = edge1
        L2 = edge2

        # Find zero of function = error(nx)

        x_fine = np.linspace(0,L1,nx_fine)
        dx = x_fine[1] - x_fine[0]
        x_fine = np.append(x_fine, x_fine[-1]+dx)
        for i in range(1, nx_fine):
            xx = 0.5*(x_fine[i]+x_fine[i+1])
            fxx = fx(xx)
            csi_fine[i] = csi_fine[i-1] + dx/fxx #integral(intfun, x_fine(i), x_fine(i+1))
        csi_fine = csi_fine/csi_fine[-1]
        x   = pchip_interpolate(csi_fine, x_fine[0:nx_fine], csi)/csi_fine[-1]
        dx0 = x[1]-x[0]
        err_L1 = dx0 - deltax_0

        x_fine = np.linspace(0,L2,nx_fine)
        dx = x_fine[1] - x_fine[0]
        x_fine = np.append(x_fine, x_fine[-1]+dx)
        for i in range(1, nx_fine):
            xx = 0.5*(x_fine[i]+x_fine[i+1])
            fxx = fx(xx)
            csi_fine[i] = csi_fine[i-1] + dx/fxx #integral(intfun, x_fine(i), x_fine(i+1))
        csi_fine = csi_fine/csi_fine[-1]
        x   = pchip_interpolate(csi_fine, x_fine[0:nx_fine], csi)/csi_fine[-1]
        dx0 = x[1]-x[0]
        err_L2 = dx0 - deltax_0

        if (err_L1*err_L2 > 0):
            print('Solution is not between L1 and L2.') 
            return None, None, None, None

        L  = (L1 + L2)/2

        err_L = 1E4
        delta_L = L
        niter = 0

        while ((niter < 100) and (err_L > 1e-4) or (delta_L > 1E-4) ):

            print('L = ' + str(L))
            L_old = L 

            x_fine = np.linspace(0,L,nx_fine)
            dx = x_fine[1] - x_fine[0]
            x_fine = np.append(x_fine, x_fine[-1]+dx)
            for i in range(1, nx_fine):
                xx = 0.5*(x_fine[i]+x_fine[i+1])
                fxx = fx(xx)
                csi_fine[i] = csi_fine[i-1] + dx/fxx #integral(intfun, x_fine(i), x_fine(i+1))
            csi_fine = csi_fine/csi_fine[-1]

            x   = pchip_interpolate(csi_fine, x_fine[0:nx_fine], csi)/csi_fine[-1]

            dx0 = x[1]-x[0]

            err_L = dx0 - deltax_0
            niter = niter + 1

            # Define nx new
            if (err_L*err_L1>0):
                L1 = L
                err_L1 = err_L
            else:
                L2 = L
                err_L2 = err_L
                
            L = (L1 + L2)/2
            delta_L = abs(L_old - L)
    else:
        print('The flag is not correct.')
        return None, None, None, None
        
   # Final distributions

    x_fine = np.linspace(0,L,nx_fine);
    dx = x_fine[1] - x_fine[0]
    x_fine = np.append(x_fine, x_fine[-1]+dx)
    for i in range(1, nx_fine):
        xx = 0.5*(x_fine[i]+x_fine[i+1])
        fxx = fx(xx)
        csi_fine[i] = csi_fine[i-1] + dx/fxx #integral(intfun, x_fine(i), x_fine(i+1))
    csi_fine = csi_fine/csi_fine[-1]
    csi = np.linspace(0,1,nx)
    x   = pchip_interpolate(csi_fine, x_fine[0:nx_fine], csi)/csi_fine[-1]
    
    # Write grid
    if write:
        with open("x.dat", 'w') as f:
            for i in range(nx - 1):
                xx = x[i]
                xxp = x[i + 1]
                dx = xxp - xx
                f.write(f"{xx:30.15f}{dx:30.15f}\n")
            
            # Scrivi l'ultimo elemento
            f.write(f"{xxp:30.15f}{dx:30.15f}\n")
    
    return csi, x, nx, L
