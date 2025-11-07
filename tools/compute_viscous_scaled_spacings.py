import numpy as np
import matplotlib.pyplot as plt

cf = np.loadtxt("POSTPRO/cf.dat")

x = cf[:, 0]             
delta99 = cf[:, 1]       
deltav  = cf[:, 12]     

x = np.loadtxt("dxg.dat")
y = np.loadtxt("dyg.dat")
z = np.loadtxt("dzg.dat")

dx = np.mean(x[:,1])
dz = np.mean(z[:,1])

ynodes = y[:,0]
dy     = y[:,1]

dx_plus = dx / deltav
dz_plus = dz / deltav

dy_wall_plus = ynodes[0] / deltav

# spacing at y = delta99 
dy_delta_plus = []
for d99, dv in zip(delta99, deltav):
    # find index of nearest y-node
    idx = np.argmin(np.abs(ynodes - d99))
    dy_loc = dy[idx]
    dy_delta_plus.append(dy_loc / dv)
    print(d99,dy_loc,dv,dy_loc / dv)

dy_delta_plus = np.array(dy_delta_plus)
print(dy_delta_plus)

# --- Plot ---
plt.figure(figsize=(10,6))
plt.plot(x[:,0], dx_plus, label=r"$\Delta x^+$")
plt.plot(x[:,0], dz_plus, label=r"$\Delta z^+$")
plt.plot(x[:,0], dy_wall_plus, label=r"$\Delta y^+$ (wall)")
plt.plot(x[:,0], dy_delta_plus, label=r"$\Delta y^+$ ($y=\delta_{99}$)")

plt.xlabel("x")
plt.ylabel("Grid spacing in wall units")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()

