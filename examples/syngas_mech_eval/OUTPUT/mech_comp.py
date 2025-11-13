import numpy as np
import matplotlib.pyplot as plt

kero_data = np.loadtxt("output_Keromnes2013.dat")
davis_data = np.loadtxt("output_Davis2005.dat")
varga_data = np.loadtxt("output_Varga2016.dat")

time_kero = kero_data[:,0]/1e-6
temp_kero = kero_data[:,2]

time_davis = davis_data[:,0]/1e-6
temp_davis = davis_data[:,2]

time_varga = varga_data[:,0]/1e-6
temp_varga = varga_data[:,2]

for j in range(len(temp_kero)): 
 if temp_kero[j]>1.05*temp_kero[0]:
  tau_kero=time_kero[j]
  temp_tau_kero=temp_kero[j]
  break

for j in range(len(temp_davis)): 
 if temp_davis[j]>1.05*temp_davis[0]:
  tau_davis=time_davis[j]
  temp_tau_davis=temp_davis[j]
  break

for j in range(len(temp_varga)): 
 if temp_varga[j]>1.05*temp_varga[0]:
  tau_varga=time_varga[j]
  temp_tau_varga=temp_varga[j]
  break

reduction_kero_davis = abs(tau_davis-tau_kero)/tau_kero * 100
reduction_kero_varga = abs(tau_varga-tau_kero)/tau_kero * 100

print(reduction_kero_davis,reduction_kero_varga)

point_kero_1 = [0,tau_kero]
point_kero_2 = [temp_tau_kero,temp_tau_kero]

point_davis_1 = [0,tau_davis]
point_davis_2 = [temp_tau_davis,temp_tau_davis]

point_varga_1 = [0,tau_varga]
point_varga_2 = [temp_tau_varga,temp_tau_varga]

plt.figure()
plt.plot(time_kero,temp_kero, label='Keromnes2013',color='black',linewidth=2)
plt.plot(point_kero_1,point_kero_2,linestyle='--',color='black')
plt.plot(tau_kero,temp_tau_kero,linestyle='',color='black',marker='o')
plt.plot(time_davis,temp_davis, label='Davis2005',color='blue',linewidth=2)
plt.plot(point_davis_1,point_davis_2,linestyle='--',color='blue')
plt.plot(tau_davis,temp_tau_davis,linestyle='',color='blue',marker='o')
plt.plot(time_varga,temp_varga, label='Varga2016',color='red',linewidth=2)
plt.plot(point_varga_1,point_varga_2,linestyle='--',color='red')
plt.plot(tau_varga,temp_tau_varga,linestyle='',color='red',marker='o')
plt.xlabel('Time [$\mu$s]')
plt.xlim([0,2000])
plt.ylabel('Temperature [K]')
plt.legend()

plt.figure()
plt.plot(kero_data[:,0]/1e-6,kero_data[:,2], label='Keromnes2013',color='black',linewidth=2)
plt.plot(point_kero_1,point_kero_2,linestyle='--',color='black')
plt.plot(tau_kero,temp_tau_kero,linestyle='',color='black',marker='o')
plt.plot(davis_data[:,0]/1e-6,davis_data[:,2], label='Davis2005',color='blue',linewidth=2)
plt.plot(point_davis_1,point_davis_2,linestyle='--',color='blue')
plt.plot(tau_davis,temp_tau_davis,linestyle='',color='blue',marker='o')
plt.plot(varga_data[:,0]/1e-6,varga_data[:,2], label='Varga2016',color='red',linewidth=2)
plt.plot(point_varga_1,point_varga_2,linestyle='--',color='red')
plt.plot(tau_varga,temp_tau_varga,linestyle='',color='red',marker='o')
plt.xlabel('Time [$\mu$s]')
plt.ylabel('Temperature [K]')
plt.xlim([250,500])
plt.legend()

plt.show()

