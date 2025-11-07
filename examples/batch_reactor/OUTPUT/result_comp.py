import numpy as np
import matplotlib.pyplot as plt

cantera_data = np.loadtxt("cantera_results.dat")
streams_data = np.loadtxt("../output_batch.dat")

tol_time = 1e-7;
tol_dens = 1e-5;
tol_temp = 1;
fail = 0;

for i in range(len(cantera_data)):
    diff_time = abs(cantera_data[i,0] - streams_data[i,0])
    diff_dens = abs(cantera_data[i,1] - streams_data[i,1])
    diff_temp = abs(cantera_data[i,2] - streams_data[i,2])

    if diff_time > tol_time:
        print(f'Error on time!')
        fail = 1
        break
    if diff_dens > tol_dens:
        print(f'Error on density!')
        fail = 1
        break
    if diff_temp > tol_temp:
        print(f'Error on temperature!')
        fail = 1
        break

if fail == 0:
    print(f'\nTest completed successfully!\n')

plt.plot(cantera_data[::8,0],cantera_data[::8,2], 'o', markerfacecolor='lightgray', markersize=7, label='Cantera')
plt.plot(streams_data[:,0],streams_data[:,2], label='STREAmS',color='black',linewidth=2)
plt.xlabel('Time [s]')
plt.ylabel('Temperature [K]')
plt.legend()
plt.xlim(0, cantera_data[-1,0])
plt.show()

