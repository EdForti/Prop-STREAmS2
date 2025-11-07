import numpy as np
import matplotlib.pyplot as plt

Cantera_data = np.loadtxt('cantera_IC.dat')
x_cantera = Cantera_data[:,0]
T_cantera = Cantera_data[:,2]
rho_cantera = Cantera_data[:,3]
YH2_cantera = Cantera_data[:,4]
YO_cantera = Cantera_data[:,7]
YOH_cantera = Cantera_data[:,8]
YH_cantera = Cantera_data[:,5]*10

Tu_cantera = T_cantera[0]
Tb_cantera = T_cantera[-1]
c_cantera = (T_cantera - Tu_cantera)/(Tb_cantera - Tu_cantera)

STREAmS_data = np.loadtxt('../output_LPF.dat')
x_STREAmS = STREAmS_data[:,0]
T_STREAmS = STREAmS_data[:,1]
rho_STREAmS = STREAmS_data[:,2]
YH2_STREAmS = STREAmS_data[:,3]
YO_STREAmS = STREAmS_data[:,4]
YOH_STREAmS = STREAmS_data[:,5]
YH_STREAmS = STREAmS_data[:,6]*10

Tu_STREAmS = T_STREAmS[0]
Tb_STREAmS = T_STREAmS[-1]
c_STREAmS = (T_STREAmS - Tu_STREAmS)/(Tb_STREAmS - Tu_STREAmS)

Ferrer_Data_1 = np.loadtxt('Ferrer_LPF_Density.dat')
c_Ferrer1 = Ferrer_Data_1[:,0]
rho_Ferrer = Ferrer_Data_1[:,1]

Ferrer_Data_2 = np.loadtxt('Ferrer_LPF_Temperature.dat')
c_Ferrer2 = Ferrer_Data_2[:,0]
temperature_Ferrer = Ferrer_Data_2[:,1]

Ferrer_Data_3 = np.loadtxt('Ferrer_LPF_YH2.dat')
c_Ferrer3 = Ferrer_Data_3[:,0]
YH2_Ferrer = Ferrer_Data_3[:,1]

Ferrer_Data_4 = np.loadtxt('Ferrer_LPF_YO.dat')
c_Ferrer4 = Ferrer_Data_4[:,0]
YO_Ferrer = Ferrer_Data_4[:,1]

Ferrer_Data_5 = np.loadtxt('Ferrer_LPF_YOH.dat')
c_Ferrer5 = Ferrer_Data_5[:,0]
YOH_Ferrer = Ferrer_Data_5[:,1]

Ferrer_Data_6 = np.loadtxt('Ferrer_LPF_YH.dat')
c_Ferrer6 = Ferrer_Data_6[:,0]
YH_Ferrer = Ferrer_Data_6[:,1]

fig, ax1 = plt.subplots(figsize=(8, 5))
color1 = 'red'
ax1.set_xlabel('x [m]')
ax1.set_ylabel('Density [Kg/m$^3$]', color=color1)
l1 = ax1.plot(x_STREAmS,rho_STREAmS,color=color1,label='STREAmS',linewidth=3)
l2 = ax1.plot(x_cantera, rho_cantera, linestyle=':',linewidth=3,color='black',label='Cantera')
ax1.tick_params(axis='y', labelcolor=color1)

ax2 = ax1.twinx()
color2 = 'darkorange'
ax2.set_ylabel('Temperature [K]', color='darkorange')
l3 = ax2.plot(x_STREAmS,T_STREAmS,color=color2,linewidth=3)
l4 = ax2.plot(x_cantera, T_cantera, linestyle=':',linewidth=3,color='black')
ax2.tick_params(axis='y', labelcolor=color2)

lines = l1+l2
labels = [line.get_label() for line in lines]
ax1.legend(lines, labels, loc="right")

fig, ax3 = plt.subplots(figsize=(8, 5))
color3 = 'red'
ax3.set_xlabel('c [-]')
ax3.set_ylabel('Density [Kg/m$^3$]', color=color3)
l5 = ax3.plot(c_STREAmS,rho_STREAmS,color='red',label='STREAmS',linewidth=3)
l6 = ax3.plot(c_cantera, rho_cantera, color='black', label="Cantera",linestyle=':',linewidth=3)
l7 = ax3.plot(c_Ferrer1[::8],rho_Ferrer[::8],marker='o',markerfacecolor='lightgray',markersize=10,color='red',label='Ferrer et al.',linestyle='',linewidth=2)
ax3.tick_params(axis='y', labelcolor=color3)

ax4 = ax3.twinx()
color4 = 'darkorange'
ax4.set_ylabel('Temperature [K]', color='darkorange')
ax4.plot(c_STREAmS,T_STREAmS,color='darkorange',linewidth=3)
ax4.plot(c_cantera, T_cantera, color='black', label="",linestyle=':',linewidth=3)
ax4.plot(c_Ferrer2[::8],temperature_Ferrer[::8],marker='o',markerfacecolor='lightgray',markersize=10,color='orange',linestyle='',linewidth=2)
ax4.tick_params(axis='y', labelcolor=color4)

liness = l5+l6+l7
labelss = [line.get_label() for line in liness]
ax3.legend(liness, labelss, loc="right")

fig, ax5 = plt.subplots(figsize=(8, 5))
color5 = 'red'
ax5.set_xlabel('c [-]')
ax5.set_ylabel('Y$_{H2}$ [-]', color=color3)
l8 = ax5.plot(c_STREAmS,YH2_STREAmS,color='red',label='STREAmS - Y$_{H2}$',linewidth=3)
l12 = ax5.plot(c_cantera, YH2_cantera, color='black', label="Cantera",linestyle=':',linewidth=3)
l13 = ax5.plot(c_Ferrer3[::8],YH2_Ferrer[::8],marker='o',markerfacecolor='lightgray',markersize=10,color='red',label='Ferrer et al.',linestyle='',linewidth=2)
ax5.tick_params(axis='y', labelcolor=color3)

ax6 = ax5.twinx()
color6 = 'darkorange'
ax6.set_ylabel('Y$_{O,OH,H}$ [-]', color=color6)
l9 = ax6.plot(c_STREAmS,YO_STREAmS,color='darkorange',linewidth=3,label='Y$_{O}$')
ax6.plot(c_cantera, YO_cantera, color='black', label="",linestyle=':',linewidth=3)
ax6.plot(c_Ferrer4[::8],YO_Ferrer[::8],marker='o',markerfacecolor='lightgray',markersize=10,color='orange',linestyle='',linewidth=2)
l10 = ax6.plot(c_STREAmS,YOH_STREAmS,color='lightgreen',linewidth=3,label='Y$_{OH}$')
ax6.plot(c_cantera, YOH_cantera, color='black', label="",linestyle=':',linewidth=3)
ax6.plot(c_Ferrer5[::8],YOH_Ferrer[::8],marker='o',markerfacecolor='lightgray',markersize=10,color='green',linestyle='',linewidth=2)
l11 = ax6.plot(c_STREAmS,YH_STREAmS,color='lightblue',linewidth=3,label='Y$_{H}$ x 10')
ax6.tick_params(axis='y', labelcolor=color4)
ax6.plot(c_Ferrer6[::8],YH_Ferrer[::8],marker='o',markerfacecolor='lightgray',markersize=10,color='lightblue',linestyle='',linewidth=2)
ax6.plot(c_cantera, YH_cantera, color='black', label="",linestyle=':',linewidth=3)


linesss = l8+l9+l10+l11+l12+l13
labelsss = [line.get_label() for line in linesss]
ax5.legend(linesss, labelsss, loc="upper center")

plt.show()


