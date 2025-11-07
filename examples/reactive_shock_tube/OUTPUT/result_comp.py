import numpy as np
import matplotlib.pyplot as plt

Ferrer_data_T_170 = np.loadtxt("Ferrer_Temp_170micro_s.dat")
x_Ferrer_170 = Ferrer_data_T_170[:,0]
T_Ferrer_170 = Ferrer_data_T_170[:,1]

Ferrer_data_T_190 = np.loadtxt("Ferrer_Temp_190micro_s.dat")
x_Ferrer_190 = Ferrer_data_T_190[:,0]
T_Ferrer_190 = Ferrer_data_T_190[:,1]

Ferrer_data_T_230 = np.loadtxt("Ferrer_Temp_230micro_s.dat")
x_Ferrer_230 = Ferrer_data_T_230[:,0]
T_Ferrer_230 = Ferrer_data_T_230[:,1]

Ferrer_data_U_170 = np.loadtxt("Ferrer_U_170micro_s.dat")
x2_Ferrer_170 = Ferrer_data_U_170[:,0]
U_Ferrer_170 = Ferrer_data_U_170[:,1]

Ferrer_data_U_190 = np.loadtxt("Ferrer_U_190micro_s.dat")
x2_Ferrer_190 = Ferrer_data_U_190[:,0]
U_Ferrer_190 = Ferrer_data_U_190[:,1]

Ferrer_data_U_230 = np.loadtxt("Ferrer_U_230micro_s.dat")
x2_Ferrer_230 = Ferrer_data_U_230[:,0]
U_Ferrer_230 = Ferrer_data_U_230[:,1]

Ferrer_data_YH2_170 = np.loadtxt("Ferrer_YH2_170micro_s.dat")
x3_Ferrer_170 = Ferrer_data_YH2_170[:,0]
YH2_Ferrer_170 = Ferrer_data_YH2_170[:,1]

Ferrer_data_YH2_190 = np.loadtxt("Ferrer_YH2_190micro_s.dat")
x3_Ferrer_190 = Ferrer_data_YH2_190[:,0]
YH2_Ferrer_190 = Ferrer_data_YH2_190[:,1]

Ferrer_data_YH2_230 = np.loadtxt("Ferrer_YH2_230micro_s.dat")
x3_Ferrer_230 = Ferrer_data_YH2_230[:,0]
YH2_Ferrer_230 = Ferrer_data_YH2_230[:,1]

data_170 = np.loadtxt("../output_1d_170micro_s.dat")
x_170 = data_170[:,0]
T_170 = data_170[:,14]
U_170 = data_170[:,10]
YH2_170 = data_170[:,3]

data_190 = np.loadtxt("../output_1d_190micro_s.dat")
x_190 = data_190[:,0]
T_190 = data_190[:,14]
U_190 = data_190[:,10]
YH2_190 = data_190[:,3]

data_230 = np.loadtxt("../output_1d_230micro_s.dat")
x_230 = data_230[:,0]
T_230 = data_230[:,14]
U_230 = data_230[:,10]
YH2_230 = data_230[:,3]

plt.figure()
plt.plot(x_170, T_170, label="t = 170 $\mu s$", color="black",linewidth=1.5)
plt.plot(x_190, T_190, label="t = 190 $\mu s$", color="blue",linewidth=1.5)
plt.plot(x_230, T_230, label="t = 230 $\mu s$", color="red",linewidth=1.5)
plt.plot(x_Ferrer_170[::6], T_Ferrer_170[::6], label="Ferrer et al.", color="black",linestyle="",marker="o",markerfacecolor='lightgray',markersize=7)
plt.plot(x_Ferrer_190[::6], T_Ferrer_190[::6], color="blue",linestyle="",marker="o",markerfacecolor='lightgray',markersize=7)
plt.plot(x_Ferrer_230[::6], T_Ferrer_230[::6], color="red",linestyle="",marker="o",markerfacecolor='lightgray',markersize=7)
plt.legend()
plt.xlim([0, 0.12])
plt.ylim([600, 3000])

plt.figure()
plt.plot(x_170, U_170, label="t = 170 $\mu s$", color="black",linewidth=1.7)
plt.plot(x_190, U_190, label="t = 190 $\mu s$", color="blue",linewidth=1.7)
plt.plot(x_230, U_230, label="t = 230 $\mu s$", color="red",linewidth=1.7)
plt.plot(x2_Ferrer_170[::10], U_Ferrer_170[::10], label="Ferrer et al.", color="black",linestyle="",marker="o",markerfacecolor='lightgray',markersize=7)
plt.plot(x2_Ferrer_190[::8], U_Ferrer_190[::8], label="", color="blue",linestyle="",marker="o",markerfacecolor='lightgray',markersize=7)
plt.plot(x2_Ferrer_230[::10], U_Ferrer_230[::10], label="", color="red",linestyle="",marker="o",markerfacecolor='lightgray',markersize=7)
plt.legend()
plt.xlim([0, 0.12])
plt.ylim([-600, 600])

plt.figure()
plt.plot(x_170, YH2_170, label="t = 170 $\mu s$", color="black",linewidth=1.7)
plt.plot(x_190, YH2_190, label="t = 190 $\mu s$", color="blue",linewidth=1.7)
plt.plot(x_230, YH2_230, label="t = 230 $\mu s$", color="red",linewidth=1.7)
plt.plot(x3_Ferrer_170[::10], YH2_Ferrer_170[::10], label="Ferrer et al.", color="black",linestyle="",marker="o",markerfacecolor='lightgray',markersize=7)
plt.plot(x3_Ferrer_190[::10], YH2_Ferrer_190[::10], label="", color="blue",linestyle="",marker="o",markerfacecolor='lightgray',markersize=7)
plt.plot(x3_Ferrer_230[::10], YH2_Ferrer_230[::10], label="", color="red",linestyle="",marker="o",markerfacecolor='lightgray',markersize=7)
plt.legend()
plt.xlim([0, 0.12])
plt.ylim([0, 0.0025])

plt.show()


