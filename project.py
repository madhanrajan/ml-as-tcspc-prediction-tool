import numpy as np
import scipy.integrate as integrate
import matplotlib.pyplot as plt


def F(t):
    
    return np.sin(t)

def G(t):
    return 0.02*t - 1

f_list = []
g_list = []

datapoints = 1000
density = 10

for i in range(datapoints):
    f_list.append(F(i/density))
    g_list.append(G(i/density))

z_list = np.convolve(f_list,g_list)

# plt.plot(range(100),f_list[:100])
# plt.plot(range(100),g_list[:100])
plt.plot(range(datapoints),f_list[:datapoints])
plt.show()
plt.plot(range(datapoints),g_list[:datapoints])
plt.show()
plt.plot(range(datapoints),z_list[:datapoints])
plt.show()