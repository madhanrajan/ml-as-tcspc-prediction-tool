import numpy as np
import matplotlib.pyplot as plt

with open("irf20000.csv") as f:
    array = np.loadtxt(f,delimiter="\n")

old_x = np.array(range(20000))
new_x = np.linspace(0,20000,1000)
new_y = np.interp(new_x,old_x,array)




plt.plot(new_y)
plt.xlabel("time (ns/100)")
plt.ylabel("Photon count (a.u)")
plt.title("Graph of Photon Count against Time")
plt.show()

if input("y"):
    np.save("irf1000_startindex0.npy",new_y)