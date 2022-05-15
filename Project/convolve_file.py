import numpy as np
import matplotlib.pyplot as plt

arr = np.load("irf1000.npy")

def f(x,a=0.35,tau=0.5):
    
    return a * np.exp(-x/tau)

x = np.array(range(1000))

y = f(x,a=0.39,tau=20) 

# + f(x,a=0.058,tau=5.2) + f(x,a=0.52,tau=18)

# y_ = np.convolve(arr,y)[50:1000] -35
y_ = np.convolve(arr,y)[:1000]

# y_[:100] = 5

y_new = y_ +  np.random.uniform(low=-y_/6,high=y_/6, size=1000)

# y_1 = np.convolve(arr,f(x,tau=25))[:1000]

# plt.plot(x,np.log(arr))
# plt.plot(x,y_new)

# with open("NACL_90_20000.csv") as f:
#     array = np.loadtxt(f,delimiter="\n")

old_x = np.array(range(20000))
new_x = np.linspace(0,20000,1000)
# new_y = np.interp(new_x,old_x,array)




plt.plot(y_new,label="Added noise")
plt.plot(x,y_,label="No added noise")
plt.xlabel("Time (ns/100)")
plt.ylabel("Intensity(a.u)")
plt.title("Plot of intensity against time for a synthetic impulse response function")
plt.legend()
plt.show()

