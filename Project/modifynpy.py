import numpy as np
import matplotlib.pyplot as plt

arr = np.load("NACL_90_1000.npy")

plt.plot(arr[50:])
plt.show()