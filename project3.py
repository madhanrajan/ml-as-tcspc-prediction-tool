# import numpy as np
# import matplotlib.pyplot as plt

density = 100

# x = np.random.normal(size=(density,))
# y = np.random.rand(density)
# z = np.convolve(x,y)[:density]


# plt.plot(range(density),x)

# plt.plot(range(density),y)

# plt.plot(range(density),z,color="r")
# plt.show()

import matplotlib.pyplot as plt
import numpy as np
import scipy.stats as stats
import math


def generate_gaussian():
    mu = np.random.uniform(low=5,high=30)
    variance = 20
    sigma = math.sqrt(variance)
    x = np.linspace(mu - 3*sigma, mu + 3*sigma, 25)
    y = stats.norm.pdf(x, mu, sigma) * np.random.uniform(low=1, high=8)

    pref_x = np.array(range(100))
    new_y = np.interp(pref_x,x,y)

    new_y += np.random.uniform(low=0,high=0.10,size=(100,))

    return (pref_x,new_y)



def f(x,a,tau):
    return a * np.exp(-x/tau)


def generate_decay_curve():
    a = np.random.random()
    tau = np.random.uniform(low=5,high=50)
    x = np.array(range(100))
    y = f(x,a,tau)

    return (y,a,tau)
    

def generate_dataset():
    (x,y) = generate_gaussian()
    (dec_y,a,t) = generate_decay_curve()
    z = np.convolve(y,dec_y)[:100]
    z += np.random.uniform(0,0.2)

    input = np.concatenate((z,y))
    output = (a,t)

    return (input,output)

(a, b) = generate_dataset()

print(a)




(x,y) = generate_gaussian()
(dec_y,a,t) = generate_decay_curve()
z = np.convolve(y,dec_y)[:100]
z += np.random.uniform(0,0.2)



plt.plot(x,y)
plt.plot(x,dec_y)
plt.plot(x,z)
plt.show()