import numpy as np

irf = np.load("irf1000.npy")

irf = irf[np.argmax(irf):]

experimental = np.load("NACL_90_1000.npy")

experimental = experimental[np.argmax(experimental):np.argmax(experimental)+len(irf)]

experimental += 55

def decayfunc(a,t):
    x = np.array(range(len(irf)))
    return a*np.exp(-x/t)



def convolve(a1,t1,a2,t2,a3,t3):
    decaycurve = decayfunc(a1,t1) +decayfunc(a2,t2) + decayfunc(a3,t3)
    y = np.convolve(decaycurve,irf)[:len(irf)]

    return y

def error_function(x,y):
    newval = np.sum(np.absolute((x-y)**2/x))
    return newval

import random

def addval(x,minn,maxx,rate):
    val = x + x/np.log(rate+2)/100
    # val = min(val,maxx)
    # val = max(val,minn)
    return val

def minnusval(x,minn,maxx,rate):
    val = x - x/np.log(rate+2)/100
    # val = min(val,maxx)
    # val = max(val,minn)
    return val

def newparams(a1,a2,a3,tau1,tau2,tau3,rate):
    if random.choice([1,0]):
        a1 = addval(a1,minn=0,maxx=1,rate=rate)
    else:
        a1 = minnusval(a1,minn=0,maxx=1,rate=rate)

    if random.choice([1,0]):
        a2 = addval(a2,minn=0,maxx=1,rate=rate)
    else:
        a2 = minnusval(a2,minn=0,maxx=1,rate=rate)

    if random.choice([1,0]):
        a3 = addval(a3,minn=0,maxx=1,rate=rate)
    else:
        a3 = minnusval(a3,minn=0,maxx=1,rate=rate)

    if random.choice([1,0]):
        tau1 = addval(tau1,minn=0,maxx=5,rate=rate)
    else:
        tau1 = minnusval(tau1,minn=0,maxx=5,rate=rate)

    if random.choice([1,0]):
        tau2 = addval(tau2,minn=5,maxx=15,rate=rate)
    else:
        tau2 = minnusval(tau2,minn=5,maxx=15,rate=rate)

    if random.choice([1,0]):
        tau3 = addval(tau3,minn=15,maxx=100,rate=rate)
    else:
        tau3 = minnusval(tau3,minn=15,maxx=100,rate=rate)

    return (a1,a2,a3,tau1,tau2,tau3)

def iterative_convolution(randomize=False):
    

    a1 = 0.36
    a2 = 0.30
    a3 = 0.17
    tau1 = 14
    tau2 = 9.94
    tau3 = 117

    if randomize:
        a1 = np.random.uniform(0,1)
        a2 = np.random.uniform(0,1)
        a3 = np.random.uniform(0,1)
        tau1 = np.random.uniform(0,5)
        tau2 = np.random.uniform(5,15)
        tau3 = np.random.uniform(15,100)

    error = np.inf

    error_list = []

    for i in range(30000):
        
        (a11,a21,a31,tau11,tau21,tau31) = newparams(a1,a2,a3,tau1,tau2,tau3,rate=i)
        y = convolve(a11,a21,a31,tau11,tau21,tau31)
        
        new_error = error_function(experimental,y)
        
        
        if new_error < error:
            error = new_error
            a1= a11
            a2=a21
            a3=a31
            tau1=tau11
            tau2=tau21
            tau3=tau31

        error_list.append(np.log(error))
        
        
    return (error_list,a1,a2,a3,tau1,tau2,tau3)


import matplotlib.pyplot as plt

(error_list2,a1,a2,a3,tau1,tau2,tau3) = iterative_convolution(randomize=True)
(error_list,a1,a2,a3,tau1,tau2,tau3) = iterative_convolution()

plt.plot(error_list2,label="Random parameters")
plt.plot(error_list,label="Parameters from ML model")
plt.xlabel("Steps")
plt.ylabel("Chi^2 error")
plt.title("Graph of error function against time")
plt.legend()
plt.show()

plt.plot(convolve(a1,a2,a3,tau1,tau2,tau3))
plt.plot(experimental)
plt.show()

print(a1,tau1)
print(a2,tau2)
print(a3,tau3)
