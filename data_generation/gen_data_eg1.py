# -*- coding: utf-8 -*-
"""
Created on Tue Jul  2 15:39:44 2019

@author: Soon Hoe Lim
"""

import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import odeint
np.random.seed(20)

# To display plots directly in the notebook
#%matplotlib inline

import pandas as pd

#parameters to be varied
epsilon=0.5
sigma=0.08 

#drift and noise coefficient
def f(x,t):
    f0=x[0]-x[0]**3+sigma*x[2]/epsilon
    f1=10*(x[2]-x[1])/epsilon**2
    f2=(28*x[1]-x[2]-x[1]*x[3])/epsilon**2
    f3=(x[1]*x[2]-8*x[3]/3)/epsilon**2
    return np.array([f0,f1,f2,f3])

T=100; N=10000; dt=T/N 
t=np.linspace(0,T,N+1) 

M=1

x0=-1.5
x1=np.random.uniform(-10,10)
x2=np.random.uniform(-10,10)
x3=np.random.uniform(-10,10)
xzero = np.array([x0,x1,x2,x3]) 

x_m = np.zeros((M,N+1))
for j in range(M):
    x_m1 = odeint(f, xzero, t) 
    x_m1 = np.array(x_m1)
    
#save data into a file
np.savetxt('xdata_eg1.csv',np.c_[t, x_m1[:,0]], delimiter=',')   

##########################################################################
#visualization:
trainbeg=0
trainlen=3700
future=315

plt.rcParams['axes.facecolor']='white'

#whole data
plt.figure(figsize=(6,3))
plt.plot(x_m1[:,0])
#plt.title('whole')
plt.show()

#training data
plt.figure(figsize=(6,3))
plt.plot(x_m1[trainbeg:trainlen,0])
#plt.title('training')
plt.show()
print(x_m1[trainbeg:trainlen,0].shape)

plt.figure(figsize=(6,3))
plt.plot(x_m1[trainlen:trainlen+future,0])
#plt.title('training')
plt.show()

data=x_m1[:,0].reshape((x_m1[:,0].shape[0],1))
forcing_true=sigma*x_m1[:,2].reshape((x_m1[:,2].shape[0],1))/epsilon

#true driving signal data
plt.figure(figsize=(6,3))
plt.plot(forcing_true)
#plt.title('true forcing')
plt.show()
