# -*- coding: utf-8 -*-
"""
Created on Wed Aug  7 20:36:11 2019

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
A= 0.5
epsilon=0.5
sigma= 0.2
alpha1=1000
sigma1=1000*epsilon

#drift and noise coefficient
def f(x,t):
    f0=x[0]*(1-x[0])*(1+x[0])*(x[0]-2)*(x[0]+2)+A*np.cos(2*np.pi*t)+sigma*x[1]
    f1=-alpha1*x[1]+sigma1*x[2]/epsilon
    f2=10*(x[3]-x[2])/epsilon**2
    f3=(28*x[2]-x[3]-x[2]*x[4])/epsilon**2
    f4=(x[2]*x[3]-8*x[4]/3)/epsilon**2
    return np.array([f0,f1,f2,f3,f4])

T=100; N=10000; dt=T/N 
t=np.linspace(0,T,N+1) 

M=1

x0=0.1
x1=np.random.uniform(-10,10)
x2=np.random.uniform(-10,10)
x3=np.random.uniform(-10,10)
x4=np.random.uniform(-10,10)
xzero = np.array([x0,x1,x2,x3,x4]) 

x_m = np.zeros((M,N+1))
for j in range(M):
    x_m1 = odeint(f, xzero, t) 
    x_m1 = np.array(x_m1)

#save data into a file
np.savetxt('xdata_eg2.csv',np.c_[t, x_m1[:,0]], delimiter=',')   

##########################################################################
#visualization:
trainbeg=0
trainlen=9400
future=600

plt.rcParams['axes.facecolor']='white'

#whole data
plt.figure(figsize=(6,3))
plt.plot(x_m1[:,0])
#plt.title('whole')
#plt.grid(b=None)
plt.show()

#training data
plt.figure(figsize=(6,3))
plt.plot(x_m1[trainbeg:trainlen,0])
#plt.title('training')
#plt.grid(b=None)
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
#plt.grid(b=None)
plt.show()
