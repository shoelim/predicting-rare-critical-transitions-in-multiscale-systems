import numpy as np
from scipy.integrate import odeint

# To display plots directly in the notebook
#%matplotlib inline

#parameters to be varied
A= 0.1
epsilon=0.5
sigma= 0.4
Num = 36
F = 8

def f(x,t):
    d=np.zeros(Num+1)
    d[0] = x[0]*(1-x[0])*(1+x[0])*(x[0]-2)*(x[0]+2)+A*np.cos(12*np.pi*t)+sigma*x[1]
    d[1] = ((x[2]-x[Num-1])*x[Num])/epsilon-x[1] + F
    d[2] = ((x[3]-x[Num])*x[1] - x[2] + F)/epsilon**2
    d[Num] = ((x[1]-x[Num-2])*x[Num-1] - x[Num] + F)/epsilon**2
    for i in range(3, Num):
        d[i] = ((x[i+1]-x[i-2])*x[i-1]-x[i] + F)/epsilon**2
    return d

T=100; N=10000; dt=T/N 
t=np.linspace(0,T,N+1) 

M=1

xzero = F*np.ones(Num+1)  # Initial state (equilibrium)
xzero[0] = 0
xzero[20] += 0.01  # Add small perturbation to 20th variable of the Lorenz-96 system

x_m = np.zeros((M,N+1))
for j in range(M):
    x_m1 = odeint(f, xzero, t) 
    x_m1 = np.array(x_m1)

#save data into a file
np.savetxt('xdata_eg3.csv',np.c_[t, x_m1[:,0]], delimiter=',') 
np.savetxt('forcing_eg3.csv',np.c_[t,sigma*x_m1[:,1]], delimiter=',')