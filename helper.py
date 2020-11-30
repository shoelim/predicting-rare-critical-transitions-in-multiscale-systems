# -*- coding: utf-8 -*-
"""
Created on Mon Nov 30 11:47:56 2020

@author: Soon Hoe Lim
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import sem

def F(x,index_data):
    if index_data == 1:
        return x-x**3
    elif index_data == 2:
        return x*(1-x)*(1+x)*(x-2)*(x+2)
    elif index_data == 3:
        return x*(1-x)*(1+x)*(x-2)*(x+2)
    else:
        print('Unexpected data!')
        
def Ft(x,t,index_data):
    if index_data == 3:
        return x*(1-x)*(1+x)*(x-2)*(x+2)+0.1*np.cos(12*np.pi*t)
    else:
        print('Unexpected data!')

def diff(x,index_data,dt):
    y=np.zeros((len(x),1))
    for i in range(len(x)-1):
        if index_data == 1:
            y[i]=(x[i+1]-x[i])/dt-F(x[i],index_data)
        elif index_data == 2:
            y[i]=(x[i+1]-x[i])/dt-F(x[i],index_data)-0.5*np.cos(2*np.pi*dt*i)
        elif index_data == 3:
            y[i]=(x[i+1]-x[i])/dt-F(x[i],index_data)-0.1*np.cos(12*np.pi*dt*i)
        else:
            print('Unexpected data!')
    return y

def select_and_plot_results(index_data, n_ens, trainlen, valid, future, trainbeg, dtau):
    
    data_orig = pd.read_csv('xdata_eg'+'{0}'.format(index_data)+'.csv',header=None)
    data_orig = np.array(data_orig)
    data_orig = data_orig[:,1]
    
    osol = pd.read_csv('Ex'+'{0}'.format(index_data)+'_'+'{0}'.format(trainlen)+'-'+'{0}'.format(valid)+'_'+'{0}'.format(n_ens)+'ens_fin.csv',header=None)
    osol = np.array(osol)
    
    #####################################################################################################      
    #visualize results
    N = 1/dtau 
    t_tr=np.linspace(trainbeg,trainlen,trainlen-trainbeg)/N
    t_res=np.linspace(trainlen,(trainlen+future),future)/N
    t_val=np.linspace(trainlen-valid-trainbeg,trainlen-trainbeg,valid)/N
    
    plt.rcParams['axes.facecolor']='white'
    
    ax1=plt.figure(figsize=(6,3))
    plt.plot(t_tr,data_orig[trainbeg:trainlen],'r^')
    plt.plot(t_res,data_orig[trainlen:trainlen+future],'r^')
    solp=[]
    for i in range(n_ens):
      solp.append(osol[i,trainlen-trainbeg:trainlen+future-trainbeg])
      plt.plot(t_res,osol[i,trainlen-trainbeg:trainlen+future-trainbeg],alpha=0.2)
    plt.plot(t_res, np.mean(solp,axis=0),'b-o')
    #ax1.text(0.1, 0.96,'(a)', fontsize=12, verticalalignment='top')
    plt.xlabel('$t$')
    plt.ylabel('$x$')
    #plt.grid(b=None)
    #plt.legend(['training','predicted','target'])
    #plt.show()
    
    #target path, multiple predicted paths, the averaged path and the 90 percent confidence interval
    sonn=[]
    ax4=plt.figure(figsize=(6,3))
    plt.plot(t_res,data_orig[trainlen:trainlen+future],'r^')
    for k in range(n_ens):
      plt.plot(t_res, osol[int(k),trainlen-trainbeg:trainlen+future-trainbeg],alpha=0.2)
      sonn.append(osol[int(k),trainlen-trainbeg:trainlen+future-trainbeg])
    stderr = sem(sonn,axis=0)  #std error of the mean (sem) provides a simple measure of uncertainty in a value
    
    #Remark: Confidence interval is calculated assuming the samples are drawn from a Gaussian distribution
    #Justification: As the sample size tends to infinity the central limit theorem guarantees that the sampling 
    #               distribution of the mean is asymptotically normal
    
    plt.plot(t_res,np.mean(sonn,axis=0),'b-o')
    y1=np.mean(sonn,axis=0)-1.645*stderr
    y2=np.mean(sonn,axis=0)+1.645*stderr
    plt.plot(t_res,y1,'--')
    plt.plot(t_res,y2,'--')
    plt.fill_between(t_res, y1, y2, facecolor='blue', alpha=0.2)
    #ax4.text(0.1, 0.96,'(b)', fontsize=12, verticalalignment='top')
    plt.xlabel('$t$')
    plt.ylabel('$x$')
    #plt.grid(False)
    #plt.show()
    
    print('################### Selecting Paths From Ensemble Using Validation Set #######################')
    
    num_std1 = 8 #increase by one if N_eff = 0 (parameter for weighting in the ensemble)
    sel = []
    it = 10 #maximum iterations
    while len(sel) == 0 and it >= 0:
      if it == 0:
        print('No path selected!')
      it -= 1

      son=[]
      #ax5=plt.figure(figsize=(6,3))
      #plt.plot(t_val,data_orig[trainlen-valid:trainlen],'r^')
      for k in range(n_ens):
      #  plt.plot(t_val, osol[int(k),trainlen-trainbeg-valid:trainlen-trainbeg],alpha=0.2)
         son.append(osol[int(k),trainlen-trainbeg-valid:trainlen-trainbeg])
      std = sem(son,axis=0)  #std error of the mean (sem) provides a simple measure of uncertainty in a value
      
      #Remark: Confidence interval is calculated assuming the samples are drawn from a Gaussian distribution
      #Justification: As the sample size tends to infinity the central limit theorem guarantees that the sampling 
      #               distribution of the mean is asymptotically normal
      #plt.plot(t_val,np.mean(son,axis=0),'b-o')
      
      y3=np.mean(son,axis=0)-num_std1*std 
      y4=np.mean(son,axis=0)+num_std1*std
      #plt.plot(t_val,y3,'--')
      #plt.plot(t_val,y4,'--')
      #plt.fill_between(t_val, y3, y4, facecolor='blue', alpha=0.2)
      #ax4.text(0.1, 0.96,'(b)', fontsize=12, verticalalignment='top')
      #plt.xlabel('$t$')
      #plt.ylabel('$x$')
      #plt.grid(False)
      #plt.show()
    
      for i in range(n_ens):
        if all(y4 > son[i]) and all(y3 < son[i]):
          sel.append(i)
      num_std1 += 1
    
    N_eff = len(sel)
    print('N_eff = ', N_eff)
    
    ############################################################################################
    print('######################### Final Prediction Results #########################')
    
    sonn=[]
    ax6=plt.figure(figsize=(6,3))
    plt.plot(t_res,data_orig[trainlen:trainlen+future],'r^')
    for k in sel:
      plt.plot(t_res, osol[int(k),trainlen-trainbeg:trainlen+future-trainbeg],alpha=0.2)
      sonn.append(osol[int(k),trainlen-trainbeg:trainlen+future-trainbeg])
    stderr = sem(sonn,axis=0)  
    
    plt.plot(t_res,np.mean(sonn,axis=0),'b-o')
    y1=np.mean(sonn,axis=0)-1.645*stderr
    y2=np.mean(sonn,axis=0)+1.645*stderr
    plt.plot(t_res,y1,'--')
    plt.plot(t_res,y2,'--')
    plt.fill_between(t_res, y1, y2, facecolor='blue', alpha=0.2)
    
    #=====================================================
    ax6.text(0.2, 0.7,'$N_{eff}$='+'{0}'.format(N_eff), fontsize=12) 
    plt.xlabel('$t$')
    plt.ylabel('$x$')
    plt.savefig('Ex'+'{0}'.format(index_data)+'_'+'{0}'.format(trainlen)+'_'+'{0}'.format(valid)+'_res.png')
    #plt.grid(False)
    #plt.show()
    
    #pathwise metric:
    #error for predicted path
    ax2=plt.figure(figsize=(6,3))
    error=[]
    for i in sel:
      diff=osol[int(i),trainlen-trainbeg:trainlen+future-trainbeg]-data_orig[trainlen-trainbeg:trainlen+future-trainbeg]
      error.append(diff)
      plt.plot(t_res,diff,alpha=0.2)
    plt.plot(t_res, np.mean(error,axis=0),'b-o')
    #ax2.text(0.1, 0.96,'(c)', fontsize=12, verticalalignment='top')
    plt.xlabel('$t$')
    plt.ylabel('$E_{out}$')
    plt.savefig('Ex'+'{0}'.format(index_data)+'_'+'{0}'.format(trainlen)+'_'+'{0}'.format(valid)+'_error.png')
    #plt.grid(b=None)
    #plt.show()
    
    #standard deviation for predicted paths
    ax3=plt.figure(figsize=(6,3))
    stdev=np.std(error,axis=0)
    plt.plot(t_res, stdev,'b-o')
    #ax3.text(0.1, 0.96,'(d)', fontsize=12, verticalalignment='top')
    plt.xlabel('$t$')
    plt.ylabel('$std(E_{out})$')
    plt.savefig('Ex'+'{0}'.format(index_data)+'_'+'{0}'.format(trainlen)+'_'+'{0}'.format(valid)+'_std.png')
    #plt.grid(b=None)
    #plt.show()