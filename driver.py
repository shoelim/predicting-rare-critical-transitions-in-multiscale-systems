# -*- coding: utf-8 -*-
"""
Created on Mon Nov 30 12:08:29 2020

@author: Soon Hoe Lim
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import sem

from deepESN import ESN
from helper import * 
import argparse
import os

import timeit

#==============================================================================
# Training settings
#==============================================================================
parser = argparse.ArgumentParser(description='Predicting Critical Transitions')
parser.add_argument('--name', type=str, default='Example1', metavar='N', help='name of dataset')
parser.add_argument('--option', type=int, default=1, metavar='N', help='option of running the baseline method (0) or our algorithm (1) (default: 1)')
parser.add_argument('--n_ens', type=int, default=50, metavar='N', help='ensemble size -- number of networks independently trained for ensemble learning (default: 50)')
parser.add_argument('--trainlen', type=int, default=3500, metavar='N', help='number of data points in accessible dataset (default: 3500)')
parser.add_argument('--valid', type=int, default=10, metavar='N', help='number of data points in validation set (default: 10)')
parser.add_argument('--future', type=int, default=500, metavar='N', help='number of data points in prediction time window (default: 500)')
parser.add_argument('--trainbeg', type=int, default=0, metavar='N', help='index where first data point of the training time series is used in training (default: 0)')
parser.add_argument('--dt', type=float, default=0.01, metavar='LR', help='time step used in numerical integration generating the data (default: 0.01)')
parser.add_argument('--num_layer', type=int, default=1, metavar='N', help='number of layer in the network (default: 1)')
parser.add_argument('--n_transient', type=int, default=0, metavar='N', help='first n_transient reservoir hidden states are discarded (default: 0)')

#==============================================================================
# Model selection settings
#==============================================================================
parser.add_argument('--noise_total', type=int, default=1, metavar='N', help='total number of noise levels in all layers to be selected from (default: 1)')
parser.add_argument('--noise_start', type=float, default=0.001, metavar='LR', help='lowest possible value of these noise levels (default: 0.001)')
parser.add_argument('--noise_delta', type=float, default=0.001, metavar='LR', help='gap between these noise levels (default: 0.001)')

parser.add_argument('--n_res_total', type=int, default=8, metavar='N', help='total number of dimensions of reservoir hidden states in all layers to be selected from (default: 8)')
parser.add_argument('--spec_rad_total', type=int, default=1, metavar='N', help='total number of spectral radii in all layers to be selected from (default: 1)')
parser.add_argument('--sparsity_total', type=int, default=1, metavar='N', help='total number of sparsity levels in all layers to be selected from (default: 1)')


#hyperparameters for the first layer
parser.add_argument('--n_res_start_1', type=int, default=600, metavar='N', help='lowest possible dimension of reservoir hidden states (default: 600)')
parser.add_argument('--n_res_delta_1', type=int, default=20, metavar='N', help='gap between these dimensions of reservoir hidden states (default: 20)')

parser.add_argument('--spec_rad_start_1', type=float, default=0.7, metavar='LR', help='lowest possible value of these spectral radii (default: 0.7)')
parser.add_argument('--spec_rad_delta_1', type=float, default=0.1, metavar='LR', help='gap between these spectral radii (default: 0.1)')

parser.add_argument('--sparsity_start_1', type=float, default=0.1, metavar='LR', help='lowest possible value of these sparsity levels (default: 0.1)')
parser.add_argument('--sparsity_delta_1', type=float, default=0.1, metavar='LR', help='gap between these sparsity levels (default: 20)')


#hyperparameters for the second layer
parser.add_argument('--n_res_start_2', type=int, default=100, metavar='N', help='lowest possible dimension of reservoir hidden states (default: 100)')
parser.add_argument('--n_res_delta_2', type=int, default=50, metavar='N', help='gap between these dimensions of reservoir hidden states (default: 50)')

parser.add_argument('--spec_rad_start_2', type=float, default=0.7, metavar='LR', help='lowest possible value of these spectral radii (default: 0.7)')
parser.add_argument('--spec_rad_delta_2', type=float, default=0.1, metavar='LR', help='gap between these spectral radii (default: 0.1)')

parser.add_argument('--sparsity_start_2', type=float, default=0.1, metavar='LR', help='lowest possible value of these sparsity levels (default: 0.1)')
parser.add_argument('--sparsity_delta_2', type=float, default=0.1, metavar='LR', help='gap between these sparsity levels (default: 20)')


#hyperparameters for the third layer
parser.add_argument('--n_res_start_3', type=int, default=100, metavar='N', help='lowest possible dimension of reservoir hidden states (default: 100)')
parser.add_argument('--n_res_delta_3', type=int, default=50, metavar='N', help='gap between these dimensions of reservoir hidden states (default: 50)')

parser.add_argument('--spec_rad_start_3', type=float, default=0.7, metavar='LR', help='lowest possible value of these spectral radii (default: 0.7)')
parser.add_argument('--spec_rad_delta_3', type=float, default=0.1, metavar='LR', help='gap between these spectral radii (default: 0.1)')

parser.add_argument('--sparsity_start_3', type=float, default=0.1, metavar='LR', help='lowest possible value of these sparsity levels (default: 0.1)')
parser.add_argument('--sparsity_delta_3', type=float, default=0.1, metavar='LR', help='gap between these sparsity levels (default: 20)')


#for forth layer and so on, please extend the codes (straightforward)

args = parser.parse_args()

if not os.path.isdir('results'):
    os.mkdir('results')

print(args)

#==============================================================================
# get dataset
#==============================================================================
if args.name == 'Example1':
  data_orig = pd.read_csv("xdata_eg1.csv",header=None)
  index_data = 1
elif args.name == 'Example2':
  data_orig = pd.read_csv("xdata_eg2.csv",header=None)
  index_data = 2
elif args.name == 'Example3':
  data_orig = pd.read_csv("xdata_eg3.csv",header=None)
  index_data = 3
else:
  print('Unexpected data!')

#==============================================================================
# Training, Model Selection and Prediction
#==============================================================================
t0 = timeit.default_timer()
option = args.option

#training/testing parameters
n_ens = args.n_ens
dt = args.dt
valid = args.valid
future = args.future+valid 
trainlen = args.trainlen-valid 
trainbeg = args.trainbeg

data_orig = np.array(data_orig)
data_orig = data_orig[:,1]
forcing = diff(data_orig,index_data,dt)

if option == 1:
    data = forcing
elif option == 0:
    data = data_orig
else:
    print('Unexpected option!')

#print info
print('Total number of data points given: ',trainlen+valid) 
print('Training data starts from data point #: ', trainbeg)
print('Training data ends at data point #: ',trainlen)
print('Total number of data points used for validation: ', valid)
print('Total number of data points used for training: ',trainlen-trainbeg)

otest_error = np.zeros((n_ens,1))
oNum = int(trainlen+future)
osol = np.zeros((n_ens,oNum+1-trainbeg))
dtau = dt

num_layer = args.num_layer
n_transient = args.n_transient

noise_total = args.noise_total
noise_start = args.noise_start
noise_delta = args.noise_delta

n_res_total = args.n_res_total
spec_rad_total = args.spec_rad_total
sparsity_total = args.sparsity_total

n_res_start_1 = args.n_res_start_1
n_res_delta_1 = args.n_res_delta_1
spec_rad_start_1 = args.spec_rad_start_1
spec_rad_delta_1 = args.spec_rad_delta_1
sparsity_start_1 = args.sparsity_start_1
sparsity_delta_1 = args.sparsity_delta_1

if num_layer >= 2:
    n_res_start_2 = args.n_res_start_2
    n_res_delta_2 = args.n_res_delta_2
    spec_rad_start_2 = args.spec_rad_start_2
    spec_rad_delta_2 = args.spec_rad_delta_2
    sparsity_start_2 = args.sparsity_start_2
    sparsity_delta_2 = args.sparsity_delta_2

if num_layer >= 3:
    n_res_start_3 = args.n_res_start_3
    n_res_delta_3 = args.n_res_delta_3
    spec_rad_start_3 = args.spec_rad_start_3
    spec_rad_delta_3 = args.spec_rad_delta_3
    sparsity_start_3 = args.sparsity_start_3
    sparsity_delta_3 = args.sparsity_delta_3

k = 0
count = 0
max_iter = 1000
threshold = 1000  #start with an initial threshold and slowly decrease it if necessary
#special case: when the threshold is big enough such that a given random seed will be used for simulation

valid_error_rmse = np.zeros((n_ens,n_res_total,spec_rad_total,sparsity_total,noise_total))
seeds = np.zeros(valid_error_rmse.shape)

opred_training = []
oprediction = []

while k < n_ens and count <= max_iter:
    for h1 in range(n_res_total): 
        for h2 in range(spec_rad_total): 
            for h3 in range(sparsity_total): 
                for h4 in range(noise_total): 
                    print("======================= optimizing over a hyperparameter space ====================== ")
    
                    if num_layer == 1:
                        esn = ESN(n_inputs = 1,
                                  n_outputs = 1,
                                  n_reservoir = [n_res_start_1+h1*n_res_delta_1], 
                                  n_layer = num_layer,
                                  spectral_radius = [spec_rad_start_1+h2*spec_rad_delta_1],
                                  sparsity = [sparsity_start_1+h3*sparsity_delta_1], 
                                  noise = noise_start+noise_delta*h4,
                              	  n_transient = n_transient,
                                  random_state = count+100)
                    elif num_layer == 2:
                        esn = ESN(n_inputs = 1,
                                  n_outputs = 1,
                                  n_reservoir = [n_res_start_1+h1*n_res_delta_1,n_res_start_2+h1*n_res_delta_2], 
                                  n_layer = num_layer,
                                  spectral_radius = [spec_rad_start_1+h2*spec_rad_delta_1,spec_rad_start_2+h2*spec_rad_delta_2],
                                  sparsity = [sparsity_start_1+h3*sparsity_delta_1,sparsity_start_2+h3*sparsity_delta_2], 
                                  noise = noise_start+noise_delta*h4,
                                  n_transient = n_transient,
                                  random_state = count+100)
                    elif num_layer == 3:
                         esn = ESN(n_inputs = 1,
                                   n_outputs = 1,
                                   n_reservoir = [n_res_start_1+h1*n_res_delta_1,n_res_start_2+h1*n_res_delta_2,n_res_start_3+h1*n_res_delta_3], 
                                   n_layer = num_layer,
                                   spectral_radius = [spec_rad_start_1+h2*spec_rad_delta_1,spec_rad_start_2+h2*spec_rad_delta_2,spec_rad_start_3+h2*spec_rad_delta_3],
                                   sparsity = [sparsity_start_1+h3*sparsity_delta_1,sparsity_start_2+h3*sparsity_delta_2,sparsity_start_3+h3*sparsity_delta_3], 
                                   noise = noise_start+noise_delta*h4,
                                   n_transient = n_transient,
                                   random_state = count+100)
                    else:
                        print("Number of layers >= 4!")

                    fitt = esn.fit(np.ones(trainlen-trainbeg),data[trainbeg:trainlen], inspect = False)
                    predd =  esn.predict(np.ones(valid))
                    valid_error = np.sqrt(np.mean((predd.flatten() - data[trainlen:trainlen+valid])**2))

                    seeds[k,h1,h2,h3,h4] = count+100

                    print("validation RMSE: \n"+str(valid_error))
                    valid_error_rmse[k,h1,h2,h3,h4] = valid_error
                   
    count += 1
    k_valid_error_rmse = valid_error_rmse[k,:,:,:,:]
    print('=============> index of min validation RMSE', np.unravel_index(np.argmin(k_valid_error_rmse, axis=None), k_valid_error_rmse.shape))
    ind_vec = np.unravel_index(np.argmin(k_valid_error_rmse, axis=None), k_valid_error_rmse.shape)
    min_error = k_valid_error_rmse[ind_vec]    
    
    if min_error <= threshold:
        print('minimum RMSE of selected model = ', min_error)

        opt_ind = np.array(ind_vec).T
        opt_seeds = seeds[k,ind_vec[0],ind_vec[1],ind_vec[2],ind_vec[3]]
        print('Seed number: ', opt_seeds)

        if num_layer == 1:
            oesn = ESN(n_inputs = 1,
                       n_outputs = 1,
                       n_reservoir = [n_res_start_1+int(opt_ind[0]*n_res_delta_1)],
                       n_layer = num_layer,
                       spectral_radius = [spec_rad_start_1+opt_ind[1]*spec_rad_delta_1], 
                       sparsity = [sparsity_start_1+opt_ind[2]*sparsity_delta_1], 
                       noise = noise_start+noise_delta*opt_ind[3], 
                       n_transient = n_transient,
                       random_state = int(opt_seeds)) 
        elif num_layer == 2:
            oesn = ESN(n_inputs = 1,
                       n_outputs = 1,
                       n_reservoir = [n_res_start_1+int(opt_ind[0]*n_res_delta_1),n_res_start_2+int(opt_ind[0]*n_res_delta_2)],
                       n_layer = num_layer,
                       spectral_radius = [spec_rad_start_1+opt_ind[1]*spec_rad_delta_1,spec_rad_start_2+opt_ind[1]*spec_rad_delta_2], 
                       sparsity = [sparsity_start_1+opt_ind[2]*sparsity_delta_1,sparsity_start_2+opt_ind[2]*sparsity_delta_2], 
                       noise = noise_start+noise_delta*opt_ind[3], 
                       n_transient = n_transient,
                       random_state = int(opt_seeds)) 
        elif num_layer == 3:
            oesn = ESN(n_inputs = 1,
                       n_outputs = 1,
                       n_reservoir = [n_res_start_1+int(opt_ind[0]*n_res_delta_1),n_res_start_2+int(opt_ind[0]*n_res_delta_2),n_res_start_3+int(opt_ind[0]*n_res_delta_3)],
                       n_layer = num_layer,
                       spectral_radius = [spec_rad_start_1+opt_ind[1]*spec_rad_delta_1,spec_rad_start_2+opt_ind[1]*spec_rad_delta_2,spec_rad_start_3+opt_ind[1]*spec_rad_delta_3], 
                       sparsity = [sparsity_start_1+opt_ind[2]*sparsity_delta_1,sparsity_start_2+opt_ind[2]*sparsity_delta_2,sparsity_start_3+opt_ind[2]*sparsity_delta_3], 
                       noise = noise_start+noise_delta*opt_ind[3], 
                       n_transient = n_transient,
                       random_state = int(opt_seeds))
        else:
            print("Number of layers >= 4!")            

        opred_training.append(oesn.fit(np.ones(trainlen-trainbeg),data[trainbeg:trainlen], inspect = False))
        oprediction.append(oesn.predict(np.ones(future)))

        otest_error[k] = np.sqrt(np.mean((oprediction[k].flatten() - data[trainlen:trainlen+future])**2))
        print("----------------------------------------------------------------------")
        print("=====================================================>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>> Run #: ", k)
        print("Prediction RMSE: \n"+str(otest_error[k]))

        oq = oprediction[k] 

        if option == 0:
            osol[k,:] = np.concatenate((data_orig[trainbeg:trainlen+1].reshape(-1,1),oq),axis=0).reshape(-1,)
            k += 1
        elif option == 1:
            oq=oq.reshape((oq.shape[0],1))
            oq=np.concatenate((forcing[trainbeg:trainlen],oq),axis=0)
    
            osol[k,0] = data_orig[trainbeg]
            for n in range(trainbeg,oNum):
                if index_data == 1 or index_data == 2:
                    k1 = dtau*(F(osol[k,n-trainbeg],index_data))
                    k2 = dtau*F(osol[k,n-trainbeg]+k1/2,index_data)
                    k3 = dtau*F(osol[k,n-trainbeg]+k2/2,index_data)
                    k4 = dtau*F(osol[k,n-trainbeg]+k3,index_data)
                elif index_data == 3:
                    k1 = dtau*(Ft(osol[k,n-trainbeg],dtau*n,index_data))
                    k2 = dtau*Ft(osol[k,n-trainbeg]+k1/2,dtau*n + dtau/2,index_data)
                    k3 = dtau*Ft(osol[k,n-trainbeg]+k2/2,dtau*n + dtau/2,index_data)
                    k4 = dtau*Ft(osol[k,n-trainbeg]+k3,dtau*n + dtau,index_data)
                else:
                    print('Unexpected data!')
                osol[k,n+1-trainbeg] = osol[k,n-trainbeg]+(k1+2*k2+2*k3+k4)/6+dtau*oq[n-trainbeg] 
            k += 1
        else:
            print('Unexpected option!')

        print("-------------------------- run completed -----------------------------")        
        print("----------------------------------------------------------------------")

if option == 1:    
  np.savetxt('Ex'+'{0}'.format(index_data)+'_'+'{0}'.format(trainlen+valid)+'-'+'{0}'.format(valid)+'_'+'{0}'.format(n_ens)+'ens_fin.csv', np.c_[osol], delimiter=',')
elif option == 0:
  np.savetxt('Baseline_Ex'+'{0}'.format(index_data)+'_'+'{0}'.format(trainlen+valid)+'-'+'{0}'.format(valid)+'_'+'{0}'.format(n_ens)+'ens_fin.csv', np.c_[osol], delimiter=',')
else:
  print('Unexpected option!')

print('Plotting results...')
select_and_plot_results(index_data, n_ens, trainlen+valid, valid, future-valid, trainbeg, dtau, option)

print('Total time: ', timeit.default_timer() - t0 )
