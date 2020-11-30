# -*- coding: utf-8 -*-
"""
Created on Mon Jul  1 17:34:24 2019
@author: Soon Hoe Lim
Some comments:
The following codes were modified and extended from the sample codes from 
[https://github.com/cknd/pyESN], which implements only a simple version of shallow echo state network.
At the moment the maximum number of layers is restricted to four, but extension to more layers 
can also be considered. 
Note that in general ESNs are very sensitive to hyperparameters, making tuning the optimal ones challenging. 
Training the network becomes more challenging as the number of layers increases.
"""

import numpy as np
import matplotlib.pyplot as plt


def correct_dimensions(s, targetlength):
    """checks the dimensionality of some numeric argument s, broadcasts it
       to the specified length if possible.
    Args:
        s: None, scalar or 1D array
        targetlength: expected length of s
    Returns:
        None if s is None, else numpy vector of length targetlength
    """
    if s is not None:
        s = np.array(s)
        if s.ndim == 0:
            s = np.array([s] * targetlength)
        elif s.ndim == 1:
            if not len(s) == targetlength:
                raise ValueError("arg must have length " + str(targetlength))
        else:
            raise ValueError("Invalid argument")
    return s


def identity(x):
    return x


class ESN():

    def __init__(self, n_inputs=1, n_outputs=1, n_reservoir=500, n_layer=1, nonlin = 1,
                 spectral_radius=0.95, sparsity=0, noise=0.001, noise_option = 0,
                 input_shift=None, input_scaling=None, teacher_forcing=False, feedback_scaling=None,
                 teacher_scaling=None, teacher_shift=None, n_transient=0, noise_in_prediction=True,
                 out_activation=identity, inverse_out_activation=identity,
                 random_state=None, silent=True):
        """
        Args:
            n_inputs: nr of input dimensions
            n_outputs: nr of output dimensions
            n_reservoir: nr of reservoir neurons
            n_layer: nr of stacked layers
            nonlin: highest power of reservoir states used for read out
            spectral_radius: spectral radius of the recurrent weight matrix
            sparsity: proportion of recurrent weights set to zero
            noise: noise added to each neuron (regularization)
            noise_option: specify choice of distribution for sampling noise (0: Unif(-1/2,1/2) or 1: Normal(0,1))
            noise_in_prediction: specify whether we add noise (the same used in training phase) also in the auto prediction phase
            n_transient: number of first hidden states discarded 
            input_shift: scalar or vector of length n_inputs to add to each
                        input dimension before feeding it to the network.
            input_scaling: scalar or vector of length n_inputs to multiply
                        with each input dimension before feeding it to the netw.
            teacher_forcing: if True, feed the target back into output units
            teacher_scaling: factor applied to the target signal
            teacher_shift: additive term applied to the target signal
            out_activation: output activation function (applied to the readout)
            inverse_out_activation: inverse of the output activation function
            random_state: positive integer seed, np.rand.RandomState object,
                          or None to use numpy's builting RandomState.
            silent: supress messages
        """
        self.n_inputs = n_inputs
        self.n_reservoir = n_reservoir
        self.n_outputs = n_outputs
        self.n_layer = n_layer
        self.n_transient = n_transient
        self.nonlin = nonlin
        self.spectral_radius = spectral_radius
        self.sparsity = sparsity
        self.noise = noise
        self.noise_option = noise_option
        self.noise_in_prediction = noise_in_prediction
        self.input_shift = correct_dimensions(input_shift, n_inputs)
        self.input_scaling = correct_dimensions(input_scaling, n_inputs)
        self.teacher_scaling = teacher_scaling
        self.teacher_shift = teacher_shift
        self.out_activation = out_activation
        self.inverse_out_activation = inverse_out_activation
        self.random_state = random_state

        # the given random_state might be either an actual RandomState object,
        # a seed or None (in which case we use numpy's builtin RandomState)
        if isinstance(random_state, np.random.RandomState):
            self.random_state_ = random_state
        elif random_state:
            try:
                self.random_state_ = np.random.RandomState(random_state)
            except TypeError as e:
                raise Exception("Invalid seed: " + str(e))
        else:
            self.random_state_ = np.random.mtrand._rand

        self.teacher_forcing = teacher_forcing
        self.silent = silent
        self.initweights()

    def initweights(self):
        # initialize recurrent weights:
        # begin with a random matrix centered around zero:
        radius = np.zeros((self.n_layer,1))
        
        if self.n_layer >= 1:
            W1 =self.random_state_.rand(self.n_reservoir[0], self.n_reservoir[0]) -0.5
            W1 = W1/(np.linalg.norm(W1)+0.001)
            # delete the fraction of connections given by (self.sparsity):
            W1[self.random_state_.rand(*W1.shape) < self.sparsity[0]] = 0
            # compute the spectral radius of these weights:
            radius[0] = np.max(np.abs(np.linalg.eigvals(W1)))
            # rescale them to reach the requested spectral radius:
            self.W1 = W1 * (self.spectral_radius[0]/ radius[0])
            if self.silent == False: 
                print('shape of W1: ', self.W1.shape)
                plt.imshow(self.W1)
                plt.colorbar()
                plt.show()
                
        if self.n_layer >= 2:
            W2 =self.random_state_.rand(self.n_reservoir[1], self.n_reservoir[1]) -0.5 
            W2 = W2/(np.linalg.norm(W2)+0.001)
            # delete the fraction of connections given by (self.sparsity):
            W2[self.random_state_.rand(*W2.shape) < self.sparsity[1]] = 0
            # compute the spectral radius of these weights:
            radius[1] = np.max(np.abs(np.linalg.eigvals(W2)))
            # rescale them to reach the requested spectral radius:
            self.W2 = W2 * (self.spectral_radius[1]/ radius[1])
            if self.silent == False: 
                print('shape of W2: ', self.W2.shape)
                plt.imshow(self.W2)
                plt.colorbar()
                plt.show()
        
            self.W_in_2 =self.random_state_.rand(self.n_reservoir[1], self.n_reservoir[0]) -0.5
            self.W_in_2 = self.W_in_2/(np.linalg.norm(self.W_in_2)+0.001)
            if self.silent == False: 
                print('shape of W_in_2: ', self.W_in_2.shape)
                #print('norm of W_in_2: ', np.linalg.norm(self.W_in_2))
                plt.imshow(self.W_in_2)
                plt.colorbar()
                plt.show()
                
        if self.n_layer >= 3:
            W3 =self.random_state_.rand(self.n_reservoir[2], self.n_reservoir[2])-0.5 
            W3 = W3/(np.linalg.norm(W3)+0.001) 
            # delete the fraction of connections given by (self.sparsity):
            W3[self.random_state_.rand(*W3.shape) < self.sparsity[2]] = 0
            # compute the spectral radius of these weights:
            radius[2] = np.max(np.abs(np.linalg.eigvals(W3)))
            # rescale them to reach the requested spectral radius:
            self.W3 = W3 * (self.spectral_radius[2]/ radius[2])
            if self.silent == False: 
                print('shape of W3: ', self.W3.shape)
                plt.imshow(self.W3)
                plt.colorbar()
                plt.show()
                
            self.W_in_3 =self.random_state_.rand(self.n_reservoir[2], self.n_reservoir[1]) -0.5
            self.W_in_3 = self.W_in_3/(np.linalg.norm(self.W_in_3)+0.001)
            if self.silent == False: 
                print('shape of W_in_3: ', self.W_in_3.shape)
                #print('norm of W_in_3: ', np.linalg.norm(self.W_in_3))
                plt.imshow(self.W_in_3)
                plt.colorbar()
                plt.show()
                
        if self.n_layer >= 4:
            W4 =self.random_state_.rand(self.n_reservoir[3], self.n_reservoir[3])-0.5 
            W4 = W4/(np.linalg.norm(W4)+0.001) 
            # delete the fraction of connections given by (self.sparsity):
            W4[self.random_state_.rand(*W4.shape) < self.sparsity[3]] = 0
            # compute the spectral radius of these weights:
            radius[3] = np.max(np.abs(np.linalg.eigvals(W4)))
            # rescale them to reach the requested spectral radius:
            self.W4 = W4 * (self.spectral_radius[3]/ radius[3])
            if self.silent == False: 
                print('shape of W4: ', self.W4.shape)
                plt.imshow(self.W4)
                plt.colorbar()
                plt.show()
                
            self.W_in_4 =self.random_state_.rand(self.n_reservoir[3], self.n_reservoir[2]) -0.5
            self.W_in_4 = self.W_in_4/(np.linalg.norm(self.W_in_4)+0.001)
            if self.silent == False: 
                print('shape of W_in_4: ', self.W_in_4.shape)
                #print('norm of W_in_4: ', np.linalg.norm(self.W_in_4))
                plt.imshow(self.W_in_4)
                plt.colorbar()
                plt.show()        
                
        if self.n_layer >= 5:
            print('why need n_layer >=5')
            
        # random input weights:
        self.W_in = self.random_state_.rand(self.n_reservoir[0], self.n_inputs) * 2 - 1
        if self.silent == False: 
            print('shape of W_in: ', self.W_in.shape)
            plt.imshow(self.W_in)
            plt.colorbar()
            plt.show()
        
        # random feedback (teacher forcing) weights:
        self.W_feedb = self.random_state_.rand(self.n_reservoir[self.n_layer-1], self.n_outputs) * 2 - 1
        if self.silent == False: 
            print('shape of W_feedb: ', self.W_feedb.shape)
            plt.imshow(self.W_feedb)
            plt.colorbar()
            plt.show()

        
    def _update(self, statel, input_pattern, output_pattern, l,option):
        """performs one update step.
        i.e., computes the next network state by applying the recurrent weights
        to the last state & and feeding in the current "input" and "output" patterns
        """
        if l==0:
            if self.n_layer == 1:    
                preactivation = (np.dot(self.W1, statel)
                                 + np.dot(self.W_in, input_pattern)
                                 + np.dot(self.W_feedb, output_pattern))
            else:
                preactivation = (np.dot(self.W1, statel)
                                 + np.dot(self.W_in, input_pattern))
        elif l==1:
            if self.n_layer == 2:
                preactivation = (np.dot(self.W2, statel) + np.dot(self.W_in_2,input_pattern)
                                     + np.dot(self.W_feedb, output_pattern))
            else:
                preactivation = (np.dot(self.W2, statel)  + np.dot(self.W_in_2,input_pattern)) 
        elif l==2:
            if self.n_layer == 3:
                preactivation = (np.dot(self.W3, statel) + np.dot(self.W_in_3, input_pattern)
                                     + np.dot(self.W_feedb, output_pattern))
            else:
                preactivation = (np.dot(self.W3, statel)  + np.dot(self.W_in_3,input_pattern)) 
        elif l==3:
            if self.n_layer == 4:
                preactivation = (np.dot(self.W4, statel) + np.dot(self.W_in_4, input_pattern)
                                     + np.dot(self.W_feedb, output_pattern))
            else:
                print('why need > 4 layers?') 
        else:
            print('why need > 4 layers?')
            
        if option == True:
            if self.noise_option == 0:
                return (np.tanh(preactivation) + self.noise * (self.random_state_.rand(self.n_reservoir[l])-0.5))
            elif self.noise_option == 1:
                return (np.tanh(preactivation) + self.noise * (self.random_state_.randn(self.n_reservoir[l])))
            else:
                raise ValueError("Invalid argument")
        else:
            return np.tanh(preactivation) 
       

    def _scale_inputs(self, inputs):
        """for each input dimension j: multiplies by the j'th entry in the
        input_scaling argument, then adds the j'th entry of the input_shift
        argument."""
        if self.input_scaling is not None:
            inputs = np.dot(inputs, np.diag(self.input_scaling))
        if self.input_shift is not None:
            inputs = inputs + self.input_shift
        return inputs

    
    def _scale_teacher(self, teacher):
        """multiplies the teacher/target signal by the teacher_scaling argument,
        then adds the teacher_shift argument to it."""
        if self.teacher_scaling is not None:
            teacher = teacher * self.teacher_scaling
        if self.teacher_shift is not None:
            teacher = teacher + self.teacher_shift
        return teacher

    
    def _unscale_teacher(self, teacher_scaled):
        """inverse operation of the _scale_teacher method."""
        if self.teacher_shift is not None:
            teacher_scaled = teacher_scaled - self.teacher_shift
        if self.teacher_scaling is not None:
            teacher_scaled = teacher_scaled / self.teacher_scaling
        return teacher_scaled

    
    def fit(self, inputs, outputs, inspect=False):
        """
        Collect the network's reaction to training data, train readout weights.
        Args:
            inputs: array of dimensions (N_training_samples x n_inputs)
            outputs: array of dimension (N_training_samples x n_outputs)
            inspect: show a visualisation of the collected reservoir states
        Returns:
            the network's output on the training data, using the trained weights
        """
        # transform any vectors of shape (x,) into vectors of shape (x,1):
        if inputs.ndim < 2:
            inputs = np.reshape(inputs, (len(inputs), -1))
        if outputs.ndim < 2:
            outputs = np.reshape(outputs, (len(outputs), -1))
        # transform input and teacher signal:
        inputs_scaled = self._scale_inputs(inputs)
        teachers_scaled = self._scale_teacher(outputs)
      
        if not self.silent:
            print("harvesting states...")
            
        # step the reservoir through the given input,output pairs:
        # statesi[0,:] = zero is the initial condition
        if self.n_layer >= 1:
            states1 = np.zeros((inputs.shape[0], self.n_reservoir[0]))
        if self.n_layer >= 2:
            states2 = np.zeros((inputs.shape[0], self.n_reservoir[1]))
        if self.n_layer >= 3:
            states3 = np.zeros((inputs.shape[0], self.n_reservoir[2]))
        if self.n_layer >= 4:
            states4 = np.zeros((inputs.shape[0], self.n_reservoir[3]))
        if self.n_layer >=5:
            print('why need > 4 layers?')
        
        for n in range(1, inputs.shape[0]):
            if self.n_layer==1:
                states1[n,:] = self._update(states1[n-1,:], inputs_scaled[n,:],teachers_scaled[n-1,:],0,True)
            else:
                if self.n_layer>=1:
                    states1[n,:] = self._update(states1[n-1,:], inputs_scaled[n,:],np.zeros(teachers_scaled[n-1,:].shape),0,True)
                if self.n_layer>=2 and self.n_layer != 2:
                    states2[n,:] = self._update(states2[n-1,:], states1[n,:], np.zeros(teachers_scaled[n-1,:].shape),1,True)
                if self.n_layer==2:
                    states2[n,:] = self._update(states2[n-1,:], states1[n,:], teachers_scaled[n-1,:],1,True)
                if self.n_layer>=3 and self.n_layer != 3:
                    states3[n,:] = self._update(states3[n-1,:], states2[n,:], np.zeros(teachers_scaled[n-1,:].shape),2,True)
                if self.n_layer==3:
                    states3[n,:] = self._update(states3[n-1,:], states2[n,:], teachers_scaled[n-1,:],2,True)
                if self.n_layer>=4 and self.n_layer != 4:
                    states4[n,:] = self._update(states4[n-1,:], states3[n,:], np.zeros(teachers_scaled[n-1,:].shape),3,True)
                if self.n_layer==4:
                    states4[n,:] = self._update(states4[n-1,:], states3[n,:], teachers_scaled[n-1,:],3,True)                    
                if self.n_layer >=5:
                    print('why need > 4 layers?')
                        
        # learn the weights, i.e. find the linear combination of collected
        # network states that is closest to the target output
        if not self.silent:
            print("fitting...")
            
        # disregard the first few states:
        transient = self.n_transient
        
        if self.n_layer == 1:
            if self.nonlin == 1:
                extended_states = np.hstack((states1, inputs_scaled))
            elif self.nonlin == 2:
                extended_states = np.hstack((states1, states1**2, inputs_scaled))
            else:
                print('why need nonlin > 2?')
            self.laststate1 = states1[-1, :]
        elif self.n_layer == 2:
            if self.nonlin == 1:
                extended_states = np.hstack((states1, states2, inputs_scaled))
            elif self.nonlin == 2:
                extended_states = np.hstack((states1, states2, states1**2, states2**2, inputs_scaled))
            else:
                print('why need nonlin > 2?')
            self.laststate1= states1[-1,:]
            self.laststate2 = states2[-1, :]
        elif self.n_layer == 3:
            if self.nonlin == 1:
                extended_states = np.hstack((states1, states2, states3, inputs_scaled))
            elif self.nonlin == 2:
                extended_states = np.hstack((states1, states2, states3, states1**2, states2**2, states3**2, inputs_scaled))
            else:
                print('why need nonlin > 2?')    
            self.laststate1= states1[-1,:]
            self.laststate2 = states2[-1, :]
            self.laststate3 = states3[-1, :]
        elif self.n_layer == 4:
            if self.nonlin == 1:
                extended_states = np.hstack((states1, states2, states3, states4, inputs_scaled))
            elif self.nonlin == 2:
                extended_states = np.hstack((states1, states2, states3,states4, states1**2, states2**2, states3**2, 
                                             states4**2, inputs_scaled))
            else:
                print('why need nonlin > 2?')    
            self.laststate1= states1[-1,:]
            self.laststate2 = states2[-1, :]
            self.laststate3 = states3[-1, :]            
            self.laststate4 = states4[-1, :]
            
        elif self.n_layer >= 5:
            print('Are you sure to run with n_layer > 4?')
        
        # Solve for W_out:
        self.W_out = np.dot(np.linalg.pinv(extended_states[transient:, :]),
                            self.inverse_out_activation(teachers_scaled[transient:, :])).T

        # remember the last state for later:
        self.lastinput = inputs[-1, :]
        self.lastoutput = teachers_scaled[-1, :]

        # optionally visualize the collected states
        if inspect:
            #print(inputs_scaled.shape)
            #from matplotlib import pyplot as plt
            # (^-- we depend on matplotlib only if this option is used)
            plt.figure(figsize=(12,10))
            plt.imshow(extended_states.T, aspect='auto',
                       interpolation='nearest')
            plt.colorbar()

        if not self.silent:
            print("training error:")
            
        # apply learned weights to the collected states:
        pred_train = self._unscale_teacher(self.out_activation(np.dot(extended_states, self.W_out.T)))
        if not self.silent:
            print(np.sqrt(np.mean((pred_train - outputs)**2)))
        return pred_train

    
    def predict(self, inputs):
        """
        Apply the learned weights to the network's reactions 
        Args:
            inputs: array of dimensions (N_test_samples x n_inputs)
            continuation: if True, start the network from the last training state
        Returns:
            Array of output activations
        """
        if inputs.ndim < 2:
            inputs = np.reshape(inputs, (len(inputs), -1)) 
        n_samples = inputs.shape[0] # inputs vector has same dimension as the teacher signal vector

        laststate1 = self.laststate1
        if self.n_layer >= 2:
            laststate2 = self.laststate2
        if self.n_layer >= 3:
            laststate3 = self.laststate3
        if self.n_layer >= 4:
            laststate4 = self.laststate4
        if self.n_layer >= 5:
            print('why need > 4 layers?')
        lastinput = self.lastinput
        lastoutput = self.lastoutput

        inputs = np.vstack([lastinput, self._scale_inputs(inputs)])
        
        states1 = np.vstack([laststate1, np.zeros((n_samples, self.n_reservoir[0]))]) 
        if self.n_layer >= 2:
            states2 = np.vstack([laststate2, np.zeros((n_samples, self.n_reservoir[1]))])
        if self.n_layer >= 3:
            states3 = np.vstack([laststate3, np.zeros((n_samples, self.n_reservoir[2]))])
        if self.n_layer >= 4:
            states4 = np.vstack([laststate4, np.zeros((n_samples, self.n_reservoir[3]))]) 
        if self.n_layer >= 5:
            print('why need > 4 layers?')
        outputs = np.vstack([lastoutput, np.zeros((n_samples, self.n_outputs))])

        choice = self.noise_in_prediction
        
        for n in range(n_samples):
            if self.n_layer == 1:
                states1[n+1,:] = self._update(states1[n, :], inputs[n + 1, :], outputs[n, :],0,choice)
            else:
                if self.n_layer >= 1:
                    states1[n+1,:] = self._update(states1[n, :], inputs[n + 1, :], np.zeros(outputs[n, :].shape),0,choice)
                if self.n_layer>=2 and self.n_layer != 2:
                    states2[n + 1, :] = self._update(states2[n,:], states1[n+1,:], np.zeros(outputs[n, :].shape),1,choice) 
                if self.n_layer==2:
                    states2[n + 1, :] = self._update(states2[n,:], states1[n+1,:], outputs[n, :],1,choice)
                if self.n_layer >= 3 and self.n_layer !=3:
                    states3[n + 1, :] = self._update(states3[n,:], states2[n+1,:], np.zeros(outputs[n, :].shape),2,choice)
                if self.n_layer == 3:
                    states3[n + 1, :] = self._update(states3[n,:], states2[n+1,:], outputs[n, :],2,choice)
                if self.n_layer >= 4 and self.n_layer !=4:
                    states4[n + 1, :] = self._update(states4[n,:], states3[n+1,:], np.zeros(outputs[n, :].shape),3,choice)
                if self.n_layer == 4:
                    states4[n + 1, :] = self._update(states4[n,:], states3[n+1,:], outputs[n, :],3,choice)
                if self.n_layer >= 5:
                    print('why need > 4 layers?')

            if self.n_layer == 1:
                q=states1[n+1,:]
                if self.nonlin == 2:
                    q2=states1[n+1,:]**2
                    q=np.concatenate([q,q2])
                if self.nonlin > 2:
                    print('why need nonlin > 2?')
            elif self.n_layer == 2:
                q=np.concatenate([states1[n+1,:],states2[n+1,:]])
                if self.nonlin == 2:
                    q2=np.concatenate([states1[n+1,:]**2,states2[n+1,:]**2])
                    q=np.concatenate([q,q2])
                if self.nonlin > 2:
                    print('why need nonlin > 2?')
            elif self.n_layer == 3:
                q=np.concatenate([states1[n+1,:],states2[n+1,:],states3[n+1,:]])
                if self.nonlin == 2:
                    q2=np.concatenate([states1[n+1,:]**2,states2[n+1,:]**2,states3[n+1,:]**2])
                    q=np.concatenate([q,q2])
                if self.nonlin > 2:
                    print('why need nonlin > 2?')
            elif self.n_layer == 4:
                q=np.concatenate([states1[n+1,:],states2[n+1,:],states3[n+1,:],states4[n+1,:]])
                if self.nonlin == 2:
                    q2=np.concatenate([states1[n+1,:]**2,states2[n+1,:]**2,states3[n+1,:]**2,states4[n+1,:]**2])
                    q=np.concatenate([q,q2])
                if self.nonlin > 2:
                    print('why need nonlin > 2?')
            else:
                print('Are you sure to run more than 4 layers?')

            outputs[n + 1, :] = self.out_activation(np.dot(self.W_out, np.concatenate([q, inputs[n + 1, :]])))

        return self._unscale_teacher(self.out_activation(outputs[1:]))