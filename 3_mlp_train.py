# -*- coding: utf-8 -*-
"""
Created on Wed Jan 10 17:18:28 2024

@author: ludej
"""

import numpy as np
from random import random

#Save the activations
#implement backpropagation
#implement gradient descent
#implement train method
#train our network with some dummy dataset
#make some predictions

class MLP(object):
    '''
    A Multilayer Perceptron class
    '''
    
    def __init__(self, num_inputs=3, hidden_layers=[3, 3], num_outputs=2):
        """Constructor for the MLP. Takes the number of inputs,
        a variable number of hidden layers, and number of outputs
        
        Args:
            num_inputs (int): Number of inputs
            hidden_layers (list): A list of ints for the hidden layers
            num_outputs (int): Number of outputs
        
        """

        self.num_inputs = num_inputs
        self.hidden_layers = hidden_layers
        self.num_outputs = num_outputs
        
        #create a generic representation of the layers
        layers = [num_inputs] + hidden_layers + [num_outputs]
        
        #create random connection weights for the layers
        weights = []
        
        for i in range(len(layers)-1):
             w = np.random.rand(layers[i], layers[i+1])
             weights.append(w)
        self.weights = weights
        
        activations = []
        for i in range(len(layers)):
            a = np.zeros(layers[i]) #we use the number of zeros equal to the number of neuron at each layer
            activations.append(a)
        self.activations = activations  #we store all the results in an instance varibale called activations
            
        
        derivatives = []
        for i in range(len(layers)-1): #We use '-1' here because the weight matrixes are inbetween layers, so it's usually 1-less in dimension
            d = np.zeros((layers[i], layers[i+1])) #'layers[i] reps the number of neurons in the current layer, and 'layers[i+1]' reps the number of neurons in the subsequent layer
            #Leo: not having the double brackets above could cause an error
            derivatives.append(d)
        self.derivatives = derivatives  #we store all the results in an instance varibale called derivatiaves
            
            
             
    def forward_propagate(self, inputs):
        """
        
        Computes forward propagation of the network based on input signals

        Parameters
        ----------
        inputs : TYPE
            DESCRIPTION.

        Returns
        -------
        activations : TYPE
            DESCRIPTION.

        """
        
        activations = inputs
        self.activations[0] = inputs
        
        #iterate through the network layers
        for i, w in enumerate(self.weights):
            #Calculate matrix multiplication between previous activation and weight matrix
            net_inputs = np.dot(activations, w)
            
            
            #Apply the activation function
            activations = self._sigmoid(net_inputs)
            self.activations[i+1] = activations #save the activations
        
        #a_3 = s(h_3)
        #h_3 = a_2 * W_2
            
        #return output layer activation   
        return activations
    
    def back_propagate(self, error, verbose=False):
        #dE/dW_i = (y-a_[i+1]) s'(h_[i+1])) a_i
        #s'(h_[i+1]) = s(h_[i+1])(1 - s(h_[i+1]))
        #s(h_[i+1]) = a_[i+1]
        
        #dE/dW_[i-1] = (y - a_[i+1]) s'(h_[i+1]) W_i S'(h_i) a_[i-1]
        
        for i in reversed(range(len(self.derivatives))): #we use reversed because we want to go from right to left
            activations = self.activations[i+1]
            delta = error * self._sigmoid_derivative(activations) #ndarray([0.1, 0.2]) --> ndarray([[0.1], [0.2]])
            delta_reshaped = delta.reshape(delta.shape[0], -1).T
            current_activations = self.activations[i] #ndarray([0.1, 0.2]) --> ndarray([[0.1], [0.2]])
            current_activations_reshaped = current_activations.reshape(current_activations.shape[0], -1)
            
            self.derivatives[i] = np.dot(current_activations_reshaped, delta_reshaped)
            error = np.dot(delta, self.weights[i].T)
            
            if verbose:
                print("Derivatives for W{}: {}".format(i, self.derivatives[i]))
            
        return error 
    
    def gradient_descent(self, learning_rate): #Leo: Try misspelling 'self' on this line to "slef" and see what happens
        #Now, we will loop through all the weights
        for i in range(len(self.weights)):
            weights = self.weights[i]
            print("Original W{} {}".format(i, weights))
            
            derivatives = self.derivatives[i]
            
            #weights = weights + derivatives * learning_rate
            weights += derivatives * learning_rate #Line above rewritten
            print("Updated W{} {}".format(i, weights))
            
            
    #Now we create the Train method
    def train(self, inputs, targets, epochs, learning_rate):
        #The number of epochs is basically how many time you want to feed the whole dataset to the Neural Network
        for i in range(epochs):
            sum_error=0
            for input, target in zip(inputs, targets):
                #perform forward propagation
                output = self.forward_propagate(input)
                
                #calculate the error/loss
                error = target - output
                
                #back propagation
                self.back_propagate(error)
                
                #Apply gradient descent
                self.gradient_descent(learning_rate)#Leo: if the learning rate is changed to 1 the difference in the original and corresponding updated weight will be larger
                
                sum_error += self._mse(target, output)
                
            #Report Error for each epoch so we can see whether we are improving
            print("Error: {} at epoch {}".format(sum_error / len(inputs), i))
            
            
    
    
    def _mse(self, target, output):
        return np.average((target - output)**2)

                 
            
    def _sigmoid_derivative(self, x):
        return x * (1.0 - x)
    
    
    def _sigmoid(self, x):
        y = 1.0 / (1 + np.exp(-x))
        return y
            

if __name__ == "__main__":
    
    #Create a dataset to train a network for the sum operation
    inputs = np.array([[random() / 2 for _ in range(2)] for _ in range(1000)]) #array([[0.1, 0.2], [0.3, 0.4]])
    
    
    targets = np.array([[i[0] + i[1]] for i in inputs])#array([[0.1, 0.2], [0.3, 0.4]])
    
    
    #create an MLP
    mlp = MLP(2, [5], 1)
    
    #train our mlp
    mlp.train(inputs, targets, 50, 0.1)
    
    
    #create dummy data
    input = np.array([0.3, 0.1])
    target = np.array([0.4]) #the expect target is the sum of 0.3 and 0.1, so we enter 0.4
    
    output = mlp.forward_propagate(input)
    print()
    print()
    print("Our network believes that {} + {} is equal to {}".format(input[0], input[1], output[0]))
    
    
    
    '''
    #create some inputs/data
    input = np.array([0.1, 0.2])
    target = np.array([0.3])
    '''
    
    '''
    #perform forward propagation
    output = mlp.forward_propagate(input)
    
    #calculate the error/loss
    error = target - output
    
    #back propagation
    mlp.back_propagate(error)
    
    #Apply gradient descent
    mlp.gradient_descent(learning_rate=0.1)#Leo: if the learning rate is changed to 1 the difference in the original and corresponding updated weight will be larger
    
    #print the results
    #print("The network input is: {}".format(inputs))
    #print("The network output is: {}".format(outputs))
    '''
    
    
    
    
    
    