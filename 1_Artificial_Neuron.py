# -*- coding: utf-8 -*-
"""
Created on Mon Jan  8 15:47:33 2024

@author: ludej
"""

import math

def sigmoid(x):
    y = 1.0/(1 + math.exp(-x))
    return y

def activate(inputs, weights):
    #perfrom net input
    h=0
    for x, w in zip(inputs, weights): #The zip function helps us pass variables bundled together
        h += x*w
        
    #perform activation
    return sigmoid(h)

if __name__ == "__main__":
    #We represent the inputs and weights with lists
    inputs = [.5, .3, .2]
    weights = [.4, .7, .2]
    output = activate(inputs, weights) #the activate function takes inputs and weights as arguments or parameters
    print(output)