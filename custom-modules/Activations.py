import numpy as np

def sigmoid(num):
    return 1/(1+np.exp(num))

def relu(num):
    return np.max(0,num)

def tanh(num):
    num = np.exp(num)-np.exp(-num)
    den = np.exp(num)+np.exp(-num)

def drelu(num):
    if num <0:
        return 0
    return 1

def dtanh(num):
    return 1 - (tanh(num)**2)

def dsigmoid(num):
    return sigmoid(num)*(1-sigmoid(num))