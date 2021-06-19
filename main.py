import numpy as np

def InitializeParameters(dimension):
    input_size = dimension[0]
    hidden_layer_size = dimension[1]
    output_size = dimension[2]

    w1 = np.random.randn((hidden_layer_size,input_size))*0.001
    b1 = np.zeros((hidden_layer_size,1))
    w2 = np.random.randn((output_size,hidden_layer_size))*0.001
    b2 = np.zeros((output_size,1))

    data = {"w1": w1 ,"w2": w2 ,"b1": b1 ,"b2": b2}
    return data

def ForwardPropagation(data,input_data,i):
    z = np.dot(data['w'+str(i)],input_data)
    a = np.sigmoid(z)
    data['z'+str(i)] = z
    data['a'+str(i)] = a
    return data

def BackwardPropagation(data,i):



