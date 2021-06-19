import numpy as np

X = np.random.randn(3,1)
Y = np.zeros((2,1))

def layer_sizes(X,Y):
    n_0 = X.shape[0]
    n_1 = 4
    n_2 = Y.shape[0]
    return [n_0,n_1,n_2]


def InitializeParameters(dimension=layer_sizes(X,Y)):
    input_size = dimension[0]
    hidden_layer_size = dimension[1]
    output_size = dimension[2]

    w1 = np.random.randn(hidden_layer_size,input_size)*0.001
    b1 = np.zeros((hidden_layer_size,1))
    w2 = np.random.randn(output_size,hidden_layer_size)*0.001
    b2 = np.zeros((output_size,1))

    data = {"w1": w1 ,"w2": w2 ,"b1": b1 ,"b2": b2}
    data['X']=X
    data['Y'] = Y
    CheckDimension(data)
    return data

def ForwardPropagation(data,input_data,i):
    z = np.dot(data['w'+str(i)],input_data)
    a = np.sigmoid(z)
    data['z'+str(i)] = z
    data['a'+str(i)] = a
    return data

def CheckDimension(data):
    print("Input Shape: ",data['X'].shape)
    for i in range(1,3):
        print("Layer ",i,": ",data['w'+str(i)].shape,end='')
        print("  Bias ",i,": ",data['b'+str(i)].shape)
    print("Output Shape: ",data['Y'].shape)

def BackwardPropagation(data,i):
    pass

if __name__ == "__main__":
    InitializeParameters()
