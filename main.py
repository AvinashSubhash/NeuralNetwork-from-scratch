import numpy as np

class Simple_NN:
    
    def __init__(self):
        self.X = np.random.randn(3,1)
        self.Y = np.zeros((2,1))

    def layer_sizes(self):
        n_0 = self.X.shape[0]
        n_1 = 4
        n_2 = self.Y.shape[0]
        return [n_0,n_1,n_2]


    def InitializeParameters(self,dimension=layer_sizes()):
        input_size = dimension[0]
        hidden_layer_size = dimension[1]
        output_size = dimension[2]

        w1 = np.random.randn(hidden_layer_size,input_size)*0.001
        b1 = np.zeros((hidden_layer_size,1))
        w2 = np.random.randn(output_size,hidden_layer_size)*0.001
        b2 = np.zeros((output_size,1))

        data = {"w1": w1 ,"w2": w2 ,"b1": b1 ,"b2": b2}
        data['X']=self.X
        data['Y'] = self.Y
        self.CheckDimension(data)
        return data

    def ForwardPropagation(self,data,input_data,i):
        z = np.dot(data['w'+str(i)],input_data)
        a = np.sigmoid(z)
        print("Output Layer 1: ",z.shape,"   Normalized Output Layer 1: ",a.shape)
        data['z'+str(i)] = z
        data['a'+str(i)] = a
        return data

    def CheckDimension(self,data):
        print("Input Shape: ",data['X'].shape)
        for i in range(1,3):
            print("Layer ",i,": ",data['w'+str(i)].shape,end='')
            print("  Bias ",i,": ",data['b'+str(i)].shape)
        print("Output Shape: ",data['Y'].shape)

    def BackwardPropagation(self,data,i):
        m = self.X.shape[1]
        dz2 = data['a'+str(2)] - self.Y
        dw2 = (1/m)*np.multiply(dz2,data['a'+str(1)].T)
        db2 = (1/m)*np.sum(dz2,axis=1,keepdims=True)
        dz1 = np.dot(data['w'+str(2)].T,(np.multiply(dz2,data['a'+str(1)])))
        dw1 = (1/m)*np.dot(dz1,self.X.T)
        db1 = (1/m)*np.sum(dz1,axis=1,keepdims=True)
        
        

if __name__ == "__main__":
    nn_network = Simple_NN()
    nn_network.InitializeParameters()
