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


    def InitializeParameters(self):
        dimension = self.layer_sizes()
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

    def ForwardPropagation(self,data,input_data):
        
        z = np.dot(data['w'+str(1)],input_data)
        a = 1/(1 + np.exp(-z))
        #print("Output Layer 1: ",z.shape,"   Normalized Output Layer 1: ",a.shape)
        data['z'+str(1)] = z
        data['a'+str(1)] = a
        z = np.dot(data['w'+str(2)],a)
        a = 1/(1 + np.exp(-z))
        #print("Output Layer 2: ",z.shape,"   Normalized Output Layer 1: ",a.shape)
        data['z'+str(2)] = z
        data['a'+str(2)] = a
        return data

    def CheckDimension(self,data):
        print("Input Shape: ",data['X'].shape)
        for i in range(1,3):
            print("Layer ",i,": ",data['w'+str(i)].shape,end='')
            print("  Bias ",i,": ",data['b'+str(i)].shape)
        print("Output Shape: ",data['Y'].shape,"\n\n")

    def BackwardPropagation(self,data):
        m = self.Y.shape[1]
        dz2 = data['a'+str(2)] - self.Y
        dw2 = (1/m)*np.multiply(dz2,np.transpose(data['a'+str(1)]))
        db2 = (1/m)*np.sum(dz2,axis=1,keepdims=True)
        dz1 = np.dot(np.transpose(data['w'+str(2)]),dz2)*(1-np.power(data['a'+str(1)],2))        
        dw1 = (1/m)*np.dot(dz1,np.transpose(self.X))
        db1 = (1/m)*np.sum(dz1,axis=1,keepdims=True)
        data['dw'+str(1)] = dw1
        data['dw'+str(2)] = dw2
        data['db'+str(1)] = db1
        data['db'+str(2)] = db2
        return data

    def CostCalculation(self,data):
        m = self.Y.shape[1]
        logprob = np.multiply(data['a'+str(2)],self.Y)+np.multiply(np.log(1-data['a'+str(2)]),1-self.Y)
        cost = (-1/m)*np.sum(logprob)
        cost = float(np.squeeze(cost))
        return cost

    def UpdateValues(self,data,learning_rate=1.12):
        for i in range(1,3):
            data['w'+str(i)] = data['w'+str(i)] - (learning_rate*data['dw'+str(i)])
            data['b'+str(i)] = data['b'+str(i)] - (learning_rate*data['db'+str(i)])
        return data

        

    def Driver(self,num_iterations=10000):

        data = self.InitializeParameters()

        for i in range(num_iterations):
            data = self.ForwardPropagation(data,self.X)
            if i in list(range(0,11000,1000)):
                print("Cost at Iteration: ",i," = ",self.CostCalculation(data))
            data = self.BackwardPropagation(data)
            data = self.UpdateValues(data)



        
        

if __name__ == "__main__":
    nn_network = Simple_NN()
    nn_network.Driver()
