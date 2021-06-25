import numpy as np

def ForwardPropagation(data,input_data,no_of_layers,debug=False):
        data['a'+str(-1)] = input_data
        for i in range(no_of_layers):

            z = np.dot(data['w'+str(i)],input_data)
            a = 1/(1 + np.exp(-z))
            print("Output Layer ",i+1,": ",z.shape,"   Activation Output Layer ",i+1,": ",a.shape)
            if debug == True:
                print("\nz",i+1,": ",z)
                print("\na",i+1,": ",a)
            data['z'+str(i)] = z
            data['a'+str(i)] = a
            
        return data