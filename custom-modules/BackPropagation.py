import numpy as np

def BackwardPropagation(data,no_of_layers,debug=False):
        m = data['Y'].shape[1]
        for i in range(no_of_layers,0,-1):
            
            if i == no_of_layers:
                dz = data['a'+str(no_of_layers)] - data['Y']
            else:
                dz = np.dot(np.transpose(data['w'+str(i+1)]),data['dz'+str(i+1)]*(1-np.power(data['a'+str(i)],2)))
            dw = (1/m)*np.dot(data['dz'+str(i)],np.transpose(data['a'+str(i-1)]))
            db = (1/m)*np.sum(data['dz'+str(i)],axis=1,keepdims=True)
            if debug == True:
                print("\n dz",i,": ",dz)
                print("\n dw",i,": ",dw)
                print("\n db",i,": ",db)
            data['dw'+str(i)] = dw
            data['db'+str(i)] = db
            
        return data

BackwardPropagation(0,5)