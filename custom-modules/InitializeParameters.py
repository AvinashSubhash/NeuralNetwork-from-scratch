import numpy as np
def InitializeParameters(dimensions,debug=False):

    input_size = dimensions[0]
    output_size = dimensions[-1]
    data = {}
    for i in range(len(dimensions)-1):
        w = np.random.randn(dimensions[i+1],dimensions[i])*0.001
        b = np.zeros((dimensions[i+1],1))
        if debug == True:
            print("Layer ",i+1,": ")
            print("W",str(i+1),": ",w.shape)
            print("b",str(i+1),": ",b.shape,"\n")
        data['w'+str(i+1)] = w
        data['b'+str(i+1)] = b
    if debug == True:
        print("Output Size: ",output_size,"\n")
    #CheckDimension(data)
    return data