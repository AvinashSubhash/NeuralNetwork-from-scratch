import numpy as np
def CostCalculation(data,debug=False):
        m = data['Y'].shape[1]
        logprob = np.multiply(data['a'+str(data['count'])],data['Y'])+np.multiply(np.log(1-data['a'+str(data['count'])]),1-data['Y'])
        cost = (-1/m)*np.sum(logprob)
        cost = float(np.squeeze(cost))
        if debug == True:
            print("m: ",m)
            print("LogProb: ",logprob)
            print("Cost: ",cost)
        return cost