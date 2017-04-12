import numpy as np

def Sigmoid(x, deriv=False):
    #Returns the derivative of the sigmoid function
    if(deriv==True):
        return (x*(1-x))
    #Cals the sigmoid value of the input
    return 1/(1+np.exp(-x))

#Input data
x = np.array([
[0,0,1],
[0,1,1],
[1,0,1],
[1,1,1]
])

#output for each inputs
y = np.array([
[0],
[1],
[1],
[0]
])

#Random seed
np.random.seed(1)

#synapses
syn0 = 2*np.random.random((3,4)) - 1
syn1 = 2*np.random.random((4,1)) - 1

#training
for j in xrange(60000):

    #creating layers
    l0 = x
    l1 = Sigmoid(np.dot(l0, syn0))
    l2 = Sigmoid(np.dot(l1, syn1))

    #backpropogation
    l2_error = y-l2
    if(j % 10000) == 0 :
        print 'error:' + str(np.mean(np.abs(l2_error)))

    #cal delta
    l2_delta = l2_error*Sigmoid(l2, deriv = True)
    l1_error = l2_delta.dot(syn1.T)
    l1_delta = l1_error*Sigmoid(l1, deriv = True)

    #update synapses or weights
    syn1 += l1.T.dot(l2_delta)
    syn0 += l0.T.dot(l1_delta)

print 'output after training'
print l2
