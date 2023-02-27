import autograd.numpy as np

'''
A list of cost functions for supervised learning.  Use the choose_cost function
to choose the desired cost with input data (x_in,y_in).  The aim here was to 
create a library of cost functions while keeping things as simple as possible 
(i.e., without the use of object oriented programming).  
'''

def choose_cost(x_in,y_in,cost,**kwargs):
    # define x and y as globals so all cost functions are aware of them
    global x,y
    x = x_in
    y = y_in
    
    # make any other variables not explicitly input into cost functions globally known
    global lam
    lam = 0
    if 'lam' in kwargs:
        lam = kwargs['lam']
    
    # make cost function choice
    cost_func = 0
    if cost == 'least_squares':
        cost_func = least_squares
    if cost == 'least_absolute_deviations':
        cost_func = least_absolute_deviations
    if cost == 'softmax':
        cost_func = softmax
    if cost == 'relu':
        cost_func = relu
    if cost == 'counter':
        cost_func = counting_cost

    if cost == 'multiclass_perceptron':
        cost_func = multiclass_perceptron
    if cost == 'multiclass_softmax':
        cost_func = multiclass_softmax
    if cost == 'multiclass_counter':
        cost_func = multiclass_counting_cost
        
    return cost_func

###### basic model ######
# compute linear combination of input point
def model(x,w):
    a = w[0] + np.dot(x.T,w[1:])
    return a.T

###### cost functions #####
# an implementation of the least squares cost function for linear regression
def least_squares(w):
    cost = np.sum((model(x,w) - y)**2)
    return cost/float(np.size(y))

# a compact least absolute deviations cost function
def least_absolute_deviations(w):
    cost = np.sum(np.abs(model(x,w) - y))
    return cost/float(np.size(y))

# the convex softmax cost function
def softmax(w):
    cost = np.sum(np.log(1 + np.exp(-y*model(x,w))))
    return cost/float(np.size(y))

# the convex relu cost function
def relu(w):
    cost = np.sum(np.maximum(0,-y*model(x,w)))
    return cost/float(np.size(y))

# the counting cost function
def counting_cost(w):
    cost = np.sum((np.sign(model(x,w)) - y)**2)
    return 0.25*cost 

# multiclass perceptron
def multiclass_perceptron(w):        
    # pre-compute predictions on all points
    all_evals = model(x,w)
    
    # compute maximum across data points
    a = np.max(all_evals,axis = 0)    

    # compute cost in compact form using numpy broadcasting
    b = all_evals[y.astype(int).flatten(),np.arange(np.size(y))]
    cost = np.sum(a - b)

    # return average
    return cost/float(np.size(y))

# multiclass softmax
def multiclass_softmax(w):        
    # pre-compute predictions on all points
    all_evals = model(x,w)
    
    # compute softmax across data points
    a = np.log(np.sum(np.exp(all_evals),axis = 0)) 
    
    # compute cost in compact form using numpy broadcasting
    b = all_evals[y.astype(int).flatten(),np.arange(np.size(y))]
    cost = np.sum(a - b)

    # return average
    return cost/float(np.size(y))

# multiclass misclassification cost function - aka the fusion rule
def multiclass_counting_cost(w):                
    # pre-compute predictions on all points
    all_evals = model(x,w)

    # compute predictions of each input point
    y_predict = (np.argmax(all_evals,axis = 0))[np.newaxis,:]

    # compare predicted label to actual label
    count = np.sum(np.abs(np.sign(y - y_predict)))

    # return number of misclassifications
    return count