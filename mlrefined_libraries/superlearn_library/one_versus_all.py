# import custom library
import copy
import sys
sys.path.append('../')
from mlrefined_libraries import math_optimization_library as optlib
from mlrefined_libraries import superlearn_library as superlearn
import autograd.numpy as np

# demos for this notebook
optimizers = optlib.optimizers
cost_lib = superlearn.cost_functions

### compare grad descent runs - given cost to counting cost ###
def train(x,y,**kwargs):    
    # get and run optimizer to solve two-class problem
    N = np.shape(x)[0]
    C = np.size(np.unique(y))
    max_its = 100; alpha_choice = 1; cost_name = 'softmax'; w = 0.1*np.random.randn(N+1,1); optimizer = 'gradient_descent';
    
    # switches for user choices
    if 'max_its' in kwargs:
        max_its = kwargs['max_its']
    if 'alpha_choice' in kwargs:
        alpha_choice = kwargs['alpha_choice']
    if 'cost_name' in kwargs:
        cost_name = kwargs['cost_name']
    if 'w' in kwargs:
        w = kwargs['w']
    if 'optimizer' in kwargs:
        optimizer = kwargs['optimizer']
    epsilon = 10**(-7)
    if 'epsilon' in kwargs:
        epsilon = kwargs['epsilon']
    
    # loop over subproblems and solve
    weight_histories = []
    for c in range(0,C):
        # prepare temporary C vs notC sub-probem labels
        y_temp = copy.deepcopy(y)
        ind = np.argwhere(y_temp.astype(int) == c)
        ind = ind[:,1]
        ind2 = np.argwhere(y_temp.astype(int) != c)
        ind2 = ind2[:,1]
        y_temp[0,ind] = 1
        y_temp[0,ind2] = -1

        # store best weight for final classification 
        cost = cost_lib.choose_cost(x,y_temp,cost_name)
        
        # run optimizer
        weight_history = 0; cost_history = 0;
        if optimizer == 'gradient_descent':
            weight_history,cost_history = optimizers.gradient_descent(cost,alpha_choice,max_its,w)
        if optimizer == 'newtons_method':
            weight_history,cost_history = optimizers.newtons_method(cost,max_its,w=w,epsilon = epsilon)

        # store each weight history
        weight_histories.append(copy.deepcopy(weight_history))
        
    # combine each individual classifier weights into single weight 
    # matrix per step
    R = len(weight_histories[0])
    combined_weights = []
    for r in range(R):
        a = []
        for c in range(C):
            a.append(weight_histories[c][r])
        a = np.array(a).T
        a = a[0,:,:]
        combined_weights.append(a)
        
    # run combined weight matrices through fusion rule to calculate
    # number of misclassifications per step
    counter = cost_lib.choose_cost(x,y,'multiclass_counter')
    count_history = [counter(v) for v in combined_weights]
        
    return combined_weights, count_history