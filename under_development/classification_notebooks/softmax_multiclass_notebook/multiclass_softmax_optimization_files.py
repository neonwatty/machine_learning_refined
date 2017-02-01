import numpy as np
import math

# calculate number of misclassifications value for a given input weight W=[w_1,...,w_C]
def calculate_misclass(X,y,W):
    # define limits
    P = len(y)
    
    # loop for cost function
    num_misclass = 0
    for p in range(0,P):
        p_c = int(y[p])-1
        guess = np.argmax(np.dot(X[:,p].T,W))
        if p_c != guess:
            num_misclass+=1
    return num_misclass

# compute cth class gradient for single data point
def compute_grad(x_p,y_p,W,c,C):
    # produce gradient for each class weights
    temp = 0
    for j in range(0,C):
        temp+=np.exp(np.dot(x_p.T,W[:,j] - W[:,c]))
    temp = (np.divide(1,temp) - int(y_p == (c+1)))*x_p
    
    return temp

# learn all C separators together running stochastic gradient descent
# note: here the matrix X is assumed to have the bias data 1 in its first column
def stochastic_softmax_multi(X,y,max_its):
    X = X.T      
    
    # initialize variables
    C = len(np.unique(y))
    N,P = np.shape(X)
    W = np.random.randn(N,C)
    
    # record number of misclassifications on training set at each epoch 
    num_misclasses = []
    m = np.random.permutation(P)    # mix up samples

    # outer descent loop
    k = 1
    best_misclass = P      # best number of misclassifications so far
    best_W = W             # best W associated with best classification rate so far
    while k <= max_its:
        # set step length
        alpha = 1/math.sqrt(float(k))
        
        # take stochastic step in pth point
        for p in range(0,P):
            # re-initialize full gradient with zeros
            grad = np.zeros((np.shape(W)))
            
            # update each classifier's weights on pth point
            for c in range(0,C):
                # compute cth class gradient in pth point
                x_p = X[:,m[p]]
                y_p = y[m[p]]
                temp = compute_grad(x_p,y_p,W,c,C)
                grad[:,c] = temp.ravel()

            # take stochastic gradient step in all weights
            W = W - alpha*grad

        # update misclass container and associated best W
        current_misclasses = calculate_misclass(X,y,W)
        num_misclasses.append(current_misclasses)
        if current_misclasses <= best_misclass:
            best_misclass = current_misclasses
            best_W = W
        
        # kick up epoch count
        k+= 1
    
    # return goodies
    num_misclasses = np.asarray(num_misclasses)
    num_misclasses.shape = (max_its,1)
    return best_W,num_misclasses