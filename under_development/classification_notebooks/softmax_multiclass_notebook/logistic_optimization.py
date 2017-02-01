import numpy as np
import math

# calculate the objective value of the softmax / logistic regression cost for a given input weight W=[w_1,...,w_C]
def calculate_obj(X,y,w):
    # define limits
    P = len(y)
    
    # loop for cost function
    cost = 0
    for p in range(0,P):
        y_p = y[p]
        x_p = X[:,p]
        temp = (1 + my_exp(-y_p*np.dot(x_p.T,w)))
        
        # update cost sum 
        cost+=np.log(temp)
    return cost

# calculate number of misclassifications value for a given input weight W=[w_1,...,w_C]
def calculate_misclass(X,y,w):
    # loop for cost function
    num_misclass = 0
    for p in range(0,len(y)):
        x_p = X[:,p]
        y_p = int(y[p])
        yhat_p = np.sign(np.dot(x_p.T,w))
        
        if y_p != yhat_p:
            num_misclass+=1
    return num_misclass

# make your own exponential function that ignores cases where exp = inf
def my_exp(val):
    newval = 0
    if val > 100:
        newval = np.inf
    if val < -100:
        newval = 0
    if val < 100 and val > -100:
        newval = np.exp(val)
    return newval

# compute cth class gradient for single data point
def compute_grad(X,y,w):
    # produce gradient for each class weights
    grad = 0
    for p in range(0,len(y)):
        x_p = X[:,p]
        y_p = y[p]
        grad+= -1/(1 + my_exp(y_p*np.dot(x_p.T,w)))*y_p*x_p
    
    grad.shape = (len(grad),1)
    return grad

# learn all C separators together running stochastic gradient descent
def softmax_2class_descent(x,y,max_its):
    X = x.T
    
    # set step length - using Lipschitz constant
    L = 0.25*np.linalg.norm(X,ord = 2)**2
    alpha = 1/L
    
    # initialize variables
    N,P = np.shape(X)
    w = np.zeros((N,1))
    
    # record number of misclassifications on training set at each epoch 
    num_misclasses = []

    # outer descent loop
    best_misclass = P      # best number of misclassifications so far
    best_w = w             # best W associated with best classification rate so far
    for k in range(0,max_its):
        alpha = 1/float(k+1)
        alpha = 0.001

        
        # take gradient step
        grad = compute_grad(X,y,w)
        w = w - alpha*grad
        
        # update misclass container and associated best W
        current_misclasses = calculate_misclass(X,y,w)
        num_misclasses.append(current_misclasses)
        if current_misclasses < best_misclass:
            best_misclass = current_misclasses
            best_w = w
        
    # return goodies
    num_misclasses = np.asarray(num_misclasses)
    num_misclasses.shape = (max_its,1)
    return best_w,num_misclasses

def softmax_2class_newton(x,y,max_its):
    X = x.T
    
    # initialize variables
    N,P = np.shape(X)
    w = np.zeros((N,1))
    
    # record number of misclassifications on training set at each epoch 
    num_misclasses = []

    # outer descent loop
    best_misclass = P      # best number of misclassifications so far
    best_w = w             # best W associated with best classification rate so far
    for k in range(0,max_its):

        # compute gradient and Hessian
        grad,hess = compute_grad_and_hess(X,y,w)
        
        # take step
        temp = np.dot(hess,w) - grad
        w = np.dot(np.linalg.pinv(hess),temp)
        
        # update misclass container and associated best W
        current_misclasses = calculate_misclass(X,y,w)
        num_misclasses.append(current_misclasses)
        if current_misclasses < best_misclass:
            best_misclass = current_misclasses
            best_w = w
        
    # return goodies
    num_misclasses = np.asarray(num_misclasses)
    num_misclasses.shape = (max_its,1)
    return best_w,num_misclasses    

# calculate grad and Hessian for newton's method
def compute_grad_and_hess(X,y,w):
    hess = 0
    grad = 0
    for p in range(0,len(y)):
        # precompute
        x_p = X[:,p]
        y_p = y[p]
        s = 1/(1 + my_exp(y_p*np.dot(x_p.T,w)))
        g = s*(1-s)
        
        # update grad and hessian
        grad+= -s*y_p*x_p
        hess+= np.outer(x_p,x_p)*g
        
    grad.shape = (len(grad),1)
    return grad,hess
