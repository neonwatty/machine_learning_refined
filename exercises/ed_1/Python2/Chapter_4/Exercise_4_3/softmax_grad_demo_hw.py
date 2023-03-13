# softmax_grad_demo_hw runs the softmax model on a separable two 
# class dataset consisting of two dimensional data features.

# This file is associated with the book
# "Machine Learning Refined", Cambridge University Press, 2016.
# by Jeremy Watt, Reza Borhani, and Aggelos Katsaggelos.

import numpy as np
import matplotlib.pyplot as plt

# sigmoid for softmax/logistic regression minimization
def sigmoid(z): 
    y = 1/(1+np.exp(-z))
    return y
    
# import training data 
def load_data(csvname):
    # load in dataframe
    all_data = np.array(np.genfromtxt(csvname,delimiter = ','))

    # grab training data and labels
    X = all_data[:,:-1]      
    y = all_data[:,-1]
    
    return X,y

# gradient descent function for softmax cost/logistic regression 
def softmax_grad(X,y):
    # Initializations 
    w = np.random.randn(3,1);        # random initial point
    alpha = 10**-2
    k = 1
    max_its = 2000
    grad = 1
    
    while np.linalg.norm(grad) > 10**-12 and k < max_its:
        # compute gradient
        grad = 
        w = w - alpha*grad;

        # update iteration count
        k = k + 1;
        
    return w

# plots everything 
def plot_all(X,y,w):
    # custom colors for plotting points
    red = [1,0,0.4]  
    blue = [0,0.4,1]
    
    # scatter plot points
    fig = plt.figure(figsize = (4,4))
    ind = np.argwhere(y==1)
    ind = [s[0] for s in ind]
    plt.scatter(X[ind,0],X[ind,1],color = red,edgecolor = 'k',s = 25)
    ind = np.argwhere(y==-1)
    ind = [s[0] for s in ind]
    plt.scatter(X[ind,0],X[ind,1],color = blue,edgecolor = 'k',s = 25)
    plt.grid('off')
    
    # plot separator
    s = np.linspace(0,1,100) 
    plt.plot(s,(-w[0]-w[1]*s)/w[2],color = 'k',linewidth = 2)
    plt.show()
    
# load in data
X,y = load_data('imbalanced_2class.csv')

# run gradient descent
w = softmax_grad(X,y)

# plot points and separator
plot_all(X,y,w)
