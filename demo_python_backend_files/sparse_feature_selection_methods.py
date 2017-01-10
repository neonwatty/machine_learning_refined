import numpy as np
import matplotlib.pyplot as plt
plt.style.use('ggplot')
import matplotlib.patches as mpatches



def plot_genes(X, gene_id_1, gene_id_2):
    N = X.shape[1]/2
    plt.xlabel('gene #'+str(gene_id_1))
    plt.ylabel('gene #'+str(gene_id_2))
    red_patch = mpatches.Patch(color='red', label='healthy')
    blue_patch = mpatches.Patch(color='blue', label='afflicted')
    plt.legend(handles=[red_patch, blue_patch])
    plt.legend(handles=[red_patch, blue_patch], loc = 2)
    ax = plt.scatter(X[gene_id_1+1,0:N], X[gene_id_2+1,0:N], color='r', s=30) #plotting the data
    plt.scatter(X[gene_id_1+1,N+1:2*N], X[gene_id_2+1,N+1:2*N], color='b', s=30)
    plt.show()
    return



def plot_weights(w, gene_id_1, gene_id_2):
    plt.figure(figsize=(20,5))
    plt.xlabel('genes')
    plt.ylabel('learned weights')
    plt.bar(np.arange(0,len(w)), w, color='grey', alpha=.5)
    plt.bar([gene_id_1, gene_id_2],[w[gene_id_1], w[gene_id_2]], color='k', alpha=.7)
    plt.show()
    return



def compute_grad(X, y, w):
    #produce gradient for each class weights
    grad = 0
    for p in range(0,len(y)):
        x_p = X[:,p]
        y_p = y[p]
        grad+= -1/(1 + np.exp(y_p*np.dot(x_p.T,w)))*y_p*x_p
   
    grad.shape = (len(grad),1)
    return grad


def L1_logistic_regression(X, y, lam):    
        
    # initialize weights - we choose w = random for illustrative purposes
    w = np.zeros((X.shape[0],1))
   
    # set maximum number of iterations and step length
    alpha = 1
    max_its = 2000
   
    # make list to record weights at each step of algorithm
    w_history = np.zeros((len(w),max_its+1)) 
    w_history[:,0] = w.flatten()
    # gradient descent loop
    for k in range(1,max_its+1):  

        # form gradient
        grad = compute_grad(X,y,w)
      
        # take gradient descent step
        w = w - alpha*grad
        
        # take a proximal step
        w[1:] = proximal_step(w[1:], lam)

        # save new weights
        w_history[:,k] = w.flatten()
   
    # return weights from each step
    return w_history[1:,-1]


def proximal_step(w, lam):
    return np.maximum(np.abs(w) - 2*lam,0)*np.sign(w)


def logistic_regression(X, y):    
        
    # initialize weights - we choose w = random for illustrative purposes
    w = np.zeros((X.shape[0],1))
   
    # set maximum number of iterations and step length
    alpha = 1
    max_its = 2000
   
    # make list to record weights at each step of algorithm
    w_history = np.zeros((len(w),max_its+1)) 
    w_history[:,0] = w.flatten()
    # gradient descent loop
    for k in range(1,max_its+1):  

        # form gradient
        grad = compute_grad(X,y,w)
      
        # take gradient descent step
        w = w - alpha*grad

        # save new weights
        w_history[:,k] = w.flatten()
   
    # return weights from each step
    return w_history[1:,-1]






