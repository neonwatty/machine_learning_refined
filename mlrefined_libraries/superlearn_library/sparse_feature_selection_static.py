import numpy as np
import matplotlib.pyplot as plt
#.style.use('ggplot')
import matplotlib.patches as mpatches
from matplotlib import gridspec

def plot_genes(X, gene_id_1, gene_id_2):
    # create figure for plotting
    fig = plt.figure(figsize = (5,5))
    
    # plot all
    N = int(X.shape[0]/2)
    plt.xlabel('gene #'+str(gene_id_1))
    plt.ylabel('gene #'+str(gene_id_2))
    red_patch = mpatches.Patch(color='red', label='healthy')
    blue_patch = mpatches.Patch(color='blue', label='afflicted')
    plt.legend(handles=[red_patch, blue_patch])
    plt.legend(handles=[red_patch, blue_patch], loc = 2)
    ax = plt.scatter(X[0:N,gene_id_1], X[0:N,gene_id_2], color='r', s=30) #plotting the data
    plt.scatter(X[N+1:2*N,gene_id_1], X[N+1:2*N,gene_id_2], color='b', s=30)
    plt.show()

# compare multiple l1 regularized runs
def compare_lams(weights,lams,genes):       
    # initialize figure
    fig = plt.figure(figsize = (9,7))
    artist = fig

    # create subplot with 3 panels, plot input function in center plot
    num_lams = len(lams)
    gs = gridspec.GridSpec(num_lams,1) 
    for n in range(num_lams):
        lam = lams[n]
        ax = plt.subplot(gs[n]); 
        w = weights[n][1:]
        
        # plot weights for this run
        plot_weights(ax,w,genes,lam)
       # ax.set_ylim([-0.6,1.7])
    plt.show()

def plot_weights(ax,w,genes,lam):
    # mark all genes
    plt.bar(np.arange(0,len(w)), w, color='k', alpha=0.2)

    # highlight chosen genes
    for gene in genes:
        plt.bar([gene],[w[gene]], color='k', alpha=.7)
    plt.axhline(c='k',zorder = 3)
    
    # dress panel
    plt.xlabel('genes')
    plt.ylabel('learned weights')
    title = r'$\lambda = ' + str(lam)
    plt.title(title)

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
    X = X.T
    
    # initialize weights - we choose w = random for illustrative purposes
    w = np.zeros((X.shape[0],1))
    print (w.shape)
   
    # set maximum number of iterations and step length
    alpha = 1
    max_its = 2000
   
    # make list to record weights at each step of algorithm
    w_history = np.zeros((len(w),max_its+1)) 
    w_history[:,0] = w.flatten()
    print(w_history.shape)
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
    return w_history[:,-1]

def proximal_step(w, lam):
    return np.maximum(np.abs(w) - 2*lam,0)*np.sign(w)


def logistic_regression(X, y):    
    X = X.T
    
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






