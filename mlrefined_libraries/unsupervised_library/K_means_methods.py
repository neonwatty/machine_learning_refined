import random
import numpy as np
import time
import matplotlib.pylab as plt
from IPython import display
import warnings
warnings.filterwarnings('ignore')


def K_means_demo(X, C, mode):
    
    t = 1   # While-loop counter
    d = [1] # Container for centroid movements    
    eps = 1e-3  # Threshold for stopping the algorithm
    P = X.shape[1]
    K = np.shape(C)[1]
    clrs = ['r', 'b', 'g', 'm', 'y'] # Colors
       
    if mode == 'just_run_the_alg':
        
        while d[-1] > eps:
            
            #Cluster assignment
            cluster_assignments = [] #This list will contain the cluster assignments
            for p in np.arange(0, P):
                diff = []
                for k in np.arange(0, K):
                    diff.append(np.linalg.norm(X[:, p] - C[2*(t-1):2*t, k]))    
                cluster_assignments.append(diff.index(min(diff)))

            #Centroid update
            AVG = np.empty([2,0]) #This array will contain the centroid locations       
            for k in  np.arange(0, K):
                ind = [i for i, y in enumerate(cluster_assignments) if y == k] 
                AVG = np.concatenate((AVG, X[:,ind].mean(axis=1).reshape([2,1])), axis=1)   
            C = np.concatenate((C, AVG), axis=0)
            d.append(np.linalg.norm(C[2*(t-1):2*t,:] - C[2*t:2*(t+1),:]))
            t = t+1
    
    elif mode == 'plot_the_steps':
        
        fig = plt.figure()
        ax = fig.add_subplot(111)
        plt.axis([-.1, 1.1, -.1, 1.1])
        plt.scatter(X[0,:], X[1,:], color='k')
        plt.axis('off')
        time.sleep(1)

        while d[-1] > eps:

            #Cluster assignment
            cluster_assignments = [] #This list will contain the cluster assignments
            for p in np.arange(0, P):
                diff = []
                for k in np.arange(0, K):
                    diff.append(np.linalg.norm(X[:, p] - C[2*(t-1):2*t, k]))    
                cluster_assignments.append(diff.index(min(diff)))

            #Plotting the centroids 
            for k in np.arange(0, K):
                plt.scatter(C[2*(t-1), k], C[2*t-1, k], s=120, color=clrs[k], marker=(5, 2))
                display.display(plt.gcf())
                display.clear_output(wait=True)  
            time.sleep(1)

            #Centroid update
            AVG = np.empty([2,0]) #This array will contain the centroid locations       
            for k in  np.arange(0, K):
                ind = [i for i, y in enumerate(cluster_assignments) if y == k]
                plt.scatter(X[0,ind], X[1,ind], color=clrs[k])
                display.display(plt.gcf())
                display.clear_output(wait=True)  
                AVG = np.concatenate((AVG, X[:,ind].mean(axis=1).reshape([2,1])), axis=1)   
            C = np.concatenate((C, AVG), axis=0)
            d.append(np.linalg.norm(C[2*(t-1):2*t,:] - C[2*t:2*(t+1),:]))
            time.sleep(1)

            for k in np.arange(0, K):
                fig = plt.plot([C[2*(t-1),k], C[2*t,k]], [C[2*t-1,k], C[2*t+1,k]], '--', color=clrs[k])
                display.display(plt.gcf())
                display.clear_output(wait=True) 
                
            t = t+1
     
        #plt.figure() # Plotting the clustered data
        #for k in np.arange(0, K):
            #ind = [i for i, x in enumerate(cluster_assignments) if x == k]
            #plt.scatter(X[0,ind], X[1,ind], s=30, color=clrs[k])
            #plt.axis([-.1, 1.1, -.1, 1.1])
            #plt.axis('off')


    return cluster_assignments, calc_obj_val(X, C[-2:,:], cluster_assignments)


def calc_obj_val(X, C, cluster_assignments):
    W = np.zeros((C.shape[1], X.shape[1]))
    for i, cluster in enumerate(cluster_assignments):
        W[cluster,i] = 1
    obj_val = np.linalg.norm(X - np.dot(C,W), 'fro')
    return obj_val


def scree_plot(X):
  
    num_clusters = 10
    num_runs = 20
    Results = float('Inf')*np.ones((num_runs, num_clusters))

    for i in np.arange(0,num_runs):
        for k in np.arange(1,num_clusters+1):
            foo, obj_val = K_means_demo(X, X[:,random.sample(set(np.arange(0,X.shape[1])),k)], mode='just_run_the_alg') 
            if np.isnan(obj_val) == False:
                Results[i,k-1] = obj_val

    obj_val = Results.min(axis=0)            
   
    plt.figure()
    plt.style.use('ggplot')
    plt.xlabel('number of clusters')
    plt.ylabel('objective value')
    plt.axis([1-.2, num_clusters+.2, min(obj_val)-.2, max(obj_val)+.2])
    plt.xticks(np.arange(1, num_clusters+1, 1))
    foo = plt.plot(np.arange(1,num_clusters+1), obj_val,'ko-') 
    return

def normalize_blobs(blobs):
    X = np.transpose(blobs[0])
    X = (X-X.min())/(X.max()-X.min())
    return X

def plot_data(X,C):
    clrs = ['r', 'b', 'g', 'm', 'y'] # colors
    plt.axis('off')
    foo = plt.scatter(X[0,:], X[1,:], s=30, color='k') # plot data
    K = np.shape(C)[1]
    for k in np.arange(0, K):
        foo = plt.scatter(C[0, k], C[1, k], s=120, color=clrs[k], marker=(5, 2)) # plot centroids
    return    
        
