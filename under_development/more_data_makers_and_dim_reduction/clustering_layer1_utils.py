import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

# load data
def load_data(csvname):
    data = np.asarray(pd.read_csv(csvname))
    x = data[:,0]
    x.shape = (len(x),1)
    y = data[:,1]
    y.shape = (len(y),1)
    labels = data[:,2]
    return x,y,labels

# simple plotting function
def plot_orig_data(x,y,labels):
    fig = plt.figure(num=None, figsize=(5, 5), dpi=80, facecolor='w', edgecolor='k')
    ax1 = fig.add_subplot(111)
        
    # plot data 
    colors = ['r','g','b','c','m','k']
    for d in range(len(x)):
        ax1.scatter(x[d],y[d],color = str(colors[int(labels[d])]))

    # clean up the plot
    ax1.set_xlim([-2,2])
    ax1.set_ylim([-2,2])
    ax1.set_yticks([],[])
    ax1.axis('off') 

# plot clustered data
def plot_clustered_data(x,y,clf):
    # colors for clusgters
    colors = ['r','g','b','c','m','k']
    
    # loop over points and plot data w/centroids
    centroids = clf.cluster_centers_
    cluster_labels = clf.labels_
    
    # plot pts
    fig = plt.figure(num=None, figsize=(5, 5), dpi=80, facecolor='w', edgecolor='k')
    ax1 = fig.add_subplot(111)
    for d in range(len(x)):
        ax1.scatter(x[d],y[d],color = str(colors[int(cluster_labels[d])]))
        
    # plot centers
    K = np.shape(clf.cluster_centers_)[0]
    s = [125 for n in range(K)]
    for d in range(K):
        ax1.scatter(centroids[d,0],centroids[d,1],color = str(colors[d]),marker = '*',s = s)
        
    s = [175 for n in range(K)]
    for d in range(K):
        ax1.scatter(centroids[d,0],centroids[d,1],color = 'k',marker = 'o',facecolors='none',s = s)
    
    # clean up the plot
    ax1.set_xlim([-2,2])
    ax1.set_ylim([-2,2])
    ax1.set_yticks([],[])
    ax1.axis('off') 