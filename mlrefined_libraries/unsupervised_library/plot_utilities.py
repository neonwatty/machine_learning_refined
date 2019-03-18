# import standard plotting and animation
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from matplotlib import gridspec
from IPython.display import clear_output
from mpl_toolkits.mplot3d import proj3d
from matplotlib.text import Annotation
from mpl_toolkits.mplot3d.proj3d import proj_transform

# import autograd functionality
import numpy as np
import math
import time
import copy

class Visualizer:
    '''
    Various plotting functions 
    '''             
    
    # compare cost to counting
    def plot_cost_history(self,history):
        ##### setup figure to plot #####
        # initialize figure
        fig = plt.figure(figsize = (7,3))

        # create subplot with 3 panels, plot input function in center plot
        gs = gridspec.GridSpec(1, 1) 
        ax1 = plt.subplot(gs[0]); 

        # run through weights, evaluate classification and counting costs, record
        ax1.plot(history,linewidth = 4*(0.8))

        ax1.set_xlabel('iteration',fontsize = 10)
        ax1.set_ylabel('cost function value',fontsize = 10)
        plt.show()

    # compare cost to counting
    def compare_histories(self,histories,**kwargs):
        # parse input args
        label1 = ''; label2 = '';
        if 'label1' in kwargs:
            label1 = kwargs['label1']
        if 'label2' in kwargs:
            label2 = kwargs['label2']  
        plot_range = len(histories[0])
        if 'plot_range' in kwargs:
            plot_range = kwargs['plot_range']
        
        ##### setup figure to plot #####
        # initialize figure
        fig = plt.figure(figsize = (7,3))

        # create subplot with 3 panels, plot input function in center plot
        gs = gridspec.GridSpec(1, 1) 
        ax1 = plt.subplot(gs[0]); 

        # run through weights, evaluate classification and counting costs, record
        c = 1
        for history in histories:
            # plot both classification and counting cost histories
            if c == 1:
                ax1.plot(np.arange(1,len(history) + 1),history,label = label1,linewidth = 4*(0.8)**(c))
            else:
                ax1.plot(np.arange(1,len(history) + 1),history,label = label2,linewidth = 4*(0.8)**(c))
            c += 1

        ax1.set_xlabel('value of $K$',fontsize = 10)
        ax1.set_ylabel('cost function value',fontsize = 10)
        plt.legend(loc='upper right')
        ax1.set_xticks(plot_range)
        plt.show()