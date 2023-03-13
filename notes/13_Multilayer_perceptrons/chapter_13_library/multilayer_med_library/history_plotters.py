# import standard plotting and animation
import autograd.numpy as np
import matplotlib.pyplot as plt
from matplotlib import gridspec

class Setup:
    def __init__(self,cost_histories,count_histories,start,labels):
        # just plot cost history?
        if len(count_histories) == 0:
            self.plot_cost_histories(cost_histories,start,labels)
        else: # plot cost and count histories
            self.plot_cost_count_histories(cost_histories,count_histories,start,labels)
 
    #### compare cost function histories ####
    def plot_cost_histories(self,cost_histories,start,labels):
        # plotting colors
        colors = ['k','magenta','springgreen','blueviolet','chocolate']
        
        # initialize figure
        fig = plt.figure(figsize = (10,3))

        # create subplot with 1 panel
        gs = gridspec.GridSpec(1, 1) 
        ax = plt.subplot(gs[0]); 

        # run through input histories, plotting each beginning at 'start' iteration
        for c in range(len(cost_histories)):
            history = cost_histories[c]
            label = labels[c]
                
            # plot cost function history
            ax.plot(np.arange(start,len(history),1),history[start:],linewidth = 3*(0.8)**(c),color = colors[c],label = label) 

        # clean up panel / axes labels
        xlabel = 'step $k$'
        ylabel = r'$g\left(\mathbf{w}^k\right)$'
        ax.set_xlabel(xlabel,fontsize = 14)
        ax.set_ylabel(ylabel,fontsize = 14,rotation = 0,labelpad = 25)
        title = 'cost history'
        ax.set_title(title,fontsize = 18)
        
        # plot legend
        anchor = (1,1)
        plt.legend(loc='upper right', bbox_to_anchor=anchor)
        ax.set_xlim([start - 0.5,len(history) - 0.5]) 
        plt.show()
        
    #### compare multiple histories of cost and misclassification counts ####
    def plot_cost_count_histories(self,cost_histories,count_histories,start,labels):
        # plotting colors
        colors = ['k','magenta','springgreen','blueviolet','chocolate']
        
        # initialize figure
        fig = plt.figure(figsize = (10,3))

        # create subplot with 1 panel
        gs = gridspec.GridSpec(1, 2) 
        ax1 = plt.subplot(gs[0]); 
        ax2 = plt.subplot(gs[1]); 

        # run through input histories, plotting each beginning at 'start' iteration
        for c in range(len(cost_histories)):
            cost_history = cost_histories[c]
            count_history = count_histories[c]
            label = labels[c]

            # check if a label exists, if so add it to the plot
            ax1.plot(np.arange(start,len(cost_history),1),cost_history[start:],linewidth = 3*(0.8)**(c),color = colors[c],label = label) 
            
            ax2.plot(np.arange(start,len(count_history),1),count_history[start:],linewidth = 3*(0.8)**(c),color = colors[c],label = label) 

        # clean up panel
        xlabel = 'step $k$'
        ylabel = r'$g\left(\mathbf{w}^k\right)$'
        ax1.set_xlabel(xlabel,fontsize = 14)
        ax1.set_ylabel(ylabel,fontsize = 14,rotation = 0,labelpad = 25)
        title = 'cost history'
        ax1.set_title(title,fontsize = 18)

        ylabel = 'num misclasses'
        ax2.set_xlabel(xlabel,fontsize = 14)
        ax2.set_ylabel(ylabel,fontsize = 14,rotation = 90,labelpad = 10)
        title = 'misclassification history'
        ax2.set_title(title,fontsize = 18)
        
        anchor = (1,1)
        plt.legend(loc='upper right', bbox_to_anchor=anchor)
        ax1.set_xlim([start - 0.5,len(cost_history) - 0.5])
        ax2.set_xlim([start - 0.5,len(cost_history) - 0.5])
        plt.show()       
        