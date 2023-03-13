# import standard plotting and animation
import autograd.numpy as np
import matplotlib.pyplot as plt
from matplotlib import gridspec

class Setup:
    def __init__(self,train_cost_histories,train_accuracy_histories,val_cost_histories,val_accuracy_histories,start):
        # plotting colors
        self.colors = [[0,0.7,1],[1,0.8,0.5]]

        # just plot cost history?
        if len(train_accuracy_histories) == 0:
            self.plot_cost_histories(train_cost_histories,val_cost_histories,start)
        else: # plot cost and count histories
            self.plot_cost_count_histories(train_cost_histories,train_accuracy_histories,val_cost_histories,val_accuracy_histories,start)
 
    #### compare cost function histories ####
    def plot_cost_histories(self,train_cost_histories,val_cost_histories,start):        
        # initialize figure
        fig = plt.figure(figsize = (10,3))

        # create subplot with 1 panel
        gs = gridspec.GridSpec(1, 1) 
        ax = plt.subplot(gs[0]); 

        # run through input histories, plotting each beginning at 'start' iteration
        for c in range(len(train_cost_histories)):
            train_history = train_cost_histories[c]
            val_history = val_cost_histories[c]

            # plot train cost function history
            ax.plot(np.arange(start,len(train_history),1),train_history[start:],linewidth = 3*(0.8)**(c),color = self.colors[0],label = 'train cost') 
            
            if np.size(val_history) > 0:
                # plot test cost function history
                ax.plot(np.arange(start,len(val_history),1),val_history[start:],linewidth = 3*(0.8)**(c),color = self.colors[1],label = 'test cost') 

        # clean up panel / axes labels
        xlabel = 'step $k$'
        ylabel = r'$g\left(\mathbf{w}^k\right)$'
        ax.set_xlabel(xlabel,fontsize = 14)
        ax.set_ylabel(ylabel,fontsize = 14,rotation = 0,labelpad = 25)
        title = 'train vs validation cost histories'
        ax.set_title(title,fontsize = 18)
        
        # plot legend
        anchor = (1,1)
        plt.legend(loc='upper right', bbox_to_anchor=anchor)
        ax.set_xlim([start - 0.5,len(train_history) - 0.5]) 
        plt.show()
        
    #### compare multiple histories of cost and misclassification counts ####
    def plot_cost_count_histories(self,train_cost_histories,train_accuracy_histories,val_cost_histories,val_accuracy_histories,start):        
        # initialize figure
        fig = plt.figure(figsize = (10,3))

        # create subplot with 1 panel
        gs = gridspec.GridSpec(1, 2) 
        ax1 = plt.subplot(gs[0]); 
        ax2 = plt.subplot(gs[1]); 

        # run through input histories, plotting each beginning at 'start' iteration
        for c in range(len(train_cost_histories)):
            train_cost_history = train_cost_histories[c]
            train_accuracy_history = train_accuracy_histories[c]

            val_cost_history = val_cost_histories[c]
            val_accuracy_history = val_accuracy_histories[c]
            
            # check if a label exists, if so add it to the plot
            ax1.plot(np.arange(start,len(train_cost_history),1),train_cost_history[start:],linewidth = 3*(0.8)**(c),color = self.colors[0]) 
  
            ax2.plot(np.arange(start,len(train_accuracy_history),1),train_accuracy_history[start:],linewidth = 3*(0.8)**(c),color = self.colors[0],label = 'train') 
    
            if np.size(val_cost_history) > 0:
                ax1.plot(np.arange(start,len(val_cost_history),1),val_cost_history[start:],linewidth = 3*(0.8)**(c),color = self.colors[1]) 

                ax2.plot(np.arange(start,len(val_accuracy_history),1),val_accuracy_history[start:],linewidth = 3*(0.8)**(c),color = self.colors[1],label = 'validation') 
            
        # clean up panel
        xlabel = 'step $k$'
        ylabel = r'$g\left(\mathbf{w}^k\right)$'
        ax1.set_xlabel(xlabel,fontsize = 14)
        ax1.set_ylabel(ylabel,fontsize = 14,rotation = 0,labelpad = 25)
        title = 'cost history'
        ax1.set_title(title,fontsize = 15)

        ylabel = 'accuracy'
        ax2.set_xlabel(xlabel,fontsize = 14)
        ax2.set_ylabel(ylabel,fontsize = 14,rotation = 90,labelpad = 10)
        title = 'accuracy history'
        ax2.set_title(title,fontsize = 15)
        
        anchor = (1,1)
        plt.legend(loc='lower right')# bbox_to_anchor=anchor)
        ax1.set_xlim([start - 0.5,len(train_cost_history) - 0.5])
        ax2.set_xlim([start - 0.5,len(train_cost_history) - 0.5])
        ax2.set_ylim([0,1.05])
        plt.show()       
        