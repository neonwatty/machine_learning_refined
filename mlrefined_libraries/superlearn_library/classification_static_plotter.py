# import standard plotting and animation
import matplotlib.pyplot as plt
from matplotlib import gridspec
from autograd import numpy as np

class Visualizer:
    '''
    Illustrate a run of your preferred optimization algorithm for classification.  Run
    the algorithm first, and input the resulting weight history into this wrapper.
    ''' 

    # compare cost histories from multiple runs
    def plot_histories(self,cost_histories,count_histories,start,**kwargs):        
        # plotting colors
        colors = ['k','magenta','springgreen','blueviolet','chocolate']
        
        # initialize figure
        fig = plt.figure(figsize = (10,3))

        # create subplot with 1 panel
        gs = gridspec.GridSpec(1, 2) 
        ax = plt.subplot(gs[0]); 
        ax2 = plt.subplot(gs[1]); 

        # any labels to add?        
        labels = [' ',' ']
        if 'labels' in kwargs:
            labels = kwargs['labels']
            
        # plot points on cost function plot too?
        points = False
        if 'points' in kwargs:
            points = kwargs['points']

        # run through input histories, plotting each beginning at 'start' iteration
        for c in range(len(cost_histories)):
            history = cost_histories[c]
            count_hist = count_histories[c]
            label = labels[c]
                
            # check if a label exists, if so add it to the plot
            ax.plot(np.arange(start,len(history),1),history[start:],linewidth = 3*(0.8)**(c),color = colors[c]) 
            if np.size(label) == 0:
                ax2.plot(np.arange(start,len(history),1),count_hist[start:],linewidth = 3*(0.8)**(c),color = colors[c]) 
            else:          
                ax2.plot(np.arange(start,len(history),1),count_hist[start:],linewidth = 3*(0.8)**(c),color = colors[c],label = label) 
                
        # clean up panel
        xlabel = 'step $k$'
        ylabel = r'$g\left(\mathbf{w}^k\right)$'
        ylabel2 = 'num misclassifications'

        ax.set_xlabel(xlabel,fontsize = 14)
        ax.set_ylabel(ylabel,fontsize = 14,rotation = 0,labelpad = 25)
        ax.set_title('cost function history',fontsize = 14)
        ax2.set_xlabel(xlabel,fontsize = 14)
        ax2.set_ylabel(ylabel2,fontsize = 12,rotation = 90,labelpad = 10)
        ax2.set_title('misclassification history',fontsize = 14)
 
        if np.size(label) > 0:
            anchor = (1,1)
            if 'anchor' in kwargs:
                anchor = kwargs['anchor']
            plt.legend(loc='upper right', bbox_to_anchor=anchor)
            #leg = ax.legend(loc='upper left', bbox_to_anchor=(1.02, 1), borderaxespad=0)

        ax.set_xlim([start - 0.5,len(history) - 0.5])
        ax2.set_xlim([start - 0.5,len(history) - 0.5])

       # fig.tight_layout()
        plt.show()