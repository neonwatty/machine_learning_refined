# import standard plotting and animation
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from IPython.display import clear_output
import time
from matplotlib import gridspec

# import autograd functionality
from autograd import grad as compute_grad   # The only autograd function you may ever need
import autograd.numpy as np
import math

class visualizer:
    '''
    This slider toy allows you to experiment with the value of epsilon in the
    definition of the numerical derivative affects its accuracy.
    '''
        
    def __init__(self,**kwargs):
        self.g = kwargs['g']                          # input function
        self.grad = compute_grad(self.g)            # gradient of input function
        self.colors = [[0,1,0.25],[0,0.75,1]]       # custom colors for visualization purposes
                
    # compute derivative approximation and return
    def numerical_derivative(self, w,epsilon):
        return (self.g(w+epsilon) - self.g(w))/epsilon
        
    # draw numerical derivative
    def draw_it(self,**kwargs):
        # number of frames to show in animation - evenly divides the input region [-3,3]
            
        epsilon_range = np.logspace(0, -17, 18)

        # initialize figure
        fig = plt.figure(figsize = (7,3))
        artist = fig

        # create subplot with 3 panels, plot input function in center plot
        gs = gridspec.GridSpec(1, 1, width_ratios=[1]) 
        ax = plt.subplot(gs[0]); ax.axis('off');

        # generate function for plotting on each slide
        w_plot = np.linspace(-3,3,2000)          # input range for the original function
        g_plot = self.g(w_plot)                 # original function evaluated over input region
        true_grad = [self.grad(w) for w in w_plot]        # true derivative 
        
        print ('starting animation rendering...')

        # animation sub-function
        def animate(k):
            # clear the current slide
            ax.cla()
            
            # print rendering update
            if k == 17 - 1:
                print ('animation rendering complete!')
                time.sleep(1.5)
                clear_output()

            if k == 0:
                # plot original function
                ax.plot(w_plot,g_plot,color = 'k',zorder = 0,label = 'function')  
            
            # plot everything on top of function after first slide
            if k > 0:
                # select epsilon
                epsilon = epsilon_range[k-1]
                
                # plot original function
                ax.plot(w_plot,g_plot,color = 'k',zorder = 0,label = 'function') 
                
                # compute numerical derivative over input range
                dervals = [self.numerical_derivative(w,epsilon) for w in w_plot]
                
                # plot numerical derivative
                ax.plot(w_plot,dervals,color = 'r',zorder = 3,label = 'numerical derivative') 
                # plot numerical derivative
                ax.plot(w_plot,true_grad,color = 'b',linestyle ='--',zorder = 2, label = 'true derivative') 
                
                # set legend
                h, l = ax.get_legend_handles_labels()
                tra = '$\epsilon = 10^{-' + str(k) + '}$'
                ax.set_title(tra,fontsize=13)
                ax.legend(bbox_to_anchor=[0, 0.9],loc='center', ncol=1,fontsize = 12)

            # fix viewing limits
            ax.set_xlim([-3,3])
            
            # label axes
            ax.set_xlabel('$w$',fontsize = 12)
            ax.set_ylabel('$g(w)$',fontsize = 12,rotation = 0,labelpad = 25)
                
            return artist,

        anim = animation.FuncAnimation(fig, animate,frames=len(epsilon_range)+1, interval=len(epsilon_range)+1, blit=True)

        return(anim)