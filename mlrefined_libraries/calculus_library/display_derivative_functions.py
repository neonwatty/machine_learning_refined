# import standard plotting and animation
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from matplotlib import gridspec

# import autograd functionality
from autograd import grad as compute_grad   # The only autograd function you may ever need
import autograd.numpy as np
import math

# For any input function display its first three derivative functions.
class visualizer:
    '''
    For any input function display its first three derivative functions.
    '''
    def __init__(self,**args):
        self.g = args['g']                                              # input function
        self.first_derivative = compute_grad(self.g)                    # first derivative of input function
        self.second_derivative = compute_grad(self.first_derivative)    # second derivative of input function
        self.third_derivative = compute_grad(self.second_derivative)    # third derivative of input function
        self.fourth_derivative = compute_grad(self.third_derivative)    # fourth derivative of input function

        self.colors = [[0,1,0.25],[0,0.75,1],[1,0.75,0],[1,0,0.75]]     # custom colors

    # draw the derivative functions
    def draw_it(self,**args):
        # how many frames to plot
        num_frames = 5
        if 'num_frames' in args:
            num_frames = args['num_frames']
            
        # initialize figure
        fig = plt.figure(figsize = (12,3))
        artist = fig
   
        # create subplot with 3 panels, plot input function in center plot
        gs = gridspec.GridSpec(1, 4, width_ratios=[1,1,1,1]) 
  
        ### draw 2d version ###
        ax1 = plt.subplot(gs[0]); 
        ax2 = plt.subplot(gs[1]); 
        ax3 = plt.subplot(gs[2]); 
        ax4 = plt.subplot(gs[3]); 

        # generate function for plotting on each slide
        w_plot = np.linspace(-3,3,200)       # range to plot over
        g_plot = self.g(w_plot)
        
        # create first, second, and third derivatives to plot
        g_first_der = []
        g_second_der = []
        g_third_der = []
        for w in w_plot:
            g_first_der.append(self.first_derivative(w))
            g_second_der.append(self.second_derivative(w))
            g_third_der.append(self.third_derivative(w))
            
        # plot function in each panel
        ax1.plot(w_plot,g_plot,color = 'k',zorder = 0)                           
        ax2.plot(w_plot,g_first_der,color = self.colors[0],zorder = 0)                         
        ax3.plot(w_plot,g_second_der,color = self.colors[1],zorder = 0)                       
        ax4.plot(w_plot,g_third_der,color = self.colors[2],zorder = 0)                       
  
        #### fix viewing limits ####
        ax1.set_xlim([-3,3])
        g_range = max(g_plot) - min(g_plot)
        ggap = g_range*0.5
        ax1.set_ylim([min(g_plot) - ggap,max(g_plot) + ggap])
        ax1.set_yticks([],[])
        ax1.set_title('original function',fontsize = 12)

        ax2.set_xlim([-3,3])
        g_range = max(g_first_der) - min(g_first_der)
        ggap = g_range*0.5
        ax2.set_ylim([min(g_first_der) - ggap,max(g_first_der) + ggap])
        ax2.set_yticks([],[])
        ax2.set_title('first derivative',fontsize = 12)

        ax3.set_xlim([-3,3])
        g_range = max(g_second_der) - min(g_second_der)
        ggap = g_range*0.5
        ax3.set_ylim([min(g_second_der) - ggap,max(g_second_der) + ggap])
        ax3.set_yticks([],[])
        ax3.set_title('second derivative',fontsize = 12)

        ax4.set_xlim([-3,3])
        g_range = max(g_third_der) - min(g_third_der)
        ggap = g_range*0.5
        ax4.set_ylim([min(g_third_der) - ggap,max(g_third_der) + ggap])
        ax4.set_yticks([],[])
        ax4.set_title('third derivative',fontsize = 12)
        
        plt.show()