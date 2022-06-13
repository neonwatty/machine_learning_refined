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
import time

class visualizer:
    '''
    Using a single input function this illustrates the construction of its corresponding first derivative function using a slider mechanism.  As the slider is run left to right a point on the input function, along with its tangent line / derivative are shown in the left panel, while the value of the derivative is plotted simultaneously in the right panel.
    '''
    def __init__(self,**args):
        self.g = args['g']                       # input function
        self.grad = compute_grad(self.g)         # gradient of input function
        self.colors = [[0,1,0.25],[0,0.75,1]]    # set of custom colors used for plotting

    # compute first order approximation
    def draw_it(self,**args):
        num_frames = 300                          # number of slides to create - the input range [-3,3] is divided evenly by this number
        if 'num_frames' in args:
            num_frames = args['num_frames']
        
        # initialize figure
        fig = plt.figure(figsize = (16,8))
        artist = fig

        # create subplot with 3 panels, plot input function in center plot
        gs = gridspec.GridSpec(1, 2, width_ratios=[1,1],wspace=0.3, hspace=0.05) 
        ax1 = plt.subplot(gs[0]);
        ax2 = plt.subplot(gs[1]); 

        # generate a range of values over which to plot input function, and derivatives
        w_plot = np.linspace(-3,3,200)                  # input range for original function
        g_plot = self.g(w_plot)
        g_range = max(g_plot) - min(g_plot)             # used for cleaning up final plot
        ggap = g_range*0.5
        w_vals = np.linspace(-2.5,2.5,num_frames)       # range of values over which to plot first / second order approximations
        
        # create the derivative function to plot, figure out its plotting range, etc.,
        grad_plot = []
        for w in w_plot:
            grad_plot.append(self.grad(w))
            
        grad_range = max(grad_plot) - min(grad_plot)
        grad_gap = grad_range*0.5
        print ('beginning animation rendering...')
               
        # animation sub-function
        def animate(k):
            # clear the panel
            ax1.cla()
            ax2.cla()
            
            # print rendering update
            if np.mod(k+1,25) == 0:
                print ('rendering animation frame ' + str(k+1) + ' of ' + str(num_frames))
            if k == num_frames - 1:
                print ('animation rendering complete!')
                time.sleep(1.5)
                clear_output()
            
            # grab the next input/output tangency pair, the center of the next approximation(s)
            w_val = w_vals[k]
            g_val = self.g(w_val)
            grad_val = self.grad(w_val)

            #### left plot: plot the original function ####
            ax1.plot(w_plot,g_plot,color = 'k',zorder = 3)                           # plot function
            
            # plot the input/output tangency point
            ax1.scatter(w_val,g_val,s = 90,c = 'r',edgecolor = 'k',linewidth = 0.7,zorder = 4)            # plot point of tangency

            # determine width to plot the approximation
            w1 = w_val - 2
            w2 = w_val + 2

            # compute first order approximation
            wrange = np.linspace(w1,w2, 100)
            h = g_val + grad_val*(wrange - w_val)

            # plot the first order approximation
            ax1.plot(wrange,h,color = self.colors[0],linewidth = 2,zorder = 2)      # plot approx

            # fix viewing limits on panel
            ax1.set_xlim([-3,3])
            ax1.set_ylim([min(g_plot) - ggap,max(g_plot) + ggap])
            
            # label axes
            ax1.set_xlabel('$w$',fontsize = 20)
            ax1.set_ylabel('$g(w)$',fontsize = 20,rotation = 0,labelpad = 25)
            
            #### right plot: plot the derivative function ####
            # get current values plotted so far
            vals = w_vals[:k+1]
            grad_vals = []
            for w in vals:
                grad_vals.append(self.grad(w))
            
            # plot all derivative values passed over so far, including current value
            ax2.plot(vals,grad_vals,color = self.colors[0],zorder = 3)                           # plot function

            # place marker on final point plotted
            ax2.scatter(vals[-1],grad_vals[-1],s = 90,color = self.colors[0],edgecolor = 'k',linewidth = 0.7,zorder = 3)            # plot point of tangency
            
            # fix viewing limits on panel
            ax2.set_xlim([-3,3])
            ax2.set_ylim([min(grad_plot) - grad_gap,max(grad_plot) + grad_gap])
            
            # label axes
            ax2.set_xlabel('$w$',fontsize = 20)
            ax2.set_ylabel(r'$\frac{\mathrm{d}}{\mathrm{d}w}g(w)$',fontsize = 20,rotation = 0,labelpad = 25)
            return artist,
        
        anim = animation.FuncAnimation(fig, animate,frames=len(w_vals), interval=len(w_vals), blit=True)
        
        return(anim)