# import standard plotting and animation
from IPython.display import clear_output
import time
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from IPython.display import clear_output
from matplotlib import gridspec

# 3d function
from mpl_toolkits.mplot3d import proj3d

# import autograd functionality
from autograd import grad as compute_grad   # The only autograd function you may ever need
import autograd.numpy as np
import math

# go from a secant to tangent line at a pre-defined point self.w_init
class visualizer:
    '''
    Using the input anchor point self.w_init, peruse over a course set of other points
    in a neighborhood around the anchor, drawing the secant line passing through the anchor and each
    such neighboring point.  When the neighboring point == the anchor point the secant
    line becomes a tangent one, and this is shown graphically.  Peruse the various secant lines using
    a custom slider widget.
    '''
        
    def __init__(self,**kwargs):
        self.g = kwargs['g']                          # input function
        self.grad = compute_grad(self.g)            # gradient of input function
        self.colors = [[0,1,0.25],[0,0.75,1]]       # custom colors for visualization purposes

    # animate the secant line passing through self.w_init and all surrounding points
    def draw_it(self,**kwargs):
        # number of frames to show in animation - evenly divides the input region [-3,3]
        num_frames = 100
        if 'num_frames' in kwargs:
            num_frames = kwargs['num_frames']
            
        self.w_init = 0
        if 'w_init' in kwargs:
            self.w_init = kwargs['w_init']                # input point - where we draw the tangent line

        self.mark_tangent = True
        if 'mark_tangent' in kwargs:
            self.mark_tangent = kwargs['mark_tangent']
            
        # initialize figure
        fig = plt.figure(figsize = (16,8))
        artist = fig

        # create subplot with 3 panels, plot input function in center plot
        gs = gridspec.GridSpec(1, 3, width_ratios=[1,2, 1]) 
        ax1 = plt.subplot(gs[0]); ax1.axis('off');
        ax3 = plt.subplot(gs[2]); ax3.axis('off');

        # plot input function
        ax = plt.subplot(gs[1])

        # generate function for plotting on each slide
        w_plot = np.linspace(-3,3,200)          # input range for the original function
        g_plot = self.g(w_plot)                 # original function evaluated over input region
        g_range = max(g_plot) - min(g_plot)     # metric used for adjusting plotting area of panel
        ggap = g_range*0.5
        
        # define values over which to draw secant line connecting to (self.w_init,self.g(self.w_init))
        w_vals = np.linspace(max(-2.6 + self.w_init,-3),min(2.6 + self.w_init,3),num_frames)
        
        # re-assign w_init to closest point in this range (for animation purposes)
        ind = np.argmin((w_vals - self.w_init)**2)
        self.w_init = w_vals[ind]
        print ('starting animation rendering...')

        # animation sub-function
        def animate(k):
            # clear the current slide
            ax.cla()
            
            # print rendering update
            if np.mod(k+1,25) == 0:
                print ('rendering animation frame ' + str(k+1) + ' of ' + str(num_frames))
            if k == num_frames - 1:
                print ('animation rendering complete!')
                time.sleep(1.5)
                clear_output()

            # plot original function
            g_init = self.g(self.w_init)
            ax.plot(w_plot,g_plot,color = 'k',zorder = 0) 
            
            # plot the chosen point
            ax.scatter(self.w_init,g_init,s = 120,c = self.colors[0],edgecolor = 'k',linewidth = 0.7,zorder = 3)            # plot point of tangency  
            
            # plot everything on top of function after first slide
            if k > 0:
                # current second point through which we draw the secant line passing through (self.w_init, self.g(self.w_init))
                w_val = w_vals[k-1]
                g_val = self.g(w_val)
            
                # make slope of the current secant line
                slope = 0
                line_color = 'k'
                
                # switch colors and guides - when the secant point becomes the tangent point remove the vertical guides and color the line green
                line_color = 'r'
                if abs(self.w_init - w_val) > 10**-6:
                    slope = (g_init - g_val)/float(self.w_init - w_val)
                    
                    # plot vertical guiding lines at w_init and w_val
                    s = np.linspace(min(g_plot) - ggap,max(g_plot) + ggap,100)
                    o = np.ones(100)
                    ax.plot(o*w_val,s,linewidth = 1,alpha = 0.3,color = 'k',linestyle = '--')
                    ax.plot(o*self.w_init,s,linewidth = 1,alpha = 0.3,color = 'k',linestyle = '--')
                
                elif abs(self.w_init - w_val) < 10**-6 and self.mark_tangent == True:
                    slope = self.grad(self.w_init)
                    line_color = self.colors[0]

                # use point-slope form to create line (here the pt = (w_val,g_val))
                h = g_val + slope*(w_plot - w_val)

                # plot the approximation
                ax.plot(w_plot,h,color = line_color,linewidth = 2,zorder = 1)      # plot approx
 
                # plot other point of intersection of secant line and cost function
                ax.scatter(w_val,g_val,s = 120,c = 'b',edgecolor = 'k',linewidth = 0.7,zorder = 2)            # plot point of tangency                
                
            # fix viewing limits
            ax.set_xlim([-3,3])
            ax.set_ylim([min(g_plot) - ggap,max(g_plot) + ggap])
            
            # label axes
            ax.set_xlabel('$w$',fontsize = 25)
            ax.set_ylabel('$g(w)$',fontsize = 25,rotation = 0,labelpad = 25)
                
            return artist,

        anim = animation.FuncAnimation(fig, animate,frames=len(w_vals)+1, interval=len(w_vals)+1, blit=True)

        return(anim)