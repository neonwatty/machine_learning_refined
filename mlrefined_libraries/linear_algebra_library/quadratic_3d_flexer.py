# import standard plotting and animation
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from mpl_toolkits.mplot3d import Axes3D

# import autograd functionality
from autograd import grad as compute_grad   # The only autograd function you may ever need
import autograd.numpy as np
from mpl_toolkits.mplot3d import Axes3D

class visualizer:
    '''
    Draw 3d quadratic ranging from convex
    '''

    # compute first order approximation
    def draw_it(self,**args):  
        ### other options
        # size of figure
        set_figsize = 7
        if 'set_figsize' in args:
            set_figsize = args['set_figsize']
            
        # turn axis on or off
        set_axis = 'on'
        if 'set_axis' in args:
            set_axis = args['set_axis']
            
        # plot title
        set_title = ''
        if 'set_title' in args:
            set_title = args['set_title']
            
        # horizontal and vertical axis labels
        horiz_1_label = ''
        if 'horiz_1_label' in args:
            horiz_1_label = args['horiz_1_label']
            
        horiz_2_label = ''
        if 'horiz_2_label' in args:
            horiz_2_label = args['horiz_2_label']
            
        vert_label = ''
        if 'vert_label' in args:
            vert_label = args['vert_label']
            
        # set width of plot
        input_range = np.linspace(-3,3,100)                  # input range for original function
        if 'input_range' in args:
            input_range = args['input_range']
            
        # set viewing angle on plot
        view = [20,50]
        if 'view' in args:
            view = args['view']
        
        num_slides = 100
        if 'num_slides' in args:
            num_frames = args['num_slides']
            
        alpha_values = np.linspace(-2,2,num_frames)
        
        # initialize figure
        fig = plt.figure(figsize = (set_figsize,set_figsize))
        artist = fig
        ax = fig.add_subplot(111, projection='3d')

        # animation sub-function
        def animate(k):
            ax.cla()
            
            # quadratic to plot
            alpha = alpha_values[k]
            g = lambda w: w[0]**2 + alpha*w[1]**2
            
            # create grid from plotting range
            w1_vals,w2_vals = np.meshgrid(input_range,input_range)
            w1_vals.shape = (len(input_range)**2,1)
            w2_vals.shape = (len(input_range)**2,1)
            g_vals = g([w1_vals,w2_vals])
        
            # vals for cost surface
            w1_vals.shape = (len(input_range),len(input_range))
            w2_vals.shape = (len(input_range),len(input_range))
            g_vals.shape = (len(input_range),len(input_range))

            g_range = np.amax(g_vals) - np.amin(g_vals)             # used for cleaning up final plot
            ggap = g_range*0.5

            # plot original function
            ax.plot_surface(w1_vals,w2_vals,g_vals,alpha = 0.1,color = 'k',rstride=15, cstride=15,linewidth=2,edgecolor = 'k') 

            # clean up plotting area
            ax.set_title(set_title,fontsize = 15)
            ax.set_xlabel(horiz_1_label,fontsize = 15)
            ax.set_ylabel(horiz_2_label,fontsize = 15)
            ax.set_zlabel(vert_label,fontsize = 15)
            ax.view_init(view[0],view[1])
            ax.axis(set_axis)
 
            return artist,

        anim = animation.FuncAnimation(fig, animate ,frames=len(alpha_values), interval=len(alpha_values), blit=True)
        
        return(anim)
        