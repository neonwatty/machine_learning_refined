# import standard plotting and animation
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from IPython.display import clear_output
import time

# import autograd functionality
from autograd import grad as compute_grad   # The only autograd function you may ever need
import autograd.numpy as np
import math

# illustrate first and second order taylor series approximations to one-parameter input functions
class visualizer:
    '''
    Illustrate first and second order Taylor series approximations to a given input function at a
    coarsely chosen set of points.  Transition between the points using a custom slider mechanism
    to peruse how the approximations change from point-to-point.
    '''
    def __init__(self,**args):
        self.g = args['g']                       # input function
        self.grad = compute_grad(self.g)         # gradient of input function
        self.hess = compute_grad(self.grad)      # hessian of input function
        self.colors = [[0,1,0.25],[0,0.75,1]]    # set of custom colors used for plotting

    # compute first order approximation
    def draw_it(self,**args):
        num_frames = 300                          # number of slides to create - the input range [-3,3] is divided evenly by this number
        if 'num_frames' in args:
            num_frames = args['num_frames']
            
        # initialize figure
        fig = plt.figure(figsize = (6,6))
        artist = fig
        ax = fig.add_subplot(111)

        # generate a range of values over which to plot input function, and derivatives
        w_plot = np.linspace(-3,3,200)                  # input range for original function
        g_plot = self.g(w_plot)
        g_range = max(g_plot) - min(g_plot)             # used for cleaning up final plot
        ggap = g_range*0.5
        w_vals = np.linspace(-2.5,2.5,num_frames)       # range of values over which to plot first / second order approximations
        
        # which approximations to plot with the function?  Two switches: first_order and second_order
        first_order = False
        second_order = False
        if 'first_order' in args:
            first_order = args['first_order']
        if 'second_order' in args:
            second_order = args['second_order']
        print ('starting animation rendering...')
            
        # animation sub-function
        def animate(k):
            # clear the panel
            ax.cla()
            
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

            # plot original function
            ax.plot(w_plot,g_plot,color = 'k',zorder = 0)                           # plot function
            
            # plot the input/output tangency point
            ax.scatter(w_val,g_val,s = 90,c = 'r',edgecolor = 'k',linewidth = 0.7,zorder = 3)            # plot point of tangency
            
            # label axes
            ax.set_xlabel('$w$',fontsize = 17)
            ax.set_ylabel('$g(w)$',fontsize = 17)

            #### should we plot first order approximation? ####
            if first_order == True:
                # plug input into the first derivative
                g_grad_val = self.grad(w_val)

                # determine width to plot the approximation -- so its length == width
                width = 1
                div = float(1 + g_grad_val**2)
                w1 = w_val - math.sqrt(width/div)
                w2 = w_val + math.sqrt(width/div)

                # compute first order approximation
                wrange = np.linspace(w1,w2, 100)
                h = g_val + g_grad_val*(wrange - w_val)

                # plot the first order approximation
                ax.plot(wrange,h,color = self.colors[0],linewidth = 2,zorder = 2)      # plot approx

            #### should we plot second order approximation? ####
            if second_order == True:
                # plug input value into the second derivative
                g_grad_val = self.grad(w_val)
                g_hess_val = self.hess(g_grad_val)

                # determine width of plotting area for second order approximator
                width = 1
                if g_hess_val < 0:
                    width = - width

                # setup quadratic formula params
                a = 0.5*g_hess_val
                b = g_grad_val - 2*0.5*g_hess_val*w_val
                c = 0.5*g_hess_val*w_val**2 - g_grad_val*w_val - width

                # solve for zero points of the quadratic (for plotting purposes only)
                w1 = (-b + math.sqrt(b**2 - 4*a*c))/float(2*a + 0.00001)
                w2 = (-b - math.sqrt(b**2 - 4*a*c))/float(2*a + 0.00001)

                # create the second order approximation
                wrange = np.linspace(w1,w2, 100)
                h = g_val + g_grad_val*(wrange - w_val) + 0.5*g_hess_val*(wrange - w_val)**2 

                # plot the second order approximation
                ax.plot(wrange,h,color = self.colors[1],linewidth = 3,zorder = 1)      # plot approx

            # fix viewing limits on panel
            ax.set_xlim([-3,3])
            ax.set_ylim([min(g_plot) - ggap,max(g_plot) + ggap])
                
            return artist,
        
        anim = animation.FuncAnimation(fig, animate,frames=len(w_vals), interval=len(w_vals), blit=True)
        
        return(anim)