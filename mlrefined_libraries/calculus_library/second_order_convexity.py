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
    def draw_it(self,savepath,**kwargs):
        num_frames = 300                          # number of slides to create - the input range [-3,3] is divided evenly by this number
        if 'num_frames' in kwargs:
            num_frames = kwargs['num_frames']
            
        # initialize figure
        fig = plt.figure(figsize = (16,8))
        artist = fig

        # create subplot with 3 panels, plot input function in center plot
        gs = gridspec.GridSpec(1, 2, width_ratios=[1,1]) 
        fig.subplots_adjust(wspace=0.5,hspace=0.01)

        # plot input function
        ax = plt.subplot(gs[0])
        ax2 = plt.subplot(gs[1])
        
        # range over which to plot second derivative
        max_val = 2.5
        if 'max_val' in kwargs:
            max_val = kwargs['max_val']
        w_vals = np.linspace(-max_val,max_val,num_frames)       # range of values over which to plot first / second order approximations
        
        # generate a range of values over which to plot input function, and derivatives
        w_plot = np.linspace(-max_val,max_val,200)                  # input range for original function
       
        ### function evaluation and second derivative evaluation ###
        g_plot = self.g(w_plot)
        g_range = max(g_plot) - min(g_plot)             # used for cleaning up final plot
        ggap = g_range*0.5
        
        hess_plot = np.array([self.hess(v) for v in w_vals])
        hess_range = max(hess_plot) - min(hess_plot)
        hess_gap = hess_range*0.25
                 
        # animation sub-function
        print ('starting animation rendering...')
        def animate(k):
            # clear the panel
            ax.cla()
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
            
            ##### plot function and second order approximation in left panel #####
            # plot original function
            ax.plot(w_plot,g_plot,color = 'k',zorder = 0,linewidth=3)                           # plot function
            
            # plot the input/output tangency point
            ax.scatter(w_val,g_val,s = 120,c = 'lime',edgecolor = 'k',linewidth = 0.7,zorder = 3)            # plot point of tangency
            
            # label axes
            ax.set_xlabel('$w$',fontsize = 25)
            ax.set_ylabel(r'$g(w)$',fontsize=25,rotation = 0,labelpad = 50)
            
            # plug input value into the second derivative
            g_grad_val = self.grad(w_val)
            g_hess_val = self.hess(w_val)

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
            ax.plot(wrange,h,color = self.colors[1],linewidth = 5,zorder = 1)      # plot approx

            # fix viewing limits on panel
            ax.set_xlim([-max_val,max_val])
            ax.set_ylim([min(g_plot) - ggap,max(g_plot) + ggap])
            
            ##### plot second derivative value in right panel #####
            ax2.plot(w_vals[:k+1],hess_plot[:k+1],color = self.colors[1],linewidth = 3,zorder = 1) 
            ax2.plot(w_plot,w_plot*0,color = 'k',zorder = 1,linewidth = 1,linestyle = '--') 
            
            # fix viewing limits on panel
            ax2.set_xlim([-max_val,max_val])
            ax2.set_ylim([min(hess_plot) - hess_gap,max(hess_plot) + hess_gap])
            ax2.set_xlabel('$w$',fontsize = 25)            
            ax2.set_ylabel(r'$\frac{\partial^2}{\partial w}g(w)$',fontsize=25,rotation = 0,labelpad = 50)           
            
            return artist,
        
        anim = animation.FuncAnimation(fig, animate,frames=len(w_vals), interval=len(w_vals), blit=True)
        
        # produce animation and save
        fps = 50
        if 'fps' in kwargs:
            fps = kwargs['fps']
        anim.save(savepath, fps=fps, extra_args=['-vcodec', 'libx264'])
        clear_output()