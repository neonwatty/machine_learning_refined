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
        fig = plt.figure(figsize = (10,5))
        artist = fig

        # create subplot with 3 panels, plot input function in center plot
        gs = gridspec.GridSpec(1, 3, width_ratios=[1,3, 1]) 
        ax1 = plt.subplot(gs[0]); ax1.axis('off');
        ax3 = plt.subplot(gs[2]); ax3.axis('off');

        # plot input function
        ax = plt.subplot(gs[1],aspect = 'equal')
        
        max_val = 2.5
        if 'max_val' in kwargs:
            max_val = kwargs['max_val']
        w_vals = np.linspace(-max_val + 0.2,max_val - 0.2,num_frames)       # range of values over which to plot first / second order approximations
        
        # generate a range of values over which to plot input function, and derivatives
        w_plot = np.linspace(-max_val-0.5,max_val+0.5,200)                  # input range for original function
        
        g_plot = self.g(w_plot)
        g_range = max(g_plot) - min(g_plot)             # used for cleaning up final plot
        ggap = g_range*0.5
     
        # which approximations to plot with the function?  Two switches: first_order and second_order
        first_order = False
        second_order = False
        if 'first_order' in kwargs:
            first_order = kwargs['first_order']
        if 'second_order' in kwargs:
            second_order = kwargs['second_order']
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
            ax.plot(w_plot,g_plot,color = 'k',zorder = 0,linewidth=1)                           # plot function
            
            # plot the input/output tangency point
            ax.scatter(w_val,g_val,s = 80,c = 'red',edgecolor = 'k',linewidth = 1,zorder = 3,marker = 'X')            # plot point of tangency
            ax.scatter(w_val,0,s = 100,c = 'red',edgecolor = 'k',linewidth = 1,zorder = 3)            # plot point of tangency
            tempy = np.linspace(0,g_val,100)
            tempx = w_val*np.ones((100))
            ax.plot(tempx,tempy,linewidth = 0.7,color = 'k',linestyle = '--',zorder = 1)

            #### should we plot first order approximation? ####
            if first_order == True:
                # plug input into the first derivative
                g_grad_val = self.grad(w_val)

                '''
                # determine width to plot the approximation -- so its length == width
                width = 10
                div = float(1 + g_grad_val**2)
                w1 = w_val - math.sqrt(width/div)
                w2 = w_val + math.sqrt(width/div)
                '''
                
                # or just constant width
                w1 = w_val - 0.5
                w2 = w_val + 0.5

                # compute first order approximation
                wrange = np.linspace(w1,w2, 100)
                h = g_val + g_grad_val*(wrange - w_val)

                # plot the first order approximation
                ax.plot(wrange,h,color = self.colors[0],linewidth = 3,zorder = 1)      # plot approx

            #### should we plot second order approximation? ####
            if second_order == True:
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
                ax.plot(wrange,h,color = self.colors[1],linewidth = 3,zorder = 2)      # plot approx

            # fix viewing limits on panel
            ax.set_xlim([-max_val,max_val])
            ax.set_ylim([min(g_plot) - ggap,max(g_plot) + ggap])
            
            # set tickmarks
            ax.set_xticks(-np.arange(-round(max_val), round(max_val) + 1, 1.0))
            ax.set_yticks(np.arange(round(min(g_plot) - ggap), round(max(g_plot) + ggap) + 1, 1.0))
            
            # label axes
            ax.set_xlabel('$w$',fontsize = 18)
            ax.set_ylabel('$g(w)$',fontsize = 18,rotation = 0,labelpad = 25)
            ax.set_title(r'$w^0 = ' + str(np.round(w_val,2)) +  '$',fontsize = 19)
                
            # set axis 
            ax.axhline(y=0, color='k',zorder = 0,linewidth = 0.5)
            
            return artist,
        
        anim = animation.FuncAnimation(fig, animate,frames=len(w_vals), interval=len(w_vals), blit=True)
        
        # produce animation and save
        fps = 50
        if 'fps' in kwargs:
            fps = kwargs['fps']
        anim.save(savepath, fps=fps, extra_args=['-vcodec', 'libx264'])
        clear_output()