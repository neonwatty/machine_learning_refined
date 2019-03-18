# import standard plotting and animation
import matplotlib.pyplot as plt
import matplotlib.animation as animation

# import autograd functionality
from autograd import grad as compute_grad   # The only autograd function you may ever need
import autograd.numpy as np
import math
import time
from IPython.display import clear_output
from matplotlib import gridspec

# simple first order taylor series visualizer
class visualizer:
    '''
    Illustrator for the affect of step size choice on the behavior of gradient descent.  User chooses
       a) an input function
       b) an initial point 
       c) a range of step length values to try
    Several runs of gradient descent are then executed - one for each choice of step length to try -
    and a custom slider widget is used to visualize each completed run.  As the slider is moved from 
    left to right a different run - with another step size - is illustrated graphically.  Points in each
    run are colored green (if near the start of the run) to yellow (as the run approaches its maximum number
    of iterations) to red (when near completion).  Points are shown both plotted on the cost function itself,
    as well as a cost function history plotted per-iteration.
    ''' 
             
    ######## gradient descent ########
     # run gradient descent 
    def run_gradient_descent(self,alpha):
        w = self.w_init
        self.w_hist = []
        self.w_hist.append(w)
        w_old = np.inf
        j = 0
        for j in range(int(self.max_its)):
            # update old w and index
            w_old = w
            
            # plug in value into func and derivative
            grad_eval = float(self.grad(w))
            
            # normalized or unnormalized?
            if self.version == 'normalized':
                grad_norm = abs(grad_eval)
                if grad_norm == 0:
                    grad_norm += 10**-6*np.sign(2*np.random.rand(1) - 1)
                grad_eval /= grad_norm
                         
            # take gradient descent step
            w = w - alpha*grad_eval
            
            # record
            self.w_hist.append(w)
           
            
    # adaptive plotting for input function
    def plot_function(self,ax):
        big_val = np.amax(np.asarray([abs(a) for a in self.w_hist]))
        big_val = max(big_val,3)
        
        # create plotting range
        w_plot = np.linspace(-big_val,big_val,500)
        g_plot = self.g(w_plot)
        
        # plot function
        ax.plot(w_plot,g_plot,color = 'k',zorder = 0)               # plot function
            
    # animate the method
    def animate_it(self,savepath,**kwargs):
        # presets
        self.g = kwargs['g']                            # input function
        self.grad = compute_grad(self.g)              # gradient of input function
        self.w_init =float( -2)                       # user-defined initial point (adjustable when calling each algorithm)
        self.max_its = 20                             # max iterations to run for each algorithm
        self.w_hist = []                              # container for algorithm path
        wmin = -3.1                                   # max and min viewing
        wmax = 3.1  
        self.steplength_range = np.linspace(10**-4,1,20)      # default range of alpha (step length) values to try, adjustable
        
        # adjust range of step values to illustrate as well as initial point for all runs
        if 'steplength_range' in kwargs:
            self.steplength_range = kwargs['steplength_range']
        if 'wmin' in kwargs:            
            wmin = kwargs['wmin']
        if 'wmax' in kwargs:
            wmax = kwargs['wmax']
        
        # get new initial point if desired
        if 'w_init' in kwargs:
            self.w_init = kwargs['w_init']
            
        # take in user defined step length
        if 'steplength' in kwargs:
            self.steplength = kwargs['steplength']
            
        # take in user defined maximum number of iterations
        if 'max_its' in kwargs:
            self.max_its = float(kwargs['max_its'])
            
        # version of gradient descent to use (normalized or unnormalized)
        self.version = 'unnormalized'
        if 'version' in kwargs:
            self.version = kwargs['version']
            
        # turn on first order approximation illustrated at each step
        tracers = 'off'
        if 'tracers' in kwargs:
            tracers = kwargs['tracers']
           
        # initialize figure
        fig = plt.figure(figsize = (9,4))
        artist = fig
        
        # create subplot with 2 panels, plot input function in center plot
        gs = gridspec.GridSpec(1, 2, width_ratios=[1,1]) 
        ax1 = plt.subplot(gs[0]);
        ax2 = plt.subplot(gs[1],sharey=ax1); 
        gs.update(wspace=0.5, hspace=0.1) 

        # generate function for plotting on each slide
        w_plot = np.linspace(wmin,wmax,500)
        g_plot = self.g(w_plot)
        g_range = max(g_plot) - min(g_plot)
        ggap = g_range*0.1
        width = 30
        
        # animation sub-function
        num_frames = len(self.steplength_range)+1
        print ('starting animation rendering...')
        def animate(k):
            ax1.cla()
            ax2.cla()
            
            # print rendering update            
            if np.mod(k+1,25) == 0:
                print ('rendering animation frame ' + str(k+1) + ' of ' + str(num_frames))
            if k == num_frames - 1:
                print ('animation rendering complete!')
                time.sleep(1.5)
                clear_output()
                
            # plot initial point and evaluation
            if k == 0:
                w_val = self.w_init
                g_val = self.g(w_val)
                ax1.scatter(w_val,g_val,s = 100,c = 'm',edgecolor = 'k',linewidth = 0.7,zorder = 2)            # plot point of tangency
                # ax1.scatter(w_val,0,s = 100,c = 'm',edgecolor = 'k',linewidth = 0.7, zorder = 2, marker = 'X')
                # plot function 
                ax1.plot(w_plot,g_plot,color = 'k',zorder = 0)               # plot function

            # plot function alone first along with initial point
            if k > 0:
                alpha = self.steplength_range[k-1]
                
                # run gradient descent method
                self.w_hist = []
                self.run_gradient_descent(alpha = alpha)
                
                # plot function
                self.plot_function(ax1)
        
                # colors for points
                s = np.linspace(0,1,len(self.w_hist[:round(len(self.w_hist)/2)]))
                s.shape = (len(s),1)
                t = np.ones(len(self.w_hist[round(len(self.w_hist)/2):]))
                t.shape = (len(t),1)
                s = np.vstack((s,t))
                self.colorspec = []
                self.colorspec = np.concatenate((s,np.flipud(s)),1)
                self.colorspec = np.concatenate((self.colorspec,np.zeros((len(s),1))),1)
        
                # plot everything for each iteration 
                for j in range(len(self.w_hist)):  
                    w_val = self.w_hist[j]
                    g_val = self.g(w_val)
                    grad_val = self.grad(w_val)
                    ax1.scatter(w_val,g_val,s = 90,c = self.colorspec[j],edgecolor = 'k',linewidth = 0.7,zorder = 3)            # plot point of tangency
                    
                    # ax1.scatter(w_val,0,s = 90,facecolor = self.colorspec[j],marker = 'X',edgecolor = 'k',linewidth = 0.7, zorder = 2)
                    
                    # determine width to plot the approximation -- so its length == width defined above
                    div = float(1 + grad_val**2)
                    w1 = w_val - math.sqrt(width/div)
                    w2 = w_val + math.sqrt(width/div)

                    # use point-slope form of line to plot
                    wrange = np.linspace(w1,w2, 100)
                    h = g_val + grad_val*(wrange - w_val)
                
                    # plot tracers connecting consecutive points on the cost (for visualization purposes)
                    if tracers == 'on':
                        if j > 0:
                            w_old = self.w_hist[j-1]
                            w_new = self.w_hist[j]
                            g_old = self.g(w_old)
                            g_new = self.g(w_new)
                            ax1.quiver(w_old, g_old, w_new - w_old, g_new - g_old, scale_units='xy', angles='xy', scale=1, color = self.colorspec[j],linewidth = 1.5,alpha = 0.2,linestyle = '-',headwidth = 4.5,edgecolor = 'k',headlength = 10,headaxislength = 7)
            
                    ### plot all on cost function decrease plot
                    ax2.scatter(j,g_val,s = 90,c = self.colorspec[j],edgecolor = 'k',linewidth = 0.7,zorder = 3)            # plot point of tangency
                    
                    # clean up second axis, set title on first
                    ax2.set_xticks(np.arange(len(self.w_hist)))
                    ax1.set_title(r'$\alpha = $' + r'{:.2f}'.format(alpha),fontsize = 14)

                    # plot connector between points for visualization purposes
                    if j > 0:
                        w_old = self.w_hist[j-1]
                        w_new = self.w_hist[j]
                        g_old = self.g(w_old)
                        g_new = self.g(w_new)
                        ax2.plot([j-1,j],[g_old,g_new],color = self.colorspec[j],linewidth = 2,alpha = 0.4,zorder = 1)      # plot approx
 
            ### clean up function plot ###
            # fix viewing limits on function plot
            #ax1.set_xlim([-3,3])
            #ax1.set_ylim([min(g_plot) - ggap,max(g_plot) + ggap])
            
            # draw axes and labels
            ax1.set_xlabel(r'$w$',fontsize = 13)
            ax1.set_ylabel(r'$g(w)$',fontsize = 13,rotation = 0,labelpad = 25)   

            ax2.set_xlabel('iteration',fontsize = 13)
            ax2.set_ylabel(r'$g(w)$',fontsize = 13,rotation = 0,labelpad = 25)
            ax1.axhline(y=0, color='k',zorder = 0,linewidth = 0.5)
            ax2.axhline(y=0, color='k',zorder = 0,linewidth = 0.5)

            return artist,

        anim = animation.FuncAnimation(fig, animate ,frames=num_frames, interval=num_frames, blit=True)

        # produce animation and save
        fps = 50
        if 'fps' in kwargs:
            fps = kwargs['fps']
        anim.save(savepath, fps=fps, extra_args=['-vcodec', 'libx264'])
        
        clear_output()    