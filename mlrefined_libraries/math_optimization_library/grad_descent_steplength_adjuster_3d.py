# import standard plotting and animation
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from mpl_toolkits.mplot3d import Axes3D
from IPython.display import clear_output

# import autograd functionality
from autograd import grad as compute_grad   # The only autograd function you may ever need
import autograd.numpy as np
from autograd import jacobian
from autograd import hessian
import math
import time
from matplotlib import gridspec

# simple first order taylor series visualizer
class visualizer:
    '''
    Illustrator for the affect of step size choice on the behavior of gradient descent on a 3d cost function (2 inputs).  User chooses
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
        w_val = self.w_init
        self.w_hist = []
        self.w_hist.append(w_val)
        w_old = np.inf
        j = 0
        while np.linalg.norm(w_old - w_val)**2 > 10**-5 and j < self.max_its:
            # update old w and index
            w_old = w_val
            j+=1
            
            # plug in value into func and derivative
            grad_val = self.grad(w_val)
            grad_val.shape = (2,1)
            
            # take newtons step
            w_val = w_val - alpha*grad_val
            
            # record
            self.w_hist.append(w_val)
            
    # determine plotting area for function based on current gradient descent run
    def plot_function(self,ax):
        big_val1 = np.amax(np.asarray([abs(a[0]) for a in self.w_hist]))
        big_val2 = np.amax(np.asarray([abs(a[1]) for a in self.w_hist]))
        big_val = max(big_val1,big_val2,3)
        
        # create plotting range
        r = np.linspace(-big_val,big_val,100)

        # create grid from plotting range
        w1_vals,w2_vals = np.meshgrid(r,r)
        w1_vals.shape = (len(r)**2,1)
        w2_vals.shape = (len(r)**2,1)
        g_vals = self.g([w1_vals,w2_vals])

        # vals for cost surface
        w1_vals.shape = (len(r),len(r))
        w2_vals.shape = (len(r),len(r))
        g_vals.shape = (len(r),len(r))
        
        # vals for plotting range
        gmin = np.amin(g_vals)
        gmax = np.amax(g_vals)
        ggap = (gmax - gmin)*0.1
        gmin = gmin - ggap
        gmax = gmax + ggap
        
        # plot and fix up panel
        strider = int(round(45/float(big_val)))
        strider = max(strider,2)
        ax.plot_surface(w1_vals,w2_vals,g_vals,alpha = 0.1,color = 'k',rstride=strider, cstride=strider ,linewidth=1,edgecolor = 'k')  
        
    # animate the method
    def animate_it(self,savepath,**kwargs):
        self.g = kwargs['g']                               # input function defined by user
        self.grad = compute_grad(self.g)                 # first derivative of input
        self.hess = compute_grad(self.grad)              # second derivative of input
        self.alpha_range = np.linspace(10**-4,1,20)      # default range of alpha (step length) values to try, adjustable
        self.max_its = 20
        
        # adjust range of step values to illustrate as well as initial point for all runs
        if 'alpha_range' in kwargs:
            self.alpha_range = kwargs['alpha_range']
            
        if 'max_its' in kwargs:
            self.max_its = kwargs['max_its']
 
        if 'w_init' in kwargs:
            w_init = kwargs['w_init']
            w_init = [float(a) for a in w_init]
            self.w_init = np.asarray(w_init)
            self.w_init.shape = (2,1)
            
        view = [10,50]
        if 'view' in kwargs:
            view = kwargs['view']
            
        # initialize figure
        fig = plt.figure(figsize = (9,5))
        artist = fig
        
        # create subplot with 3 panels, plot input function in center plot
        gs = gridspec.GridSpec(1, 2, width_ratios=[3,1]) 
        ax1 = plt.subplot(gs[0],projection='3d'); 
        ax2 = plt.subplot(gs[1]); 

        # animation sub-function
        print ('starting animation rendering...')
        num_frames = len(self.alpha_range)+1
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
                ax1.scatter(w_val[0],w_val[1],g_val,s = 100,c = 'm',edgecolor = 'k',linewidth = 0.7,zorder = 2)            # plot point of tangency
                
                # plot function 
                r = np.linspace(-3,3,100)

                # create grid from plotting range
                w1_vals,w2_vals = np.meshgrid(r,r)
                w1_vals.shape = (len(r)**2,1)
                w2_vals.shape = (len(r)**2,1)
                g_vals = self.g([w1_vals,w2_vals])

                # vals for cost surface
                w1_vals.shape = (len(r),len(r))
                w2_vals.shape = (len(r),len(r))
                g_vals.shape = (len(r),len(r))

                ax1.plot_surface(w1_vals,w2_vals,g_vals,alpha = 0.1,color = 'k',rstride=15, cstride=15,linewidth=1,edgecolor = 'k')    

            # plot function alone first along with initial point
            if k > 0:
                alpha = self.alpha_range[k-1]
                
                # setup axes
                ax1.set_title(r'$\alpha = $' + r'{:.2f}'.format(alpha),fontsize = 14)
                ax2.set_xlabel('iteration',fontsize = 13)
                ax2.set_ylabel('cost function value',fontsize = 13)          
                
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
                    ax1.scatter(w_val[0],w_val[1],g_val,s = 90,c = self.colorspec[j],edgecolor = 'k',linewidth = 0.7,zorder = 3)            # plot point of tangency
            
                    ### plot all on cost function decrease plot
                    ax2.scatter(j,g_val,s = 90,c = self.colorspec[j],edgecolor = 'k',linewidth = 0.7,zorder = 3)            # plot point of tangency
                    
                    # clean up second axis
                    ax2.set_xticks(np.arange(len(self.w_hist)))
                    
                    # plot connector between points for visualization purposes
                    if j > 0:
                        w_old = self.w_hist[j-1]
                        w_new = self.w_hist[j]
                        g_old = self.g(w_old)
                        g_new = self.g(w_new)
                        ax2.plot([j-1,j],[g_old,g_new],color = self.colorspec[j],linewidth = 2,alpha = 0.4,zorder = 1)      # plot approx
                        
            # clean up plot
            ax1.view_init(view[0],view[1])
            ax1.set_axis_off()
 
            return artist,

        anim = animation.FuncAnimation(fig, animate ,frames=num_frames, interval=num_frames, blit=True)
        
        # produce animation and save
        fps = 50
        if 'fps' in kwargs:
            fps = kwargs['fps']
        anim.save(savepath, fps=fps, extra_args=['-vcodec', 'libx264'])
        
        clear_output()    