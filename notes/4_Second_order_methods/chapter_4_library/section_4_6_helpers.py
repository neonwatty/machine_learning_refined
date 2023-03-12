import matplotlib.pyplot as plt
import matplotlib.animation as animation
from mpl_toolkits.mplot3d import Axes3D
from matplotlib import gridspec
from IPython.display import clear_output

from autograd import grad as compute_grad
import autograd.numpy as np
import time
import math

class function_addition_visualizer:
    '''
    This file illlustrates the sum of two functions in 3d.  Both functions are defined by the user.
    ''' 

    # animate the method
    def draw_it(self,h1,h2,savepath,**kwargs):
        # user input functions to add
        self.h1 = h1                            # input function
        self.h2 = h2                            # input function
        num_frames = 100
        if 'num_frames' in kwargs:
            num_frames = kwargs['num_frames']
            
        # turn axis on or off
        set_axis = 'on'
        if 'set_axis' in kwargs:
            set_axis = kwargs['set_axis']
  
        # set viewing angle on plot
        view = [20,50]
        if 'view' in kwargs:
            view = kwargs['view']
            
        epsmax = 2
        if 'epsmax' in kwargs:
            epsmax = kwargs['epsmax']
            
        # initialize figure
        fig = plt.figure(figsize = (15,5))
        artist = fig
        
        gs = gridspec.GridSpec(1, 3, width_ratios=[1,1, 1]) 
        ax1 = plt.subplot(gs[0],projection='3d');
        ax2 = plt.subplot(gs[1],projection='3d');
        ax3 = plt.subplot(gs[2],projection='3d');
        
        # generate input range for functions
        r = np.linspace(-3,3,100)
        w1_vals,w2_vals = np.meshgrid(r,r)
        w1_vals.shape = (len(r)**2,1)
        w2_vals.shape = (len(r)**2,1)
        h1_vals = self.h1([w1_vals,w2_vals])
        h2_vals = self.h2([w1_vals,w2_vals])
        
        # vals for cost surface
        w1_vals.shape = (len(r),len(r))
        w2_vals.shape = (len(r),len(r))
        h1_vals.shape = (len(r),len(r))
        h2_vals.shape = (len(r),len(r))

        # decide on number of slides
        epsilon_vals = np.linspace(0,epsmax,num_frames)

        # animation sub-function
        print ('starting animation rendering...')
        def animate(t):
            # clear panels for next slide
            ax1.cla()
            ax2.cla()
            ax3.cla()
            
            # print rendering update
            if np.mod(t+1,25) == 0:
                print ('rendering animation frame ' + str(t+1) + ' of ' + str(num_frames))
            if t == num_frames - 1:
                print ('animation rendering complete!')
                time.sleep(1.5)
                clear_output()
              
            # plot function 1
            ax1.plot_surface(w1_vals,w2_vals,h1_vals,alpha = 0.3,color = 'w',rstride=10, cstride=10,linewidth=2,edgecolor = 'k') 
            ax1.set_title("$h_1$",fontsize = 15)
            ax1.view_init(view[0],view[1])
            ax1.grid(False)


            # plot function 2
            ax2.plot_surface(w1_vals,w2_vals,h2_vals,alpha = 0.3,color = 'w',rstride=10, cstride=10,linewidth=2,edgecolor = 'k') 
            ax2.set_title("$h_2$",fontsize = 15)
            ax2.view_init(view[0],view[1])
            ax2.grid(False)
                
            # plot combination of both
            epsilon = epsilon_vals[t]
            h3_vals = h1_vals + epsilon*h2_vals
            ax3.plot_surface(w1_vals,w2_vals,h3_vals,alpha = 0.3,color = 'w',rstride=10, cstride=10,linewidth=2,edgecolor = 'k')
            ax3.grid(False)

            title = r'$h_1 + ' +  r'{:.2f}'.format(epsilon)  + 'h_2$'
            ax3.set_title(title,fontsize = 15)
            ax3.view_init(view[0],view[1])
            
            return artist,

        anim = animation.FuncAnimation(fig, animate ,frames=num_frames, interval=num_frames, blit=True)

        # produce animation and save
        fps = 50
        if 'fps' in kwargs:
            fps = kwargs['fps']
        anim.save(savepath, fps=fps, extra_args=['-vcodec', 'libx264'])
        clear_output()



class regularized_newton_visualizer:
    '''
    Illustrating how to regularize Newton's method to deal with nonconvexity.  Using a custom slider
    widget we can visualize the result of adding a pure weighted quadratic to the second derivative
    at each step of Newton's method.  Each time the slider is moved a new complete run of regularized
    Newton's method is illustrated, where at each step going from left to right the weight on the 
    pure quadratic is increased.
    
    For a non-convex function we can see how that - without reglarizing - we will climb to a local maximum,
    since at each step the quadratic approximation is concave.  However if the regularization parameter is set
    large enough the sum quadratic is made convex, and we can descend.  If the weight is made too high we 
    completely drown out second derivative and have gradient descent.
    ''' 
    
    def __init__(self,**args):
        self.g = args['g']                          # input function
        self.grad = compute_grad(self.g)            # first derivative of input function
        self.hess = compute_grad(self.grad)         # second derivative of input function
        self.w_init = float( -2.3)                  # initial point
        self.w_hist = []
        self.epsilon_range = np.linspace(0,2,20)       # range of regularization parameter to try
        self.max_its = 10
        
    ######## newton's method ########
    # run newton's method
    def run_newtons_method(self,epsilon):
        w_val = self.w_init
        self.w_hist = []
        self.w_hist.append(w_val)
        w_old = np.inf
        for j in range(int(self.max_its)):
            # update old w and index
            w_old = w_val
            
            # plug in value into func and derivative
            grad_val = float(self.grad(w_val))
            hess_val = float(self.hess(w_val))

            # take newtons step
            curvature = hess_val + epsilon
            if abs(curvature) > 10**-6:
                w_val = w_val - grad_val/curvature
            
            # record
            self.w_hist.append(w_val)

    # animate the method
    def animate_it(self,epsilon_range,savepath,**kwargs):
        self.epsilon_range = epsilon_range
        if 'w_init' in kwargs:
            self.w_init = float(kwargs['w_init'])
        if 'max_its' in kwargs:
            self.max_its = float(kwargs['max_its'])
        wmax = 3
        if 'wmax' in kwargs:
            wmax = kwargs['wmax']
    
        # initialize figure
        fig = plt.figure(figsize = (10,4))
        artist = fig
        
        gs = gridspec.GridSpec(1, 2, width_ratios=[2,1]) 
        ax1 = plt.subplot(gs[0],aspect = 'auto');
        ax2 = plt.subplot(gs[1],sharey=ax1);

        # generate function for plotting on each slide
        w_plot = np.linspace(-wmax,wmax,1000)
        g_plot = self.g(w_plot)
        g_range = max(g_plot) - min(g_plot)
        ggap = g_range*0.5
        w_vals = np.linspace(-2.5,2.5,50)
 
        # animation sub-function
        print ('starting animation rendering...')
        num_frames = len(self.epsilon_range) + 1
        def animate(k):
            # clear the previous panel for next slide
            ax1.cla()
            ax2.cla()
            
            # plot function 
            ax1.plot(w_plot,g_plot,color = 'k',zorder = 0)               # plot function
            
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
                ax1.scatter(w_val,g_val,s = 100,color = 'm',edgecolor = 'k',linewidth = 0.7,zorder = 2)            # plot point of tangency
                ax1.scatter(w_val,0,s = 100,color = 'm',edgecolor = 'k',linewidth = 0.7, zorder = 2, marker = 'X')

            # plot function alone first along with initial point
            if k > 0:
                epsilon = self.epsilon_range[k-1]
                
                # run gradient descent method
                self.w_hist = []
                self.run_newtons_method(epsilon)
        
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
                    ax1.scatter(w_val,g_val,s = 90,color = self.colorspec[j],edgecolor = 'k',linewidth = 0.7,zorder = 3)            # plot point of tangency
                    ax1.scatter(w_val,0,s = 90,facecolor = self.colorspec[j],marker = 'X',edgecolor = 'k',linewidth = 0.7, zorder = 2)
                    
                    # plug in value into func and derivative
                    g_val = self.g(w_val)
                    g_grad_val = self.grad(w_val)
                    g_hess_val = self.hess(w_val)

                    # determine width of plotting area for second order approximator
                    width = 0.5
                    if g_hess_val < 0:
                        width = - width

                    # compute second order approximation
                    wrange = np.linspace(w_val - 3,w_val + 3, 100)
                    h = g_val + g_grad_val*(wrange - w_val) + 0.5*(g_hess_val + epsilon)*(wrange - w_val)**2 

                    # plot all
                    ax1.plot(wrange,h,color = self.colorspec[j],linewidth = 2,alpha = 0.4,zorder = 1)      # plot approx
            
                    ### plot all on cost function decrease plot
                    ax2.scatter(j,g_val,s = 90,color = self.colorspec[j],edgecolor = 'k',linewidth = 0.7,zorder = 3)            # plot point of tangency
                    
                    # place title
                    title = r'$\epsilon = $' + r'{:.2f}'.format(epsilon)
                    ax1.set_title(title,fontsize = 15)
            
                    # plot connector between points for visualization purposes
                    if j > 0:
                        w_old = self.w_hist[j-1]
                        w_new = self.w_hist[j]
                        g_old = self.g(w_old)
                        g_new = self.g(w_new)
                        ax2.plot([j-1,j],[g_old,g_new],color = self.colorspec[j],linewidth = 2,alpha = 0.4,zorder = 1)      # plot approx
            else:
                title = r'$\,\,\,$'
                ax1.set_title(title,fontsize = 15)

            # clean up axis in each panel
            ax2.set_xlabel('iteration',fontsize = 13)
            ax2.set_ylabel(r'$g(w)$',fontsize = 13,labelpad = 15,rotation = 0)
            ax1.set_xlabel(r'$w$',fontsize = 13)
            ax1.set_ylabel(r'$g(w)$',fontsize = 13,labelpad = 15,rotation = 0)
                    
            # fix viewing limits
            ax1.set_xlim([-wmax,wmax])
            ax1.set_ylim([min(g_plot) - ggap,max(g_plot) + ggap])
            
            ax2.set_xlim([-0.5,self.max_its + 0.5])
            ax2.set_ylim([min(g_plot) - ggap,max(g_plot) + ggap])
            
            # set tickmarks
            ax1.set_xticks(np.arange(-round(wmax), round(wmax) + 1, 1.0))
            ax1.set_yticks(np.arange(round(min(g_plot) - ggap), round(max(g_plot) + ggap) + 1, 1.0))
           
            ax2.set_xticks(np.arange(0,self.max_its + 1, 1.0))
                        
            return artist,

        anim = animation.FuncAnimation(fig, animate,frames=num_frames, interval=num_frames, blit=True)

        # produce animation and save
        fps = 50
        if 'fps' in kwargs:
            fps = kwargs['fps']
        anim.save(savepath, fps=fps, extra_args=['-vcodec', 'libx264'])
        clear_output()