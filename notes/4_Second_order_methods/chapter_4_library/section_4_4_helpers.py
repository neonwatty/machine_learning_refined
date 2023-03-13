import matplotlib.pyplot as plt
import matplotlib.animation as animation
from matplotlib import gridspec
from IPython.display import clear_output
from mpl_toolkits.mplot3d import Axes3D
from mpl_toolkits.mplot3d import proj3d
from matplotlib.patches import FancyArrowPatch
from matplotlib.text import Annotation
from mpl_toolkits.mplot3d.proj3d import proj_transform

from autograd import grad as compute_grad 
from autograd.misc.flatten import flatten_func
from autograd import hessian as compute_hess
from autograd import value_and_grad   
import autograd.numpy as np
from autograd import jacobian
from autograd import hessian
import math
import time
import copy



# illustrate first and second order taylor series approximations to one-parameter input functions
class taylor_2d_visualizer:
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
            ax.scatter(w_val,g_val,s = 80,color = 'red',edgecolor = 'k',linewidth = 1,zorder = 3,marker = 'X')            # plot point of tangency
            ax.scatter(w_val,0,s = 100,color = 'red',edgecolor = 'k',linewidth = 1,zorder = 3)            # plot point of tangency
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




class taylor_3d_visualizer:
    '''
    Illustrate first and second order Taylor series approximations to a given input function at a
    user defined point in 3-dimensions.
    '''
    def __init__(self,**args):
        self.g = args['g']                      # user-defined input function
        self.grad = jacobian(self.g)            # first derivative of input point
        self.hess = hessian(self.g)             # second derivative of input point
        
        # default colors
        self.colors = [[0,1,0.25],[0,0.75,1]]   # custom colors

    # draw taylor series approximation to 3d function
    def draw_it(self,**kwargs):
        # which approximations to plot with the function?  first_order == True (draw first order approximation), second_order == True (draw second order approximation)
        first_order = False
        second_order = False
        if 'first_order' in kwargs:
            first_order = kwargs['first_order']
        if 'second_order' in kwargs:
            second_order = kwargs['second_order']
        view = [20,20]
        if 'view' in kwargs:
            view = kwargs['view']
            
        # get user-defined point
        w_val = kwargs['w_val']
        
        # initialize figure
        fig = plt.figure(figsize = (9,3))

        # create subplot with 3 panels, plot input function in center plot
        gs = gridspec.GridSpec(1, 3, width_ratios=[1,2, 1]) 
        ax1 = plt.subplot(gs[0]); ax1.axis('off');
        ax3 = plt.subplot(gs[2]); ax3.axis('off');

        # create panel for input function
        ax2 = plt.subplot(gs[1], projection='3d')

        ax2.set_xlabel('$w_1$',fontsize = 17)
        ax2.set_ylabel('$w_2$',fontsize = 17)
        ax2.zaxis.set_rotate_label(False)  # disable automatic rotation
        ax2.set_zlabel('$g(w_1,w_2)$',fontsize = 17,labelpad = 30,rotation = 0)
        
        # create plotting range
        r = np.linspace(-3,3,100)

        # create grid from plotting range
        w1_vals,w2_vals = np.meshgrid(r,r)
        w1_vals.shape = (len(r)**2,1)
        w2_vals.shape = (len(r)**2,1)
        g_vals = self.g([w1_vals,w2_vals])

        # plot cost surface
        w1_vals.shape = (len(r),len(r))
        w2_vals.shape = (len(r),len(r))
        g_vals.shape = (len(r),len(r))
        ax2.plot_surface(w1_vals,w2_vals,g_vals,alpha = 0.1,color = 'w',rstride=15, cstride=15,linewidth=1,edgecolor = 'k')
        
        # get input/output pairs
        w_val = [float(a) for a in w_val]
        w_val = np.asarray(w_val)
        w1_val = w_val[0]
        w2_val = w_val[1]
        g_val = self.g(w_val)
        grad_val = self.grad(w_val)
        grad_val.shape = (2,1)
        
       # plot tangency point
        ax2.scatter(w1_val,w2_val,g_val,s = 50,color = 'lime',edgecolor = 'k',linewidth = 0.7,zorder = 3)            # plot point of tangency

        # plot first order approximation
        if first_order == True:
            # compute first order approximation
            t1 = np.linspace(-1.5, 1.5,100)
            t2 = np.linspace(-1.5, 1.5,100)
            wrange1,wrange2 = np.meshgrid(t1,t2)
            wrange1.shape = (len(t1)**2,1) 
            wrange2.shape = (len(t1)**2,1) 
            wrange =  np.hstack((wrange1,wrange2))

            # first order function
            h = lambda weh: g_val + np.dot(weh - w_val ,grad_val)
            h_val = h(wrange + w_val)

            # # plot all
            wrange1 = wrange1 + w_val[0]
            wrange2 = wrange2 + w_val[1]
            wrange1.shape = (len(t1),len(t1)) 
            wrange2.shape = (len(t1),len(t1)) 
            h_val.shape = (len(t1),len(t1))
            ax2.plot_surface(wrange1,wrange2,h_val,alpha = 0.2,color = 'lime',rstride=15, cstride=15,linewidth=1,edgecolor = 'k')

        # print second order approximation
        if second_order == True:
            # compute hessian at input point
            hess = self.hess(w_val)
            
            # setup grid - without compensation 
            t1 = np.linspace(-1.5, 1.5,100)
            t2 = np.linspace(-1.5, 1.5,100)
            wrange1,wrange2 = np.meshgrid(t1,t2)
            wrange1.shape = (len(t1)**2,1) 
            wrange2.shape = (len(t1)**2,1) 
            wrange =  np.hstack((wrange1,wrange2))
            temp =  0.5*np.dot(np.dot(wrange - w_val,hess).T,wrange - w_val)
            
            # first order function
            h = lambda weh: g_val + np.dot(weh - w_val ,grad_val) + 0.5*np.dot(np.dot(weh - w_val,hess).T,weh - w_val)
            h_val = []
            for i in range(len(wrange)):
                pt = wrange[i] + w_val
                h_pt = h(pt)
                h_val.append(h_pt)
            h_val = np.asarray(h_val)

            # # plot all
            wrange1 = wrange1 + w_val[0]
            wrange2 = wrange2 + w_val[1]
            wrange1.shape = (len(t1),len(t1)) 
            wrange2.shape = (len(t1),len(t1)) 
            h_val.shape = (len(t1),len(t1))
            ax2.plot_surface(wrange1,wrange2,h_val,alpha = 0.4,color = self.colors[1],rstride=15, cstride=15,linewidth=1,edgecolor = 'k')

        # clean up plot
        ax2.grid(False);
        ax2.view_init(view[0],view[1]);
        plt.show()   




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

class majorizer_visualizer:
    '''
    Illustrate majorization of second order Taylor series approximations to a function
    '''
    def __init__(self,**args):
        self.g = args['g']                       # input function
        self.grad = compute_grad(self.g)         # gradient of input function
        self.hess = compute_grad(self.grad)      # hessian of input function
        self.colors = [[0,1,0.25],[0,0.75,1]]    # set of custom colors used for plotting

    # compute first order approximation
    def animate_it(self,savepath,**kwargs):
        num_frames = 300                          # number of slides to create - the input range [-3,3] is divided evenly by this number
        if 'num_frames' in kwargs:
            num_frames = kwargs['num_frames']
            
        # initialize figure
        fig = plt.figure(figsize = (10,5))
        artist = fig

        # create subplot with 3 panels, plot input function in center plot
        gs = gridspec.GridSpec(1, 3, width_ratios=[1,5, 1]) 
        ax1 = plt.subplot(gs[0]); ax1.axis('off');
        ax3 = plt.subplot(gs[2]); ax3.axis('off');

        # plot input function
        ax = plt.subplot(gs[1],aspect = 'equal')
        
        max_val = 2.5
        if 'max_val' in kwargs:
            max_val = kwargs['max_val']
        w_vals = np.linspace(-max_val+0.1,max_val-0.1,num_frames)       # range of values over which to plot first / second order approximations
        
        # generate a range of values over which to plot input function, and derivatives
        w_plot = np.linspace(-max_val,max_val,1000)                  # input range for original function
        
        g_plot = self.g(w_plot)
        g_range = max(g_plot) - min(g_plot)             # used for cleaning up final plot
        ggap = g_range*0.1
     
        # animation sub-function
        print ('starting animation rendering...')
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
            
            # grab the next input/output tangency pair, minimum of gradient quadratic
            w_val = w_vals[k]
            g_val = self.g(w_val)
            grad_val = self.grad(w_val)
            hess_val = self.hess(w_val)

            # plot original function
            ax.plot(w_plot,g_plot,color = 'k',zorder = 0,linewidth=1)                           # plot function
            
            # plot the input/output tangency point
            ax.scatter(w_val,g_val,s = 60,color = 'r',edgecolor = 'k',linewidth = 0.7,marker = 'X',zorder = 3)            # plot point of tangency
            ax.scatter(w_val,0,s = 80,color = 'r',edgecolor = 'k',linewidth = 0.7,zorder = 3)            # plot point of tangency
            
            # plot visual aid for old point
            tempy = np.linspace(0,g_val,100)
            tempx = w_val*np.ones((100))
            ax.plot(tempx,tempy,linewidth = 0.7,color = 'k',linestyle = '--',zorder = 1)

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
            w_major = np.linspace(w1,w2, 1000)
            h = lambda w: g_val + g_grad_val*(w - w_val) + 0.5*g_hess_val*(w - w_val)**2 
            h_major = h(w_major)
            
            # compute minimum point
            eps = 0
            if abs(hess_val) < 10**-5:
                eps = 10**-5
            w_step = w_val - grad_val/(hess_val + eps)
            h_step = h(w_step)
            g_step = self.g(w_step)
            
            ax.scatter(w_step,h_step,s = 60,color = 'blue',edgecolor = 'k',linewidth = 0.7,marker = 'X',zorder = 3)           
            ax.scatter(w_step,g_step,s = 60,color = 'lime',edgecolor = 'k',linewidth = 0.7,marker = 'X',zorder = 3)            # plot point of tangency
            ax.scatter(w_step,0,s = 80,color = 'lime',edgecolor = 'k',linewidth = 0.7,zorder = 3)            # plot point of tangency
            
            # plot visual aid for new point
            tempy = np.linspace(0,h_step,100)
            tempx = w_step*np.ones((100))
            ax.plot(tempx,tempy,linewidth = 0.7,color = 'k',linestyle = '--',zorder = 1)
            
            # plot approximation
            ax.plot(w_major,h_major,color = self.colors[1],zorder = 1,linewidth=2)                           # plot function
          
            # label axes
            ax.set_xlabel('$w$',fontsize = 12)
            ax.set_ylabel('$g(w)$',fontsize = 12,rotation = 0,labelpad = 12)

            # fix viewing limits on panel
            ax.set_xlim([-max_val,max_val])
            ax.set_ylim([min(-0.3,min(g_plot) - ggap),max(max(g_plot) + ggap,0.3)])
            
            # set tickmarks
            ax.set_xticks(-np.arange(-round(max_val), round(max_val) + 1, 1.0))
            ax.set_yticks(np.arange(round(min(g_plot) - ggap), round(max(g_plot) + ggap) + 1, 1.0))
                
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


        # gradient descent function - inputs: g (input function), alpha (steplength parameter), max_its (maximum number of iterations), w (initialization)
def gradient_descent(g,alpha_choice,max_its,w):
    # flatten the input function to more easily deal with costs that have layers of parameters
    g_flat, unflatten, w = flatten_func(g, w) # note here the output 'w' is also flattened

    # compute the gradient function of our input function - note this is a function too
    # that - when evaluated - returns both the gradient and function evaluations (remember
    # as discussed in Chapter 3 we always ge the function evaluation 'for free' when we use
    # an Automatic Differntiator to evaluate the gradient)
    gradient = value_and_grad(g_flat)

    # run the gradient descent loop
    weight_history = []      # container for weight history
    cost_history = []        # container for corresponding cost function history
    alpha = 0
    for k in range(1,max_its+1):
        # check if diminishing steplength rule used
        if alpha_choice == 'diminishing':
            alpha = 1/float(k)
        else:
            alpha = alpha_choice
        
        # evaluate the gradient, store current (unflattened) weights and cost function value
        cost_eval,grad_eval = gradient(w)
        weight_history.append(unflatten(w))
        cost_history.append(cost_eval)

        # take gradient descent step
        w = w - alpha*grad_eval
            
    # collect final weights
    weight_history.append(unflatten(w))
    # compute final cost function value via g itself (since we aren't computing 
    # the gradient at the final step we don't get the final cost function value 
    # via the Automatic Differentiatoor) 
    cost_history.append(g_flat(w))  
    return weight_history,cost_history



class static_visualizer:
    '''
    Illustrate a run of your preferred optimization algorithm on a one or two-input function.  Run
    the algorithm first, and input the resulting weight history into this wrapper.
    ''' 
        
    ##### draw picture of function and run for two-input function ####       
    def compare_runs_contour_plots(self,g,weight_histories,**kwargs):
        ##### construct figure with panels #####
        # construct figure
        fig = plt.figure(figsize = (10,4.5))
        self.edgecolor = 'k'
         
        # create figure with single plot for contour
        gs = gridspec.GridSpec(1, 2) 
        ax1 = plt.subplot(gs[0],aspect='equal'); 
        ax2 = plt.subplot(gs[1],aspect='equal'); 

        # remove whitespace from figure
        fig.subplots_adjust(left=0, right=1, bottom=0, top=1) # remove whitespace
        fig.subplots_adjust(wspace=0.01,hspace=0.01)
        
        ### make contour right plot - as well as horizontal and vertical axes ###
        self.contour_plot_setup(g,ax1,**kwargs)  # draw contour plot
        w_hist = weight_histories[0]
        self.draw_weight_path(ax1,w_hist)        # draw path on contour plot
        
        self.contour_plot_setup(g,ax2,**kwargs)  # draw contour plot
        w_hist = weight_histories[1]
        self.draw_weight_path(ax2,w_hist)        # draw path on contour plot
        
        # plot
        plt.show()   
        
        
    ########################################################################################
    #### utility functions - for setting up / making contour plots, 3d surface plots, etc., ####
    # show contour plot of input function
    def contour_plot_setup(self,g,ax,**kwargs):
        xmin = -3.1
        xmax = 3.1
        ymin = -3.1
        ymax = 3.1
        if 'xmin' in kwargs:            
            xmin = kwargs['xmin']
        if 'xmax' in kwargs:
            xmax = kwargs['xmax']
        if 'ymin' in kwargs:            
            ymin = kwargs['ymin']
        if 'ymax' in kwargs:
            ymax = kwargs['ymax']      
        num_contours = 20
        if 'num_contours' in kwargs:
            num_contours = kwargs['num_contours']   
            
        # choose viewing range using weight history?
        if 'view_by_weights' in kwargs:
            view_by_weights = True
            weight_history = kwargs['weight_history']
            if view_by_weights == True:
                xmin = min([v[0] for v in weight_history])[0]
                xmax = max([v[0] for v in weight_history])[0]
                xgap = (xmax - xmin)*0.25
                xmin -= xgap
                xmax += xgap

                ymin = min([v[1] for v in weight_history])[0]
                ymax = max([v[1] for v in weight_history])[0]
                ygap = (ymax - ymin)*0.25
                ymin -= ygap
                ymax += ygap
 
        ### plot function as contours ###
        self.draw_contour_plot(g,ax,num_contours,xmin,xmax,ymin,ymax)
        
        ### cleanup panel ###
        ax.set_xlabel('$w_0$',fontsize = 14)
        ax.set_ylabel('$w_1$',fontsize = 14,labelpad = 15,rotation = 0)
        ax.axhline(y=0, color='k',zorder = 0,linewidth = 0.5)
        ax.axvline(x=0, color='k',zorder = 0,linewidth = 0.5)
        # ax.set_xticks(np.arange(round(xmin),round(xmax)+1))
        # ax.set_yticks(np.arange(round(ymin),round(ymax)+1))
        
        # set viewing limits
        ax.set_xlim(xmin,xmax)
        ax.set_ylim(ymin,ymax)

    ### function for creating contour plot
    def draw_contour_plot(self,g,ax,num_contours,xmin,xmax,ymin,ymax):
        #### define input space for function and evaluate ####
        w1 = np.linspace(xmin,xmax,400)
        w2 = np.linspace(ymin,ymax,400)
        w1_vals, w2_vals = np.meshgrid(w1,w2)
        w1_vals.shape = (len(w1)**2,1)
        w2_vals.shape = (len(w2)**2,1)
        h = np.concatenate((w1_vals,w2_vals),axis=1)
        func_vals = np.asarray([ g(np.reshape(s,(2,1))) for s in h])

        w1_vals.shape = (len(w1),len(w1))
        w2_vals.shape = (len(w2),len(w2))
        func_vals.shape = (len(w1),len(w2)) 
        
        ### make contour right plot - as well as horizontal and vertical axes ###
        # set level ridges
        levelmin = min(func_vals.flatten())
        levelmax = max(func_vals.flatten())
        cutoff = 1
        cutoff = (levelmax - levelmin)*cutoff
        numper = 4
        levels1 = np.linspace(cutoff,levelmax,numper)
        num_contours -= numper

        # produce generic contours
        levels2 = np.linspace(levelmin,cutoff,min(num_contours,numper))
        levels = np.unique(np.append(levels1,levels2))
        num_contours -= numper
        while num_contours > 0:
            cutoff = levels[1]
            levels2 = np.linspace(levelmin,cutoff,min(num_contours,numper))
            levels = np.unique(np.append(levels2,levels))
            num_contours -= numper
   
        # plot the contours
        ax.contour(w1_vals, w2_vals, func_vals,levels = levels[1:],colors = 'k')
        ax.contourf(w1_vals, w2_vals, func_vals,levels = levels,cmap = 'Blues')

        ###### clean up plot ######
        ax.set_xlabel('$w_0$',fontsize = 12)
        ax.set_ylabel('$w_1$',fontsize = 12,rotation = 0)
        ax.axhline(y=0, color='k',zorder = 0,linewidth = 0.5)
        ax.axvline(x=0, color='k',zorder = 0,linewidth = 0.5)
        
        
    ### makes color spectrum for plotted run points - from green (start) to red (stop)
    def make_colorspec(self,w_hist):
        # make color range for path
        s = np.linspace(0,1,len(w_hist[:round(len(w_hist)/2)]))
        s.shape = (len(s),1)
        t = np.ones(len(w_hist[round(len(w_hist)/2):]))
        t.shape = (len(t),1)
        s = np.vstack((s,t))
        colorspec = []
        colorspec = np.concatenate((s,np.flipud(s)),1)
        colorspec = np.concatenate((colorspec,np.zeros((len(s),1))),1)
        return colorspec


    ### function for drawing weight history path
    def draw_grads(self,ax,directions,**kwargs):
        # make colors for plot
        colorspec = self.make_colorspec(directions)

        arrows = True
        if 'arrows' in kwargs:
            arrows = kwargs['arrows']
            
        # plot axes
        ax.axhline(y=0, color='k',zorder = 0,linewidth = 0.5)
        ax.axvline(x=0, color='k',zorder = 0,linewidth = 0.5)
        
        ### plot function decrease plot in right panel
        for j in range(len(directions)):  
            # get current direction
            direction = directions[j]
            
            # draw arrows connecting pairwise points
            head_length = 0.1
            head_width = 0.1
            ax.arrow(0,0,direction[0],direction[1], head_width=head_width, head_length=head_length, fc='k', ec='k',linewidth=1,zorder = 2,length_includes_head=True)
            ax.arrow(0,0,direction[0],direction[1], head_width=0.1, head_length=head_length, fc=colorspec[j], ec=colorspec[j],linewidth=0.25,zorder = 2,length_includes_head=True)

    ### function for drawing weight history path
    def draw_grads_v2(self,ax,directions,**kwargs):
        arrows = True
        if 'arrows' in kwargs:
            arrows = kwargs['arrows']
            
        # plot axes
        ax.axhline(y=0, color='k',zorder = 0,linewidth = 0.5)
        ax.axvline(x=0, color='k',zorder = 0,linewidth = 0.5)
        
        ### plot function decrease plot in right panel
        head_length = 0.1
        head_width = 0.1
        alpha = 0.1
        for j in range(len(directions)-1):  
            # get current direction
            direction = directions[j]
            
            # draw arrows connecting pairwise points
            ax.arrow(0,0,direction[0],direction[1], head_width=head_width, head_length=head_length, fc='k', ec='k',linewidth=3.5,zorder = 2,length_includes_head=True,alpha = alpha)
            ax.arrow(0,0,direction[0],direction[1], head_width=0.1, head_length=head_length, fc=self.colorspec[j], ec=self.colorspec[j],linewidth=3,zorder = 2,length_includes_head=True,alpha = alpha)
            
        # plot most recent direction
        direction = directions[-1]
        num_dirs = len(directions)
  
        # draw arrows connecting pairwise points
        ax.arrow(0,0,direction[0],direction[1], head_width=head_width, head_length=head_length, fc='k', ec='k',linewidth=4,zorder = 2,length_includes_head=True)
        ax.arrow(0,0,direction[0],direction[1], head_width=0.1, head_length=head_length, fc=self.colorspec[num_dirs], ec=self.colorspec[num_dirs],linewidth=3,zorder = 2,length_includes_head=True)            
            
    ### function for drawing weight history path
    def draw_weight_path(self,ax,w_hist,**kwargs):
        # make colors for plot
        colorspec = self.make_colorspec(w_hist)
        
        arrows = True
        if 'arrows' in kwargs:
            arrows = kwargs['arrows']

        ### plot function decrease plot in right panel
        for j in range(len(w_hist)):  
            w_val = w_hist[j]

            # plot each weight set as a point
            ax.scatter(w_val[0],w_val[1],s = 80,color = colorspec[j],edgecolor = self.edgecolor,linewidth = 2*math.sqrt((1/(float(j) + 1))),zorder = 3)

            # plot connector between points for visualization purposes
            if j > 0:
                pt1 = w_hist[j-1]
                pt2 = w_hist[j]
                
                # produce scalar for arrow head length
                pt_length = np.linalg.norm(pt1 - pt2)
                head_length = 0.1
                alpha = (head_length - 0.35)/pt_length + 1
                
                # if points are different draw error
                if np.linalg.norm(pt1 - pt2) > head_length and arrows == True:
                    if np.ndim(pt1) > 1:
                        pt1 = pt1.flatten()
                        pt2 = pt2.flatten()
                        
                        
                    # draw color connectors for visualization
                    w_old = pt1
                    w_new = pt2
                    ax.plot([w_old[0],w_new[0]],[w_old[1],w_new[1]],color = colorspec[j],linewidth = 2,alpha = 1,zorder = 2)      # plot approx
                    ax.plot([w_old[0],w_new[0]],[w_old[1],w_new[1]],color = 'k',linewidth = 3,alpha = 1,zorder = 1)      # plot approx
                
                
                    # draw arrows connecting pairwise points
                    #ax.arrow(pt1[0],pt1[1],(pt2[0] - pt1[0])*alpha,(pt2[1] - pt1[1])*alpha, head_width=0.1, head_length=head_length, fc='k', ec='k',linewidth=4,zorder = 2,length_includes_head=True)
                    #ax.arrow(pt1[0],pt1[1],(pt2[0] - pt1[0])*alpha,(pt2[1] - pt1[1])*alpha, head_width=0.1, head_length=head_length, fc='w', ec='w',linewidth=0.25,zorder = 2,length_includes_head=True)
        
    ### draw surface plot
    def draw_surface(self,g,ax,**kwargs):
        xmin = -3.1
        xmax = 3.1
        ymin = -3.1
        ymax = 3.1
        if 'xmin' in kwargs:            
            xmin = kwargs['xmin']
        if 'xmax' in kwargs:
            xmax = kwargs['xmax']
        if 'ymin' in kwargs:            
            ymin = kwargs['ymin']
        if 'ymax' in kwargs:
            ymax = kwargs['ymax']   
            
        #### define input space for function and evaluate ####
        w1 = np.linspace(xmin,xmax,200)
        w2 = np.linspace(ymin,ymax,200)
        w1_vals, w2_vals = np.meshgrid(w1,w2)
        w1_vals.shape = (len(w1)**2,1)
        w2_vals.shape = (len(w2)**2,1)
        h = np.concatenate((w1_vals,w2_vals),axis=1)
        func_vals = np.asarray([g(np.reshape(s,(2,1))) for s in h])

        ### plot function as surface ### 
        w1_vals.shape = (len(w1),len(w2))
        w2_vals.shape = (len(w1),len(w2))
        func_vals.shape = (len(w1),len(w2))
        ax.plot_surface(w1_vals, w2_vals, func_vals, alpha = 0.1,color = 'w',rstride=25, cstride=25,linewidth=1,edgecolor = 'k',zorder = 2)

        # plot z=0 plane 
        ax.plot_surface(w1_vals, w2_vals, func_vals*0, alpha = 0.1,color = 'w',zorder = 1,rstride=25, cstride=25,linewidth=0.3,edgecolor = 'k') 
                
        # clean up axis
        ax.xaxis.pane.fill = False
        ax.yaxis.pane.fill = False
        ax.zaxis.pane.fill = False

        ax.xaxis.pane.set_edgecolor('white')
        ax.yaxis.pane.set_edgecolor('white')
        ax.zaxis.pane.set_edgecolor('white')

        ax.xaxis._axinfo["grid"]['color'] =  (1,1,1,0)
        ax.yaxis._axinfo["grid"]['color'] =  (1,1,1,0)
        ax.zaxis._axinfo["grid"]['color'] =  (1,1,1,0)
        
        ax.set_xlabel('$w_0$',fontsize = 14)
        ax.set_ylabel('$w_1$',fontsize = 14,rotation = 0)
        ax.set_title('$g(w_0,w_1)$',fontsize = 14)
        

    ### plot points and connectors in input space in 3d plot        
    def show_inputspace_path(self,w_hist,ax):
        # make colors for plot
        colorspec = self.make_colorspec(w_hist)
        
        for k in range(len(w_hist)):
            pt1 = w_hist[k]
            ax.scatter(pt1[0],pt1[1],0,s = 60,color = colorspec[k],edgecolor = 'k',linewidth = 0.5*math.sqrt((1/(float(k) + 1))),zorder = 3)
            if k < len(w_hist)-1:
                pt2 = w_hist[k+1]
                if np.linalg.norm(pt1 - pt2) > 10**(-3):
                    # draw arrow in left plot
                    a = Arrow3D([pt1[0],pt2[0]], [pt1[1],pt2[1]], [0, 0], mutation_scale=10, lw=2, arrowstyle="-|>", color="k")
                    ax.add_artist(a)
        
#### custom 3d arrow and annotator functions ###    
# nice arrow maker from https://stackoverflow.com/questions/11140163/python-matplotlib-plotting-a-3d-cube-a-sphere-and-a-vector
class Arrow3D(FancyArrowPatch):

    def __init__(self, xs, ys, zs, *args, **kwargs):
        FancyArrowPatch.__init__(self, (0, 0), (0, 0), *args, **kwargs)
        self._verts3d = xs, ys, zs

    def draw(self, renderer):
        xs3d, ys3d, zs3d = self._verts3d
        xs, ys, zs = proj3d.proj_transform(xs3d, ys3d, zs3d, renderer.M)
        self.set_positions((xs[0], ys[0]), (xs[1], ys[1]))
        FancyArrowPatch.draw(self, renderer)



class animation_visualizer:
    '''
    Animate runs of gradient descent and Newton's method, showing the correspnoding Taylor Series approximations as you go along.
    Run the algorithm first, and input the resulting weight history into this wrapper.
    '''           
            
    ###### animate the method ######
    def newtons_method(self,g,w_hist,savepath,**kwargs):
        # compute gradient and hessian of input
        grad = compute_grad(g)              # gradient of input function
        hess = compute_hess(g)           # hessian of input function
        
        # set viewing range
        wmax = 3
        if 'wmax' in kwargs:
            wmax = kwargs['wmax']
        wmin = -wmax
        if 'wmin' in kwargs:
            wmin = kwargs['wmin']
            
        # initialize figure
        fig = plt.figure(figsize = (9,4))
        artist = fig

        # create subplot with 3 panels, plot input function in center plot
        gs = gridspec.GridSpec(1, 3, width_ratios=[1,4,1]) 

        ax1 = plt.subplot(gs[0]); ax1.axis('off')
        ax3 = plt.subplot(gs[2]); ax3.axis('off')
        ax = plt.subplot(gs[1]); 
        
        # generate function for plotting on each slide
        w_plot = np.linspace(wmin,wmax,1000)
        g_plot = g(w_plot)
        g_range = max(g_plot) - min(g_plot)
        ggap = g_range*0.1
        w_vals = np.linspace(-2.5,2.5,50)
        width = 1
        
        # make color spectrum for points
        colorspec = self.make_colorspec(w_hist)
        
        # animation sub-function
        print ('starting animation rendering...')
        num_frames = 2*len(w_hist)+2
        def animate(t):
            ax.cla()
            k = math.floor((t+1)/float(2))
            
            # print rendering update
            if np.mod(k+1,25) == 0:
                print ('rendering animation frame ' + str(k+1) + ' of ' + str(num_frames))
            if t == num_frames - 1:
                print ('animation rendering complete!')
                time.sleep(1.5)
                clear_output()

            # plot function
            ax.plot(w_plot,g_plot,color = 'k',zorder = 1)                           # plot function
            
            # plot initial point and evaluation
            if k == 0:
                w_val = w_hist[0]
                g_val = g(w_val)
                ax.scatter(w_val,g_val,s = 100,color = colorspec[k],edgecolor = 'k',linewidth = 0.7, marker = 'X',zorder = 2)            # plot point of tangency
                ax.scatter(w_val,0,s = 100,color = colorspec[k],edgecolor = 'k',linewidth = 0.7, zorder = 2)
                # draw dashed line connecting w axis to point on cost function
                s = np.linspace(0,g_val)
                o = np.ones((len(s)))
                ax.plot(o*w_val,s,'k--',linewidth=1,zorder = 0)
                
            # plot all input/output pairs generated by algorithm thus far
            if k > 0:
                # plot all points up to this point
                for j in range(min(k-1,len(w_hist))):  
                    w_val = w_hist[j]
                    g_val = g(w_val)
                    ax.scatter(w_val,g_val,s = 90,color = colorspec[j],edgecolor = 'k',marker = 'X',linewidth = 0.7,zorder = 3)            # plot point of tangency
                    ax.scatter(w_val,0,s = 90,facecolor = colorspec[j],edgecolor = 'k',linewidth = 0.7, zorder = 2)
                          
            # plot surrogate function and travel-to point
            if k > 0 and k < len(w_hist) + 1:          
                # grab historical weight, compute function and derivative evaluations    
                w_eval = w_hist[k-1]
                if type(w_eval) != float:
                    w_eval = float(w_eval)

                # plug in value into func and derivative
                g_eval = g(w_eval)
                g_grad_eval = grad(w_eval)
                g_hess_eval = hess(w_eval)

                # determine width of plotting area for second order approximator
                width = 0.5
                if g_hess_eval < 0:
                    width = - width

                # setup quadratic formula params
                a = 0.5*g_hess_eval
                b = g_grad_eval - 2*0.5*g_hess_eval*w_eval
                c = 0.5*g_hess_eval*w_eval**2 - g_grad_eval*w_eval - width

                # solve for zero points
                w1 = (-b + math.sqrt(b**2 - 4*a*c))/float(2*a + 0.00001)
                w2 = (-b - math.sqrt(b**2 - 4*a*c))/float(2*a + 0.00001)

                # compute second order approximation
                wrange = np.linspace(w1,w2, 100)
                h = g_eval + g_grad_eval*(wrange - w_eval) + 0.5*g_hess_eval*(wrange - w_eval)**2 

                # plot tangent curve
                ax.plot(wrange,h,color = colorspec[k-1],linewidth = 2,zorder = 2)      # plot approx

                # plot tangent point
                ax.scatter(w_eval,g_eval,s = 100,color = 'm',edgecolor = 'k', marker = 'X',linewidth = 0.7,zorder = 3)            # plot point of tangency
            
                # plot next point learned from surrogate
                if np.mod(t,2) == 0:
                    # create next point information
                    w_zero = w_eval - g_grad_eval/(g_hess_eval + 10**-5)
                    g_zero = g(w_zero)
                    h_zero = g_eval + g_grad_eval*(w_zero - w_eval) + 0.5*g_hess_eval*(w_zero - w_eval)**2

                    # draw dashed line connecting the three
                    vals = [0,h_zero,g_zero]
                    vals = np.sort(vals)

                    s = np.linspace(vals[0],vals[2])
                    o = np.ones((len(s)))
                    ax.plot(o*w_zero,s,'k--',linewidth=1)

                    # draw intersection at zero and associated point on cost function you hop back too
                    ax.scatter(w_zero,h_zero,s = 100,color = 'b',linewidth=0.7, marker = 'X',edgecolor = 'k',zorder = 3)
                    ax.scatter(w_zero,0,s = 100,color = 'm',edgecolor = 'k',linewidth = 0.7, zorder = 3)
                    ax.scatter(w_zero,g_zero,s = 100,color = 'm',edgecolor = 'k',linewidth = 0.7, marker = 'X',zorder = 3)            # plot point of tangency
            
            # fix viewing limits on panel
            ax.set_xlim([wmin,wmax])
            ax.set_ylim([min(-0.3,min(g_plot) - ggap),max(max(g_plot) + ggap,0.3)])
            
            # add horizontal axis
            ax.axhline(y=0, color='k',zorder = 0,linewidth = 0.5)
            
            # label axes
            ax.set_xlabel(r'$w$',fontsize = 14)
            ax.set_ylabel(r'$g(w)$',fontsize = 14,rotation = 0,labelpad = 25)
            
            # set tickmarks
            ax.set_xticks(np.arange(round(wmin), round(wmax) + 1, 1.0))
            ax.set_yticks(np.arange(round(min(g_plot) - ggap), round(max(g_plot) + ggap) + 1, 1.0))

            return artist,

        anim = animation.FuncAnimation(fig, animate ,frames=num_frames, interval=num_frames, blit=True)

        # produce animation and save
        fps = 50
        if 'fps' in kwargs:
            fps = kwargs['fps']
        anim.save(savepath, fps=fps, extra_args=['-vcodec', 'libx264'])
        clear_output()
            
    ### makes color spectrum for plotted run points - from green (start) to red (stop)
    def make_colorspec(self,w_hist):
        # make color range for path
        s = np.linspace(0,1,len(w_hist[:round(len(w_hist)/2)]))
        s.shape = (len(s),1)
        t = np.ones(len(w_hist[round(len(w_hist)/2):]))
        t.shape = (len(t),1)
        s = np.vstack((s,t))
        colorspec = []
        colorspec = np.concatenate((s,np.flipud(s)),1)
        colorspec = np.concatenate((colorspec,np.zeros((len(s),1))),1)
        return colorspec        
             