# import standard plotting and animation
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from mpl_toolkits.mplot3d import Axes3D
from matplotlib import gridspec

# import autograd functionality
from autograd import grad as compute_grad   # The only autograd function you may ever need
import autograd.numpy as np
from autograd import jacobian
from autograd import hessian
import math

class visualizer:
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
        ax2.scatter(w1_val,w2_val,g_val,s = 50,c = 'lime',edgecolor = 'k',linewidth = 0.7,zorder = 3)            # plot point of tangency

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