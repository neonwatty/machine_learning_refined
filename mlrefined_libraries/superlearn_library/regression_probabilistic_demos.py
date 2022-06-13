# import standard plotting and animation
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from mpl_toolkits.mplot3d import Axes3D
from IPython.display import clear_output

# import autograd functionality
from autograd import grad as compute_grad   # The only autograd function you may ever need
import autograd.numpy as np
from autograd import hessian as compute_hess
import math
import time
from matplotlib import gridspec
import copy

class Visualizer:
    '''
    Visualize linear regression in 2 and 3 dimensions.  For single input cases (2 dimensions) the path of gradient descent on the cost function can be animated.
    '''
    #### initialize ####
    def __init__(self,data):
        # grab input
        self.x = data[:,:-1]
        self.y = data[:,-1]
        
    def center_data(self):
        # center data
        self.x = self.x - np.mean(self.x)
        self.y = self.y - np.mean(self.y)
        
    def run_algo(self,algo,**kwargs):
        # Get function and compute gradient
        self.g = self.least_squares
        self.grad = compute_grad(self.g)
        
        # choose algorithm
        self.algo = algo
        if self.algo == 'gradient_descent':
            self.alpha = 10**-3
            if 'alpha' in kwargs:
                self.alpha = kwargs['alpha']
        
        self.max_its = 10
        if 'max_its' in kwargs:
            self.max_its = kwargs['max_its']
            
        self.w_init = np.random.randn(2)
        if 'w_init' in kwargs:
            self.w_init = kwargs['w_init']
            self.w_init = np.asarray([float(s) for s in self.w_init])
            self.w_init.shape = (len(self.w_init),1)
            
        # run algorithm of choice
        if self.algo == 'gradient_descent':
            self.w_hist = []
            self.gradient_descent()
        if self.algo == 'newtons_method':
            self.hess = compute_hess(self.g)           # hessian of input function
            self.beta = 0
            if 'beta' in kwargs:
                self.beta = kwargs['beta']
            self.w_hist = []
            self.newtons_method()            
    
    ######## linear regression functions ########    
    def least_squares(self,w):
        cost = 0
        for p in range(0,len(self.y)):
            x_p = copy.deepcopy(self.x[p,:])
            x_p.shape = (len(x_p),1)
            y_p = self.y[p]
            cost +=(w[0] + np.dot(w[1:].T,x_p) - y_p)**2
        return cost
    
    def predict(self,w,x_new):
        model = w[0] + np.dot(w[1:].T,x_new)
        return model
    
    def compute_errors(self,w):
        errors = []
        for p in range(len(self.y)):
            x_p = copy.deepcopy(self.x[p,:])
            x_p.shape = (len(x_p),1)
            y_p = self.y[p]
            y_predict =  w[0] + np.dot(w[1:].T,x_p)
            error = y_predict - y_p
            errors.append(error)
        errors = np.asarray([s[0] for s in errors])
        return errors
        
    #### run newton's method ####
    def newtons_method(self):
        w = self.w_init
        self.w_hist = []
        self.w_hist.append(w)
        for k in range(self.max_its):
            # plug in value into func and derivative
            grad_eval = self.grad(w)
            hess_eval = self.hess(w)
            
            # reshape for numpy linalg functionality
            hess_eval.shape = (int((np.size(hess_eval))**(0.5)),int((np.size(hess_eval))**(0.5)))

            # solve linear system for weights
            w = w - np.dot(np.linalg.pinv(hess_eval + self.beta*np.eye(np.size(w))),grad_eval)
                                
            # record
            self.w_hist.append(w)
            
    ### plot data, fit, and histogram of errors ###
    def error_hist(self,**kwargs):
        fig = plt.figure(figsize = (8,3))
        
        # create subplot with 3 panels, plot input function in center plot
        gs = gridspec.GridSpec(1, 2, width_ratios=[1,1]) 
        ax1 = plt.subplot(gs[0]); 
        ax2 = plt.subplot(gs[1]);
    
        ### plot data, fit linear regression, plot fit
        # scatter data in left panel
        self.scatter_pts(ax1)
        
        # find best fit         
        w = self.w_hist[1]
        
        # run model with best weight
        xmin = copy.deepcopy(min(self.x))
        xmax = copy.deepcopy(max(self.x))
        xgap = (xmax - xmin)*0.1
        xmin-=xgap
        xmax+=xgap
        x_fit = np.linspace(xmin,xmax,300)
        y_fit = w[0] + w[1]*x_fit
        
        # plot fit 
        ax1.plot(x_fit,y_fit,color = 'r',linewidth = 3) 
        
        ### plot histogram of errors in right plot ###
        # compute actual error of fully trained model
        errors = self.compute_errors(w)
        num_bins = 5
        if 'num_bins' in kwargs:
            num_bins = kwargs['num_bins']
        ax2.hist(errors, normed=True, bins=num_bins,facecolor='blue', alpha=0.5,edgecolor = 'k')
        
        # label axes
        if 'xlabel' in kwargs:
            xlabel = kwargs['xlabel']
            ax1.set_xlabel(xlabel,fontsize = 12)
        if 'ylabel' in kwargs:
            ylabel = kwargs['ylabel']
            ax1.set_ylabel(ylabel,fontsize = 12,rotation = 90)          
        plt.show()

    # scatter points
    def scatter_pts(self,ax):
        if np.shape(self.x)[1] == 1:
            # set plotting limits
            xmax = copy.deepcopy(max(self.x))
            xmin = copy.deepcopy(min(self.x))
            xgap = (xmax - xmin)*0.2
            xmin -= xgap
            xmax += xgap
            
            ymax = max(self.y)
            ymin = min(self.y)
            ygap = (ymax - ymin)*0.2
            ymin -= ygap
            ymax += ygap    

            # initialize points
            ax.scatter(self.x,self.y,color = 'k', edgecolor = 'w',linewidth = 0.9,s = 20)

            # clean up panel
            ax.set_xlim([xmin,xmax])
            ax.set_ylim([ymin,ymax])
            
            # label axes
            ax.set_xlabel(r'$x$', fontsize = 12)
            ax.set_ylabel(r'$y$', rotation = 0,fontsize = 12)
            ax.set_title('data', fontsize = 13)
            
        if np.shape(self.x)[1] == 2:
            # set plotting limits
            xmax1 = copy.deepcopy(max(self.x[:,0]))
            xmin1 = copy.deepcopy(min(self.x[:,0]))
            xgap1 = (xmax1 - xmin1)*0.35
            xmin1 -= xgap1
            xmax1 += xgap1
            
            xmax2 = copy.deepcopy(max(self.x[:,0]))
            xmin2 = copy.deepcopy(min(self.x[:,0]))
            xgap2 = (xmax2 - xmin2)*0.35
            xmin2 -= xgap2
            xmax2 += xgap2
            
            ymax = max(self.y)
            ymin = min(self.y)
            ygap = (ymax - ymin)*0.2
            ymin -= ygap
            ymax += ygap    

            # initialize points
            ax.scatter(self.x[:,0],self.x[:,1],self.y,s = 40,color = 'k', edgecolor = 'w',linewidth = 0.9)

            # clean up panel
            ax.set_xlim([xmin1,xmax1])
            ax.set_ylim([xmin2,xmax2])
            ax.set_zlim([ymin,ymax])
            
            ax.set_xticks(np.arange(round(xmin1) +1, round(xmax1), 1.0))
            ax.set_yticks(np.arange(round(xmin2) +1, round(xmax2), 1.0))

            # label axes
            ax.set_xlabel(r'$x_1$', fontsize = 12,labelpad = 5)
            ax.set_ylabel(r'$x_2$', rotation = 0,fontsize = 12,labelpad = 5)
            ax.set_zlabel(r'$y$', rotation = 0,fontsize = 12,labelpad = -3)

            # clean up panel
            ax.xaxis.pane.fill = False
            ax.yaxis.pane.fill = False
            ax.zaxis.pane.fill = False

            ax.xaxis.pane.set_edgecolor('white')
            ax.yaxis.pane.set_edgecolor('white')
            ax.zaxis.pane.set_edgecolor('white')

            ax.xaxis._axinfo["grid"]['color'] =  (1,1,1,0)
            ax.yaxis._axinfo["grid"]['color'] =  (1,1,1,0)
            ax.zaxis._axinfo["grid"]['color'] =  (1,1,1,0)
