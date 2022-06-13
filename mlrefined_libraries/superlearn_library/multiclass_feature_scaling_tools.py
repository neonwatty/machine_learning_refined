# import standard plotting and animation
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from matplotlib import gridspec
from IPython.display import clear_output
from mpl_toolkits.mplot3d import proj3d
from matplotlib.patches import FancyArrowPatch
from matplotlib.text import Annotation
from mpl_toolkits.mplot3d.proj3d import proj_transform

# import autograd functionality
from autograd import grad as compute_grad   
from autograd import hessian as compute_hess
import autograd.numpy as np
import math
import time

class Visualizer:
    '''
    Animate how normalizing the input of a single input supervised cost function re-shapes 
    its contours, equalizing the penalty assigned to violating either the ideal bias or slope
    parameter.
    ''' 

    # load in data, in particular input and normalized input
    def __init__(self,x,x_normalized,y,cost):
        self.x_original = x
        self.x_normalized = x_normalized
        self.y = y
        
        # make cost function choice
        self.cost_func = 0
        if cost == 'multiclass_softmax':
            self.cost_func = self.multiclass_softmax
        if cost == 'multiclass_perceptron':
            self.cost_func = self.multiclass_perceptron
        if cost == 'fusion_rule':
            self.cost_func = self.fusion_rule
                            
    #####   #####
    def animate_transition(self,savepath,num_frames,**kwargs):
        # initialize figure
        fig = plt.figure(figsize = (10,4.5))
        artist = fig

        # create subplot with 3 panels, plot input function in center plot
        gs = gridspec.GridSpec(1, 1) 
        ax = plt.subplot(gs[0]); ax.set_aspect('equal')

        # animation sub-function
        lams = np.linspace(0,1,num_frames)
        print ('starting animation rendering...')
        def animate(k):
            ax.cla()
            lam = lams[k]
            
            # print rendering update            
            if np.mod(k+1,25) == 0:
                print ('rendering animation frame ' + str(k+1) + ' of ' + str(num_frames))
            if k == num_frames - 1:
                print ('animation rendering complete!')
                time.sleep(1.5)
                clear_output()
                
            # re-assign inputs as weighted average of original and normalized input
            self.x = (1 - lam)*self.x_original + lam*self.x_normalized
            
            # plot contour
            self.contour_plot_setup(ax,**kwargs)  # draw contour plot
            ax.set_title(r'$\lambda = ' + str(np.round(lam,2)) + '$',fontsize = 14)

            return artist,

        anim = animation.FuncAnimation(fig, animate ,frames=num_frames, interval=num_frames, blit=True)
        
        # produce animation and save
        fps = 50
        if 'fps' in kwargs:
            fps = kwargs['fps']
        anim.save(savepath, fps=fps, extra_args=['-vcodec', 'libx264'])
        clear_output()
            
    ########################################################################################
    ###### predict and cost functions #####
    ###### basic model ######
    # compute linear combination of input point
    def model(self,x,w):
        # tack a 1 onto the top of each input point all at once
        o = np.ones((1,np.shape(x)[1]))
        x = np.vstack((o,x))

        # compute linear combination and return
        a = np.dot(x.T,w)
        return a
            
    ###### cost functions #####
    # multiclass perceptron
    def multiclass_perceptron(self,w):        
        # pre-compute predictions on all points
        all_evals = self.model(self.x,w)

        # compute maximum across data points
        a =  np.max(all_evals,axis = 1)        

        # compute cost in compact form using numpy broadcasting
        b = all_evals[np.arange(len(self.y)),self.y.astype(int).flatten()-1]
        cost = np.sum(a - b)

        # add regularizer
        cost = cost + self.lam*np.linalg.norm(w[1:,:],'fro')**2

        # return average
        return cost/float(len(self.y))
    
    # multiclass softmax
    def multiclass_softmax(self,w):        
        # pre-compute predictions on all points
        all_evals = self.model(self.x,w)

        # compute softmax across data points
        a = np.log(np.sum(np.exp(all_evals),axis = 1)) 

        # compute cost in compact form using numpy broadcasting
        b = all_evals[np.arange(len(self.y)),self.y.astype(int).flatten()-1]
        cost = np.sum(a - b)

        # add regularizer
        cost = cost + self.lam*np.linalg.norm(w[1:,:],'fro')**2

        # return average
        return cost/float(len(self.y))
    
    # multiclass misclassification cost function - aka the fusion rule
    def fusion_rule(self,w):        
        # pre-compute predictions on all points
        all_evals = self.model(self.x,w)

        # compute predictions of each input point
        y_predict = (np.argmax(all_evals,axis = 1) + 1)[:,np.newaxis]

        # compare predicted label to actual label
        count = np.sum(np.abs(np.sign(self.y - y_predict)))

        # return number of misclassifications
        return count

    ########################################################################################
    #### utility functions - for setting up / making contour plots, 3d surface plots, etc., ####
    # show contour plot of input function
    def contour_plot_setup(self,ax,**kwargs):
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
        self.draw_contour_plot(ax,num_contours,xmin,xmax,ymin,ymax)
        
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
    def draw_contour_plot(self,ax,num_contours,xmin,xmax,ymin,ymax):
        #### define input space for function and evaluate ####
        w1 = np.linspace(xmin,xmax,400)
        w2 = np.linspace(ymin,ymax,400)
        w1_vals, w2_vals = np.meshgrid(w1,w2)
        w1_vals.shape = (len(w1)**2,1)
        w2_vals.shape = (len(w2)**2,1)
        h = np.concatenate((w1_vals,w2_vals),axis=1)
        func_vals = np.asarray([ self.cost_func(np.reshape(s,(2,1))) for s in h])

        w1_vals.shape = (len(w1),len(w1))
        w2_vals.shape = (len(w2),len(w2))
        func_vals.shape = (len(w1),len(w2)) 

        ### make contour right plot - as well as horizontal and vertical axes ###
        # set level ridges
        levelmin = min(func_vals.flatten())
        levelmax = max(func_vals.flatten())
        cut = 0.4
        cutoff = (levelmax - levelmin)
        levels = [levelmin + cutoff*cut**(num_contours - i) for i in range(0,num_contours+1)]
        levels = [levelmin] + levels
        levels = np.asarray(levels)
   
        a = ax.contour(w1_vals, w2_vals, func_vals,levels = levels,colors = 'k')
        b = ax.contourf(w1_vals, w2_vals, func_vals,levels = levels,cmap = 'Blues')
      
    # a small Python function for plotting the distributions of input features
    def feature_distributions(self,x):
        # create figure 
        fig = plt.figure(figsize = (10,4))

        # create subplots
        N = x.shape[0]
        gs = 0
        if N <= 5:
            gs = gridspec.GridSpec(1,N)
        else:
            gs = gridspec.GridSpec(2,5)


        # remove whitespace from figure
        fig.subplots_adjust(left=0, right=1, bottom=0, top=1) # remove whitespace
        fig.subplots_adjust(wspace=0.01,hspace=0.01)

        # loop over input and plot each individual input dimension value
        all_bins = []
        for n in range(N):
            hist, bins = np.histogram(x[n,:], bins=30)
            all_bins.append(bins.ravel())
            
        # determine range for all subplots
        maxview = np.max(all_bins)
        minview = np.min(all_bins)
        viewrange = (maxview - minview)*0.1
        maxview += viewrange
        minview -= viewrange
        
        for n in range(N):
            # make subplot
            ax = plt.subplot(gs[n]); 
            hist, bins = np.histogram(x[n,:], bins=30)
            width = 0.7 * (bins[1] - bins[0])
            center = (bins[:-1] + bins[1:]) / 2
            ax.barh(center, hist,width)
            ax.set_title(r'$x_' + str(n+1) + '$',fontsize=14)
            ax.set_ylim([minview,maxview])
        plt.show()