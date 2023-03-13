import math, time, copy

# import standard plotting and animation
from matplotlib import gridspec
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from matplotlib.patches import FancyArrowPatch
from matplotlib.text import Annotation
from mpl_toolkits.mplot3d import Axes3D
from IPython.display import clear_output
from mpl_toolkits.mplot3d import proj3d
from mpl_toolkits.mplot3d.proj3d import proj_transform
from matplotlib.axes._axes import _log as matplotlib_axes_logger
matplotlib_axes_logger.setLevel('ERROR')

# import autograd functionality
import autograd.numpy as np
from autograd import grad as compute_grad   
from autograd import value_and_grad 
from autograd import hessian as compute_hess
from autograd.misc.flatten import flatten_func


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

def choose_cost(x_in,y_in,cost,**kwargs):
    '''
    A list of cost functions for supervised learning.  Use the choose_cost function
    to choose the desired cost with input data (x_in,y_in).  The aim here was to 
    create a library of cost functions while keeping things as simple as possible 
    (i.e., without the use of object oriented programming).  
    '''

    
    # define x and y as globals so all cost functions are aware of them
    global x,y
    x = x_in
    y = y_in
    
    # make any other variables not explicitly input into cost functions globally known
    global lam
    lam = 0
    if 'lam' in kwargs:
        lam = kwargs['lam']
    
    # make cost function choice
    cost_func = 0
    if cost == 'least_squares':
        cost_func = least_squares
    if cost == 'least_absolute_deviations':
        cost_func = least_absolute_deviations
    if cost == 'softmax':
        cost_func = softmax
    if cost == 'relu':
        cost_func = relu
    if cost == 'counter':
        cost_func = counting_cost

    if cost == 'multiclass_perceptron':
        cost_func = multiclass_perceptron
    if cost == 'multiclass_softmax':
        cost_func = multiclass_softmax
    if cost == 'multiclass_counter':
        cost_func = multiclass_counting_cost
        
    return cost_func

###### basic model ######
# compute linear combination of input point
def model(x,w):
    a = w[0] + np.dot(x.T,w[1:])
    return a.T

###### cost functions #####
# an implementation of the least squares cost function for linear regression
def least_squares(w):
    cost = np.sum((model(x,w) - y)**2)
    return cost/float(np.size(y))

# a compact least absolute deviations cost function
def least_absolute_deviations(w):
    cost = np.sum(np.abs(model(x,w) - y))
    return cost/float(np.size(y))

# the convex softmax cost function
def softmax(w):
    cost = np.sum(np.log(1 + np.exp(-y*model(x,w))))
    return cost/float(np.size(y))

# the convex relu cost function
def relu(w):
    cost = np.sum(np.maximum(0,-y*model(x,w)))
    return cost/float(np.size(y))

# the counting cost function
def counting_cost(w):
    cost = np.sum((np.sign(model(x,w)) - y)**2)
    return 0.25*cost 

# multiclass perceptron
def multiclass_perceptron(w):        
    # pre-compute predictions on all points
    all_evals = model(x,w)
    
    # compute maximum across data points
    a = np.max(all_evals,axis = 0)    

    # compute cost in compact form using numpy broadcasting
    b = all_evals[y.astype(int).flatten(),np.arange(np.size(y))]
    cost = np.sum(a - b)

    # return average
    return cost/float(np.size(y))

# multiclass softmax
def multiclass_softmax(w):        
    # pre-compute predictions on all points
    all_evals = model(x,w)
    
    # compute softmax across data points
    a = np.log(np.sum(np.exp(all_evals),axis = 0)) 
    
    # compute cost in compact form using numpy broadcasting
    b = all_evals[y.astype(int).flatten(),np.arange(np.size(y))]
    cost = np.sum(a - b)

    # return average
    return cost/float(np.size(y))

# multiclass misclassification cost function - aka the fusion rule
def multiclass_counting_cost(w):                
    # pre-compute predictions on all points
    all_evals = model(x,w)

    # compute predictions of each input point
    y_predict = (np.argmax(all_evals,axis = 0))[np.newaxis,:]

    # compare predicted label to actual label
    count = np.sum(np.abs(np.sign(y - y_predict)))

    # return number of misclassifications
    return count


class StaticVisualizer:
    '''
    Illustrate a run of your preferred optimization algorithm on a one or two-input function.  Run
    the algorithm first, and input the resulting weight history into this wrapper.
    ''' 

    ##### draw picture of function and run for single-input function ####       
    def single_input_plot(self,g,weight_histories,cost_histories,**kwargs):        
        # adjust viewing range
        wmin = -3.1
        wmax = 3.1
        if 'wmin' in kwargs:            
            wmin = kwargs['wmin']
        if 'wmax' in kwargs:
            wmax = kwargs['wmax']
            
        onerun_perplot = False
        if 'onerun_perplot' in kwargs:
            onerun_perplot = kwargs['onerun_perplot']
            
        ### initialize figure
        fig = plt.figure(figsize = (15,5))
        artist = fig
        
        # remove whitespace from figure
        #fig.subplots_adjust(left=0, right=1, bottom=0, top=1) # remove whitespace
        #fig.subplots_adjust(wspace=0.01,hspace=0.01)

        # create subplot with 2 panels, plot input function in center plot
        gs = gridspec.GridSpec(1, 2, width_ratios=[1,1]) 
        ax1 = plt.subplot(gs[0]); 
        ax2 = plt.subplot(gs[1]); 

        ### plot function in both panels
        w_plot = np.linspace(wmin,wmax,500)
        g_plot = g(w_plot)
        gmin = np.min(g_plot)
        gmax = np.max(g_plot)
        g_range = gmax - gmin
        ggap = g_range*0.1
        gmin -= ggap
        gmax += ggap
        
        # plot function, axes lines
        ax1.plot(w_plot,g_plot,color = 'k',zorder = 2)                           # plot function
        ax1.axhline(y=0, color='k',zorder = 1,linewidth = 0.25)
        ax1.axvline(x=0, color='k',zorder = 1,linewidth = 0.25)
        ax1.set_xlabel(r'$w$',fontsize = 13)
        ax1.set_ylabel(r'$g(w)$',fontsize = 13,rotation = 0,labelpad = 25)            
        ax1.set_xlim(wmin,wmax)
        ax1.set_ylim(gmin,gmax)
        
        ax2.plot(w_plot,g_plot,color = 'k',zorder = 2)                           # plot function
        ax2.axhline(y=0, color='k',zorder = 1,linewidth = 0.25)
        ax2.axvline(x=0, color='k',zorder = 1,linewidth = 0.25)
        ax2.set_xlabel(r'$w$',fontsize = 13)
        ax2.set_ylabel(r'$g(w)$',fontsize = 13,rotation = 0,labelpad = 25)   
        ax2.set_xlim(wmin,wmax)
        ax2.set_ylim(gmin,gmax)        
       
        #### loop over histories and plot each
        for j in range(len(weight_histories)):
            w_hist = weight_histories[j]
            c_hist = cost_histories[j]
            
            # colors for points --> green as the algorithm begins, yellow as it converges, red at final point
            s = np.linspace(0,1,len(w_hist[:round(len(w_hist)/2)]))
            s.shape = (len(s),1)
            t = np.ones(len(w_hist[round(len(w_hist)/2):]))
            t.shape = (len(t),1)
            s = np.vstack((s,t))
            self.colorspec = []
            self.colorspec = np.concatenate((s,np.flipud(s)),1)
            self.colorspec = np.concatenate((self.colorspec,np.zeros((len(s),1))),1)
            
            ### plot all history points
            ax = ax2
            if onerun_perplot == True:
                if j == 0:
                    ax = ax1
                if j == 1:
                    ax = ax2
            for k in range(len(w_hist)):
                # pick out current weight and function value from history, then plot
                w_val = w_hist[k]
                g_val = c_hist[k]
                ax.scatter(w_val,g_val,s = 90,c = self.colorspec[k],edgecolor = 'k',linewidth = 0.5*((1/(float(k) + 1)))**(0.4),zorder = 3,marker = 'X')            # evaluation on function
                ax.scatter(w_val,0,s = 90,facecolor = self.colorspec[k],edgecolor = 'k',linewidth = 0.5*((1/(float(k) + 1)))**(0.4), zorder = 3)

    ##### draw picture of function and run for two-input function ####       
    def two_input_surface_contour_plot(self,g,w_hist,**kwargs):
        ### input arguments ###        
        num_contours = 10
        if 'num_contours' in kwargs:
            num_contours = kwargs['num_contours']
           
        view = [20,20]
        if 'view' in kwargs:
            view = kwargs['view']
            
        ##### construct figure with panels #####
        # construct figure
        fig = plt.figure(figsize = (15,5))

        # create subplot with 3 panels, plot input function in center plot
        # this seems to be the best option for whitespace management when using
        # both a surface and contour plot in the same figure
        gs = gridspec.GridSpec(1, 3, width_ratios=[1,5,10]) 
        ax1 = plt.subplot(gs[1],projection='3d'); 
        ax2 = plt.subplot(gs[2],aspect='equal'); 
        
        # remove whitespace from figure
        fig.subplots_adjust(left=0, right=1, bottom=0, top=1) # remove whitespace
        fig.subplots_adjust(wspace=0.01,hspace=0.01)
        
        # plot 3d surface and path in left panel
        self.draw_surface(g,ax1,**kwargs)
        self.show_inputspace_path(w_hist,ax1)
        ax1.view_init(view[0],view[1])
        
        ### make contour right plot - as well as horizontal and vertical axes ###
        self.contour_plot_setup(g,ax2,**kwargs)  # draw contour plot
        self.draw_weight_path(ax2,w_hist)              # draw path on contour plot

        # plot
        plt.show()
        
    ##### draw picture of function and run for two-input function ####       
    def two_input_contour_plot(self,g,w_hist,**kwargs):
        ##### construct figure with panels #####
        # construct figure
        fig = plt.figure(figsize = (15,5))
        
        # show original contour function as well?
        show_original = True
        if 'show_original' in kwargs:
            show_original = kwargs['show_original']

        # create figure with single plot for contour
        gs = gridspec.GridSpec(1, 2) 
        ax1 = plt.subplot(gs[0],aspect='equal'); 
        ax2 = plt.subplot(gs[1],aspect='equal'); 
        
        if show_original == False:
            gs = gridspec.GridSpec(1, 1) 
            ax2 = plt.subplot(gs[0],aspect='equal'); 

        # remove whitespace from figure
        fig.subplots_adjust(left=0, right=1, bottom=0, top=1) # remove whitespace
        fig.subplots_adjust(wspace=0.01,hspace=0.01)
        
        ### make contour right plot - as well as horizontal and vertical axes ###
        self.contour_plot_setup(g,ax2,**kwargs)  # draw contour plot
        self.draw_weight_path(ax2,w_hist,**kwargs)        # draw path on contour plot
        
        if show_original == True:
            self.contour_plot_setup(g,ax1,**kwargs)  # draw contour plot

        # plot
        plt.show()
        
    ##### draw picture of function and run for two-input function ####       
    def compare_runs_contour_plots(self,g,weight_histories,**kwargs):
        ##### construct figure with panels #####
        # construct figure
        fig = plt.figure(figsize = (15,5))
         
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
        
        
    # compare cost histories from multiple runs
    def plot_cost_histories(self,histories,start,**kwargs):
        # plotting colors
        colors = ['k','magenta','springgreen','blueviolet','chocolate']
        
        # initialize figure
        fig = plt.figure(figsize = (15,5))

        # create subplot with 1 panel
        gs = gridspec.GridSpec(1, 1) 
        ax = plt.subplot(gs[0]); 
        
        # any labels to add?        
        labels = [' ',' ']
        if 'labels' in kwargs:
            labels = kwargs['labels']
            
        # plot points on cost function plot too?
        points = False
        if 'points' in kwargs:
            points = kwargs['points']

        # run through input histories, plotting each beginning at 'start' iteration
        for c in range(len(histories)):
            history = histories[c]
            label = 0
            if c == 0:
                label = labels[0]
            else:
                label = labels[1]
                
            # check if a label exists, if so add it to the plot
            if np.size(label) == 0:
                ax.plot(np.arange(start,len(history),1),history[start:],linewidth = 3*(0.8)**(c),color = colors[c]) 
            else:               
                ax.plot(np.arange(start,len(history),1),history[start:],linewidth = 3*(0.8)**(c),color = colors[c],label = label) 
                
            # check if points should be plotted for visualization purposes
            if points == True:
                ax.scatter(np.arange(start,len(history),1),history[start:],s = 90,color = colors[c],edgecolor = 'w',linewidth = 2,zorder = 3) 


        # clean up panel
        xlabel = 'step $k$'
        if 'xlabel' in kwargs:
            xlabel = kwargs['xlabel']
        ylabel = r'$g\left(\mathbf{w}^k\right)$'
        if 'ylabel' in kwargs:
            ylabel = kwargs['ylabel']
        ax.set_xlabel(xlabel,fontsize = 14)
        ax.set_ylabel(ylabel,fontsize = 14,rotation = 0,labelpad = 25)
        if np.size(label) > 0:
            anchor = (1,1)
            if 'anchor' in kwargs:
                anchor = kwargs['anchor']
            plt.legend(loc='upper right', bbox_to_anchor=anchor)
            #leg = ax.legend(loc='upper left', bbox_to_anchor=(1.02, 1), borderaxespad=0)

        ax.set_xlim([start - 0.5,len(history) - 0.5])
 
        
       # fig.tight_layout()
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
        cut = 0.4
        cutoff = (levelmax - levelmin)
        levels = [levelmin + cutoff*cut**(num_contours - i) for i in range(0,num_contours+1)]
        levels = [levelmin] + levels
        levels = np.asarray(levels)
   
        a = ax.contour(w1_vals, w2_vals, func_vals,levels = levels,colors = 'k')
        b = ax.contourf(w1_vals, w2_vals, func_vals,levels = levels,cmap = 'Blues')
        
        
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
            ax.scatter(w_val[0],w_val[1],s = 80,c = colorspec[j],edgecolor = 'k',linewidth = 2*math.sqrt((1/(float(j) + 1))),zorder = 3)

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
                    
                    ax.arrow(pt1[0],pt1[1],(pt2[0] - pt1[0])*alpha,(pt2[1] - pt1[1])*alpha, head_width=0.1, head_length=head_length, fc='k', ec='k',linewidth=4,zorder = 2,length_includes_head=True)
                    ax.arrow(pt1[0],pt1[1],(pt2[0] - pt1[0])*alpha,(pt2[1] - pt1[1])*alpha, head_width=0.1, head_length=head_length, fc='w', ec='w',linewidth=0.25,zorder = 2,length_includes_head=True)
        
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
            ax.scatter(pt1[0],pt1[1],0,s = 60,c = colorspec[k],edgecolor = 'k',linewidth = 0.5*math.sqrt((1/(float(k) + 1))),zorder = 3)
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
        
        
class RegressionVisualizer(StaticVisualizer):
    '''
    Visualize linear regression in 2 and 3 dimensions.  For single input cases (2 dimensions) the path of gradient descent on the cost function can be animated.
    '''
    #### initialize ####
    def __init__(self,data):
        # grab input
        self.x = data[:-1,:].T
        self.y = data[-1:,:].T
        
    def center_data(self):
        # center data
        self.x = self.x - np.mean(self.x)
        self.y = self.y - np.mean(self.y)
        
    ######## linear regression functions ########    
    def least_squares(self,w):
        cost = 0
        for p in range(0,len(self.y)):
            x_p = copy.deepcopy(self.x[p,:])
            x_p.shape = (len(x_p),1)
            y_p = self.y[p]
            cost +=(w[0] + np.dot(w[1:].T,x_p) - y_p)**2
        return cost/float(np.size(self.y))
    
     ######## 3d animation function ########
    # animate gradient descent or newton's method
    def animate_it_3d(self,savepath,w_hist,**kwargs):         
        self.w_hist = w_hist 
        
        ##### setup figure to plot #####
        # initialize figure
        fig = plt.figure(figsize = (15,5))
        artist = fig
        
        # create subplot with 3 panels, plot input function in center plot
        gs = gridspec.GridSpec(1, 2, width_ratios=[2,1]) 
        ax1 = plt.subplot(gs[0],projection='3d'); 
        ax2 = plt.subplot(gs[1]);

        # produce color scheme
        s = np.linspace(0,1,len(self.w_hist[:round(len(self.w_hist)/2)]))
        s.shape = (len(s),1)
        t = np.ones(len(self.w_hist[round(len(self.w_hist)/2):]))
        t.shape = (len(t),1)
        s = np.vstack((s,t))
        self.colorspec = []
        self.colorspec = np.concatenate((s,np.flipud(s)),1)
        self.colorspec = np.concatenate((self.colorspec,np.zeros((len(s),1))),1)
        
        # seed left panel plotting range
        viewmax = 3
        if 'viewmax' in kwargs:
            viewmax = kwargs['viewmax']
        r = np.linspace(-viewmax,viewmax,200)

        # create grid from plotting range
        x1_vals,x2_vals = np.meshgrid(r,r)
        x1_vals.shape = (len(r)**2,1)
        x2_vals.shape = (len(r)**2,1)
        
        x1_vals.shape = (np.size(r),np.size(r))
        x2_vals.shape = (np.size(r),np.size(r))

        # seed left panel view 
        view = [20,50]
        if 'view' in kwargs:
            view = kwargs['view']
        
        # set zaxis to the left
        self.move_axis_left(ax1)
            
        # start animation
        num_frames = len(self.w_hist)
        print ('starting animation rendering...')
        def animate(k):
            # clear panels
            ax1.cla()
            
            # set axis in left panel
            self.move_axis_left(ax1)
            
            # current color
            color = self.colorspec[k]

            # print rendering update
            if np.mod(k+1,25) == 0:
                print ('rendering animation frame ' + str(k+1) + ' of ' + str(num_frames))
            if k == num_frames - 1:
                print ('animation rendering complete!')
                time.sleep(1.5)
                clear_output()
            
            ###### make left panel - plot data and fit ######
            # initialize fit
            w = self.w_hist[k]
                    
            # reshape and plot the surface, as well as where the zero-plane is        
            y_fit = w[0] + w[1]*x1_vals + w[2]*x2_vals

            # plot cost surface
            ax1.plot_surface(x1_vals,x2_vals,y_fit,alpha = 0.1,color = color,rstride=25, cstride=25,linewidth=0.25,edgecolor = 'k',zorder = 2)  
            
            # scatter data
            self.scatter_pts(ax1)
            #ax1.view_init(view[0],view[1])
            
            # plot connector between points for visualization purposes
            if k == 0:
                w_new = self.w_hist[k]
                g_new = self.least_squares(w_new)[0]
                ax2.scatter(k,g_new,s = 0.1,color = 'w',linewidth = 2.5,alpha = 0,zorder = 1)      # plot approx
                
            if k > 0:
                w_old = self.w_hist[k-1]
                w_new = self.w_hist[k]
                g_old = self.least_squares(w_old)[0]
                g_new = self.least_squares(w_new)[0]
     
                ax2.plot([k-1,k],[g_old,g_new],color = color,linewidth = 2.5,alpha = 1,zorder = 2)      # plot approx
                ax2.plot([k-1,k],[g_old,g_new],color = 'k',linewidth = 3.5,alpha = 1,zorder = 1)      # plot approx
            
            # set viewing limits for second panel
            ax2.axhline(y=0, color='k',zorder = 0,linewidth = 0.5)
            ax2.set_xlabel('iteration',fontsize = 12)
            ax2.set_ylabel(r'$g(\mathbf{w})$',fontsize = 12,rotation = 0,labelpad = 25)
            ax2.set_xlim([-0.5,len(self.w_hist)])
            
            # set axis in left panel
            self.move_axis_left(ax1)
                
            return artist,

        anim = animation.FuncAnimation(fig, animate ,frames=num_frames, interval=num_frames, blit=True)
        
        # produce animation and save
        fps = 50
        if 'fps' in kwargs:
            fps = kwargs['fps']
        anim.save(savepath, fps=fps, extra_args=['-vcodec', 'libx264'])
        clear_output()

    # set axis in left panel
    def move_axis_left(self,ax):
        tmp_planes = ax.zaxis._PLANES 
        ax.zaxis._PLANES = ( tmp_planes[2], tmp_planes[3], 
                             tmp_planes[0], tmp_planes[1], 
                             tmp_planes[4], tmp_planes[5])
        view_1 = (25, -135)
        view_2 = (25, -45)
        init_view = view_2
        ax.view_init(*init_view)
    
    ######## 2d animation function ########
    # animate gradient descent or newton's method
    def animate_it_2d(self,savepath,w_hist,**kwargs):       
        self.w_hist = w_hist
        
        ##### setup figure to plot #####
        # initialize figure
        fig = plt.figure(figsize = (15,5))
        artist = fig
        
        # create subplot with 3 panels, plot input function in center plot
        gs = gridspec.GridSpec(1, 2, width_ratios=[1,1]) 
        ax1 = plt.subplot(gs[0]); 
        ax2 = plt.subplot(gs[1]);

        # produce color scheme
        s = np.linspace(0,1,len(self.w_hist[:round(len(self.w_hist)/2)]))
        s.shape = (len(s),1)
        t = np.ones(len(self.w_hist[round(len(self.w_hist)/2):]))
        t.shape = (len(t),1)
        s = np.vstack((s,t))
        self.colorspec = []
        self.colorspec = np.concatenate((s,np.flipud(s)),1)
        self.colorspec = np.concatenate((self.colorspec,np.zeros((len(s),1))),1)
        
        # seed left panel plotting range
        xmin = np.min(copy.deepcopy(self.x))
        xmax = np.max(copy.deepcopy(self.x))
        xgap = (xmax - xmin)*0.1
        xmin-=xgap
        xmax+=xgap
        x_fit = np.linspace(xmin,xmax,300)
        
        # seed right panel contour plot
        viewmax = 3
        if 'viewmax' in kwargs:
            viewmax = kwargs['viewmax']
        view = [20,100]
        if 'view' in kwargs:
            view = kwargs['view']
        num_contours = 15
        if 'num_contours' in kwargs:
            num_contours = kwargs['num_contours']        
        self.contour_plot(ax2,viewmax,num_contours)
        
        # start animation
        num_frames = len(self.w_hist)
        print ('starting animation rendering...')
        def animate(k):
            # clear panels
            ax1.cla()
            
            # current color
            color = self.colorspec[k]

            # print rendering update
            if np.mod(k+1,25) == 0:
                print ('rendering animation frame ' + str(k+1) + ' of ' + str(num_frames))
            if k == num_frames - 1:
                print ('animation rendering complete!')
                time.sleep(1.5)
                clear_output()
            
            ###### make left panel - plot data and fit ######
            # initialize fit
            w = self.w_hist[k]
            y_fit = w[0] + x_fit*w[1]
            
            # scatter data
            self.scatter_pts(ax1)
            
            # plot fit to data
            ax1.plot(x_fit,y_fit,color = color,linewidth = 3) 

            ###### make right panel - plot contour and steps ######
            if k == 0:
                ax2.scatter(w[0],w[1],s = 90,facecolor = color,edgecolor = 'k',linewidth = 0.5, zorder = 3)
            if k > 0 and k < num_frames:
                self.plot_pts_on_contour(ax2,k,color)
            if k == num_frames -1:
                ax2.scatter(w[0],w[1],s = 90,facecolor = color,edgecolor = 'k',linewidth = 0.5, zorder = 3)
               
            return artist,

        anim = animation.FuncAnimation(fig, animate ,frames=num_frames, interval=num_frames, blit=True)
        
        # produce animation and save
        fps = 50
        if 'fps' in kwargs:
            fps = kwargs['fps']
        anim.save(savepath, fps=fps, extra_args=['-vcodec', 'libx264'])
        clear_output()

    ### animate only the fit ###
    def animate_it_2d_fit_only(self,savepath,w_hist,**kwargs):       
        self.w_hist = w_hist
        
        ##### setup figure to plot #####
        # initialize figure
        fig = plt.figure(figsize = (4,4))
        artist = fig
        
        # create subplot with 3 panels, plot input function in center plot
        gs = gridspec.GridSpec(1, 1) 
        ax1 = plt.subplot(gs[0]); 

        # produce color scheme
        s = np.linspace(0,1,len(self.w_hist[:round(len(self.w_hist)/2)]))
        s.shape = (len(s),1)
        t = np.ones(len(self.w_hist[round(len(self.w_hist)/2):]))
        t.shape = (len(t),1)
        s = np.vstack((s,t))
        self.colorspec = []
        self.colorspec = np.concatenate((s,np.flipud(s)),1)
        self.colorspec = np.concatenate((self.colorspec,np.zeros((len(s),1))),1)
        
        # seed left panel plotting range
        xmin = np.min(copy.deepcopy(self.x))
        xmax = np.max(copy.deepcopy(self.x))
        xgap = (xmax - xmin)*0.1
        xmin-=xgap
        xmax+=xgap
        x_fit = np.linspace(xmin,xmax,300)
        
        # seed right panel contour plot
        viewmax = 3
        if 'viewmax' in kwargs:
            viewmax = kwargs['viewmax']
        view = [20,100]
        if 'view' in kwargs:
            view = kwargs['view']
        
        # start animation
        num_frames = len(self.w_hist)
        print ('starting animation rendering...')
        def animate(k):
            # clear panels
            ax1.cla()
            
            # current color
            color = self.colorspec[k]

            # print rendering update
            if np.mod(k+1,25) == 0:
                print ('rendering animation frame ' + str(k+1) + ' of ' + str(num_frames))
            if k == num_frames - 1:
                print ('animation rendering complete!')
                time.sleep(1.5)
                clear_output()
            
            ###### make left panel - plot data and fit ######
            # initialize fit
            w = self.w_hist[k]
            y_fit = w[0] + x_fit*w[1]
            
            # scatter data
            self.scatter_pts(ax1)
            
            # plot fit to data
            ax1.plot(x_fit,y_fit,color = color,linewidth = 3) 

            return artist,

        anim = animation.FuncAnimation(fig, animate ,frames=num_frames, interval=num_frames, blit=True)
        
        # produce animation and save
        fps = 50
        if 'fps' in kwargs:
            fps = kwargs['fps']
        anim.save(savepath, fps=fps, extra_args=['-vcodec', 'libx264'])
        clear_output()
         
    ###### plot plotting functions ######
    def plot_data(self):
        # construct figure
        fig, axs = plt.subplots(1, 3, figsize=(15,4))

        # create subplot with 2 panels
        gs = gridspec.GridSpec(1, 3, width_ratios=[1,5,1]) 
        ax1 = plt.subplot(gs[0]); ax1.axis('off') 
        ax2 = plt.subplot(gs[1]); 
        ax3 = plt.subplot(gs[2]); ax3.axis('off')
        
        if np.shape(self.x)[1] == 2:
            ax2 = plt.subplot(gs[1],projection='3d'); 

        # scatter points
        self.scatter_pts(ax2)
        
    def plot_regression_fits(self,final_weights):
        # construct figure
        fig, axs = plt.subplots(1, 3, figsize=(15,5))

        # create subplot with 2 panels
        gs = gridspec.GridSpec(1, 3, width_ratios=[1,2,1]) 
        ax1 = plt.subplot(gs[0]); ax1.axis('off') 
        ax2 = plt.subplot(gs[1]); 
        ax3 = plt.subplot(gs[2]); ax3.axis('off')
 
        # scatter points
        self.scatter_pts(ax2)      
        
        # print regression fits
        for weights in final_weights:
            ax2.plot_fit(ax2,weights)
        
    # plot regression fits
    def plot_fit(self,plotting_weights,**kwargs):
        # construct figure
        fig, axs = plt.subplots(1, 3, figsize=(15,5))

        # create subplot with 2 panels
        gs = gridspec.GridSpec(1, 3, width_ratios=[1,5,1]) 
        ax1 = plt.subplot(gs[0]); ax1.axis('off') 
        ax = plt.subplot(gs[1]); 
        ax3 = plt.subplot(gs[2]); ax3.axis('off')
        
        # set plotting limits
        xmin = np.min(copy.deepcopy(self.x))
        xmax = np.max(copy.deepcopy(self.x))
        xgap = (xmax - xmin)*0.25
        xmin -= xgap
        xmax += xgap

        ymin = np.min(copy.deepcopy(self.y))
        ymax = np.max(copy.deepcopy(self.y))
        ygap = (ymax - ymin)*0.25
        ymin -= ygap
        ymax += ygap    

        # initialize points
        ax.scatter(self.x,self.y,color = 'k', edgecolor = 'w',linewidth = 0.9,s = 40,zorder = 0)

        # clean up panel
        ax.set_xlim([xmin,xmax])
        ax.set_ylim([ymin,ymax])

        # label axes
        ax.set_xlabel(r'$x$', fontsize = 12)
        ax.set_ylabel(r'$y$', rotation = 0,fontsize = 12)
        
        # create fit
        s = np.linspace(xmin,xmax,300)
        colors = ['k','magenta']
        if 'colors' in kwargs:
            colors = kwargs['colors']
 
        transformers = [lambda a: a for i in range(len(plotting_weights))]
        if 'transformers' in kwargs:
            transformers = kwargs['transformers']

        for i in range(len(plotting_weights)):
            weights = plotting_weights[i]
            transformer = transformers[i]
            t = weights[0] + weights[1]*transformer(s).flatten()
            ax.plot(s,t,linewidth = 2,color = colors[i],zorder = 3)
            #c+=1
    
    # scatter points
    def scatter_pts(self,ax):
        if np.shape(self.x)[1] == 1:
            # set plotting limits
            xmin = np.min(copy.deepcopy(self.x))
            xmax = np.max(copy.deepcopy(self.x))
            xgap = (xmax - xmin)*0.2
            xmin -= xgap
            xmax += xgap
            
            ymin = np.min(copy.deepcopy(self.y))
            ymax = np.max(copy.deepcopy(self.y))
            ygap = (ymax - ymin)*0.2
            ymin -= ygap
            ymax += ygap    

            # initialize points
            ax.scatter(self.x,self.y,color = 'k', edgecolor = 'w',linewidth = 0.9,s = 40)

            # clean up panel
            ax.set_xlim([xmin,xmax])
            ax.set_ylim([ymin,ymax])
            
            # label axes
            ax.set_xlabel(r'$x$', fontsize = 16)
            ax.set_ylabel(r'$y$', rotation = 0,fontsize = 16,labelpad = 15)
            
        if np.shape(self.x)[1] == 2:
            # set plotting limits
            xmin1 = np.min(copy.deepcopy(self.x[:,0]))
            xmax1 = np.max(copy.deepcopy(self.x[:,0])) 
            xgap1 = (xmax1 - xmin1)*0.35
            xmin1 -= xgap1
            xmax1 += xgap1
            
            xmin2 = np.min(copy.deepcopy(self.x[:,1]))
            xmax2 = np.max(copy.deepcopy(self.x[:,1])) 
            xgap2 = (xmax2 - xmin2)*0.35
            xmin2 -= xgap2
            xmax2 += xgap2
            
            ymin = np.min(copy.deepcopy(self.y))
            ymax = np.max(copy.deepcopy(self.y))
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
           
    # plot points on contour
    def plot_pts_on_contour(self,ax,j,color):
        # plot connector between points for visualization purposes
        w_old = self.w_hist[j-1]
        w_new = self.w_hist[j]
        g_old = self.least_squares(w_old)
        g_new = self.least_squares(w_new)
     
        ax.plot([w_old[0],w_new[0]],[w_old[1],w_new[1]],color = color,linewidth = 3,alpha = 1,zorder = 2)      # plot approx
        ax.plot([w_old[0],w_new[0]],[w_old[1],w_new[1]],color = 'k',linewidth = 3 + 1,alpha = 1,zorder = 1)      # plot approx
    
    ###### function plotting functions #######
    def plot_ls_cost(self,**kwargs):
        # construct figure
        fig, axs = plt.subplots(1, 2, figsize=(15,5))

        # create subplot with 2 panels
        gs = gridspec.GridSpec(1, 2, width_ratios=[1,1]) 
        ax1 = plt.subplot(gs[0],aspect = 'equal'); 
        ax2 = plt.subplot(gs[1],projection='3d'); 
        
        # pull user-defined args
        viewmax = 3
        if 'viewmax' in kwargs:
            viewmax = kwargs['viewmax']
        view = [20,100]
        if 'view' in kwargs:
            view = kwargs['view']
        num_contours = 15
        if 'num_contours' in kwargs:
            num_contours = kwargs['num_contours']
        
        # make contour plot in left panel
        self.contour_plot(ax1,viewmax,num_contours)
        
        # make contour plot in right panel
        self.surface_plot(ax2,viewmax,view)
        
        plt.show()
        
    ### visualize the surface plot of cost function ###
    def surface_plot(self,ax,wmax,view):
        ##### Produce cost function surface #####
        wmax += wmax*0.1
        r = np.linspace(-wmax,wmax,200)

        # create grid from plotting range
        w1_vals,w2_vals = np.meshgrid(r,r)
        w1_vals.shape = (len(r)**2,1)
        w2_vals.shape = (len(r)**2,1)
        w_ = np.concatenate((w1_vals,w2_vals),axis = 1)
        g_vals = []
        for i in range(len(r)**2):
            g_vals.append(self.least_squares(w_[i,:]))
        g_vals = np.asarray(g_vals)

        # reshape and plot the surface, as well as where the zero-plane is
        w1_vals.shape = (np.size(r),np.size(r))
        w2_vals.shape = (np.size(r),np.size(r))
        g_vals.shape = (np.size(r),np.size(r))
        
        # plot cost surface
        ax.plot_surface(w1_vals,w2_vals,g_vals,alpha = 0.1,color = 'w',rstride=25, cstride=25,linewidth=1,edgecolor = 'k',zorder = 2)  
        
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

        ax.set_xlabel(r'$w_0$',fontsize = 12)
        ax.set_ylabel(r'$w_1$',fontsize = 12,rotation = 0)
        ax.set_title(r'$g\left(w_0,w_1\right)$',fontsize = 13)

        ax.view_init(view[0],view[1])
        
    ### visualize contour plot of cost function ###
    def contour_plot(self,ax,wmax,num_contours):
        #### define input space for function and evaluate ####
        w1 = np.linspace(-wmax,wmax,100)
        w2 = np.linspace(-wmax,wmax,100)
        w1_vals, w2_vals = np.meshgrid(w1,w2)
        w1_vals.shape = (len(w1)**2,1)
        w2_vals.shape = (len(w2)**2,1)
        h = np.concatenate((w1_vals,w2_vals),axis=1)
        func_vals = np.asarray([self.least_squares(s) for s in h])
        w1_vals.shape = (len(w1),len(w1))
        w2_vals.shape = (len(w2),len(w2))
        func_vals.shape = (len(w1),len(w2)) 

        ### make contour right plot - as well as horizontal and vertical axes ###
        # set level ridges
        levelmin = min(func_vals.flatten())
        levelmax = max(func_vals.flatten())
        cutoff = 0.5
        cutoff = (levelmax - levelmin)*cutoff
        numper = 3
        levels1 = np.linspace(cutoff,levelmax,numper)
        num_contours -= numper

        levels2 = np.linspace(levelmin,cutoff,min(num_contours,numper))
        levels = np.unique(np.append(levels1,levels2))
        num_contours -= numper
        while num_contours > 0:
            cutoff = levels[1]
            levels2 = np.linspace(levelmin,cutoff,min(num_contours,numper))
            levels = np.unique(np.append(levels2,levels))
            num_contours -= numper

        a = ax.contour(w1_vals, w2_vals, func_vals,levels = levels,colors = 'k')
        ax.contourf(w1_vals, w2_vals, func_vals,levels = levels,cmap = 'Blues')
                
        # clean up panel
        ax.set_xlabel('$w_0$',fontsize = 12)
        ax.set_ylabel('$w_1$',fontsize = 12,rotation = 0)
        ax.set_title(r'$g\left(w_0,w_1\right)$',fontsize = 13)

        ax.axhline(y=0, color='k',zorder = 0,linewidth = 0.5)
        ax.axvline(x=0, color='k',zorder = 0,linewidth = 0.5)
        ax.set_xlim([-wmax,wmax])
        ax.set_ylim([-wmax,wmax])
        
        
class NormalizeVisualizer:
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
        self.cost_func = cost
        
        # make cost function choice
#         self.cost_func = cost
        if cost == 'least_squares':
            self.cost_func = self.least_squares
        if cost == 'least_absolute_deviations':
            self.cost_func = self.least_absolute_deviations
        if cost == 'softmax':
            self.cost_func = self.softmax
        if cost == 'relu':
            self.cost_func = self.relu

                            
    #####   #####
    def animate_transition(self,savepath,num_frames,**kwargs):
        # initialize figure
        fig = plt.figure(figsize = (15,5))
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
    # an implementation of the least squares cost function for linear regression
    def least_squares(self,w):
        cost = np.sum((self.model(self.x,w) - self.y)**2)
        return cost/float(len(self.y))
    
    # a compact least absolute deviations cost function
    def least_absolute_deviations(self,w):
        cost = np.sum(np.abs(self.model(self.x,w) - self.y))
        return cost/float(len(self.y))

    # the convex softmax cost function
    def softmax(self,w):
        cost = np.sum(np.log(1 + np.exp(-self.y*self.model(self.x,w))))
        return cost/float(np.size(self.y))

    # the convex relu cost function
    def relu(self,w):
        cost = np.sum(np.maximum(0,-self.y*self.model(self.x,w)))
        return cost/float(len(self.y))

    # the counting cost function
    def counting_cost(self,w):
        cost = np.sum((np.sign(self.model(self.x,w)) - self.y)**2)
        return 0.25*cost 
    
    
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
        fig = plt.figure(figsize = (15,5))

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
        
        

class ClassificationVisualizer:
    '''
    Visualize linear classification and fits in 2d (for N=1 dimensional input datasets)
    '''
    #### initialize ####
    def __init__(self,data):
        # grab input
        data = data.T
        self.x = data[:,:-1]
        self.y = data[:,-1]
   
    ###### plot plotting functions ######
    def plot_data(self):
        # construct figure
        fig, axs = plt.subplots(1, 3, figsize=(15,5))

        # create subplot with 2 panels
        gs = gridspec.GridSpec(1, 3, width_ratios=[1,5,1]) 
        ax1 = plt.subplot(gs[0]); ax1.axis('off') 
        ax2 = plt.subplot(gs[1]); 
        ax3 = plt.subplot(gs[2]); ax3.axis('off')

        # scatter points
        self.scatter_pts(ax2)
        
    # plot regression fits
    def plot_fit(self,plotting_weights,**kwargs):
        # construct figure
        fig, axs = plt.subplots(1, 3, figsize=(15,5))

        # create subplot with 2 panels
        gs = gridspec.GridSpec(1, 3, width_ratios=[1,5,1]) 
        ax1 = plt.subplot(gs[0]); ax1.axis('off') 
        ax = plt.subplot(gs[1]); 
        ax3 = plt.subplot(gs[2]); ax3.axis('off')
        
        # set plotting limits
        xmax = copy.deepcopy(max(self.x))
        xmin = copy.deepcopy(min(self.x))
        xgap = (xmax - xmin)*0.25
        xmin -= xgap
        xmax += xgap

        ymax = max(self.y)
        ymin = min(self.y)
        ygap = (ymax - ymin)*0.25
        ymin -= ygap
        ymax += ygap    

        # initialize points
        ax.scatter(self.x,self.y,color = 'k', edgecolor = 'w',linewidth = 0.9,s = 80,zorder = 3)

        # clean up panel
        ax.set_xlim([xmin,xmax])
        ax.set_ylim([ymin,ymax])

        # label axes
        ax.set_xlabel(r'$x$', fontsize = 12)
        ax.set_ylabel(r'$y$', rotation = 0,fontsize = 12)
        
        # create fit
        s = np.linspace(xmin,xmax,300)
        colors = ['k','magenta']
        if 'colors' in kwargs:
            colors = kwargs['colors']
        
        transformers = [lambda a: a for i in range(len(plotting_weights))]
        if 'transformers' in kwargs:
            transformers = kwargs['transformers']

        for i in range(len(plotting_weights)):
            weights = plotting_weights[i]
            transformer = transformers[i]
            
            # plot approximation
            l = weights[0] + weights[1]*transformer(s)
            t = np.tanh(l).flatten()
            ax.plot(s,t,linewidth = 2,color = colors[i],zorder = 2)
        
        # plot counting cost 
        #t = np.sign(l).flatten()
        #ax.plot(s,t,linewidth = 4,color = 'b',zorder = 1)
    
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
            ax.scatter(self.x,self.y,color = 'k', edgecolor = 'w',linewidth = 0.9,s = 80)

            # clean up panel
            ax.set_xlim([xmin,xmax])
            ax.set_ylim([ymin,ymax])
            
            # label axes
            ax.set_xlabel(r'$x$', fontsize = 16)
            ax.set_ylabel(r'$y$', rotation = 0,fontsize = 16,labelpad = 15)
            
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
           
    # plot points on contour
    def plot_pts_on_contour(self,ax,j,color):
        # plot connector between points for visualization purposes
        w_old = self.w_hist[j-1]
        w_new = self.w_hist[j]
        g_old = self.least_squares(w_old)
        g_new = self.least_squares(w_new)
     
        ax.plot([w_old[0],w_new[0]],[w_old[1],w_new[1]],color = color,linewidth = 3,alpha = 1,zorder = 2)      # plot approx
        ax.plot([w_old[0],w_new[0]],[w_old[1],w_new[1]],color = 'k',linewidth = 3 + 1,alpha = 1,zorder = 1)      # plot approx
    
    ###### function plotting functions #######
    def plot_ls_cost(self,**kwargs):
        # construct figure
        fig, axs = plt.subplots(1, 2, figsize=(15,5))

        # create subplot with 2 panels
        gs = gridspec.GridSpec(1, 2, width_ratios=[1,1]) 
        ax1 = plt.subplot(gs[0],aspect = 'equal'); 
        ax2 = plt.subplot(gs[1],projection='3d'); 
        
        # pull user-defined args
        viewmax = 3
        if 'viewmax' in kwargs:
            viewmax = kwargs['viewmax']
        view = [20,100]
        if 'view' in kwargs:
            view = kwargs['view']
        num_contours = 15
        if 'num_contours' in kwargs:
            num_contours = kwargs['num_contours']
        
        # make contour plot in left panel
        self.contour_plot(ax1,viewmax,num_contours)
        
        # make contour plot in right panel
        self.surface_plot(ax2,viewmax,view)
        
        plt.show()
        
    ### visualize the surface plot of cost function ###
    def surface_plot(self,ax,wmax,view):
        ##### Produce cost function surface #####
        wmax += wmax*0.1
        r = np.linspace(-wmax,wmax,200)

        # create grid from plotting range
        w1_vals,w2_vals = np.meshgrid(r,r)
        w1_vals.shape = (len(r)**2,1)
        w2_vals.shape = (len(r)**2,1)
        w_ = np.concatenate((w1_vals,w2_vals),axis = 1)
        g_vals = []
        for i in range(len(r)**2):
            g_vals.append(self.least_squares(w_[i,:]))
        g_vals = np.asarray(g_vals)

        # reshape and plot the surface, as well as where the zero-plane is
        w1_vals.shape = (np.size(r),np.size(r))
        w2_vals.shape = (np.size(r),np.size(r))
        g_vals.shape = (np.size(r),np.size(r))
        
        # plot cost surface
        ax.plot_surface(w1_vals,w2_vals,g_vals,alpha = 0.1,color = 'w',rstride=25, cstride=25,linewidth=1,edgecolor = 'k',zorder = 2)  
        
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

        ax.set_xlabel(r'$w_0$',fontsize = 12)
        ax.set_ylabel(r'$w_1$',fontsize = 12,rotation = 0)
        ax.set_title(r'$g\left(w_0,w_1\right)$',fontsize = 13)

        ax.view_init(view[0],view[1])
        
    ### visualize contour plot of cost function ###
    def contour_plot(self,ax,wmax,num_contours):
        #### define input space for function and evaluate ####
        w1 = np.linspace(-wmax,wmax,100)
        w2 = np.linspace(-wmax,wmax,100)
        w1_vals, w2_vals = np.meshgrid(w1,w2)
        w1_vals.shape = (len(w1)**2,1)
        w2_vals.shape = (len(w2)**2,1)
        h = np.concatenate((w1_vals,w2_vals),axis=1)
        func_vals = np.asarray([self.least_squares(s) for s in h])
        w1_vals.shape = (len(w1),len(w1))
        w2_vals.shape = (len(w2),len(w2))
        func_vals.shape = (len(w1),len(w2)) 

        ### make contour right plot - as well as horizontal and vertical axes ###
        # set level ridges
        levelmin = min(func_vals.flatten())
        levelmax = max(func_vals.flatten())
        cutoff = 0.5
        cutoff = (levelmax - levelmin)*cutoff
        numper = 3
        levels1 = np.linspace(cutoff,levelmax,numper)
        num_contours -= numper

        levels2 = np.linspace(levelmin,cutoff,min(num_contours,numper))
        levels = np.unique(np.append(levels1,levels2))
        num_contours -= numper
        while num_contours > 0:
            cutoff = levels[1]
            levels2 = np.linspace(levelmin,cutoff,min(num_contours,numper))
            levels = np.unique(np.append(levels2,levels))
            num_contours -= numper

        a = ax.contour(w1_vals, w2_vals, func_vals,levels = levels,colors = 'k')
        ax.contourf(w1_vals, w2_vals, func_vals,levels = levels,cmap = 'Blues')
                
        # clean up panel
        ax.set_xlabel('$w_0$',fontsize = 12)
        ax.set_ylabel('$w_1$',fontsize = 12,rotation = 0)
        ax.set_title(r'$g\left(w_0,w_1\right)$',fontsize = 13)

        ax.axhline(y=0, color='k',zorder = 0,linewidth = 0.5)
        ax.axvline(x=0, color='k',zorder = 0,linewidth = 0.5)
        ax.set_xlim([-wmax,wmax])
        ax.set_ylim([-wmax,wmax])