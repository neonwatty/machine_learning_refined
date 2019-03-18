# import standard plotting and animation
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from IPython.display import clear_output
import time
from matplotlib import gridspec
from mpl_toolkits.mplot3d import proj3d
from matplotlib.patches import FancyArrowPatch
from matplotlib.text import Annotation
from mpl_toolkits.mplot3d.proj3d import proj_transform

# import autograd functionality
from autograd import grad as compute_grad   # The only autograd function you may ever need
import autograd.numpy as np
import math
import copy

# random local search function
def random_local_search(func,pt,max_steps,num_samples,steplength):
    # starting point evaluation
    current_eval = func(pt)
    current_pt = pt
    
    # loop over max_its descend until no improvement or max_its reached
    pt_history = [current_pt]
    eval_history = [current_eval]
    for i in range(max_steps):
        # loop over num_samples, randomly sample direction and evaluate, move to best evaluation
        swap = 0
        keeper_pt = current_pt
        
        # check if diminishing steplength rule used
        if steplength == 'diminish':
            steplength_temp = 1/(1 + i)
        else:
            steplength_temp = steplength
        
        for j in range(num_samples):            
            # produce direction
            theta = np.random.rand(1)
            x = steplength_temp*np.cos(2*np.pi*theta)
            y = steplength_temp*np.sin(2*np.pi*theta)
            new_pt = np.asarray([x,y])
            temp_pt = copy.deepcopy(keeper_pt) 
            new_pt += temp_pt
            
            # evaluate new point
            new_eval = func(new_pt)
            if new_eval < current_eval:
                current_pt = new_pt
                current_eval = new_eval
                swap = 1
        
        # if nothing has changed
        if swap == 1:
            pt_history.append(current_pt)
            eval_history.append(current_eval)
    
    # translate to array, reshape appropriately
    pt_history = np.asarray(pt_history)
    pt_history.shape = (np.shape(pt_history)[0],np.shape(pt_history)[1])

    eval_history = np.asarray(eval_history)
    eval_history.shape = (np.shape(eval_history)[0],np.shape(eval_history)[1])

    return pt_history,eval_history

# random local search function
def random_local_search_2d(func,pt,max_steps,num_samples,steplength):
    # starting point evaluation
    current_eval = func(pt)
    current_pt = pt
    
    # loop over max_its descend until no improvement or max_its reached
    pt_history = [current_pt]
    eval_history = [current_eval]
    for i in range(max_steps):
        # loop over num_samples, randomly sample direction and evaluate, move to best evaluation
        swap = 0
        keeper_pt = current_pt
        
        # check if diminishing steplength rule used
        if steplength == 'diminish':
            steplength_temp = 1/(1 + i)
        else:
            steplength_temp = steplength
        
        for j in range(num_samples):            
            # produce direction
            new_pt = steplength*np.sign(2*np.random.rand(1) - 1)
            temp_pt = copy.deepcopy(keeper_pt) 
            new_pt += temp_pt
            
            # evaluate new point
            new_eval = func(new_pt)
            if new_eval < current_eval:
                current_pt = new_pt
                current_eval = new_eval
                swap = 1
        
            # if nothing has changed
            if swap == 1:
                pt_history.append(current_pt)
                eval_history.append(current_eval)
    
    # translate to array, reshape appropriately
    pt_history = np.asarray(pt_history)
    eval_history = np.asarray(eval_history)

    return pt_history,eval_history

##### draw still image of gradient descent on single-input function ####       
def draw_2d(g,steplength,max_steps,w_inits,num_samples,**kwargs):
    wmin = -3.1
    wmax = 3.1
    if 'wmin' in kwargs:            
        wmin = kwargs['wmin']
    if 'wmax' in kwargs:
        wmax = kwargs['wmax'] 
        
    # initialize figure
    fig = plt.figure(figsize = (9,4))
    artist = fig

    # create subplot with 3 panels, plot input function in center plot
    gs = gridspec.GridSpec(1, 3, width_ratios=[1,4,1]) 
    ax1 = plt.subplot(gs[0]); ax1.axis('off')
    ax3 = plt.subplot(gs[2]); ax3.axis('off')
    ax = plt.subplot(gs[1]); 

    # generate function for plotting on each slide
    w_plot = np.linspace(wmin,wmax,500)
    g_plot = [g(s) for s in w_plot]
    g_range = max(g_plot) - min(g_plot)
    ggap = g_range*0.1
    width = 30
       
    #### loop over all initializations, run gradient descent algorithm for each and plot results ###
    for j in range(len(w_inits)):
        # get next initialization
        w_init = w_inits[j]

        # run grad descent for this init
        func = g
        pt_history,eval_history = random_local_search_2d(func,w_init,max_steps,num_samples,steplength)

        # colors for points --> green as the algorithm begins, yellow as it converges, red at final point
        s = np.linspace(0,1,len(pt_history[:round(len(eval_history)/2)]))
        s.shape = (len(s),1)
        t = np.ones(len(eval_history[round(len(eval_history)/2):]))
        t.shape = (len(t),1)
        s = np.vstack((s,t))
        colorspec = []
        colorspec = np.concatenate((s,np.flipud(s)),1)
        colorspec = np.concatenate((colorspec,np.zeros((len(s),1))),1)
        
        # plot function, axes lines
        ax.plot(w_plot,g_plot,color = 'k',zorder = 2)                           # plot function
        ax.axhline(y=0, color='k',zorder = 1,linewidth = 0.25)
        ax.axvline(x=0, color='k',zorder = 1,linewidth = 0.25)
        ax.set_xlabel(r'$w$',fontsize = 13)
        ax.set_ylabel(r'$g(w)$',fontsize = 13,rotation = 0,labelpad = 25)            
            
        ### plot all local search points ###
        for k in range(len(eval_history)):
            # pick out current weight and function value from history, then plot
            w_val = pt_history[k]
            g_val = eval_history[k]

            ax.scatter(w_val,g_val,s = 90,c = colorspec[k],edgecolor = 'k',linewidth = 0.5*((1/(float(k) + 1)))**(0.4),zorder = 3,marker = 'X')            # evaluation on function
            ax.scatter(w_val,0,s = 90,facecolor = colorspec[k],edgecolor = 'k',linewidth = 0.5*((1/(float(k) + 1)))**(0.4), zorder = 3)

            
# animator for random local search
def visualize3d(func,pt_history,eval_history,**kwargs):
    ### input arguments ###        
    wmax = 1
    if 'wmax' in kwargs:
        wmax = kwargs['wmax'] + 0.5
        
    view = [20,-50]
    if 'view' in kwargs:
        view = kwargs['view']
        
    axes = False
    if 'axes' in kwargs:
        axes = kwargs['axes']
       
    plot_final = False
    if 'plot_final' in kwargs:
        plot_final = kwargs['plot_final']
      
    num_contours = 10
    if 'num_contours' in kwargs:
        num_contours = kwargs['num_contours']
        
    pt = [0,0]
    if 'pt' in kwargs:
        pt = kwargs['pt']
    pt = np.asarray(pt)
    pt.shape = (2,1)
     
    max_steps = 10
    if 'max_steps' in kwargs:
        max_steps = kwargs['max_steps']
    num_samples = 10
    if 'num_samples' in kwargs:
        num_samples = kwargs['num_samples'] 
    steplength = 1
    if 'steplength' in kwargs:
        steplength = kwargs['steplength']     
        
    ##### construct figure with panels #####
    # construct figure
    fig = plt.figure(figsize = (9,3))
          
    # remove whitespace from figure
    fig.subplots_adjust(left=0, right=1, bottom=0, top=1) # remove whitespace
        
    # create subplot with 3 panels, plot input function in center plot
    gs = gridspec.GridSpec(1, 2, width_ratios=[1,2]) 
    ax = plt.subplot(gs[0],projection='3d'); 
    ax2 = plt.subplot(gs[1],aspect='equal'); 
    
    #### define input space for function and evaluate ####
    w = np.linspace(-wmax,wmax,200)
    w1_vals, w2_vals = np.meshgrid(w,w)
    w1_vals.shape = (len(w)**2,1)
    w2_vals.shape = (len(w)**2,1)
    h = np.concatenate((w1_vals,w2_vals),axis=1)
    func_vals = np.asarray([func(s) for s in h])
    w1_vals.shape = (len(w),len(w))
    w2_vals.shape = (len(w),len(w))
    func_vals.shape = (len(w),len(w))
    
    # plot function 
    ax.plot_surface(w1_vals, w2_vals, func_vals, alpha = 0.1,color = 'w',rstride=25, cstride=25,linewidth=1,edgecolor = 'k',zorder = 2)

    # plot z=0 plane 
    ax.plot_surface(w1_vals, w2_vals, func_vals*0, alpha = 0.1,color = 'w',zorder = 1,rstride=25, cstride=25,linewidth=0.3,edgecolor = 'k') 
    
    ### make contour right plot - as well as horizontal and vertical axes ###
    ax2.contour(w1_vals, w2_vals, func_vals,num_contours,colors = 'k')
    if axes == True:
        ax2.axhline(linestyle = '--', color = 'k',linewidth = 1)
        ax2.axvline(linestyle = '--', color = 'k',linewidth = 1)
    
    ### plot circle on which point lies, as well as step length circle - used only for simple quadratic
    if plot_final == True:
        # plot contour of quadratic on which final point was plotted
        f = pt_history[-1]
        val = np.linalg.norm(f)
        theta = np.linspace(0,1,400)
        x = val*np.cos(2*np.pi*theta) 
        y = val*np.sin(2*np.pi*theta) 
        ax2.plot(x,y,color = 'r',linestyle = '--',linewidth = 1)

        # plot direction sampling circle centered at final point
        x = steplength*np.cos(2*np.pi*theta) + f[0]
        y = steplength*np.sin(2*np.pi*theta) + f[1]
        ax2.plot(x,y,color = 'b',linewidth = 1)    
    
    # colors for points
    s = np.linspace(0,1,len(eval_history[:round(len(eval_history)/2)]))
    s.shape = (len(s),1)
    t = np.ones(len(eval_history[round(len(eval_history)/2):]))
    t.shape = (len(t),1)
    s = np.vstack((s,t))
    colorspec = []
    colorspec = np.concatenate((s,np.flipud(s)),1)
    colorspec = np.concatenate((colorspec,np.zeros((len(s),1))),1)
    
    #### scatter path points ####
    for k in range(len(eval_history)):
        ax.scatter(pt_history[k][0],pt_history[k][1],0,s = 60,c = colorspec[k],edgecolor = 'k',linewidth = 0.5*math.sqrt((1/(float(k) + 1))),zorder = 3)
        
        ax2.scatter(pt_history[k][0],pt_history[k][1],s = 60,c = colorspec[k],edgecolor = 'k',linewidth = 1.5*math.sqrt((1/(float(k) + 1))),zorder = 3)

    #### connect points with arrows ####
    for i in range(len(eval_history)-1):
        pt1 = pt_history[i]
        pt2 = pt_history[i+1]

        if np.linalg.norm(pt1 - pt2) > 0.5:
            # draw arrow in left plot
            a = Arrow3D([pt1[0],pt2[0]], [pt1[1],pt2[1]], [0, 0], mutation_scale=10, lw=2, arrowstyle="-|>", color="k")
            ax.add_artist(a)

            # draw 2d arrow in right plot
            ax2.arrow(pt1[0],pt1[1],(pt2[0] - pt1[0])*0.78,(pt2[1] - pt1[1])*0.78, head_width=0.1, head_length=0.1, fc='k', ec='k',linewidth=3,zorder = 2,length_includes_head=True)

    ### cleanup panels ###
    ax.set_xlabel('$w_0$',fontsize = 12)
    ax.set_ylabel('$w_1$',fontsize = 12,rotation = 0)
    ax.set_title('$g(w_0,w_1)$',fontsize = 12)
    ax.view_init(view[0],view[1])
    
    ax2.set_xlabel('$w_0$',fontsize = 12)
    ax2.set_ylabel('$w_1$',fontsize = 12,rotation = 0)
    
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
    
    # plot
    plt.show()

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

# great solution for annotating 3d objects - from https://datascience.stackexchange.com/questions/11430/how-to-annotate-labels-in-a-3d-matplotlib-scatter-plot
class Annotation3D(Annotation):
    '''Annotate the point xyz with text s'''

    def __init__(self, s, xyz, *args, **kwargs):
        Annotation.__init__(self,s, xy=(0,0), *args, **kwargs)
        self._verts3d = xyz        

    def draw(self, renderer):
        xs3d, ys3d, zs3d = self._verts3d
        xs, ys, zs = proj_transform(xs3d, ys3d, zs3d, renderer.M)
        self.xy=(xs,ys)
        Annotation.draw(self, renderer)        

def annotate3D(ax, s, *args, **kwargs):
    '''add anotation text s to to Axes3d ax'''

    tag = Annotation3D(s, *args, **kwargs)
    ax.add_artist(tag)