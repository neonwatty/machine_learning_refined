import matplotlib.pyplot as plt
import matplotlib.animation as animation
from IPython.display import clear_output
import time
from matplotlib import gridspec
from mpl_toolkits.mplot3d import proj3d
from matplotlib.patches import FancyArrowPatch
from matplotlib.text import Annotation
from mpl_toolkits.mplot3d.proj3d import proj_transform

from autograd import grad as compute_grad 
import autograd.numpy as np
import math
 
# animate ascent direction given by derivative for single input function over a range of values
def animate_visualize2d(savepath,**kwargs):
    g = kwargs['g']                       # input function
    grad = compute_grad(g)         # gradient of input function
    colors = [[0,1,0.25],[0,0.75,1]]    # set of custom colors used for plotting
         
    num_frames = 300                          # number of slides to create - the input range [-3,3] is divided evenly by this number
    if 'num_frames' in kwargs:
        num_frames = kwargs['num_frames']
         
    plot_descent = False
    if 'plot_descent' in kwargs:
        plot_descent = kwargs['plot_descent']
             
    # initialize figure
    fig = plt.figure(figsize = (16,8))
    artist = fig
 
    # create subplot with 3 panels, plot input function in center plot
    gs = gridspec.GridSpec(1, 3, width_ratios=[1,4, 1]) 
    ax1 = plt.subplot(gs[0]); ax1.axis('off');
    ax3 = plt.subplot(gs[2]); ax3.axis('off');
 
    # plot input function
    ax = plt.subplot(gs[1])
 
    # generate a range of values over which to plot input function, and derivatives
    w_plot = np.linspace(-3,3,200)                  # input range for original function
    g_plot = g(w_plot)
    g_range = max(g_plot) - min(g_plot)             # used for cleaning up final plot
    ggap = g_range*0.2
    w_vals = np.linspace(-3,3,num_frames)       # range of values over which to plot first / second order approximations
                     
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
 
        # grab the next input/output tangency pair, the center of the next approximation(s)
        w_val = w_vals[k]
        g_val = g(w_val)
 
        # plot original function
        ax.plot(w_plot,g_plot,color = 'k',zorder = 1,linewidth=4)                           # plot function
 
        # plot the input/output tangency point
        ax.scatter(w_val,g_val,s = 200,c = 'lime',edgecolor = 'k',linewidth = 2,zorder = 3)            # plot point of tangency
 
        #### plot first order approximation ####
        # plug input into the first derivative
        g_grad_val = grad(w_val)
 
        # determine width to plot the approximation -- so its length == width
        width = 1
        div = float(1 + g_grad_val**2)
        w1 = w_val - math.sqrt(width/div)
        w2 = w_val + math.sqrt(width/div)
 
        # compute first order approximation
        wrange = np.linspace(w1,w2, 100)
        h = g_val + g_grad_val*(wrange - w_val)
 
        # plot the first order approximation
        ax.plot(wrange,h,color = 'lime',alpha = 0.5,linewidth = 6,zorder = 2)      # plot approx
             
        #### plot derivative as vector ####
        func = lambda w: g_val + g_grad_val*w
        name = r'$\frac{\mathrm{d}}{\mathrm{d}w}g(' + r'{:.2f}'.format(w_val) + r')$'    
        if abs(func(1) - func(0)) >=0:
            head_width = 0.08*(func(1) - func(0))
            head_length = 0.2*(func(1) - func(0))
 
            # annotate arrow and annotation
            if func(1)-func(0) >= 0:
                ax.arrow(0, 0, func(1)-func(0),0, head_width=head_width, head_length=head_length, fc='k', ec='k',linewidth=2.5,zorder = 3)
         
                ax.annotate(name, xy=(2, 1), xytext=(func(1 + 0.3)-func(0),0),fontsize=22)
            elif func(1)-func(0) < 0:
                ax.arrow(0, 0, func(1)-func(0),0, head_width=-head_width, head_length=-head_length, fc='k', ec='k',linewidth=2.5,zorder = 3)
                 
                ax.annotate(name, xy=(2, 1), xytext=(func(1+0.3) - 1.3 - func(0),0),fontsize=22)
             
        #### plot negative derivative as vector ####
        if plot_descent == True:
            ax.scatter(0,0,c = 'k',edgecolor = 'w',s = 100, linewidth = 0.5,zorder = 4)
             
            func = lambda w: g_val - g_grad_val*w
            name = r'$-\frac{\mathrm{d}}{\mathrm{d}w}g(' + r'{:.2f}'.format(w_val) + r')$'    
            if abs(func(1) - func(0)) >=0:
                head_width = 0.08*(func(1) - func(0))
                head_length = 0.2*(func(1) - func(0))
 
                # annotate arrow and annotation
                if func(1)-func(0) >= 0:
                    ax.arrow(0, 0, func(1)-func(0),0, head_width=head_width, head_length=head_length, fc='r', ec='r',linewidth=2.5,zorder = 3)
         
                    ax.annotate(name, xy=(2, 1), xytext=(func(1 + 0.3)-0.2-func(0),0),fontsize=22)
                elif func(1)-func(0) < 0:
                    ax.arrow(0, 0, func(1)-func(0),0, head_width=-head_width, head_length=-head_length, fc='r', ec='r',linewidth=2.5,zorder = 3)
                 
                    ax.annotate(name, xy=(2, 1), xytext=(func(1+0.3) - 1.6 - func(0),0),fontsize=22)     
             
        #### clean up panel ####
        # fix viewing limits on panel
        ax.set_xlim([-5,5])
        ax.set_ylim([min(min(g_plot) - ggap,-0.5),max(max(g_plot) + ggap,0.5)])
 
        # label axes
        ax.set_xlabel('$w$',fontsize = 25)
        ax.set_ylabel('$g(w)$',fontsize = 25,rotation = 0,labelpad = 50)
        ax.grid(False)
        ax.yaxis.set_visible(False)
        ax.spines['right'].set_visible(False)
        ax.spines['top'].set_visible(False)
        ax.spines['left'].set_visible(False)
        for tick in ax.xaxis.get_major_ticks():
            tick.label.set_fontsize(18) 
 
        return artist,
         
    anim = animation.FuncAnimation(fig, animate,frames=len(w_vals), interval=len(w_vals), blit=True)
         
    # produce animation and save
    fps = 50
    if 'fps' in kwargs:
        fps = kwargs['fps']
    anim.save(savepath, fps=fps, extra_args=['-vcodec', 'libx264'])
    clear_output()           

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