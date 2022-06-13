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

def compare_2d3d(func1,func2,**kwargs):
    view = [20,-50]
    if 'view' in kwargs:
        view = kwargs['view']
        
    # construct figure
    fig = plt.figure(figsize = (12,4))
          
    # remove whitespace from figure
    fig.subplots_adjust(left=0, right=1, bottom=0, top=1) # remove whitespace
    fig.subplots_adjust(wspace=0.01,hspace=0.01)
        
    # create subplot with 3 panels, plot input function in center plot
    gs = gridspec.GridSpec(1, 3, width_ratios=[1,2,4]) 
  
    ### draw 2d version ###
    ax1 = plt.subplot(gs[1]); 
    grad = compute_grad(func1)
    
    # generate a range of values over which to plot input function, and derivatives
    w_plot = np.linspace(-3,3,200)                  # input range for original function
    g_plot = func1(w_plot)
    g_range = max(g_plot) - min(g_plot)             # used for cleaning up final plot
    ggap = g_range*0.2
    w_vals = np.linspace(-2.5,2.5,200)      

    # grab the next input/output tangency pair, the center of the next approximation(s)
    w_val = float(0)
    g_val = func1(w_val)

    # plot original function
    ax1.plot(w_plot,g_plot,color = 'k',zorder = 1,linewidth=2)                       

    # plot axis
    ax1.plot(w_plot,g_plot*0,color = 'k',zorder = 1,linewidth=1)                       
    # plot the input/output tangency point
    ax1.scatter(w_val,g_val,s = 80,c = 'lime',edgecolor = 'k',linewidth = 2,zorder = 3)            # plot point of tangency

    #### plot first order approximation ####
    # plug input into the first derivative
    g_grad_val = grad(w_val)

    # determine width to plot the approximation -- so its length == width
    width = 4
    div = float(1 + g_grad_val**2)
    w1 = w_val - math.sqrt(width/div)
    w2 = w_val + math.sqrt(width/div)

    # compute first order approximation
    wrange = np.linspace(w1,w2, 100)
    h = g_val + g_grad_val*(wrange - w_val)

    # plot the first order approximation
    ax1.plot(wrange,h,color = 'lime',alpha = 0.5,linewidth = 3,zorder = 2)      # plot approx
    
    #### clean up panel ####
    # fix viewing limits on panel
    v = 5
    ax1.set_xlim([-v,v])
    ax1.set_ylim([-1 - 0.3,v - 0.3])

    # label axes
    ax1.set_xlabel('$w$',fontsize = 12,labelpad = -60)
    ax1.set_ylabel('$g(w)$',fontsize = 25,rotation = 0,labelpad = 50)
    ax1.grid(False)
    ax1.yaxis.set_visible(False)
    ax1.spines['right'].set_visible(False)
    ax1.spines['top'].set_visible(False)
    ax1.spines['left'].set_visible(False)
    
    ### draw 3d version ###
    ax2 = plt.subplot(gs[2],projection='3d'); 
    grad = compute_grad(func2)
    w_val = [float(0),float(0)]
    
    # define input space
    w_in = np.linspace(-2,2,200)
    w1_vals, w2_vals = np.meshgrid(w_in,w_in)
    w1_vals.shape = (len(w_in)**2,1)
    w2_vals.shape = (len(w_in)**2,1)
    w_vals = np.concatenate((w1_vals,w2_vals),axis=1).T
    g_vals = func2(w_vals) 
      
    # evaluation points
    w_val = np.array([float(w_val[0]),float(w_val[1])])
    w_val.shape = (2,1)
    g_val = func2(w_val)
    grad_val = grad(w_val)
    grad_val.shape = (2,1)  

    # create and evaluate tangent hyperplane
    w_tan = np.linspace(-1,1,200)
    w1tan_vals, w2tan_vals = np.meshgrid(w_tan,w_tan)
    w1tan_vals.shape = (len(w_tan)**2,1)
    w2tan_vals.shape = (len(w_tan)**2,1)
    wtan_vals = np.concatenate((w1tan_vals,w2tan_vals),axis=1).T

    #h = lambda weh: g_val +  np.dot( (weh - w_val).T,grad_val)
    h = lambda weh: g_val + (weh[0]-w_val[0])*grad_val[0] + (weh[1]-w_val[1])*grad_val[1]     
    h_vals = h(wtan_vals + w_val)
    zmin = min(np.min(h_vals),-0.5)
    zmax = max(np.max(h_vals),+0.5)

    # vals for cost surface, reshape for plot_surface function
    w1_vals.shape = (len(w_in),len(w_in))
    w2_vals.shape = (len(w_in),len(w_in))
    g_vals.shape = (len(w_in),len(w_in))
    w1tan_vals += w_val[0]
    w2tan_vals += w_val[1]
    w1tan_vals.shape =  (len(w_tan),len(w_tan))
    w2tan_vals.shape =  (len(w_tan),len(w_tan))
    h_vals.shape = (len(w_tan),len(w_tan))

    ### plot function ###
    ax2.plot_surface(w1_vals, w2_vals, g_vals, alpha = 0.5,color = 'w',rstride=25, cstride=25,linewidth=1,edgecolor = 'k',zorder = 2)

    ### plot z=0 plane ###
    ax2.plot_surface(w1_vals, w2_vals, g_vals*0, alpha = 0.1,color = 'w',zorder = 1,rstride=25, cstride=25,linewidth=0.3,edgecolor = 'k') 

    ### plot tangent plane ###
    ax2.plot_surface(w1tan_vals, w2tan_vals, h_vals, alpha = 0.4,color = 'lime',zorder = 1,rstride=50, cstride=50,linewidth=1,edgecolor = 'k')     

    # scatter tangency 
    ax2.scatter(w_val[0],w_val[1],g_val,s = 70,c = 'lime',edgecolor = 'k',linewidth = 2)
    
    ### clean up plot ###
    # plot x and y axes, and clean up
    ax2.xaxis.pane.fill = False
    ax2.yaxis.pane.fill = False
    ax2.zaxis.pane.fill = False

    ax2.xaxis.pane.set_edgecolor('white')
    ax2.yaxis.pane.set_edgecolor('white')
    ax2.zaxis.pane.set_edgecolor('white')

    # remove axes lines and tickmarks
    ax2.w_zaxis.line.set_lw(0.)
    ax2.set_zticks([])
    ax2.w_xaxis.line.set_lw(0.)
    ax2.set_xticks([])
    ax2.w_yaxis.line.set_lw(0.)
    ax2.set_yticks([])

    # set viewing angle
    ax2.view_init(20,-65)

    # set vewing limits
    y = 4
    ax2.set_xlim([-y,y])
    ax2.set_ylim([-y,y])
    ax2.set_zlim([zmin,zmax])

    # label plot
    fontsize = 12
    ax2.set_xlabel(r'$w_1$',fontsize = fontsize,labelpad = -35)
    ax2.set_ylabel(r'$w_2$',fontsize = fontsize,rotation = 0,labelpad=-40)
        
    plt.show()
    
# animator for recursive function
def visualize3d(func,**kwargs):
    grad = compute_grad(func)           # gradient of input function
    colors = [[0,1,0.25],[0,0.75,1]]    # set of custom colors used for plotting
        
    num_frames = 10
    if 'num_frames' in kwargs:
        num_frames = kwargs['num_frames']
    
    view = [20,-50]
    if 'view' in kwargs:
        view = kwargs['view']
        
    plot_descent = False
    if 'plot_descent' in kwargs:
        plot_descent = kwargs['plot_descent']
        
    pt1 = [0,0]
    pt2 = [-0.5,0.5]
    if 'pt' in kwargs:
        pt1 = kwargs['pt']
    if 'pt2' in kwargs:
        pt2 = kwargs['pt2']
       
    # construct figure
    fig = plt.figure(figsize = (9,6))
          
    # remove whitespace from figure
    fig.subplots_adjust(left=0, right=1, bottom=0, top=1) # remove whitespace
    fig.subplots_adjust(wspace=0.01,hspace=0.01)
        
    # create subplotting mechanism
    gs = gridspec.GridSpec(1, 1) 
    ax1 = plt.subplot(gs[0],projection='3d'); 
    
    # define input space
    w_in = np.linspace(-2,2,200)
    w1_vals, w2_vals = np.meshgrid(w_in,w_in)
    w1_vals.shape = (len(w_in)**2,1)
    w2_vals.shape = (len(w_in)**2,1)
    w_vals = np.concatenate((w1_vals,w2_vals),axis=1).T
    g_vals = func(w_vals) 
    cont = 1
    for pt in [pt1]:
        # create axis for plotting
        if cont == 1:
            ax = ax1
        if cont == 2:
            ax = ax2

        cont+=1
        # evaluation points
        w_val = np.array([float(pt[0]),float(pt[1])])
        w_val.shape = (2,1)
        g_val = func(w_val)
        grad_val = grad(w_val)
        grad_val.shape = (2,1)  

        # create and evaluate tangent hyperplane
        w_tan = np.linspace(-1,1,200)
        w1tan_vals, w2tan_vals = np.meshgrid(w_tan,w_tan)
        w1tan_vals.shape = (len(w_tan)**2,1)
        w2tan_vals.shape = (len(w_tan)**2,1)
        wtan_vals = np.concatenate((w1tan_vals,w2tan_vals),axis=1).T

        #h = lambda weh: g_val +  np.dot( (weh - w_val).T,grad_val)
        h = lambda weh: g_val + (weh[0]-w_val[0])*grad_val[0] + (weh[1]-w_val[1])*grad_val[1]     
        h_vals = h(wtan_vals + w_val)
        zmin = min(np.min(h_vals),-0.5)
        zmax = max(np.max(h_vals),+0.5)

        # vals for cost surface, reshape for plot_surface function
        w1_vals.shape = (len(w_in),len(w_in))
        w2_vals.shape = (len(w_in),len(w_in))
        g_vals.shape = (len(w_in),len(w_in))
        w1tan_vals += w_val[0]
        w2tan_vals += w_val[1]
        w1tan_vals.shape =  (len(w_tan),len(w_tan))
        w2tan_vals.shape =  (len(w_tan),len(w_tan))
        h_vals.shape = (len(w_tan),len(w_tan))

        ### plot function ###
        ax.plot_surface(w1_vals, w2_vals, g_vals, alpha = 0.1,color = 'w',rstride=25, cstride=25,linewidth=1,edgecolor = 'k',zorder = 2)

        ### plot z=0 plane ###
        ax.plot_surface(w1_vals, w2_vals, g_vals*0, alpha = 0.1,color = 'w',zorder = 1,rstride=25, cstride=25,linewidth=0.3,edgecolor = 'k') 

        ### plot tangent plane ###
        ax.plot_surface(w1tan_vals, w2tan_vals, h_vals, alpha = 0.1,color = 'lime',zorder = 1,rstride=50, cstride=50,linewidth=1,edgecolor = 'k')     

        ### plot particular points - origins and tangency ###
        # scatter origin
        ax.scatter(0,0,0,s = 60,c = 'k',edgecolor = 'w',linewidth = 2)

        # scatter tangency 
        ax.scatter(w_val[0],w_val[1],g_val,s = 70,c = 'lime',edgecolor = 'k',linewidth = 2)

        ##### add arrows and annotations for steepest ascent direction #####
        # re-assign func variable to tangent
        cutoff_val = 0.1
        an = 1.7
        pname = 'g(' + str(pt[0]) + ',' + str(pt[1]) + ')'
        s = h([1,0]) - h([0,0])
        if abs(s) > cutoff_val:
            # draw arrow
            a = Arrow3D([0, s], [0, 0], [0, 0], mutation_scale=20,
                        lw=2, arrowstyle="-|>", color="b")
            ax.add_artist(a)

            # label arrow
            q = h([an,0]) - h([0,0])
            name = r'$\left(\frac{\mathrm{d}}{\mathrm{d}w_1}' + pname + r',0\right)$'
            annotate3D(ax, s=name, xyz=[q,0,0], fontsize=12, xytext=(-3,3),textcoords='offset points', ha='center',va='center') 

        t = h([0,1]) - h([0,0])
        if abs(t) > cutoff_val:
            # draw arrow
            a = Arrow3D([0, 0], [0, t], [0, 0], mutation_scale=20,
                        lw=2, arrowstyle="-|>", color="b")
            ax.add_artist(a)  

            # label arrow
            q = h([0,an]) - h([0,0])
            name = r'$\left(0,\frac{\mathrm{d}}{\mathrm{d}w_2}' + pname + r'\right)$'
            annotate3D(ax, s=name, xyz=[0,q,0], fontsize=12, xytext=(-3,3),textcoords='offset points', ha='center',va='center') 

        # full gradient
        if abs(s) > cutoff_val and abs(t) > cutoff_val:
            a = Arrow3D([0, h([1,0])- h([0,0])], [0, h([0,1])- h([0,0])], [0, 0], mutation_scale=20,
                        lw=2, arrowstyle="-|>", color="k")
            ax.add_artist(a)  

            s = h([an+0.2,0]) - h([0,0])
            t = h([0,an+0.2]) - h([0,0])
            name = r'$\left(\frac{\mathrm{d}}{\mathrm{d}w_1}' + pname + r',\frac{\mathrm{d}}{\mathrm{d}w_2}' + pname + r'\right)$'
            annotate3D(ax, s=name, xyz=[s,t,0], fontsize=12, xytext=(-3,3),textcoords='offset points', ha='center',va='center') 

        ###### add arrow and text for steepest descent direction #####
        if plot_descent == True:
            # full negative gradient
            if abs(s) > cutoff_val and abs(t) > cutoff_val:
                a = Arrow3D([0, - (h([1,0])- h([0,0]))], [0, - (h([0,1])- h([0,0]))], [0, 0], mutation_scale=20,
                        lw=2, arrowstyle="-|>", color="r")
                ax.add_artist(a)  

                s = - (h([an+0.2,0]) - h([0,0]))
                t = - (h([0,an+0.2]) - h([0,0]))
                name = r'$\left(-\frac{\mathrm{d}}{\mathrm{d}w_1}' + pname + r',-\frac{\mathrm{d}}{\mathrm{d}w_2}' + pname + r'\right)$'
                annotate3D(ax, s=name, xyz=[s,t,0], fontsize=12, xytext=(-3,3),textcoords='offset points', ha='center',va='center') 

            
        ### clean up plot ###
        # plot x and y axes, and clean up
        ax.xaxis.pane.fill = False
        ax.yaxis.pane.fill = False
        ax.zaxis.pane.fill = False

        ax.xaxis.pane.set_edgecolor('white')
        ax.yaxis.pane.set_edgecolor('white')
        ax.zaxis.pane.set_edgecolor('white')

        # remove axes lines and tickmarks
        ax.w_zaxis.line.set_lw(0.)
        ax.set_zticks([])
        ax.w_xaxis.line.set_lw(0.)
        ax.set_xticks([])
        ax.w_yaxis.line.set_lw(0.)
        ax.set_yticks([])

        # set viewing angle
        ax.view_init(view[0],view[1])

        # set vewing limits
        y = 4.5
        ax.set_xlim([-y,y])
        ax.set_ylim([-y,y])
        ax.set_zlim([zmin,zmax])

        # label plot
        fontsize = 14
        ax.set_xlabel(r'$w_1$',fontsize = fontsize,labelpad = -20)
        ax.set_ylabel(r'$w_2$',fontsize = fontsize,rotation = 0,labelpad=-30)
    # plot
    plt.show()

 
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