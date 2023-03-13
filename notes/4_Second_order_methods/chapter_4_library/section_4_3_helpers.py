import matplotlib.pyplot as plt
import matplotlib.animation as animation
from IPython.display import clear_output
from matplotlib import gridspec
from mpl_toolkits.mplot3d import proj3d
from matplotlib.patches import FancyArrowPatch
from matplotlib.text import Annotation
from mpl_toolkits.mplot3d.proj3d import proj_transform

from autograd import grad as compute_grad 
import autograd.numpy as np
import math
import copy
import time

# illustrate first and second order taylor series approximations to one-parameter input functions
class visualizer:
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
        fig = plt.figure(figsize = (16,8))
        artist = fig

        # create subplot with 3 panels, plot input function in center plot
        gs = gridspec.GridSpec(1, 2, width_ratios=[1,1]) 
        fig.subplots_adjust(wspace=0.5,hspace=0.01)

        # plot input function
        ax = plt.subplot(gs[0])
        ax2 = plt.subplot(gs[1])
        
        # range over which to plot second derivative
        max_val = 2.5
        if 'max_val' in kwargs:
            max_val = kwargs['max_val']
        w_vals = np.linspace(-max_val,max_val,num_frames)       # range of values over which to plot first / second order approximations
        
        # generate a range of values over which to plot input function, and derivatives
        w_plot = np.linspace(-max_val,max_val,200)                  # input range for original function
       
        ### function evaluation and second derivative evaluation ###
        g_plot = self.g(w_plot)
        g_range = max(g_plot) - min(g_plot)             # used for cleaning up final plot
        ggap = g_range*0.5
        
        hess_plot = np.array([self.hess(v) for v in w_vals])
        hess_range = max(hess_plot) - min(hess_plot)
        hess_gap = hess_range*0.25
                 
        # animation sub-function
        print ('starting animation rendering...')
        def animate(k):
            # clear the panel
            ax.cla()
            ax2.cla()
            
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
            
            ##### plot function and second order approximation in left panel #####
            # plot original function
            ax.plot(w_plot,g_plot,color = 'k',zorder = 0,linewidth=3)                           # plot function
            
            # plot the input/output tangency point
            ax.scatter(w_val,g_val,s = 120,c = 'lime',edgecolor = 'k',linewidth = 0.7,zorder = 3)            # plot point of tangency
            
            # label axes
            ax.set_xlabel('$w$',fontsize = 25)
            ax.set_ylabel(r'$g(w)$',fontsize=25,rotation = 0,labelpad = 50)
            
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
            ax.plot(wrange,h,color = self.colors[1],linewidth = 5,zorder = 1)      # plot approx

            # fix viewing limits on panel
            ax.set_xlim([-max_val,max_val])
            ax.set_ylim([min(g_plot) - ggap,max(g_plot) + ggap])
            
            ##### plot second derivative value in right panel #####
            ax2.plot(w_vals[:k+1],hess_plot[:k+1],color = self.colors[1],linewidth = 3,zorder = 1) 
            ax2.plot(w_plot,w_plot*0,color = 'k',zorder = 1,linewidth = 1,linestyle = '--') 
            
            # fix viewing limits on panel
            ax2.set_xlim([-max_val,max_val])
            ax2.set_ylim([min(hess_plot) - hess_gap,max(hess_plot) + hess_gap])
            ax2.set_xlabel('$w$',fontsize = 25)            
            ax2.set_ylabel(r'$\frac{\partial^2}{\partial w}g(w)$',fontsize=25,rotation = 0,labelpad = 50)           
            
            return artist,
        
        anim = animation.FuncAnimation(fig, animate,frames=len(w_vals), interval=len(w_vals), blit=True)
        
        # produce animation and save
        fps = 50
        if 'fps' in kwargs:
            fps = kwargs['fps']
        anim.save(savepath, fps=fps, extra_args=['-vcodec', 'libx264'])
        clear_output()



# visualize derivatie
def compare_2d3d(func1,func2,**kwargs):
    # input arguments
    view = [20,-65]
    if 'view' in kwargs:
        view = kwargs['view']
        
    # define input space
    w = np.linspace(-3,3,200)                  # input range for original function
    if 'w' in kwargs:
        w = kwargs['w']
        
    # define pts
    pt1 = 0
    if 'pt1' in kwargs:
        pt1 = kwargs['pt1']
        
    pt2 = [0,0]
    if 'pt2' in kwargs:
        pt2 = kwargs['pt2']
    
    # construct figure
    fig = plt.figure(figsize = (6,3))
          
    # remove whitespace from figure
    fig.subplots_adjust(left=0, right=1, bottom=0, top=1) # remove whitespace
    fig.subplots_adjust(wspace=0.01,hspace=0.01)
        
    # create subplot with 3 panels, plot input function in center plot
    gs = gridspec.GridSpec(1, 2, width_ratios=[1,2]) 
  
    ### draw 2d version ###
    ax1 = plt.subplot(gs[0]); 
    grad = compute_grad(func1)
    
    # generate a range of values over which to plot input function, and derivatives
    g_plot = func1(w)
    g_range = max(g_plot) - min(g_plot)             # used for cleaning up final plot
    ggap = g_range*0.2
    
    # grab the next input/output tangency pair, the center of the next approximation(s)
    pt1 = float(pt1)
    g_val = func1(pt1)

    # plot original function
    ax1.plot(w,g_plot,color = 'k',zorder = 1,linewidth=2)                          
    
    # plot the input/output tangency point
    ax1.scatter(pt1,g_val,s = 60,c = 'lime',edgecolor = 'k',linewidth = 2,zorder = 3)            # plot point of tangency

    #### plot first order approximation ####
    # plug input into the first derivative
    g_grad_val = grad(pt1)

    # compute first order approximation
    w1 = pt1 - 3
    w2 = pt1 + 3
    wrange = np.linspace(w1,w2, 100)
    h = g_val + g_grad_val*(wrange - pt1)

    # plot the first order approximation
    ax1.plot(wrange,h,color = 'lime',alpha = 0.5,linewidth = 3,zorder = 2)      # plot approx
    
    # make new x-axis
    ax1.plot(w,g_plot*0,linewidth=3,color = 'k')
    
    #### clean up panel ####
    # fix viewing limits on panel
    ax1.set_xlim([min(w),max(w)])
    ax1.set_ylim([min(min(g_plot) - ggap,-4),max(max(g_plot) + ggap,0.5)])

    # label axes
    ax1.set_xlabel('$w$',fontsize = 12,labelpad = -50)
    ax1.set_ylabel('$g(w)$',fontsize = 25,rotation = 0,labelpad = 50)
    
    ax1.grid(False)
    ax1.yaxis.set_visible(False)
    ax1.spines['right'].set_visible(False)
    ax1.spines['top'].set_visible(False)
    ax1.spines['left'].set_visible(False)
    
    
    ### draw 3d version ###
    ax2 = plt.subplot(gs[1],projection='3d'); 
    grad = compute_grad(func2)
    w_val = [float(0),float(0)]
    
    # define input space
    w1_vals, w2_vals = np.meshgrid(w,w)
    w1_vals.shape = (len(w)**2,1)
    w2_vals.shape = (len(w)**2,1)
    w_vals = np.concatenate((w1_vals,w2_vals),axis=1).T
    g_vals = func2(w_vals) 
      
    # evaluation points
    w_val = np.array([float(pt2[0]),float(pt2[1])])
    w_val.shape = (2,1)
    g_val = func2(w_val)
    grad_val = grad(w_val)
    grad_val.shape = (2,1)  

    # create and evaluate tangent hyperplane
    w1tan_vals, w2tan_vals = np.meshgrid(w,w)
    w1tan_vals.shape = (len(w)**2,1)
    w2tan_vals.shape = (len(w)**2,1)
    wtan_vals = np.concatenate((w1tan_vals,w2tan_vals),axis=1).T

    #h = lambda weh: g_val +  np.dot( (weh - w_val).T,grad_val)
    h = lambda weh: g_val + (weh[0]-w_val[0])*grad_val[0] + (weh[1]-w_val[1])*grad_val[1]     
    h_vals = h(wtan_vals + w_val)

    # vals for cost surface, reshape for plot_surface function
    w1_vals.shape = (len(w),len(w))
    w2_vals.shape = (len(w),len(w))
    g_vals.shape = (len(w),len(w))
    w1tan_vals += w_val[0]
    w2tan_vals += w_val[1]
    w1tan_vals.shape =  (len(w),len(w))
    w2tan_vals.shape =  (len(w),len(w))
    h_vals.shape = (len(w),len(w))

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

    #ax2.xaxis.pane.set_edgecolor('white')
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
    ax2.view_init(view[0],view[1])

    # set vewing limits
    wgap = (max(w) - min(w))*0.4
    y = max(w) + wgap
    ax2.set_xlim([-y,y])
    ax2.set_ylim([-y,y])
    
    zmin = min(np.min(g_vals),-0.5)
    zmax = max(np.max(g_vals),+0.5)
    ax2.set_zlim([zmin,zmax])

    # label plot
    fontsize = 12
    ax2.set_xlabel(r'$w_1$',fontsize = fontsize,labelpad = -30)
    ax2.set_ylabel(r'$w_2$',fontsize = fontsize,rotation = 0,labelpad=-30)
        
    plt.show()

    
def show_stationary_1func(func,**kwargs):
    '''
    Input one functions, draw each highlighting its stationary points 
    '''
        
    # define input space
    wmax = -3
    if 'wmax' in kwargs:
        wmax = kwargs['wmax']
    w = np.linspace(-wmax,wmax,5000)                  # input range for original function

    # construct figure
    fig = plt.figure(figsize = (6,3))
          
    # remove whitespace from figure
    #fig.subplots_adjust(left=0, right=1, bottom=0, top=1) # remove whitespace
    fig.subplots_adjust(wspace=0.3,hspace=0.4)
       
    # create subplot with 3 panels, plot input function in center plot
    gs = gridspec.GridSpec(1, 2, width_ratios=[1,1]) 
  
    ###### draw function, tangent lines, etc., ######
    ax = plt.subplot(gs[0]); 
    ax2 =  plt.subplot(gs[1],sharey=ax);  

    # generate a range of values over which to plot input function, and derivatives
    g_plot = func(w)
    grad = compute_grad(func)
    grad_plot = np.array([grad(s) for s in w])
    wgap = (max(w) - min(w))*0.1
    ggap = (max(g_plot) - min(g_plot))*0.1
    grad_gap = (max(grad_plot) - min(grad_plot))*0.1
        
    # plot first in top panel, derivative in bottom panel
    ax.plot(w,g_plot,color = 'k',zorder = 1,linewidth=2)  
    ax.set_title(r'$g(w)$',fontsize = 12)
    ax.set_xlim([min(w)-wgap,max(w)+wgap])
    ax.set_ylim([min(g_plot) - ggap, max(g_plot) + ggap])
        
    # plot function with stationary points marked 
    ax2.plot(w,g_plot,color = 'k',zorder = 1,linewidth = 2) 
    ax2.set_title(r'$g(w)$',fontsize = 12)
    ax2.set_ylim([min(g_plot) - ggap, max(g_plot) + ggap])

    # clean up and label axes 
    ax.tick_params(labelsize=6)
    ax2.tick_params(labelsize=6)

    # determine zero derivative points 'visually'
    grad_station = copy.deepcopy(grad_plot)
    grad_station = np.sign(grad_station)
    ind = []
    for i in range(len(grad_station)-1):
        pt1 = grad_station[i]
        pt2 = grad_station[i+1]
        plot_pt1 = grad_plot[i]
        plot_pt2 = grad_plot[i+1]

        # if either point is zero add to list
        if pt1 == 0 or abs(plot_pt1) < 10**-5:
            ind.append(i)
        if pt2 == 0:
            ind.append(i+1)

        # if grad difference is small then sign change has taken place, add to list
        gap = abs(pt1 + pt2)
        if gap < 2 and pt1 !=0 and pt2 != 0:
            ind.append(i)

    # keep unique pts
    ind = np.unique(ind)
        
    # plot the input/output tangency points and tangent line
    wtan = np.linspace(-1,1,500)                  # input range for original function
    for pt in ind:
        # plot point
        w_val = w[pt]
        g_val = func(w_val)
        grad_val = grad(w_val)
        ax2.scatter(w_val,g_val,s = 40,c = 'lime',edgecolor = 'k',linewidth = 2,zorder = 3)            # plot point of tangency
    plt.show()
    
    
    
def show_stationary(func1,func2,func3,**kwargs):
    '''
    Input three functions, draw each highlighting their stationary points and draw tangent lines, mark evaluations on first derivative as well
    '''
        
    # define input space
    w = np.linspace(-3,3,5000)                  # input range for original function
    if 'w' in kwargs:
        w = kwargs['w']

    # construct figure
    fig = plt.figure(figsize = (7,5))
          
    # remove whitespace from figure
    #fig.subplots_adjust(left=0, right=1, bottom=0, top=1) # remove whitespace
    fig.subplots_adjust(wspace=0.3,hspace=0.4)
       
    # create subplot with 3 panels, plot input function in center plot
    gs = gridspec.GridSpec(2, 3, width_ratios=[1,1,1]) 
  
    ###### draw function, tangent lines, etc., ######
    for k in range(3):
        ax = plt.subplot(gs[k]); 
        ax2 =  plt.subplot(gs[k+3],sharex=ax);  
        
        func = func1
        if k == 1:
            func = func2
        if k == 2:
            func = func3

        # generate a range of values over which to plot input function, and derivatives
        g_plot = func(w)
        grad = compute_grad(func)
        grad_plot = np.array([grad(s) for s in w])
        wgap = (max(w) - min(w))*0.1
        ggap = (max(g_plot) - min(g_plot))*0.1
        grad_gap = (max(grad_plot) - min(grad_plot))*0.1
        
        # plot first in top panel, derivative in bottom panel
        ax.plot(w,g_plot,color = 'k',zorder = 1,linewidth=2)   
        ax.set_title(r'$g(w)$',fontsize = 12)
        ax.set_xlim([min(w)-wgap,max(w)+wgap])
        ax.set_ylim([min(g_plot) - ggap, max(g_plot) + ggap])
        
        # plot derivative and horizontal axis
        ax2.plot(w,grad_plot,color = 'k',zorder = 1,linewidth = 2) 
        ax2.plot(w,grad_plot*0,color = 'k',zorder = 1,linewidth = 1,linestyle = '--') 
        ax2.set_title(r'$\frac{\mathrm{d}}{\mathrm{d}w}g(w)$',fontsize = 12)
        ax2.set_ylim([min(grad_plot) - grad_gap, max(grad_plot) + grad_gap])

        # clean up and label axes 
        ax.tick_params(labelsize=6)
        ax2.tick_params(labelsize=6)

        # determine zero derivative points 'visually'
        grad_station = copy.deepcopy(grad_plot)
        grad_station = np.sign(grad_station)
        ind = []
        for i in range(len(grad_station)-1):
            pt1 = grad_station[i]
            pt2 = grad_station[i+1]
            plot_pt1 = grad_plot[i]
            plot_pt2 = grad_plot[i+1]

            # if either point is zero add to list
            if pt1 == 0 or abs(plot_pt1) < 10**-5:
                ind.append(i)
            if pt2 == 0:
                ind.append(i+1)

            # if grad difference is small then sign change has taken place, add to list
            gap = abs(pt1 + pt2)
            if gap < 2 and pt1 !=0 and pt2 != 0:
                ind.append(i)

        # keep unique pts
        ind = np.unique(ind)
        
        # plot the input/output tangency points and tangent line
        wtan = np.linspace(-1,1,500)                  # input range for original function
        for pt in ind:
            # plot point
            w_val = w[pt]
            g_val = func(w_val)
            grad_val = grad(w_val)
            ax.scatter(w_val,g_val,s = 40,c = 'lime',edgecolor = 'k',linewidth = 2,zorder = 3)            # plot point of tangency
            ax2.scatter(w_val,grad_val,s = 40,c = 'lime',edgecolor = 'k',linewidth = 2,zorder = 3)            # plot point of tangency

            # plot tangent line in original space
            w1 = w_val - 1
            w2 = w_val + 1
            wrange = np.linspace(w1,w2, 100)
            h = g_val + 0*(wrange - w_val)
            ax.plot(wrange,h,color = 'lime',alpha = 0.5,linewidth = 1.5,zorder = 2)      # plot approx
    plt.show()
    
def show_stationary_v2(func1,func2,func3,**kwargs):
    '''
    Input three functions, draw each highlighting their stationary points and draw tangent lines, draw the first and second derivatives stationary point evaluations  on each as well
    '''
        
    # define input space
    w = np.linspace(-3,3,5000)                  # input range for original function
    if 'w' in kwargs:
        w = kwargs['w']

    # construct figure
    fig = plt.figure(figsize = (7,5))
          
    # remove whitespace from figure
    #fig.subplots_adjust(left=0, right=1, bottom=0, top=1) # remove whitespace
    fig.subplots_adjust(wspace=0.2,hspace=0.8)
       
    # create subplot with 3 panels, plot input function in center plot
    gs = gridspec.GridSpec(3, 3, width_ratios=[1,1,1]) 
  
    ###### draw function, tangent lines, etc., ######
    for k in range(3):
        ax = plt.subplot(gs[k]); 
        ax2 =  plt.subplot(gs[k+3],sharex=ax);  
        ax3 =  plt.subplot(gs[k+6],sharex=ax);  
        
        func = func1
        if k == 1:
            func = func2
        if k == 2:
            func = func3

        # generate a range of values over which to plot input function, and derivatives
        g_plot = func(w)
        grad = compute_grad(func)
        grad_plot = np.array([grad(s) for s in w])
        wgap = (max(w) - min(w))*0.1
        ggap = (max(g_plot) - min(g_plot))*0.1
        grad_gap = (max(grad_plot) - min(grad_plot))*0.1

        hess = compute_grad(grad)
        hess_plot = np.array([hess(s) for s in w])
        hess_gap = (max(hess_plot) - min(hess_plot))*0.1
            
        # plot first in top panel, derivative in bottom panel
        ax.plot(w,g_plot,color = 'k',zorder = 1,linewidth=2)   
        ax.set_title(r'$g(w)$',fontsize = 12)
        ax.set_xlim([min(w)-wgap,max(w)+wgap])
        ax.set_ylim([min(g_plot) - ggap, max(g_plot) + ggap])
        
        # plot derivative and horizontal axis
        ax2.plot(w,grad_plot,color = 'k',zorder = 1,linewidth = 2) 
        ax2.plot(w,grad_plot*0,color = 'k',zorder = 1,linewidth = 1,linestyle = '--') 
        ax2.set_title(r'$\frac{\mathrm{d}}{\mathrm{d}w}g(w)$',fontsize = 12)
        ax2.set_ylim([min(grad_plot) - grad_gap, max(grad_plot) + grad_gap])

        # plot second derivative and horizontal axis
        ax3.plot(w,hess_plot,color = 'k',zorder = 1,linewidth = 2) 
        ax3.plot(w,hess_plot*0,color = 'k',zorder = 1,linewidth = 1,linestyle = '--') 
        ax3.set_title(r'$\frac{\mathrm{d}^2}{\mathrm{d}w^2}g(w)$',fontsize = 12)
        ax3.set_ylim([min(hess_plot) - hess_gap, max(hess_plot) + hess_gap])
       
        # clean up and label axes 
        ax.tick_params(labelsize=6)
        ax2.tick_params(labelsize=6)
        ax3.tick_params(labelsize=6)

        # determine zero derivative points 'visually'
        grad_station = copy.deepcopy(grad_plot)
        grad_station = np.sign(grad_station)
        ind = []
        for i in range(len(grad_station)-1):
            pt1 = grad_station[i]
            pt2 = grad_station[i+1]
            plot_pt1 = grad_plot[i]
            plot_pt2 = grad_plot[i+1]

            # if either point is zero add to list
            if pt1 == 0 or abs(plot_pt1) < 10**-5:
                ind.append(i)
            if pt2 == 0:
                ind.append(i+1)

            # if grad difference is small then sign change has taken place, add to list
            gap = abs(pt1 + pt2)
            if gap < 2 and pt1 !=0 and pt2 != 0:
                ind.append(i)

        # keep unique pts
        ind = np.unique(ind)
        
        # plot the input/output tangency points and tangent line
        wtan = np.linspace(-1,1,500)                  # input range for original function
        for pt in ind:
            # plot point
            w_val = w[pt]
            g_val = func(w_val)
            grad_val = grad(w_val)
            hess_val = hess(w_val)
            ax.scatter(w_val,g_val,s = 40,c = 'lime',edgecolor = 'k',linewidth = 2,zorder = 3)            # plot point of tangency
            ax2.scatter(w_val,grad_val,s = 40,c = 'lime',edgecolor = 'k',linewidth = 2,zorder = 3)            # plot point of tangency
            ax3.scatter(w_val,hess_val,s = 40,c = 'lime',edgecolor = 'k',linewidth = 2,zorder = 3)            # plot point of tangency
            
            # plot tangent line in original space
            w1 = w_val - 1
            w2 = w_val + 1
            wrange = np.linspace(w1,w2, 100)
            h = g_val + 0*(wrange - w_val)
            ax.plot(wrange,h,color = 'lime',alpha = 0.5,linewidth = 1.5,zorder = 2)      # plot approx
    plt.show()
                
