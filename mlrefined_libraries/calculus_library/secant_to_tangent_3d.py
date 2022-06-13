# import standard plotting and animation
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from IPython.display import clear_output
import time
from matplotlib import gridspec
from mpl_toolkits.mplot3d import proj3d
from mpl_toolkits.mplot3d.proj3d import proj_transform

# import autograd functionality
from autograd import grad as compute_grad   # The only autograd function you may ever need
import autograd.numpy as np
import math
import copy
        
# function for producing fixed image of tangency along each input axis, along with full tangent hyperplane (first order Taylor Series approximation)
def draw_it(func,**kwargs):
    view =  [33,50]
    if 'view' in kwargs:
        view = kwargs['view']
    
    # compute gradient, points
    anchor = [0,0]
    anchor = np.array([float(anchor[0]),float(anchor[1])])
    anchor.shape = (2,1)
    g_anchor = func(anchor)
    
    # file tracer
    tracer = np.asarray([0,10**-5])
    tracer = np.array([float(tracer[0]),float(tracer[1])])
    tracer.shape = (2,1)
    g_tracer = func(tracer) 

    # construct figure
    fig = plt.figure(figsize = (9,3))
    artist = fig
    
    # remove whitespace from figure
    fig.subplots_adjust(left=0, right=1, bottom=0, top=1) # remove whitespace
    fig.subplots_adjust(wspace=0.01,hspace=0.01)
    
    # create subplot with 3 panels, plot input function in center plot
    gs = gridspec.GridSpec(1, 3, width_ratios=[1,1,1]) 
    ax1 = plt.subplot(gs[0],projection='3d'); 
    ax2 = plt.subplot(gs[1],projection='3d');     
    ax3 = plt.subplot(gs[2],projection='3d');     
    
    ### first panel - partial with respect to w_1 ###
    # scatter anchor point 
    ax1.scatter(anchor[0],anchor[1],g_anchor,s = 50,c = 'lime',edgecolor = 'k',linewidth = 1) 
    
    # plot hyperplane connecting the anchor to tracer
    secant(func,anchor,tracer,ax1)
                
    # plot function
    plot_func(func,view,ax1)
    
    ### second panel - partial with respect to w_2 ###
    tracer = np.flipud(tracer)
    
    ax2.scatter(anchor[0],anchor[1],g_anchor,s = 50,c = 'lime',edgecolor = 'k',linewidth = 1) 
    
    # plot hyperplane connecting the anchor to tracer
    secant(func,anchor,tracer,ax2)
                
    # plot function
    plot_func(func,view,ax2)    

    ### third panel - plot full tangent hyperplane at anchor ###
    ax3.scatter(anchor[0],anchor[1],g_anchor,s = 50,c = 'lime',edgecolor = 'k',linewidth = 1) 
    
    # plot hyperplane connecting the anchor to tracer
    tangent(func,anchor,ax3)
                
    # plot function
    plot_func(func,view,ax3)    
    
    
# main function for plotting individual axes tangent approximations
def animate_it(func,**kwargs):
    view =  [33,50]
    if 'view' in kwargs:
        view = kwargs['view']
        
    num_frames = 10
    if 'num_frames' in kwargs:
        num_frames = kwargs['num_frames']
    
    # compute gradient, points
    anchor = [0,0]
    anchor = np.array([float(anchor[0]),float(anchor[1])])
    anchor.shape = (2,1)
    g_anchor = func(anchor)
    
    # compute tracer range
    z = np.zeros((num_frames,1))
    tracer_range = np.linspace(-2.5,2.5,num_frames)
    ind = np.argmin(abs(tracer_range))
    tracer_range[ind] = 10**-5
    tracer_range.shape = (num_frames,1)
    tracer_range = np.concatenate((tracer_range,z),axis=1)
    
    # construct figure
    fig = plt.figure(figsize = (9,4))
    artist = fig
    
    # remove whitespace from figure
    fig.subplots_adjust(left=0, right=1, bottom=0, top=1) # remove whitespace
    fig.subplots_adjust(wspace=0.01,hspace=0.01)
    
    # create subplot with 3 panels, plot input function in center plot
    gs = gridspec.GridSpec(1, 2, width_ratios=[1,1]) 
    ax1 = plt.subplot(gs[0],projection='3d'); 
    ax2 = plt.subplot(gs[1],projection='3d'); 
    
    # start animation
    def animate(k):
        # clear the panels
        ax1.cla()
        ax2.cla()
        
        # print rendering update
        if np.mod(k+1,25) == 0:
            print ('rendering animation frame ' + str(k+1) + ' of ' + str(num_frames))
        if k == num_frames - 1:
            print ('animation rendering complete!')
            time.sleep(1.5)
            clear_output()

        if k > 0:
            # pull current tracer
            tracer = tracer_range[k-1]
            tracer = np.array([float(tracer[0]),float(tracer[1])])
            tracer.shape = (2,1)
            g_tracer = func(tracer)
                
        ### draw 3d version ###
        for ax in [ax1,ax2]:
            # plot function
            plot_func(func,view,ax)
            
            if k > 0:
                # scatter anchor point 
                ax.scatter(anchor[0],anchor[1],g_anchor,s = 50,c = 'lime',edgecolor = 'k',linewidth = 1)  
                # plot hyperplane connecting the anchor to tracer
                secant(func,anchor,tracer,ax)
                
                # reset tracer
                tracer = np.flipud(tracer)
            
        return artist,
    
    anim = animation.FuncAnimation(fig, animate,frames=num_frames+1, interval=num_frames+1, blit=True)
        
    return(anim)

# plot secant hyperplane, as well as guides for both anchor and tracer point
def secant(func,anchor,tracer,ax): 
    # evaluate function at anchor and tracer
    g_anchor = func(anchor)  
    g_tracer = func(tracer) 
    anchor_orig = copy.deepcopy(anchor)
    tracer_orig = copy.deepcopy(tracer)
    
    # determine non-zero component of tracer, compute slope of secant line
    anchor = anchor.flatten()
    tracer = tracer.flatten()
    ind = np.argwhere(tracer != 0) 
    anchor = anchor[ind]
    tracer = tracer[ind]
   
    # plot secant plane
    color = 'lime'
    if abs(anchor - tracer) > 10**-4:
        # scatter tracer point
        ax.scatter(tracer_orig[0],tracer_orig[1],g_tracer,s = 50,c = 'b',edgecolor = 'k',linewidth = 1)   
        
        # change color to red
        color = 'r'
        
        # plot visual guide for tracer
        w = np.linspace(0,g_tracer,100)
        o = np.ones(100)
        ax.plot(o*tracer_orig[0],o*tracer_orig[1],w,linewidth = 1.5,alpha = 1,color = 'k',linestyle = '--')

        w = np.linspace(0,g_anchor,100)
        o = np.ones(100)
        ax.plot(o*anchor_orig[0],o*anchor_orig[1],w,linewidth = 1.5,alpha = 1,color = 'k',linestyle = '--')
        
    # compute slope of secant plane
    slope = (g_anchor - g_tracer)/float(anchor - tracer)
    
    # create function for hyperplane connecting anchor to tracer
    w_tan = np.linspace(-2.5,2.5,200)

    w1tan_vals, w2tan_vals = np.meshgrid(w_tan,w_tan)
    w1tan_vals.shape = (len(w_tan)**2,1)
    w2tan_vals.shape = (len(w_tan)**2,1)
    wtan_vals = np.concatenate((w1tan_vals,w2tan_vals),axis=1).T    
    
    # create tangent hyperplane formula, evaluate  
    h = lambda w: g_anchor + slope*(w[ind] - anchor)
    h_vals = h(wtan_vals) 

    # reshape everything and prep for plotting
    w1tan_vals.shape = (len(w_tan),len(w_tan))
    w2tan_vals.shape = (len(w_tan),len(w_tan))
    h_vals.shape = (len(w_tan),len(w_tan))

    # plot hyperplane and guides based on proximity of tracer to anchor
    ax.plot_surface(w1tan_vals, w2tan_vals, h_vals, alpha = 0.2,color = color,zorder = 3,rstride=50, cstride=50,linewidth=0.5,edgecolor = 'k')  

# form tangent hyperplane
def tangent(func,anchor,ax):
    # compute gradient
    grad = compute_grad(func)
    grad_val = grad(anchor)
    grad_val.shape = (2,1)  
    g_val = func(anchor)
    
    # create input for tangent hyperplane
    w_tan = np.linspace(-2.5,2.5,200)
    w1tan_vals, w2tan_vals = np.meshgrid(w_tan,w_tan)
    w1tan_vals.shape = (len(w_tan)**2,1)
    w2tan_vals.shape = (len(w_tan)**2,1)
    wtan_vals = np.concatenate((w1tan_vals,w2tan_vals),axis=1).T    
    
    # create tangent hyperplane formula, evaluate
    h = lambda weh: g_val + (weh[0]-anchor[0])*grad_val[0] + (weh[1]-anchor[1])*grad_val[1]     
    h_vals = h(wtan_vals + anchor)

    # vals for tangent
    w1tan_vals += anchor[0]
    w2tan_vals += anchor[1]
    w1tan_vals.shape = (len(w_tan),len(w_tan))
    w2tan_vals.shape = (len(w_tan),len(w_tan))
    h_vals.shape = (len(w_tan),len(w_tan))

    ### plot tangent plane ###
    ax.plot_surface(w1tan_vals, w2tan_vals, h_vals, alpha = 0.4,color = 'lime',zorder = 1,rstride=50, cstride=50,linewidth=0.5,edgecolor = 'k')      
    
# plot the input function and clean up panel
def plot_func(func,view,ax):
    # define input space
    w_func = np.linspace(-2.5,2.5,200)
    w1_vals, w2_vals = np.meshgrid(w_func,w_func)
    w1_vals.shape = (len(w_func)**2,1)
    w2_vals.shape = (len(w_func)**2,1)
    w_vals = np.concatenate((w1_vals,w2_vals),axis=1).T
    g_vals = func(w_vals) 
    w1_vals.shape = (len(w_func),len(w_func))
    w2_vals.shape = (len(w_func),len(w_func))
    g_vals.shape = (len(w_func),len(w_func))
    
    ### plot function ###
    ax.plot_surface(w1_vals, w2_vals, g_vals, alpha = 0.1,color = 'w',rstride=25, cstride=25,linewidth=0.75,edgecolor = 'k',zorder = 2)
    
    # clean up the plot while you're at it
    cleanup(g_vals,view,ax)
    
# cleanup an input panel
def cleanup(g_vals,view,ax):
    ### clean up plot ###
    # plot x and y axes, and clean up
    ax.xaxis.pane.fill = False
    ax.yaxis.pane.fill = False
    ax.zaxis.pane.fill = False

    ax.xaxis.pane.set_edgecolor('white')
    ax.yaxis.pane.set_edgecolor('white')
    ax.zaxis.pane.set_edgecolor('white')

    ### plot z=0 plane ###
    w_zplane = np.linspace(-3,3,200)
    w1_zplane_vals, w2_zplane_vals = np.meshgrid(w_zplane,w_zplane)
    ax.plot_surface(w1_zplane_vals, w2_zplane_vals, np.zeros(np.shape(w1_zplane_vals)), alpha = 0.1,color = 'w',zorder = 1,rstride=25, cstride=25,linewidth=0.3,edgecolor = 'k') 
    
    # bolden axis on z=0 plane
    ax.plot(w_zplane,w_zplane*0,w_zplane*0,color = 'k',linewidth = 1.5)
    ax.plot(w_zplane*0,w_zplane,w_zplane*0,color = 'k',linewidth = 1.5)

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
    y = 3
    ax.set_xlim([-y,y])
    ax.set_ylim([-y,y])
    zmin = min(np.min(g_vals),-0.5)
    zmax = max(np.max(g_vals),+0.5)
    ax.set_zlim([zmin,zmax])

    # label plot
    fontsize = 12
    ax.set_xlabel(r'$w_1$',fontsize = fontsize,labelpad = -20)
    ax.set_ylabel(r'$w_2$',fontsize = fontsize,rotation = 0,labelpad=-20)
        