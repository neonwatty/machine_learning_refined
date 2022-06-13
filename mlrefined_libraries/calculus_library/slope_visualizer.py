import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from matplotlib import gridspec
from IPython.display import display, HTML, clear_output
import copy
from mpl_toolkits.mplot3d import proj3d
from matplotlib.patches import FancyArrowPatch
from matplotlib.text import Annotation
from mpl_toolkits.mplot3d.proj3d import proj_transform

from matplotlib import gridspec
from matplotlib.ticker import FormatStrFormatter
import matplotlib.animation as animation
from mpl_toolkits.mplot3d import Axes3D
from IPython.display import clear_output
import matplotlib.ticker as ticker

# import custom JS animator
# from mlrefined_libraries.JSAnimation_slider_only import IPython_display_slider_only
import time
import math

### animate 2d slope visualization ###
# animator for recursive function
def animate_visualize2d(func,num_frames,savepath,**kwargs):
    # define input space
    w = np.linspace(-10,10,500)
    guides = 'on'
    
    # define slopes
    func_orig = func
    s = func(1) - func(0)  # slope of input function
    slopes = np.linspace(-abs(s),abs(s),num_frames)

    # construct figure
    fig = plt.figure(figsize = (12,4))
    artist = fig

    # create subplot with 3 panels, plot input function in center plot
    gs = gridspec.GridSpec(1, 3, width_ratios=[1,3, 1]) 
    ax1 = plt.subplot(gs[0]); ax1.axis('off');
    ax3 = plt.subplot(gs[2]); ax3.axis('off');

    # plot input function
    ax2 = plt.subplot(gs[1])    

    # animate
    def animate(k):
        # clear the panel
        ax2.cla()
        
        # setup function
        slope = slopes[k]
        func = lambda w: slope*w + func_orig(0)
        
        # print rendering update
        if np.mod(k+1,25) == 0:
            print ('rendering animation frame ' + str(k+1) + ' of ' + str(num_frames))
        if k == num_frames - 1:
            print ('animation rendering complete!')
            time.sleep(1.5)
            clear_output()
            
        # plot function
        ax2.plot(w,func(w), c='lime', linewidth=2,zorder = 3)
     
        ### plot slope as vector
        if abs(func(1) - func(0)) > 0.2:
            head_width = 0.166*(func(1) - func(0))
            head_length = 0.25*(func(1) - func(0))
        
            # annotate arrow and annotation
            if func(1)-func(0) > 0.1:
                ax2.arrow(0, 0, func(1)-func(0),0, head_width=head_width, head_length=head_length, fc='k', ec='k',linewidth=2.5,zorder = 3)
                
                ax2.annotate('$b$', xy=(2, 1), xytext=(func(1.3)-func(0),0),fontsize=15
            )
            elif func(1)-func(0) < -0.1:
                ax2.arrow(0, 0, func(1)-func(0),0, head_width=-head_width, head_length=-head_length, fc='k', ec='k',linewidth=2.5,zorder = 3)
                
                ax2.annotate('$b$', xy=(2, 1), xytext=(func(1.5)-func(0),0),fontsize=15
            )
                
        ### plot negative slope as vector
        if abs(func(1) - func(0)) >=0:
            head_width = 0.166*(func(0) - func(1))
            head_length = -0.25*(func(0) - func(1))

            # annotate arrow and annotation
            if func(1)-func(0) >= 0:
                ax2.arrow(0, 0, func(0)-func(1),0, head_width=head_width, head_length=head_length, fc='r', ec='r',linewidth=2.5,zorder = 3)
        
                ax2.annotate('$-b$', xy=(2, 1), xytext=(func(0) - func(1) - head_length - 0.7,0),fontsize=15)
        
            elif func(1)-func(0) < 0:
                ax2.arrow(0, 0, func(0)-func(1),0, head_width=-head_width, head_length=-head_length, fc='r', ec='r',linewidth=2.5,zorder = 3)
                
                ax2.annotate('$-b$', xy=(2, 1), xytext=( func(0) - func(1+0.3),0),fontsize=15)      
                    
        # set viewing limits
        wgap = (max(w) - min(w))*0.3
        ax2.set_xlim([-5,5])
        ax2.set_ylim([-5,5])

        # plot x and y axes, and clean up
        ax2.grid(True, which='both')
        
        # label plot
        ax2.set_xlabel('$w$',fontsize = 15)
        ax2.set_ylabel('$g(w)$',fontsize = 15,rotation = 0,labelpad = 20)
        title = r'$g(w) = {:.1f}'.format(func_orig(0)) + '+ {:.1f}'.format(slope)+'w$'
        if slope < 0:
            title = r'$g(w) = {:.1f}'.format(func_orig(0)) + '{:.1f}'.format(slope)+'w$'

        ax2.set_title(title,fontsize = 18)
        return artist,        
        
    anim = animation.FuncAnimation(fig, animate ,frames=num_frames, interval=num_frames, blit=True)

    # produce animation and save
    fps = 50
    if 'fps' in kwargs:
        fps = kwargs['fps']
    anim.save(savepath, fps=fps, extra_args=['-vcodec', 'libx264'])
    clear_output()        
        
    return(anim)

### animate 3d slope visualization ###
# animator for recursive function
def animate_visualize3d(func,savepath,**kwargs):
    
    num_frames = 10
    if 'num_frames' in kwargs:
        num_frames = kwargs['num_frames']
    
    view = [20,-50]
    if 'view' in kwargs:
        view = kwargs['view']
       
    # construct figure
    fig = plt.figure(figsize = (6,6))
    artist = fig
    
    # create subplot with 3 panels, plot input function in center plot
    gs = gridspec.GridSpec(1, 1, width_ratios=[1]) 
    ax = plt.subplot(gs[0],projection='3d'); 

    # determine slope range of input function
    func_orig = func
    bias = func([0,0])
    s = func([1,0]) - bias   # slope 1 of input
    t = func([0,1]) - bias   # slope 2 of input
    slopes1 = np.linspace(s,-s,num_frames)
    slopes1.shape = (len(slopes1),1)
    slopes2 = np.linspace(t,-t,num_frames)
    slopes2.shape = (len(slopes2),1)
    slopes = np.concatenate((slopes1,slopes2),axis=1)
    
    # define input space
    w_in = np.linspace(-2,2,200)
    w1_vals, w2_vals = np.meshgrid(w_in,w_in)
    w1_vals.shape = (len(w_in)**2,1)
    w2_vals.shape = (len(w_in)**2,1)
    g_vals_orig = func([w1_vals,w2_vals]) 
    zmin = np.min(g_vals_orig)
    zmax = np.max(g_vals_orig)
    
    # animate
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
            
        # create mesh for surface
        w1_vals.shape = (len(w_in)**2,1)
        w2_vals.shape = (len(w_in)**2,1)
        
        # create and evaluate function
        slope = slopes[k,:]
        func = lambda w: slope[0]*w[0] + slope[1]*w[1]  + bias
        g_vals = func([w1_vals,w2_vals]) 

        # vals for cost surface, reshape for plot_surface function
        w1_vals.shape = (len(w_in),len(w_in))
        w2_vals.shape = (len(w_in),len(w_in))
        g_vals.shape = (len(w_in),len(w_in))
        
        ### plot function and z=0 for visualization ###
        ax.plot_surface(w1_vals, w2_vals, g_vals, alpha = 0.3,color = 'lime',rstride=25, cstride=25,linewidth=0.7,edgecolor = 'k',zorder = 2)

        ax.plot_surface(w1_vals, w2_vals, g_vals*0, alpha = 0.1,color = 'w',zorder = 1,rstride=25, cstride=25,linewidth=0.5,edgecolor = 'k') 
        
        ### add arrows and annotations ###
        # add arrow for slope visualization
        s = func([1,0]) - func([0,0])
        if abs(s) > 0.5:
            # draw arrow
            a = Arrow3D([0, s], [0, 0], [0, 0], mutation_scale=20,
                    lw=2, arrowstyle="-|>", color="b")
            ax.add_artist(a)
            
            # label arrow
            q = func([1.3,0]) - func([0,0])
            annotate3D(ax, s='$(b_1,0)$', xyz=[q,0,0], fontsize=13, xytext=(-3,3),
               textcoords='offset points', ha='center',va='center') 

        t = func([0,1]) - func([0,0])
        if abs(t) > 0.5:
            # draw arrow
            a = Arrow3D([0, 0], [0, t], [0, 0], mutation_scale=20,
                    lw=2, arrowstyle="-|>", color="b")
            ax.add_artist(a)  
            
            # label arrow
            q = func([0,1.3]) - func([0,0])
            annotate3D(ax, s='$(0,b_2)$', xyz=[0,q,0], fontsize=13, xytext=(-3,3),
               textcoords='offset points', ha='center',va='center') 
                
        # full gradient
        if abs(s) > 0.5 and abs(t) > 0.5:
            a = Arrow3D([0, func([1,0])- func([0,0])], [0, func([0,1])- func([0,0])], [0, 0], mutation_scale=20,
                    lw=2, arrowstyle="-|>", color="k")
            ax.add_artist(a)  
            
            s = func([1.2,0]) - func([0,0])
            t = func([0,1.2]) - func([0,0])
            annotate3D(ax, s='$(b_1,b_2)$', xyz=[s,t,0], fontsize=13, xytext=(-3,3),
               textcoords='offset points', ha='center',va='center') 
            
        # full negative gradient
        if abs(s) > 0.5 and abs(t) > 0.5:
            a = Arrow3D([0, - (func([1,0])- func([0,0]))], [0, - (func([0,1])- func([0,0]))], [0, 0], mutation_scale=20,
                        lw=2, arrowstyle="-|>", color="r")
            ax.add_artist(a)  
            an = 1
            s = - (func([an+0.2,0]) - func([0,0]))
            t = - (func([0,an+0.2]) - func([0,0]))
            name = '$-(b_1,b_2)$'
            annotate3D(ax, s=name, xyz=[s,t,0], fontsize= 13, xytext=(-3,3),textcoords='offset points', ha='center',va='center') 

        # draw negative coordinate-wise slope vector        
        if abs(s) > 0.5 and abs(t) < 0.5:
            a = Arrow3D([0, - (func([1,0])- func([0,0]))], [0, - (func([0,1])- func([0,0]))], [0, 0], mutation_scale=20,
                        lw=2, arrowstyle="-|>", color="r")
            ax.add_artist(a)  
            an = 1.2
            s = - (func([an+0.2,0]) - func([0,0]))
            t = - (func([0,an+0.2]) - func([0,0]))
            name = '$-(b_1,0)$'
            annotate3D(ax, s=name, xyz=[s,t,0], fontsize=13, xytext=(-3,3),textcoords='offset points', ha='center',va='center') 
 
            
        # draw negative coordinate-wise slope vector        
        if abs(t) > 0.5 and abs(s) < 0.5:
            a = Arrow3D([0, - (func([1,0])- func([0,0]))], [0, - (func([0,1])- func([0,0]))], [0, 0], mutation_scale=20,
                        lw=2, arrowstyle="-|>", color="r")
            ax.add_artist(a)  
            an = 1.2
            s = - (func([an+0.2,0]) - func([0,0]))
            t = - (func([0,an+0.2]) - func([0,0]))
            name = '$-(0,b_2)$'
            annotate3D(ax, s=name, xyz=[s,t,0], fontsize=13, xytext=(-3,3),textcoords='offset points', ha='center',va='center') 
           
            
        ### clean up plot ###
        # plot x and y axes, and clean up
        ax.grid(False)
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
        
        # remove whitespace from figure
        fig.subplots_adjust(left=0, right=1, bottom=0, top=1) # remove whitespace
        fig.subplots_adjust(wspace=0.01,hspace=0.01)

        # set vewing limits
        y = 2.3
        ax.set_xlim([-y,y])
        ax.set_ylim([-y,y])
        ax.set_zlim([zmin,zmax])

        # label plot
        fontsize = 15
        ax.set_xlabel(r'$w_1$',fontsize = fontsize,labelpad = -10)
        ax.set_ylabel(r'$w_2$',fontsize = fontsize,rotation = 0,labelpad=-10)
        
        
        sig = '+'
        if slope[0] < 0:
            sig = '-'
        sig2 = '+'
        if slope[1] < 0:
            sig2 = '-'
       
            
        part2 = sig + '{:.1f}'.format(abs(slope[0])) + 'w_1 '
        if abs(slope[0]) < 0.01:
            part2 = ''
            
        part3 = sig2 + '{:.1f}'.format(abs(slope[1])) + 'w_2'
        if abs(slope[1]) < 0.01:
            part3 = ''
        
        part1 = '{:.1f}'.format(abs(bias))
        if abs(bias) < 0.01:
            part3 = ''
        
        
        ax.set_title(r'$g(w_1,w_2) = ' + part1 + part2  + part3 + '$' ,fontsize = 15)

        return artist,
              
    anim = animation.FuncAnimation(fig, animate ,frames=num_frames, interval=num_frames, blit=True)

    # produce animation and save
    fps = 50
    if 'fps' in kwargs:
        fps = kwargs['fps']
    anim.save(savepath, fps=fps, extra_args=['-vcodec', 'libx264'])
    clear_output()            
        
    return(anim)
    
### static slope visualizer functions ###
# custom plot for spiffing up plot of a single mathematical function
def visualize2d(func,**kwargs):
    # define input space
    w = np.linspace(-10,10,500)
    if 'w' in kwargs:
        w = kwargs['w']
    guides = 'on'
    if 'guides' in kwargs:
        guides = kwargs['guides']
    
    # construct figure
    fig = plt.figure(figsize = (12,4))

    # create subplot with 3 panels, plot input function in center plot
    gs = gridspec.GridSpec(1, 3, width_ratios=[1,3, 1]) 
    ax1 = plt.subplot(gs[0]); ax1.axis('off');
    ax3 = plt.subplot(gs[2]); ax3.axis('off');

    # plot input function
    ax2 = plt.subplot(gs[1])

    # plot function
    ax2.plot(w,func(w), c='r', linewidth=2,zorder = 3)
     
    ### plot slope as vector
    if abs(func(1)) > 0.2:
        head_width = 0.166*func(1)
        head_length = 0.25*func(1)
        
        # plot slope guide as arrow
        ax2.arrow(0, 0, func(1),0, head_width=head_width, head_length=head_length, fc='k', ec='k',linewidth=2.5,zorder = 3)
    
        # annotate arrow
        if func(1) > 0.1:
            ax2.annotate('$b$', xy=(2, 1), xytext=(func(1.3),0),fontsize=15
            )
        elif func(1) < -0.1:
            ax2.annotate('$b$', xy=(2, 1), xytext=(func(1.5),0),fontsize=15
            )
            
    # set viewing limits
    wgap = (max(w) - min(w))*0.3
    ax2.set_xlim([-5,5])
    ax2.set_ylim([-5,5])

    # plot x and y axes, and clean up
    ax2.grid(True, which='both')
    #ax2.axhline(y=0, color='k', linewidth=1)
    #ax2.axvline(x=0, color='k', linewidth=1)
        
    # label plot
    ax2.set_xlabel('$w$',fontsize = 15)
    ax2.set_ylabel('$g(w)$',fontsize = 15,rotation = 0,labelpad = 20)
    plt.show()
    
# custom plot for spiffing up plot of a single mathematical function
def visualize3d(func1,func2,func3,**kwargs):
    # define input space
    w = np.linspace(-2,2,200)

    if 'w' in kwargs:
        w = kwargs['w']
    guides = 'on'
    if 'guides' in kwargs:
        guides = kwargs['guides']
        
    view = [20,20]
    if 'view' in kwargs:
        view = kwargs['view']
        
    # create mesh
    w1_vals,w2_vals = np.meshgrid(w,w)
    w1_vals.shape = (len(w)**2,1)
    w2_vals.shape = (len(w)**2,1)
    g_vals1 = func1([w1_vals,w2_vals])
    g_vals2 = func2([w1_vals,w2_vals])
    g_vals3 = func3([w1_vals,w2_vals])

    # vals for cost surface
    w1_vals.shape = (len(w),len(w))
    w2_vals.shape = (len(w),len(w))
    g_vals1.shape = (len(w),len(w))
    g_vals2.shape = (len(w),len(w))
    g_vals3.shape = (len(w),len(w))
       
    # construct figure
    fig = plt.figure(figsize = (9,4),edgecolor = 'k')

    # create subplot with 3 panels, plot input function in center plot
    gs = gridspec.GridSpec(1, 3, width_ratios=[1,1,1]) 
    ax1 = plt.subplot(gs[0],projection='3d'); 
    ax2 = plt.subplot(gs[1],projection='3d')
    ax3 = plt.subplot(gs[2],projection='3d')
    
    for i in range(3):
        ax = 0
        func = 0
        g_vals = 0
        if i == 0:
            ax = ax1
            func = func1
            g_vals = g_vals1
        if i == 1:
            ax = ax2
            func = func2
            g_vals = g_vals2
        if i == 2:
            ax = ax3
            func = func3
            g_vals = g_vals3
            
        # add arrow for slope visualization
        s = func([1,0]) - func([0,0])
        t = func([0,1]) - func([0,0])

        # plot coordinate-wise slope vector
        if abs(s) > 0.5:
            # draw arrow
            a = Arrow3D([0, s], [0, 0], [0, 0], mutation_scale=20,
                    lw=2, arrowstyle="-|>", color="b")
            ax.add_artist(a)
            
            # label arrow
            s = func([1.5,0]) - func([0,0])
            annotate3D(ax, s='$(b_1,0)$', xyz=[s,0,0], fontsize=14, xytext=(-3,3),
               textcoords='offset points', ha='center',va='center') 
            
        # draw negative coordinate-wise slope vector        
        if abs(s) > 0.5 and abs(t) < 0.5:
            a = Arrow3D([0, - (func([1,0])- func([0,0]))], [0, - (func([0,1])- func([0,0]))], [0, 0], mutation_scale=20,
                        lw=2, arrowstyle="-|>", color="r")
            ax.add_artist(a)  
            an = 1.2
            s = - (func([an+0.2,0]) - func([0,0]))
            t = - (func([0,an+0.2]) - func([0,0]))
            name = '$-(b_1,0)$'
            annotate3D(ax, s=name, xyz=[s,t,0], fontsize=14, xytext=(-3,3),textcoords='offset points', ha='center',va='center') 
 
        # plot coordinate-wise slope vector
        if abs(t) > 0.5:
            # draw arrow
            a = Arrow3D([0, 0], [0, t], [0, 0], mutation_scale=20,
                    lw=2, arrowstyle="-|>", color="b")
            ax.add_artist(a)  
            
            # label arrow
            t = func([0,1.5]) - func([0,0])
            annotate3D(ax, s='$(0,b_2)$', xyz=[0,t,0], fontsize=14, xytext=(-3,3),
               textcoords='offset points', ha='center',va='center') 
            
        # draw negative coordinate-wise slope vector        
        if abs(t) > 0.5 and abs(s) < 0.5:
            a = Arrow3D([0, - (func([1,0])- func([0,0]))], [0, - (func([0,1])- func([0,0]))], [0, 0], mutation_scale=20,
                        lw=2, arrowstyle="-|>", color="r")
            ax.add_artist(a)  
            an = 1.2
            s = - (func([an+0.2,0]) - func([0,0]))
            t = - (func([0,an+0.2]) - func([0,0]))
            name = '$-(0,b_2)$'
            annotate3D(ax, s=name, xyz=[s,t,0], fontsize=14, xytext=(-3,3),textcoords='offset points', ha='center',va='center') 
         
                
        # full gradient
        if abs(s) > 0.5 and abs(t) > 0.5:
            a = Arrow3D([0, func([1,0])- func([0,0])], [0, func([0,1])- func([0,0])], [0, 0], mutation_scale=20,
                    lw=2, arrowstyle="-|>", color="k")
            ax.add_artist(a)  
            
            b = func([1.2,0]) - func([0,0])
            c = func([0,1.2]) - func([0,0])
            annotate3D(ax, s='$(b_1,b_2)$', xyz=[b,c,0], fontsize=14, xytext=(-3,3),
               textcoords='offset points', ha='center',va='center') 
    
        # full negative gradient
        if abs(s) > 0.5 and abs(t) > 0.5:
            a = Arrow3D([0, - (func([1,0])- func([0,0]))], [0, - (func([0,1])- func([0,0]))], [0, 0], mutation_scale=20,
                        lw=2, arrowstyle="-|>", color="r")
            ax.add_artist(a)  
            an = 1
            s = - (func([an+0.2,0]) - func([0,0]))
            t = - (func([0,an+0.2]) - func([0,0]))
            name = '$-(b_1,b_2)$'
            annotate3D(ax, s=name, xyz=[s,t,0], fontsize=14, xytext=(-3,3),textcoords='offset points', ha='center',va='center') 
 
        # plot function        
        ax.plot_surface(w1_vals, w2_vals, g_vals, alpha = 0.3,color = 'lime',rstride=25, cstride=25,linewidth=0.5,edgecolor = 'k',zorder = 2)

        ax.plot_surface(w1_vals, w2_vals, g_vals*0, alpha = 0.1,color = 'w',zorder = 1,rstride=25, cstride=25,linewidth=0.3,edgecolor = 'k') 

        # plot x and y axes, and clean up
        ax.grid(False)
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
        y = 2.3
        ax.set_xlim([-y,y])
        ax.set_ylim([-y,y])
        s = np.min(np.min(g_vals))
        t = np.max(np.max(g_vals))
        ax.set_zlim([s,t])
        
        # label plot
        fontsize = 15
        ax.set_xlabel(r'$w_1$',fontsize = fontsize,labelpad = -10)
        ax.set_ylabel(r'$w_2$',fontsize = fontsize,rotation = 0,labelpad=-10)
        
        # remove whitespace from figure
        fig.subplots_adjust(left=0, right=1, bottom=0, top=1) # remove whitespace
        fig.subplots_adjust(wspace=0.01,hspace=0.01)

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