# import standard plotting and animation
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from IPython.display import clear_output
import time
from matplotlib import gridspec
from mpl_toolkits.mplot3d import proj3d
from mpl_toolkits.mplot3d.proj3d import proj_transform
from IPython.display import display

# import autograd functionality
from autograd import grad as compute_grad   # The only autograd function you may ever need
from autograd import hessian
import autograd.numpy as np
import math
import copy
        
'''
Plot a user-defined function taking in two inputs, along with first and second derivative functions
'''
def draw_it(func,**kwargs):
    view = [10,150]
    if 'view' in kwargs:
        view = kwargs['view']
    
    # generate input space for plotting
    w_in = np.linspace(-5,5,100)
    w1_vals, w2_vals = np.meshgrid(w_in,w_in)
    w1_vals.shape = (len(w_in)**2,1)
    w2_vals.shape = (len(w_in)**2,1)
    w_vals = np.concatenate((w1_vals,w2_vals),axis=1).T
    w1_vals.shape = (len(w_in),len(w_in))
    w2_vals.shape = (len(w_in),len(w_in))
    
    # compute grad vals
    grad = compute_grad(func)
    grad_vals = [grad(s) for s in w_vals.T]
    grad_vals = np.asarray(grad_vals)

    # compute hessian
    hess = hessian(func)
    hess_vals = [hess(s) for s in w_vals.T]
    
    # define figure
    fig = plt.figure(figsize = (9,6))

    ###  plot original function ###
    ax1 = plt.subplot2grid((3, 6), (0, 3), colspan=1,projection='3d')

    # evaluate function, reshape
    g_vals = func(w_vals)   
    g_vals.shape = (len(w_in),len(w_in))

    # plot function surface
    ax1.plot_surface(w1_vals, w2_vals, g_vals, alpha = 0.1,color = 'w',zorder = 1,rstride=15, cstride=15,linewidth=0.5,edgecolor = 'k') 
    ax1.set_title(r'$g(w_1,w_2)$',fontsize = 10)
    
    # cleanup axis
    cleanup(g_vals,view,ax1)
    
    ### plot first derivative functions ###
    ax2 = plt.subplot2grid((3, 6), (1, 2), colspan=1,projection='3d')
    ax3 = plt.subplot2grid((3, 6), (1, 4), colspan=1,projection='3d')

    # plot first function
    grad_vals1 = grad_vals[:,0]
    grad_vals1.shape = (len(w_in),len(w_in))
    ax2.plot_surface(w1_vals, w2_vals, grad_vals1, alpha = 0.1,color = 'w',zorder = 1,rstride=15, cstride=15,linewidth=0.5,edgecolor = 'k') 
    ax2.set_title(r'$\frac{\partial}{\partial w_1}g(w_1,w_2)$',fontsize = 10)
        
    # cleanup axis
    cleanup(grad_vals1,view,ax2)
    
    # plot second
    grad_vals1 = grad_vals[:,1]
    grad_vals1.shape = (len(w_in),len(w_in))
    ax3.plot_surface(w1_vals, w2_vals, grad_vals1, alpha = 0.1,color = 'w',zorder = 1,rstride=15, cstride=15,linewidth=0.5,edgecolor = 'k') 
    ax3.set_title(r'$\frac{\partial}{\partial w_2}g(w_1,w_2)$',fontsize = 10)

    # cleanup axis
    cleanup(grad_vals1,view,ax3)
    
    ### plot second derivatives ###
    ax4 = plt.subplot2grid((3, 6), (2, 1), colspan=1,projection='3d')
    ax5 = plt.subplot2grid((3, 6), (2, 3), colspan=1,projection='3d')
    ax6 = plt.subplot2grid((3, 6), (2, 5), colspan=1,projection='3d')

    # plot first hessian function
    hess_vals1 = np.asarray([s[0,0] for s in hess_vals])
    hess_vals1.shape = (len(w_in),len(w_in))
    ax4.plot_surface(w1_vals, w2_vals, hess_vals1, alpha = 0.1,color = 'w',zorder = 1,rstride=15, cstride=15,linewidth=0.5,edgecolor = 'k') 
    ax4.set_title(r'$\frac{\partial}{\partial w_1}\frac{\partial}{\partial w_1}g(w_1,w_2)$',fontsize = 10)

    # cleanup axis
    cleanup(hess_vals1,view,ax4)
    
    # plot second hessian function
    hess_vals1 = np.asarray([s[1,0] for s in hess_vals])
    hess_vals1.shape = (len(w_in),len(w_in))
    ax5.plot_surface(w1_vals, w2_vals, hess_vals1, alpha = 0.1,color = 'w',zorder = 1,rstride=15, cstride=15,linewidth=0.5,edgecolor = 'k') 
    ax5.set_title(r'$\frac{\partial}{\partial w_1}\frac{\partial}{\partial w_2}g(w_1,w_2)=\frac{\partial}{\partial w_2}\frac{\partial}{\partial w_1}g(w_1,w_2)$',fontsize = 10)
    
    # cleanup axis
    cleanup(hess_vals1,view,ax5)
    
    # plot first hessian function
    hess_vals1 = np.asarray([s[1,1] for s in hess_vals])
    hess_vals1.shape = (len(w_in),len(w_in))
    ax6.plot_surface(w1_vals, w2_vals, hess_vals1, alpha = 0.1,color = 'w',zorder = 1,rstride=15, cstride=15,linewidth=0.5,edgecolor = 'k') 
    ax6.set_title(r'$\frac{\partial}{\partial w_2}\frac{\partial}{\partial w_2}g(w_1,w_2)$',fontsize = 10)
    
    # cleanup axis
    cleanup(hess_vals1,view,ax6)
    plt.show()
    
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
    ax.plot(w_zplane,w_zplane*0,w_zplane*0,color = 'k',linewidth = 0.25)
    ax.plot(w_zplane*0,w_zplane,w_zplane*0,color = 'k',linewidth = 0.25)

    # remove axes lines and tickmarks
    #ax.w_zaxis.line.set_lw(0.)
    #ax.set_zticks([])
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
        