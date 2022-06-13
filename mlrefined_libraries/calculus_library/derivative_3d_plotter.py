import sys
sys.path.append('../')
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from matplotlib import gridspec
from IPython.display import display, HTML
import copy
import autograd.numpy as np
from autograd import jacobian
from autograd import grad

# a short function for plotting function and derivative values over a large range for input function g
def autograd_3d_derval_plot(g,**kwargs):
    # use autograd to compute gradient
    gradient = grad(g)
    
    # specify range of input for our function and its derivative
    plot_size = 20
    if 'plot_size' in kwargs:
        plot_size = kwargs['plot_size']
    w = np.linspace(-1,1,plot_size) 
    if 'w' in kwargs:
        w = kwargs['w']
        
    # determine vertical plotting limit
    xx,yy = np.meshgrid(w,w)
    xx.shape = (1,xx.size)
    yy.shape = (1,yy.size)
    h = np.vstack((xx,yy))

    # compute cost func and gradient values
    vals = g(h)    
    grad_vals = np.array([gradient(v) for v in h.T])
    ders1 = grad_vals[:,0]
    ders2 = grad_vals[:,1]
        
    # re-shape everything
    xx.shape = (plot_size,plot_size)
    yy.shape = (plot_size,plot_size)
    vals.shape = (plot_size,plot_size)
    ders1.shape = (plot_size,plot_size)
    ders2.shape = (plot_size,plot_size)
    
     # plot the functions 
    fig = plt.figure(figsize = (9,4))
    
    # create subplot with 3 panels, plot input function in center plot
    gs = gridspec.GridSpec(1, 3) 
    ax1 = plt.subplot(gs[0],projection='3d'); ax1.axis('off')
    ax2 = plt.subplot(gs[1],projection='3d'); ax2.axis('off')
    ax3 = plt.subplot(gs[2],projection='3d'); ax3.axis('off')
    color = 'r'

    # plot surfaces
    ax1.plot_surface(xx, yy, vals, alpha = 0.2,color = color,rstride=2, cstride=2,linewidth=2,edgecolor = 'k')
    ax2.plot_surface(xx, yy, ders1, alpha = 0.2,color = color,rstride=2, cstride=2,linewidth=2,edgecolor = 'k')
    ax3.plot_surface(xx, yy, ders2, alpha = 0.2,color = color,rstride=2, cstride=2,linewidth=2,edgecolor = 'k')
    
    # titles
    ax1.set_title(r'$g$',fontsize = 20)
    ax2.set_title(r'$\frac{\mathrm{d}}{\mathrm{d}w_1}g$',fontsize = 20)
    ax3.set_title(r'$\frac{\mathrm{d}}{\mathrm{d}w_2}g$',fontsize = 20)

    plt.show()
    
# a short function for plotting function and derivative values over a large range for input function g
def ad_3d_derval_plot(MyTuple,g,**kwargs):
    # specify range of input for our function and its derivative
    plot_size = 20
    if 'plot_size' in kwargs:
        plot_size = kwargs['plot_size']
    w = np.linspace(-1,1,plot_size) 
    if 'w' in kwargs:
        w = kwargs['w']
        
    # determine vertical plotting limit
    xx,yy = np.meshgrid(w,w)
    xx.shape = (xx.size,1)
    yy.shape = (yy.size,1)

    # initialize objects
    vals = []
    ders1 = []
    ders2 = []
    for i in range(xx.size):
        u = xx[i]; v = yy[i];
        
        w_1 = MyTuple(val = u,der = np.array([1,0]))
        w_2 = MyTuple(val = v,der = np.array([0,1]))   
        
        s = g(w_1,w_2)
        
        # extract val and der values
        val = s.val
        der = s.der
        vals.append(val)
        ders1.append(der[0])
        ders2.append(der[1])

    # array-afy all output lists
    vals = np.array(vals)
    ders1 = np.array(ders1)
    ders2 = np.array(ders2)
    
    # re-shape everything
    xx.shape = (plot_size,plot_size)
    yy.shape = (plot_size,plot_size)
    vals.shape = (plot_size,plot_size)
    ders1.shape = (plot_size,plot_size)
    ders2.shape = (plot_size,plot_size)
    
     # plot the functions 
    fig = plt.figure(figsize = (9,4))
    
    # create subplot with 3 panels, plot input function in center plot
    gs = gridspec.GridSpec(1, 3) 
    ax1 = plt.subplot(gs[0],projection='3d'); ax1.axis('off')
    ax2 = plt.subplot(gs[1],projection='3d'); ax2.axis('off')
    ax3 = plt.subplot(gs[2],projection='3d'); ax3.axis('off')
    color = 'r'

    # plot function surfaces
    ax1.plot_surface(xx, yy, vals, alpha = 0.2,color = color,rstride=2, cstride=2,linewidth=2,edgecolor = 'k')
    ax2.plot_surface(xx, yy, ders1, alpha = 0.2,color = color,rstride=2, cstride=2,linewidth=2,edgecolor = 'k')
    ax3.plot_surface(xx, yy, ders2, alpha = 0.2,color = color,rstride=2, cstride=2,linewidth=2,edgecolor = 'k')
    
    # titles
    ax1.set_title(r'$g$',fontsize = 20)
    ax2.set_title(r'$\frac{\mathrm{d}}{\mathrm{d}w_1}g$',fontsize = 20)
    ax3.set_title(r'$\frac{\mathrm{d}}{\mathrm{d}w_2}g$',fontsize = 20)
    
    plt.show()
    
    