
import matplotlib.pyplot as plt
from matplotlib import gridspec
from autograd import grad as compute_grad   # The only autograd function you may ever need
import autograd.numpy as np
import copy

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
    ind = np.unique(np.array(ind))
        
    # plot the input/output tangency points and tangent line
    wtan = np.linspace(-1,1,500)                  # input range for original function
    for pt in ind:
        # plot point
        w_val = w[pt]
        g_val = func(w_val)
        grad_val = grad(w_val)
        ax2.scatter(w_val,g_val,s = 40,c = 'lime',edgecolor = 'k',linewidth = 2,zorder = 3)            # plot point of tangency
    plt.show()
    
    
    