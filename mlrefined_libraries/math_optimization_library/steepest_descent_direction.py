# import standard plotting and animation
import matplotlib.pyplot as plt
from matplotlib import gridspec
import matplotlib.animation as animation
import math
import numpy as np
from IPython.display import clear_output
import time

'''
A collection of animations illustrating the steepest descent direction under the L2, L1, and Linfinity norms.
'''
def L2(pt,num_frames,savepath,**kwargs):
    # initialize figure
    fig = plt.figure(figsize = (16,8))
    artist = fig

    # create subplot with 3 panels, plot input function in center plot
    gs = gridspec.GridSpec(1, 2, width_ratios=[1,1],wspace=0.3, hspace=0.05) 
    ax1 = plt.subplot(gs[0],aspect ='equal');
    ax2 = plt.subplot(gs[1],aspect = 'equal'); 

    # create dataset for unit circle
    v = np.linspace(0,2*np.pi,1000)    
    s = np.sin(v)
    s.shape = (len(s),1)
    t = np.cos(v)
    t.shape = (len(t),1)
    
    # create span of angles to plot over
    v = np.linspace(0,2*np.pi,num_frames)
    a = np.arccos(pt[0]/(pt[0]**2 + pt[1]**2)**(0.5)) + np.pi
    v = np.append(v,a)
    v = np.sort(v)
    v = np.unique(v)
    y = np.sin(v)
    x = np.cos(v)
    
    # create inner product plot
    obj = [(x[s]*pt[0] + y[s]*pt[1]) for s in range(len(v))]
    ind_min = np.argmin(obj)
    
    # rescale directions for plotting with arrows
    x = 0.96*x
    y = 0.96*y
    
    # create linspace for left panel
    w = np.linspace(0,2*np.pi,300)
            
    
    # print update
    num_frames = len(v)
    print ('starting animation rendering...')
    # animation sub-function
    def animate(k):
        # clear panels for next slide
        ax1.cla()
        ax2.cla()
        
        # print rendering update
        if np.mod(k+1,25) == 0:
            print ('rendering animation frame ' + str(k+1) + ' of ' + str(num_frames))
        if k == num_frames - 1:
            print ('animation rendering complete!')
            time.sleep(1.5)
            clear_output()
           
        ### setup left panel ###
        # plot circle with lines in left panel
        ax1.plot(s,t,color = 'k',linewidth = 3)
        
        # plot rotation arrow
        ax1.arrow(0, 0, x[k], y[k], head_width=0.1, head_length=0.1, fc='k', ec='k',linewidth=3,zorder = 3,length_includes_head = True)

        # plot input point as arrow
        ax1.arrow(0, 0, pt[0], pt[1], head_width=0.1, head_length=0.1, fc='r', ec='r',linewidth=3,zorder = 3,length_includes_head = True)
        ax1.arrow(0, 0, pt[0], pt[1], head_width=0.11, head_length=0.1, fc='k', ec='k',linewidth=5,zorder = 2,length_includes_head = True)

        # clean up panel
        ax1.grid(True, which='both')
        ax1.axhline(y=0, color='k')
        ax1.axvline(x=0, color='k')
        ax1.set_xlim([-1.5,1.5])
        ax1.set_ylim([-1.5,1.5])
        
        ### setup right panel ###
        current_angle = v[k]
        ind = np.argmin(np.abs(w - current_angle))
        p = w[:ind+1]
        
        # plot normalized objective thus far
        ax2.plot(v[:k+1],obj[:k+1],color ='k',linewidth=4,zorder = 2)
        
        # if we have reached the minimum plot it on all slides going further
        if k >= ind_min:
            # plot direction
            ax1.arrow(0, 0, x[ind_min], y[ind_min], head_width=0.1, head_length=0.1, fc='lime', ec='lime',linewidth=3,zorder = 3,length_includes_head = True)
            ax1.arrow(0, 0, x[ind_min], y[ind_min], head_width=0.11, head_length=0.1, fc='k', ec='k',linewidth=5,zorder = 2,length_includes_head = True)
        
            # mark objective plot
            ax2.scatter(v[ind_min],obj[ind_min],color ='lime',s = 200,linewidth = 1,edgecolor = 'k',zorder = 3)
        
        # cleanup plot
        ax2.grid(True, which='both')
        ax2.axhline(y=0, color='k')
        ax2.axvline(x=0, color='k')   
        ax2.set_xlim([-0.1,2*np.pi + 0.1])
        ax2.set_ylim([min(obj) - 0.2,max(obj) + 0.2])
        
        # add legend
        ax2.legend([r'$\nabla g(\mathbf{v})^T \mathbf{d}$'],loc='center left', bbox_to_anchor=(0.13, 1.05),fontsize=18,ncol=2)

        return artist,

    anim = animation.FuncAnimation(fig, animate ,frames=num_frames, interval=num_frames, blit=True)

    # produce animation and save
    fps = 50
    if 'fps' in kwargs:
        fps = kwargs['fps']
    anim.save(savepath, fps=fps, extra_args=['-vcodec', 'libx264'])
    clear_output()

def L1(pt,num_frames,savepath,**kwargs):
    # initialize figure
    fig = plt.figure(figsize = (16,8))
    artist = fig

    # create subplot with 3 panels, plot input function in center plot
    gs = gridspec.GridSpec(1, 2, width_ratios=[1,1],wspace=0.3, hspace=0.05) 
    ax1 = plt.subplot(gs[0],aspect ='equal');
    ax2 = plt.subplot(gs[1],aspect = 'equal'); 

    # create dataset for unit diamond
    v = np.linspace(0,2*np.pi,2000)
    s = np.sin(v)
    s.shape = (len(s),1)
    t = np.cos(v)
    t.shape = (len(t),1)
    diamond = np.concatenate((s,t),axis=1)
    news = []
    for a in diamond:
        a = a/np.sum(abs(a))
        news.append(a)
    news = np.asarray(news)
    s = news[:,0]
    t = news[:,1]
    
    ### create span of angles to plot over
    v = np.linspace(0,2*np.pi,num_frames)
    
    # make sure corners of the diamond are included
    v = np.append(v,np.pi*0.5)
    v = np.append(v,np.pi)
    v = np.append(v,3/float(2)*np.pi)
    v = np.sort(v)
    v = np.unique(v)
    y = np.sin(v)
    x = np.cos(v)
    
    # make l2 ball
    x.shape = (len(x),1)
    y.shape = (len(y),1)
    l2 = np.concatenate((x,y),axis=1)
        
    # make l1 ball 
    l1 = []
    for a in l2:
        a = a/np.sum(abs(a))
        l1.append(a)
    l1 = np.asarray(l1)
    x = l1[:,0]
    y = l1[:,1]
    
    # create inner product plot
    obj = [(x[s]*pt[0] + y[s]*pt[1]) for s in range(len(v))]
    ind_min = np.argmin(obj)
    
    # rescale directions for plotting with arrows
    x = 0.96*x
    y = 0.96*y
    
    # create linspace for left panel
    w = np.linspace(0,2*np.pi,300)
    pt = [0.975*a for a in pt]

    # print update
    num_frames = len(v)
    print ('starting animation rendering...')
    # animation sub-function
    def animate(k):
        # clear panels for next slide
        ax1.cla()
        ax2.cla()
        
        # print rendering update
        if np.mod(k+1,25) == 0:
            print ('rendering animation frame ' + str(k+1) + ' of ' + str(num_frames))
        if k == num_frames - 1:
            print ('animation rendering complete!')
            time.sleep(1.5)
            clear_output()
           
        ### setup left panel ###
        # plot circle with lines in left panel
        ax1.plot(s,t,color = 'k',linewidth = 3)
        
        # plot rotation arrow
        ax1.arrow(0, 0, x[k], y[k], head_width=0.1, head_length=0.1, fc='k', ec='k',linewidth=3,zorder = 3,length_includes_head = True)

        # plot input point as arrow
        ax1.arrow(0, 0, pt[0], pt[1], head_width=0.1, head_length=0.1, fc='r', ec='r',linewidth=3,zorder = 3,length_includes_head = True)
        ax1.arrow(0, 0, pt[0], pt[1], head_width=0.11, head_length=0.1, fc='k', ec='k',linewidth=5,zorder = 2,length_includes_head = True)

        # clean up panel
        ax1.grid(True, which='both')
        ax1.axhline(y=0, color='k')
        ax1.axvline(x=0, color='k')
        ax1.set_xlim([-1.5,1.5])
        ax1.set_ylim([-1.5,1.5])
        
        ### setup right panel ###
        current_angle = v[k]
        ind = np.argmin(np.abs(w - current_angle))
        p = w[:ind+1]
        
        # plot normalized objective thus far
        ax2.plot(v[:k+1],obj[:k+1],color ='k',linewidth=4,zorder = 2)
        
        # if we have reached the minimum plot it on all slides going further
        if k >= ind_min:
            # plot direction
            ax1.arrow(0, 0, x[ind_min], y[ind_min], head_width=0.1, head_length=0.1, fc='lime', ec='lime',linewidth=3,zorder = 3,length_includes_head = True)
            ax1.arrow(0, 0, x[ind_min], y[ind_min], head_width=0.11, head_length=0.1, fc='k', ec='k',linewidth=5,zorder = 2,length_includes_head = True)
        
            # mark objective plot
            ax2.scatter(v[ind_min],obj[ind_min],color ='lime',s = 200,linewidth = 1,edgecolor = 'k',zorder = 3)
        
        # cleanup plot
        ax2.grid(True, which='both')
        ax2.axhline(y=0, color='k')
        ax2.axvline(x=0, color='k')   
        ax2.set_xlim([-0.1,2*np.pi + 0.1])
        ax2.set_ylim([min(obj) - 0.2,max(obj) + 0.2])
        
        # add legend
        ax2.legend([r'$\nabla g(\mathbf{v})^T \mathbf{d}$'],loc='center left', bbox_to_anchor=(0.13, 1.05),fontsize=18,ncol=2)

        return artist,

    anim = animation.FuncAnimation(fig, animate ,frames=num_frames, interval=num_frames, blit=True)

    # produce animation and save
    fps = 50
    if 'fps' in kwargs:
        fps = kwargs['fps']
    anim.save(savepath, fps=fps, extra_args=['-vcodec', 'libx264'])
    clear_output()

def Linf(pt,num_frames,savepath,**kwargs):
    # initialize figure
    fig = plt.figure(figsize = (16,8))
    artist = fig

    # create subplot with 3 panels, plot input function in center plot
    gs = gridspec.GridSpec(1, 2, width_ratios=[1,1],wspace=0.3, hspace=0.05) 
    ax1 = plt.subplot(gs[0],aspect ='equal');
    ax2 = plt.subplot(gs[1],aspect = 'equal'); 

    # create dataset for unit square
    v = np.linspace(0,2*np.pi,2000)
    s = np.sin(v)
    s.shape = (len(s),1)
    t = np.cos(v)
    t.shape = (len(t),1)
    square = np.concatenate((s,t),axis=1)
    news = []
    for a in square:
        a = a/np.max(abs(a))
        news.append(a)
    news = np.asarray(news)
    s = news[:,0]
    t = news[:,1]
    
    ### create span of angles to plot over
    v = np.linspace(0,2*np.pi,num_frames)
    
    # make sure corners of the square are included
    v = np.append(v,np.pi/float(4))
    v = np.append(v,np.pi*3/float(4))
    v = np.append(v,np.pi*5/float(4))
    v = np.append(v,np.pi*7/float(4))
    v = np.sort(v)
    v = np.unique(v)
    y = np.sin(v)
    x = np.cos(v)
    
    # make l2 ball
    x.shape = (len(x),1)
    y.shape = (len(y),1)
    l2 = np.concatenate((x,y),axis=1)
        
    # make Linf ball
    linf = []
    for a in l2:
        a = a/np.max(abs(a))
        linf.append(a)
    linf = np.asarray(linf) 
    x = linf[:,0]
    y = linf[:,1]
    
    # create inner product plot
    obj = [(x[s]*pt[0] + y[s]*pt[1]) for s in range(len(v))]
    ind_min = np.argmin(obj)
    
    # rescale directions for plotting with arrows
    x = 0.96*x
    y = 0.96*y
    
    # create linspace for left panel
    w = np.linspace(0,2*np.pi,300)
    pt = [0.975*a for a in pt]

    # print update
    num_frames = len(v)
    print ('starting animation rendering...')
    # animation sub-function
    def animate(k):
        # clear panels for next slide
        ax1.cla()
        ax2.cla()
        
        # print rendering update
        if np.mod(k+1,25) == 0:
            print ('rendering animation frame ' + str(k+1) + ' of ' + str(num_frames))
        if k == num_frames - 1:
            print ('animation rendering complete!')
            time.sleep(1.5)
            clear_output()
           
        ### setup left panel ###
        # plot circle with lines in left panel
        ax1.plot(s,t,color = 'k',linewidth = 3)
        
        # plot rotation arrow
        ax1.arrow(0, 0, x[k], y[k], head_width=0.1, head_length=0.1, fc='k', ec='k',linewidth=3,zorder = 3,length_includes_head = True)

        # plot input point as arrow
        ax1.arrow(0, 0, pt[0], pt[1], head_width=0.1, head_length=0.1, fc='r', ec='r',linewidth=3,zorder = 3,length_includes_head = True)
        ax1.arrow(0, 0, pt[0], pt[1], head_width=0.11, head_length=0.1, fc='k', ec='k',linewidth=5,zorder = 2,length_includes_head = True)

        # clean up panel
        ax1.grid(True, which='both')
        ax1.axhline(y=0, color='k')
        ax1.axvline(x=0, color='k')
        ax1.set_xlim([-1.5,1.5])
        ax1.set_ylim([-1.5,1.5])
        
        ### setup right panel ###
        current_angle = v[k]
        ind = np.argmin(np.abs(w - current_angle))
        p = w[:ind+1]
        
        # plot normalized objective thus far
        ax2.plot(v[:k+1],obj[:k+1],color ='k',linewidth=4,zorder = 2)
        
        # if we have reached the minimum plot it on all slides going further
        if k >= ind_min:
            # plot direction
            ax1.arrow(0, 0, x[ind_min], y[ind_min], head_width=0.1, head_length=0.1, fc='lime', ec='lime',linewidth=3,zorder = 3,length_includes_head = True)
            ax1.arrow(0, 0, x[ind_min], y[ind_min], head_width=0.11, head_length=0.1, fc='k', ec='k',linewidth=5,zorder = 2,length_includes_head = True)
        
            # mark objective plot
            ax2.scatter(v[ind_min],obj[ind_min],color ='lime',s = 200,linewidth = 1,edgecolor = 'k',zorder = 3)
        
        # cleanup plot
        ax2.grid(True, which='both')
        ax2.axhline(y=0, color='k')
        ax2.axvline(x=0, color='k')   
        ax2.set_xlim([-0.1,2*np.pi + 0.1])
        ax2.set_ylim([min(obj) - 0.2,max(obj) + 0.2])
        
        # add legend
        ax2.legend([r'$\nabla g(\mathbf{v})^T \mathbf{d}$'],loc='center left', bbox_to_anchor=(0.13, 1.05),fontsize=18,ncol=2)

        return artist,

    anim = animation.FuncAnimation(fig, animate ,frames=num_frames, interval=num_frames, blit=True)

    # produce animation and save
    fps = 50
    if 'fps' in kwargs:
        fps = kwargs['fps']
    anim.save(savepath, fps=fps, extra_args=['-vcodec', 'libx264'])
    clear_output()