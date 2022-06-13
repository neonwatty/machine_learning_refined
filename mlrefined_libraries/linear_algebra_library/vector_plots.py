import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from matplotlib import gridspec
from IPython.display import display, HTML
import copy
import math

# simple plot of 2d vector addition / paralellagram law
def single_plot(vec1,**kwargs): 
    guides = False
    if 'guides' in kwargs:
        guides = kwargs['guides']
    
    # create figure
    fig = plt.figure(figsize = (12,4))

    # create subplot with 2 panels
    gs = gridspec.GridSpec(1, 2, width_ratios=[1, 1]) 
    ax1 = plt.subplot(gs[0]); 
    ax2 = plt.subplot(gs[1]);

    ### plot point in left panel
    ax1.scatter(vec1[0],vec1[1],c = 'k',edgecolor = 'w',s = 50,linewidth = 1)
    
    ### plot arrow in right panel
    head_length = 0.4
    veclen = math.sqrt(vec1[0]**2 + vec1[1]**2)
    vec1_orig = copy.deepcopy(vec1)
    vec1 = (veclen - head_length)/veclen*vec1

    ax2.arrow(0, 0, vec1[0],vec1[1], head_width=0.25, head_length=head_length, fc='k', ec='k',linewidth=2,zorder = 3)

    # draw guides?
    if guides == True:
        ax1.plot([0,vec1_orig[0]],[vec1_orig[1],vec1_orig[1]],linestyle= '--',c='b',zorder=2,linewidth = 0.75)
        ax1.plot([vec1_orig[0],vec1_orig[0]],[0,vec1_orig[1]],linestyle= '--',c='b',zorder=2,linewidth = 0.75)
        ax2.plot([0,vec1_orig[0]],[vec1_orig[1],vec1_orig[1]],linestyle= '--',c='b',zorder=2,linewidth = 0.75)
        ax2.plot([vec1_orig[0],vec1_orig[0]],[0,vec1_orig[1]],linestyle= '--',c='b',zorder=2,linewidth = 0.75)

    # plot x and y axes, and clean up
    ax1.grid(True, which='both')
    ax1.axhline(y=0, color='k', linewidth=1,zorder = 1)
    ax1.axvline(x=0, color='k', linewidth=1,zorder = 1)
    
    ax2.grid(True, which='both')
    ax2.axhline(y=0, color='k', linewidth=1,zorder = 1)
    ax2.axvline(x=0, color='k', linewidth=1,zorder = 1)
        
    # set viewing limits
    xmax = max(vec1[0],0)
    xmin = min(vec1[0],0)
    xgap = (xmax - xmin)*0.3
    xmax = xmax + xgap
    xmin = xmin - xgap

    ymax = max(vec1[1],0)
    ymin = min(vec1[1],0)
    ygap = (ymax - ymin)*0.3
    ymax = ymax + ygap
    ymin = ymin - ygap
    ax1.set_xlim([xmin,xmax])
    ax1.set_ylim([ymin,ymax])
    ax2.set_xlim([xmin,xmax])
    ax2.set_ylim([ymin,ymax])
    
    # renderer
    plt.style.use('ggplot')
    plt.show()

# simple plot of 2d vector addition / paralellagram law
def vector_add_plot(vec1,vec2):     
    # renderer
    plt.style.use('ggplot')
    
    fig = plt.figure(figsize = (12,4))

    # create subplot with 3 panels, plot input function in center plot
    gs = gridspec.GridSpec(1, 3, width_ratios=[1,3, 1]) 
    ax1 = plt.subplot(gs[0]); ax1.axis('off');
    ax3 = plt.subplot(gs[2]); ax3.axis('off');

    # plot input function
    ax2 = plt.subplot(gs[1])

    # plot each vector
    head_length = 0.4
    veclen = math.sqrt(vec1[0]**2 + vec1[1]**2)
    vec1_orig = copy.deepcopy(vec1)
    vec1 = (veclen - head_length)/veclen*vec1
    veclen = math.sqrt(vec2[0]**2 + vec2[1]**2)
    vec2_orig = copy.deepcopy(vec2)
    vec2 = (veclen - head_length)/veclen*vec2
    ax2.arrow(0, 0, vec1[0],vec1[1], head_width=0.25, head_length=head_length, fc='k', ec='k',linewidth=2,zorder = 3)
    ax2.arrow(0, 0, vec2[0],vec2[1], head_width=0.25, head_length=head_length, fc='k', ec='k',linewidth=2,zorder = 3)
    
    # plot the sum of the two vectors
    vec3 = vec1_orig + vec2_orig
    vec3_orig = copy.deepcopy(vec3)
    veclen = math.sqrt(vec3[0]**2 + vec3[1]**2)
    vec3 = (veclen - head_length)/veclen*vec3
    ax2.arrow(0, 0, vec3[0],vec3[1], head_width=0.25, head_length=head_length, fc='r', ec='r',linewidth=2,zorder=3)
    
    # connect them
    ax2.plot([vec1_orig[0],vec3_orig[0]],[vec1_orig[1],vec3_orig[1]],linestyle= '--',c='b',zorder=2,linewidth = 0.75)
    ax2.plot([vec2_orig[0],vec3_orig[0]],[vec2_orig[1],vec3_orig[1]],linestyle= '--',c='b',zorder=2,linewidth = 0.75)

    # plot x and y axes, and clean up
    ax2.grid(True, which='both')
    ax2.axhline(y=0, color='k', linewidth=1,zorder = 1)
    ax2.axvline(x=0, color='k', linewidth=1,zorder = 1)
        
    # set viewing limits
    xmax = max(vec1[0],vec2[0],vec3[0],0)
    xmin = min(vec1[0],vec2[0],vec3[0],0)
    xgap = (xmax - xmin)*0.3
    xmax = xmax + xgap
    xmin = xmin - xgap

    ymax = max(vec1[1],vec2[1],vec3[1],0)
    ymin = min(vec1[1],vec2[1],vec3[1],0)
    ygap = (ymax - ymin)*0.3
    ymax = ymax + ygap
    ymin = ymin - ygap
    ax2.set_xlim([xmin,xmax])
    ax2.set_ylim([ymin,ymax])

    plt.show()
    
    
# simple plot of 2d vector linear combination / paralellagram law
def vector_linear_combination_plot(vec1, vec2, alpha1, alpha2):     
    # renderer
    fig = plt.figure(figsize = (12,4))

    # create subplot with 3 panels, plot input function in center plot
    gs = gridspec.GridSpec(1, 3, width_ratios=[1,3, 1]) 
    ax1 = plt.subplot(gs[0]); ax1.axis('off');
    ax3 = plt.subplot(gs[2]); ax3.axis('off');

    # plot input function
    ax2 = plt.subplot(gs[1])

    # plot each vector
    head_length = 0.4
    veclen = math.sqrt(vec1[0]**2 + vec1[1]**2)
    vec1_orig = copy.deepcopy(vec1)
    vec1 = (veclen - head_length)/veclen*vec1
    veclen = math.sqrt(vec2[0]**2 + vec2[1]**2)
    vec2_orig = copy.deepcopy(vec2)
    vec2 = (veclen - head_length)/veclen*vec2
    ax2.arrow(0, 0, vec1[0],vec1[1], head_width=0.25, head_length=head_length, fc='k', ec='k',linewidth=2,zorder = 3)
    ax2.arrow(0, 0, vec2[0],vec2[1], head_width=0.25, head_length=head_length, fc='k', ec='k',linewidth=2,zorder = 3)
    
    # plot the linear combination of the two vectors
    vec3 = alpha1*vec1_orig + alpha2*vec2_orig
    vec3_orig = copy.deepcopy(vec3)
    veclen = math.sqrt(vec3[0]**2 + vec3[1]**2)
    vec3 = (veclen - head_length)/veclen*vec3
    ax2.arrow(0, 0, vec3[0],vec3[1], head_width=0.25, head_length=head_length, fc='r', ec='r',linewidth=2,zorder=3)
    
    # connect them
    ax2.plot([vec1_orig[0],alpha1*vec1_orig[0]],[vec1_orig[1],alpha1*vec1_orig[1]],linestyle= '--',c='k',zorder=2,linewidth = 0.75)
    ax2.plot([vec2_orig[0],alpha2*vec2_orig[0]],[vec2_orig[1],alpha2*vec2_orig[1]],linestyle= '--',c='k',zorder=2,linewidth = 0.75)
    
    # connect them
    ax2.plot([alpha1*vec1_orig[0],vec3_orig[0]],[alpha1*vec1_orig[1],vec3_orig[1]],linestyle= '--',c='b',zorder=2,linewidth = 0.75)
    ax2.plot([alpha2*vec2_orig[0],vec3_orig[0]],[alpha2*vec2_orig[1],vec3_orig[1]],linestyle= '--',c='b',zorder=2,linewidth = 0.75)

    # plot x and y axes, and clean up
    ax2.grid(True, which='both')
    ax2.axhline(y=0, color='k', linewidth=1,zorder = 1)
    ax2.axvline(x=0, color='k', linewidth=1,zorder = 1)
        
    # set viewing limits
    xmax = max(vec1[0], alpha1*vec1[0], vec2[0], alpha2*vec2[0], vec3[0], 0)
    xmin = min(vec1[0], alpha1*vec1[0], vec2[0], alpha2*vec2[0], vec3[0], 0)
    xgap = (xmax - xmin)*0.3
    xmax = xmax + xgap
    xmin = xmin - xgap

    ymax = max(vec1[1],vec2[1],vec3[1],0)
    ymin = min(vec1[1],vec2[1],vec3[1],0)
    ygap = (ymax - ymin)*0.3
    ymax = ymax + ygap
    ymin = ymin - ygap
    ax2.set_xlim([xmin,xmax])
    ax2.set_ylim([ymin,ymax])

    plt.show()    
    