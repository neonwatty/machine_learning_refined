import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import axes3d
import matplotlib.pyplot as plt 
from autograd import numpy as np
    
# plot multi-output regression dataset where output dimension C = 2
def plot_data(x,y,view1,view2):    
    # construct panels
    fig = plt.figure(figsize = (9,4))
    ax0 = plt.subplot(121,projection='3d')
    ax0.view_init(view1[0],view1[1])
    ax0.axis('off')

    ax1 = plt.subplot(122,projection='3d')
    ax1.view_init(view2[0],view2[1])
    ax1.axis('off')

    # scatter plot data in each panel
    ax0.scatter(x[0,:],x[1,:],y[0,:],c='k',edgecolor = 'w',linewidth = 1,s=60)
    ax1.scatter(x[0,:],x[1,:],y[1,:],c='k',edgecolor = 'w',linewidth = 1,s=60)
    plt.show()
   
# plot multi-output regression dataset with fits provided by 'predictor'
def plot_regressions(x,y,predictor,view1,view2):        
    # import all the requisite libs
    # construct panels
    fig = plt.figure(figsize = (9,4))
    ax0 = plt.subplot(121,projection='3d')
    ax0.view_init(view1[0],view1[1])
    ax0.axis('off')

    ax1 = plt.subplot(122,projection='3d')
    ax1.view_init(view2[0],view2[1])
    ax1.axis('off')

    # scatter plot data in each panel
    ax0.scatter(x[0,:],x[1,:],y[0,:],c='k',edgecolor = 'w',linewidth = 1,s=60)
    ax1.scatter(x[0,:],x[1,:],y[1,:],c='k',edgecolor = 'w',linewidth = 1,s=60)

    # construct input for each model fit
    a_ = np.linspace(0,1,15)
    a,b = np.meshgrid(a_,a_)
    a = a.flatten()[np.newaxis,:]
    b = b.flatten()[np.newaxis,:]
    c = np.vstack((a,b))

    # evaluate model 
    p = predictor(c)
    m1 = p[0,:]
    m2 = p[1,:]

    # plot each as surface
    a.shape = (a_.size,a_.size)
    b.shape = (a_.size,a_.size)
    m1.shape = (a_.size,a_.size)
    m2.shape = (a_.size,a_.size)

    ax0.plot_surface(a,b,m1,alpha = 0.25,color = 'lime',cstride = 2,rstride = 2,linewidth = 1,edgecolor ='k')
    ax1.plot_surface(a,b,m2,alpha = 0.25,color = 'lime',cstride = 2,rstride = 2,linewidth = 1,edgecolor ='k')

    plt.show()