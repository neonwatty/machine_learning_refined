import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from ipywidgets import interact
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.animation as animation
from JSAnimation import IPython_display
from demo_logistic_regression import demo_logistic_regression
# -*- coding: utf-8 -*-

class classification_optimization_sliders:
    
    def __init__(self):
        a = 0
        self.X = []
        self.y = []
        self.cost_history = []
        self.x_orig = []
        
    # load in a two-dimensional dataset from csv - input should be in first column, oiutput in second column, no headers 
    def load_data(self,*args):
        # load data
        data = np.asarray(pd.read_csv(args[0],header = None))
    
        # import data and reshape appropriately
        self.X = data[:,0:-1]
        y = data[:,-1]
        y.shape = (len(y),1)
        self.y = y

    ##### plotting functions ####
    # show the net transformation using slider
    def animate_fit(self,**args): 
        # run logistic regression 
        classifier = demo_logistic_regression()
        self.w,self.cost_history = classifier.fit(self.X,self.y,**args)

        # setup range for plots
        costs = [v[-1] for v in self.cost_history]
        X = self.X
        y = self.y
        xgap = float(max(X[1,:]) - min(X[1,:]))/float(10)
        ygap = float(max(X[2,:]) - min(X[2,:]))/float(10)
        cgap = float(max(costs) - min(costs))/float(10)
            
        # take positive and negative label indicies
        pos_inds = np.argwhere(y > 0)
        pos_inds = [v[0] for v in pos_inds]
    
        neg_inds = np.argwhere(y < 0)
        neg_inds = [v[0] for v in neg_inds]
        
        # setup surface data for left and middle plot
        colors = ['salmon','cornflowerblue']
        a = np.linspace(np.min(X[:,0]) - xgap,np.max(X[:,0]) + xgap,200)  
        b = np.linspace(np.min(X[:,1]) - ygap,np.max(X[:,1]) + ygap,200)    

        s,t = np.meshgrid(a,b)
        shape = np.shape(s)
    
        s = np.reshape(s,(np.size(s),1))
        t = np.reshape(t,(np.size(t),1))
        h = np.concatenate((s,t),1)

        s.shape = (shape)
        t.shape = (shape)
        
        # get kernelized data
        if "kernel" in args:
            h = classifier.kernelize_test(h)
        else:
            o = np.ones((np.shape(h)[0],1))
            h = np.concatenate((o,h),1)
        
        # plotting
        fig = plt.figure(num=None, figsize=(15,5), dpi=80, facecolor='w', edgecolor='k')
        ax1 = plt.subplot(131)
        ax2 = plt.subplot(132,projection='3d')
        ax3 = plt.subplot(133)

        # slider mechanism
        def show_fit(step):
            #### print left panel ####
            ax1.cla()
            ax1.scatter(X[pos_inds,0],X[pos_inds,1],color = colors[0],linewidth = 1,marker = 'o',edgecolor = 'k',s = 80)
            ax1.scatter(X[neg_inds,0],X[neg_inds,1],color = colors[1],linewidth = 1,marker = 'o',edgecolor = 'k',s = 80)

            # print approximation
            w_now = self.cost_history[step][:-1]
            w_now = np.asarray([u[0] for u in w_now])
            w_now.shape = (len(w_now),1)
            
            z = np.tanh(np.dot(h,w_now))
            z2 = np.sign(np.copy(z))
            z2.shape = (shape)
    
            unique_labels = np.unique(y)
            levels = unique_labels
            ax1.contourf(s,t,z2,colors = colors, alpha = 0.2)

            # show the classification boundary if it exists
            if len(np.unique(z2)) > 1:
                ax1.contour(s,t,z2,colors = 'k',linewidths = 2.5, levels = unique_labels)
        
            # clean up panel
            ax1.set_xlim([min(X[:,0])-xgap,max(X[:,0])+xgap])
            ax1.set_ylim([min(X[:,1])-ygap,max(X[:,1])+ygap])
            ax1.set_xticks([])
            ax1.set_yticks([])   
            
            #### print middle panel ####
            ax2.cla()
            
            # compute surface - due to bug in matplotlib surface will sometimes appear in front of points, sometimes behind, depending on angle and regardless of zorder setting
            z.shape = (shape)
            ax2.plot_surface(s,t,z,alpha = 0.1,color = 'k',zorder = 0)
            
            # plot points
            ax2.scatter(X[pos_inds,0],X[pos_inds,1],y[pos_inds],color = colors[0],linewidth = 1,marker = 'o',edgecolor = 'k',s = 80,zorder = 1)
            ax2.scatter(X[neg_inds,0],X[neg_inds,1],y[neg_inds],color = colors[1],linewidth = 1,marker = 'o',edgecolor = 'k',s = 80,zorder = 1)

            ## clean up panel
            # set viewing angle
            ax2.view_init(args['view'][0],args['view'][1])  
            
            # set viewing limits
            ax2.set_xlim([min(X[:,0])-xgap,max(X[:,0])+xgap])
            ax2.set_ylim([min(X[:,1])-ygap,max(X[:,1])+ygap])    
            
            # turn off tick labels
            ax2.set_xticklabels([])
            ax2.set_yticklabels([])
            ax2.set_zticklabels([])

            # Get rid of the spines on the 3d plot
            ax2.w_xaxis.line.set_color((1.0, 1.0, 1.0, 0.0))
            ax2.w_yaxis.line.set_color((1.0, 1.0, 1.0, 0.0))
            ax2.w_zaxis.line.set_color((1.0, 1.0, 1.0, 0.0))

            # turn off tick marks
            ax2.xaxis.set_tick_params(size=0,color = 'w')
            ax2.yaxis.set_tick_params(size=0,color = 'w')
            ax2.zaxis.set_tick_params(size=0,color = 'w')
            
            #### print right panel #### 
            # print all of the step cost function values lightly
            ax3.cla()
            ax3.plot(costs,color = 'm',marker = 'o',linewidth = 3, alpha = 0.05)            

            # print current value 
            artist = ax3.scatter(step,costs[step],marker = 'o',color = 'r',s = 60,edgecolor = 'k',linewidth = 2)            
            
            # dress up plot 
            ax3.set_xlabel('step',fontsize=15,labelpad = 5)
            ax3.set_ylabel('cost function value',fontsize=12,rotation = 90,labelpad = 5)
            ax3.set_ylim(min(costs) - cgap, max(costs) + cgap)
#             ax3.set_xticks([])
#             ax3.set_yticks([])
            
            return artist,
           
        anim = animation.FuncAnimation(fig, show_fit,frames=len(self.cost_history), interval=len(self.cost_history), blit=True)
        
        # set frames per second in animation
        IPython_display.anim_to_html(anim,fps = int(len(self.cost_history)/float(3)))
        
        return(anim)
