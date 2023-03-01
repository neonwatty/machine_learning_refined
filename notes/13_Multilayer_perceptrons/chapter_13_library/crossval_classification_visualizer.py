# import standard plotting and animation
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from mpl_toolkits.mplot3d import Axes3D
from IPython.display import clear_output

# import autograd functionality
import autograd.numpy as np
import math
import time
from matplotlib import gridspec
import copy
from matplotlib.ticker import FormatStrFormatter
from inspect import signature

class Visualizer:
    '''
    Visualize cross validation performed on N = 2 dimensional input classification datasets
    '''
    #### initialize ####
    def __init__(self,csvname):
        # grab input
        data = np.loadtxt(csvname,delimiter = ',')
        data = data.T
        self.x = data[:,:-1]
        self.y = data[:,-1]

        self.colors = ['salmon','cornflowerblue','lime','bisque','mediumaquamarine','b','m','g']

    ######## show N = 2 static image ########
    # show coloring of entire space
    def static_N2_simple(self,w_best,runner,**kwargs):
        cost = runner.cost
        predict = runner.model
        full_predict = runner.full_model
        feat = runner.feature_transforms
        normalizer = runner.normalizer
        inverse_nornalizer = runner.inverse_normalizer
        x_train = inverse_nornalizer(runner.x_train).T
        y_train = runner.y_train

        x_test = inverse_nornalizer(runner.x_test).T
        y_test = runner.y_test
      
        # or just take last weights
        self.w = w_best

        # construct figure
        fig, axs = plt.subplots(1, 1, figsize=(9,4))

        # create subplot with 2 panels
        gs = gridspec.GridSpec(1, 2,width_ratios = [3,1]) 
        ax2 = plt.subplot(gs[0],aspect = 'equal')
        ax3 = plt.subplot(gs[1]); ax3.axis('off')
        
        ### create boundary data ###
        xmin1 = np.min(self.x[:,0])
        xmax1 = np.max(self.x[:,0])
        xgap1 = (xmax1 - xmin1)*0.05
        xmin1 -= xgap1
        xmax1 += xgap1

        xmin2 = np.min(self.x[:,1])
        xmax2 = np.max(self.x[:,1])
        xgap2 = (xmax2 - xmin2)*0.05
        xmin2 -= xgap2
        xmax2 += xgap2    

        # plot boundary for 2d plot
        r1 = np.linspace(xmin1,xmax1,300)
        r2 = np.linspace(xmin2,xmax2,300)
        s,t = np.meshgrid(r1,r2)
        s = np.reshape(s,(np.size(s),1))
        t = np.reshape(t,(np.size(t),1))
        h = np.concatenate((s,t),axis = 1)
        
        # compute model on train data
        z1 = predict(normalizer(h.T),self.w)
        z1 = np.sign(z1)

        # reshape it
        s.shape = (np.size(r1),np.size(r2))
        t.shape = (np.size(r1),np.size(r2))     
        z1.shape = (np.size(r1),np.size(r2))

        ### loop over two panels plotting each ###
        for ax in [ax2]:
            # plot training points
            ind0 = np.argwhere(y_train == +1)
            ind0 = [v[1] for v in ind0]
            ax.scatter(x_train[ind0,0],x_train[ind0,1],s = 55, color = self.colors[0], edgecolor = 'k',linewidth = 2.5,zorder = 3)

            ind1 = np.argwhere(y_train == -1)
            ind1 = [v[1] for v in ind1]
            ax.scatter(x_train[ind1,0],x_train[ind1,1],s = 55, color = self.colors[1], edgecolor = 'k',linewidth = 2.5,zorder = 3)

            # plot testing points
            ind0 = np.argwhere(y_test == +1)
            ind0 = [v[1] for v in ind0]
            ax.scatter(x_test[ind0,0],x_test[ind0,1],s = 55, color = self.colors[0], edgecolor = [1,0.8,0.5],linewidth = 2.5,zorder = 3)

            ind1 = np.argwhere(y_test == -1)
            ind1 = [v[1] for v in ind1]
            ax.scatter(x_test[ind1,0],x_test[ind1,1],s = 55, color = self.colors[1], edgecolor = [1,0.8,0.5],linewidth = 2.5,zorder = 3)

            #### plot contour, color regions ####
            ax.contour(s,t,z1,colors='k', linewidths=2.5,levels = [0],zorder = 2)
            ax.contourf(s,t,z1,colors = [self.colors[1],self.colors[0]],alpha = 0.15,levels = range(-1,2))
  
            # cleanup panel
            ax.set_xlabel(r'$x_1$',fontsize = 15)
            ax.set_ylabel(r'$x_2$',fontsize = 15,rotation = 0,labelpad = 20)
            ax.xaxis.set_major_formatter(FormatStrFormatter('%.1f'))
            ax.yaxis.set_major_formatter(FormatStrFormatter('%.1f'))
        
    ###### plot plotting functions ######
    def plot_data(self):
        fig = 0
        # plot data in two and one-d
        if np.shape(self.x)[1] < 2:
            # construct figure
            fig, axs = plt.subplots(2,1, figsize=(4,4))
            gs = gridspec.GridSpec(2,1,height_ratios = [6,1]) 
            ax1 = plt.subplot(gs[0],aspect = 'equal');
            ax2 = plt.subplot(gs[1],sharex = ax1); 
            
            # set plotting limits
            xmax = copy.deepcopy(max(self.x))
            xmin = copy.deepcopy(min(self.x))
            xgap = (xmax - xmin)*0.2
            xmin -= xgap
            xmax += xgap
            
            ymax = max(self.y)
            ymin = min(self.y)
            ygap = (ymax - ymin)*0.5
            ymin -= ygap
            ymax += ygap    

            ### plot in 2-d
            ax1.scatter(self.x,self.y,color = 'k', edgecolor = 'w',linewidth = 0.9,s = 40)

            # clean up panel
            ax1.set_xlim([xmin,xmax])
            ax1.set_ylim([ymin,ymax])
            ax1.axhline(linewidth=0.5, color='k',zorder = 1)
            
            ### plot in 1-d
            ind0 = np.argwhere(self.y == +1)
            ax2.scatter(self.x[ind0],np.zeros((len(self.x[ind0]))),s = 55, color = self.colors[0], edgecolor = 'k',zorder = 3)

            ind1 = np.argwhere(self.y == -1)
            ax2.scatter(self.x[ind1],np.zeros((len(self.x[ind1]))),s = 55, color = self.colors[1], edgecolor = 'k',zorder = 3)
            ax2.set_yticks([0])
            ax2.axhline(linewidth=0.5, color='k',zorder = 1)
        
            ax2.xaxis.set_major_formatter(FormatStrFormatter('%.1f'))
            ax2.yaxis.set_major_formatter(FormatStrFormatter('%.1f'))
            
        if np.shape(self.x)[1] == 2:
            # construct figure
            fig, axs = plt.subplots(1, 2, figsize=(9,4))

            # create subplot with 2 panels
            gs = gridspec.GridSpec(1, 2) 
            ax2 = plt.subplot(gs[1],aspect = 'equal'); 
            ax1 = plt.subplot(gs[0],projection='3d'); 

            # scatter points
            self.scatter_pts(ax1,self.x)
            
            ### from above
            ax2.set_xlabel(r'$x_1$',fontsize = 15)
            ax2.set_ylabel(r'$x_2$',fontsize = 15,rotation = 0,labelpad = 20)
            ax2.xaxis.set_major_formatter(FormatStrFormatter('%.1f'))
            ax2.yaxis.set_major_formatter(FormatStrFormatter('%.1f'))
            
            # plot points in 2d and 3d
            C = len(np.unique(self.y))
            if C == 2:
                ind0 = np.argwhere(self.y == +1)
                ax2.scatter(self.x[ind0,0],self.x[ind0,1],s = 55, color = self.colors[0], edgecolor = 'k')

                ind1 = np.argwhere(self.y == -1)
                ax2.scatter(self.x[ind1,0],self.x[ind1,1],s = 55, color = self.colors[1], edgecolor = 'k')
            else:
                for c in range(C):
                    ind0 = np.argwhere(self.y == c)
                    ax2.scatter(self.x[ind0,0],self.x[ind0,1],s = 55, color = self.colors[c], edgecolor = 'k')
                    
        
            self.move_axis_left(ax1)
            ax1.set_xlabel(r'$x_1$', fontsize = 12,labelpad = 5)
            ax1.set_ylabel(r'$x_2$', rotation = 0,fontsize = 12,labelpad = 5)
            ax1.set_zlabel(r'$y$', rotation = 0,fontsize = 12,labelpad = -3)
        
    # scatter points
    def scatter_pts(self,ax,x):
        if np.shape(x)[1] == 1:
            # set plotting limits
            xmax = copy.deepcopy(max(x))
            xmin = copy.deepcopy(min(x))
            xgap = (xmax - xmin)*0.2
            xmin -= xgap
            xmax += xgap
            
            ymax = max(self.y)
            ymin = min(self.y)
            ygap = (ymax - ymin)*0.2
            ymin -= ygap
            ymax += ygap    

            # initialize points
            ax.scatter(x,self.y,color = 'k', edgecolor = 'w',linewidth = 0.9,s = 40)

            # clean up panel
            ax.set_xlim([xmin,xmax])
            ax.set_ylim([ymin,ymax])
            
        if np.shape(x)[1] == 2:
            # set plotting limits
            xmax1 = copy.deepcopy(max(x[:,0]))
            xmin1 = copy.deepcopy(min(x[:,0]))
            xgap1 = (xmax1 - xmin1)*0.1
            xmin1 -= xgap1
            xmax1 += xgap1
            
            xmax2 = copy.deepcopy(max(x[:,1]))
            xmin2 = copy.deepcopy(min(x[:,1]))
            xgap2 = (xmax2 - xmin2)*0.1
            xmin2 -= xgap2
            xmax2 += xgap2
            
            ymax = max(self.y)
            ymin = min(self.y)
            ygap = (ymax - ymin)*0.2
            ymin -= ygap
            ymax += ygap    

            # initialize points
            ax.scatter(x[:,0],x[:,1],self.y.flatten(),s = 40,color = 'k', edgecolor = 'w',linewidth = 0.9)

            # clean up panel
            ax.set_xlim([xmin1,xmax1])
            ax.set_ylim([xmin2,xmax2])
            ax.set_zlim([ymin,ymax])
            
            ax.set_xticks(np.arange(round(xmin1), round(xmax1)+1, 1.0))
            ax.set_yticks(np.arange(round(xmin2), round(xmax2)+1, 1.0))
            ax.set_zticks(np.arange(round(ymin), round(ymax)+1, 1.0))
           
            # clean up panel
            ax.xaxis.pane.fill = False
            ax.yaxis.pane.fill = False
            ax.zaxis.pane.fill = False

            ax.xaxis.pane.set_edgecolor('white')
            ax.yaxis.pane.set_edgecolor('white')
            ax.zaxis.pane.set_edgecolor('white')

            ax.xaxis._axinfo["grid"]['color'] =  (1,1,1,0)
            ax.yaxis._axinfo["grid"]['color'] =  (1,1,1,0)
            ax.zaxis._axinfo["grid"]['color'] =  (1,1,1,0)
