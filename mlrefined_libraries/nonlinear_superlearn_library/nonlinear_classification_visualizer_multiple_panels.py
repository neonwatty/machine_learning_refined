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
    Visualize linear regression in 2 and 3 dimensions.  For single input cases (2 dimensions) the path of gradient descent on the cost function can be animated.
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
    def plot_three_fits(self,run1,run2,run3,**kwargs):
        ## strip off model, normalizer, etc., ##
        model1 = run1.model
        model2 = run2.model
        model3 = run3.model

        normalizer1 = run1.normalizer
        normalizer2 = run2.normalizer
        normalizer3 = run3.normalizer

        # get weights
        cost_history1 = run1.cost_histories[0]
        ind1 = np.argmin(cost_history1)
        w1 = run1.weight_histories[0][ind1]
        cost_history2 = run2.cost_histories[0]
        ind2 = np.argmin(cost_history2)
        w2 = run2.weight_histories[0][ind2]
        cost_history3 = run3.cost_histories[0]
        ind3 = np.argmin(cost_history3)
        w3 = run3.weight_histories[0][ind3]

        # construct figure
        fig, axs = plt.subplots(1, 3, figsize=(10,4))

        # create subplot with 2 panels
        gs = gridspec.GridSpec(1, 3) 
        ax1 = plt.subplot(gs[0],aspect = 'equal'); 
        ax2 = plt.subplot(gs[1],aspect = 'equal'); 
        ax3 = plt.subplot(gs[2],aspect = 'equal'); 

        # loop over axes
        for ax in [ax1,ax2,ax3]:
            ### from above
            ax.set_xlabel(r'$x_1$',fontsize = 15)
            ax.set_ylabel(r'$x_2$',fontsize = 15,rotation = 0,labelpad = 20)
            ax.xaxis.set_major_formatter(FormatStrFormatter('%.1f'))
            ax.yaxis.set_major_formatter(FormatStrFormatter('%.1f'))

            # plot points in 2d 
            ind0 = np.argwhere(self.y == +1)
            ax.scatter(self.x[ind0,0],self.x[ind0,1],s = 55, color = self.colors[0], edgecolor = 'k')

            ind1 = np.argwhere(self.y == -1)
            ax.scatter(self.x[ind1,0],self.x[ind1,1],s = 55, color = self.colors[1], edgecolor = 'k')

            ### create surface and boundary plot ###
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
            
            # plot model
            z = 0
            if ax == ax1:
                z = model1(normalizer1(h.T),w1)
                ax.set_title('underfitting',fontsize = 14)
            if ax == ax2:
                z = model2(normalizer2(h.T),w2)
                ax.set_title('overfitting',fontsize = 14)
            if ax == ax3:
                z = model3(normalizer3(h.T),w3)
                ax.set_title(r'"just right"',fontsize = 14)
            z = np.sign(z)

            # reshape it
            s.shape = (np.size(r1),np.size(r2))
            t.shape = (np.size(r1),np.size(r2))     
            z.shape = (np.size(r1),np.size(r2))

            #### plot contour, color regions ####
            ax.contour(s,t,z,colors='k', linewidths=2.5,levels = [0],zorder = 2)
            ax.contourf(s,t,z,colors = [self.colors[1],self.colors[0]],alpha = 0.15,levels = range(-1,2))
            
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
           
    # set axis in left panel
    def move_axis_left(self,ax):
        tmp_planes = ax.zaxis._PLANES 
        ax.zaxis._PLANES = ( tmp_planes[2], tmp_planes[3], 
                             tmp_planes[0], tmp_planes[1], 
                             tmp_planes[4], tmp_planes[5])
        view_1 = (25, -135)
        view_2 = (25, -45)
        init_view = view_2
        ax.view_init(*init_view) 
        
        
    # toy plot
    def multiclass_plot(self,run,w,**kwargs):
        model = run.model
        normalizer = run.normalizer
        
        # grab args
        view = [20,-70]
        if 'view' in kwargs:
            view = kwargs['view']
 
        ### plot all input data ###
        # generate input range for functions
        minx = min(min(self.x[:,0]),min(self.x[:,1]))
        maxx = max(max(self.x[:,0]),max(self.x[:,1]))
        gapx = (maxx - minx)*0.1
        minx -= gapx
        maxx += gapx

        r = np.linspace(minx,maxx,600)
        w1_vals,w2_vals = np.meshgrid(r,r)
        w1_vals.shape = (len(r)**2,1)
        w2_vals.shape = (len(r)**2,1)
        h = np.concatenate([w1_vals,w2_vals],axis = 1).T

        g_vals = model(normalizer(h),w)
        g_vals = np.asarray(g_vals)
        g_vals = np.argmax(g_vals,axis = 0)

        # vals for cost surface
        w1_vals.shape = (len(r),len(r))
        w2_vals.shape = (len(r),len(r))
        g_vals.shape = (len(r),len(r))

        # create figure to plot
        fig = plt.figure(num=None, figsize=(12,5), dpi=80, facecolor='w', edgecolor='k')

        ### create 3d plot in left panel
        ax1 = plt.subplot(121,projection = '3d')
        ax2 = plt.subplot(122)

        fig.subplots_adjust(left=0,right=1,bottom=0,top=1)   # remove whitespace around 3d figure

        # scatter points in both panels
        class_nums = np.unique(self.y)
        C = len(class_nums)
        for c in range(C):
            ind = np.argwhere(self.y == class_nums[c])
            ind = [v[0] for v in ind]
            ax1.scatter(self.x[ind,0],self.x[ind,1],self.y[ind],s = 80,color = self.colors[c],edgecolor = 'k',linewidth = 1.5)
            ax2.scatter(self.x[ind,0],self.x[ind,1],s = 110,color = self.colors[c],edgecolor = 'k', linewidth = 2)
            
        # switch for 2class / multiclass view
        if C == 2:
            # plot regression surface
            ax1.plot_surface(w1_vals,w2_vals,g_vals,alpha = 0.1,color = 'k',rstride=20, cstride=20,linewidth=0,edgecolor = 'k') 

            # plot zplane = 0 in left 3d panel - showing intersection of regressor with z = 0 (i.e., its contour, the separator, in the 3d plot too)?
            ax1.plot_surface(w1_vals,w2_vals,g_vals*0,alpha = 0.1,rstride=20, cstride=20,linewidth=0.15,color = 'k',edgecolor = 'k') 
            
            # plot separator in left plot z plane
            ax1.contour(w1_vals,w2_vals,g_vals,colors = 'k',levels = [0],linewidths = 3,zorder = 1)

            # color parts of plane with correct colors
            ax1.contourf(w1_vals,w2_vals,g_vals+1,colors = self.colors[:],alpha = 0.1,levels = range(0,2))
            ax1.contourf(w1_vals,w2_vals,-g_vals+1,colors = self.colors[1:],alpha = 0.1,levels = range(0,2))
    
            # plot separator in right plot
            ax2.contour(w1_vals,w2_vals,g_vals,colors = 'k',levels = [0],linewidths = 3,zorder = 1)

            # adjust height of regressor to plot filled contours
            ax2.contourf(w1_vals,w2_vals,g_vals+1,colors = self.colors[:],alpha = 0.1,levels = range(0,C+1))

            ### clean up panels
            # set viewing limits on vertical dimension for 3d plot           
            minz = min(copy.deepcopy(self.y))
            maxz = max(copy.deepcopy(self.y))

            gapz = (maxz - minz)*0.1
            minz -= gapz
            maxz += gapz

        # multiclass view
        else:   
            ax1.plot_surface(w1_vals,w2_vals,g_vals,alpha = 0.1,color = 'w',rstride=45, cstride=45,linewidth=0.25,edgecolor = 'k')

            for c in range(C):
                # plot separator curve in left plot z plane
                ax1.contour(w1_vals,w2_vals,g_vals - c,colors = 'k',levels = [0],linewidths = 3,zorder = 1)

                # color parts of plane with correct colors
                ax1.contourf(w1_vals,w2_vals,g_vals - c +0.5,colors = self.colors[c],alpha = 0.4,levels = [0,1])
             
                
            # plot separator in right plot
            ax2.contour(w1_vals,w2_vals,g_vals,colors = 'k',levels = range(0,C+1),linewidths = 3,zorder = 1)
            
            # adjust height of regressor to plot filled contours
            ax2.contourf(w1_vals,w2_vals,g_vals+0.5,colors = self.colors[:],alpha = 0.2,levels = range(0,C+1))

            ### clean up panels
            # set viewing limits on vertical dimension for 3d plot 
            minz = 0
            maxz = max(copy.deepcopy(self.y))
            gapz = (maxz - minz)*0.1
            minz -= gapz
            maxz += gapz
            ax1.set_zlim([minz,maxz])

            ax1.view_init(view[0],view[1]) 

        # clean up panel
        ax1.xaxis.pane.fill = False
        ax1.yaxis.pane.fill = False
        ax1.zaxis.pane.fill = False

        ax1.xaxis.pane.set_edgecolor('white')
        ax1.yaxis.pane.set_edgecolor('white')
        ax1.zaxis.pane.set_edgecolor('white')

        ax1.xaxis._axinfo["grid"]['color'] =  (1,1,1,0)
        ax1.yaxis._axinfo["grid"]['color'] =  (1,1,1,0)
        ax1.zaxis._axinfo["grid"]['color'] =  (1,1,1,0)

        self.move_axis_left(ax1)
        ax1.set_xlabel(r'$x_1$', fontsize = 16,labelpad = 5)
        ax1.set_ylabel(r'$x_2$', rotation = 0,fontsize = 16,labelpad = 5)
        ax1.set_zlabel(r'$y$', rotation = 0,fontsize = 16,labelpad = 5)

        ax2.set_xlabel(r'$x_1$', fontsize = 18,labelpad = 10)
        ax2.set_ylabel(r'$x_2$', rotation = 0,fontsize = 18,labelpad = 15)
        
        
    # toy plot
    def show_individual_classifiers(self,run,w,**kwargs):
        model = run.model
        normalizer = run.normalizer
        feat = run.feature_transforms
        
        # grab args
        view = [20,-70]
        if 'view' in kwargs:
            view = kwargs['view']
 
        ### plot all input data ###
        # generate input range for functions
        minx = min(min(self.x[:,0]),min(self.x[:,1]))
        maxx = max(max(self.x[:,0]),max(self.x[:,1]))
        gapx = (maxx - minx)*0.1
        minx -= gapx
        maxx += gapx

        r = np.linspace(minx,maxx,600)
        w1_vals,w2_vals = np.meshgrid(r,r)
        w1_vals.shape = (len(r)**2,1)
        w2_vals.shape = (len(r)**2,1)
        h = np.concatenate([w1_vals,w2_vals],axis = 1).T

        g_vals = model(normalizer(h),w)
        g_vals = np.asarray(g_vals)
        g_new = copy.deepcopy(g_vals).T
        g_vals = np.argmax(g_vals,axis = 0)

        # vals for cost surface
        w1_vals.shape = (len(r),len(r))
        w2_vals.shape = (len(r),len(r))
        g_vals.shape = (len(r),len(r))

        # count points
        class_nums = np.unique(self.y)
        C = int(len(class_nums))
        
        fig = plt.figure(figsize = (10,7))
        gs = gridspec.GridSpec(2, C) 

        #### left plot - data and fit in original space ####
        # setup current axis
        ax1 = plt.subplot(gs[C],projection = '3d');
        ax2 = plt.subplot(gs[C+1],aspect = 'equal');
        fig.subplots_adjust(left=0,right=1,bottom=0,top=1)   # remove whitespace around 3d figure

        ##### plot top panels ####
        for d in range(C):
            # create panel
            ax = plt.subplot(gs[d],aspect = 'equal');
                       
            for c in range(C):
                # plot points
                ind = np.argwhere(self.y == class_nums[c])
                ind = [v[0] for v in ind]
                ax.scatter(self.x[ind,0],self.x[ind,1],s = 50,color = self.colors[c],edgecolor = 'k', linewidth = 2)
            
            g_2 = np.sign(g_new[:,d])
            g_2.shape = (len(r),len(r))

            # plot separator curve 
            ax.contour(w1_vals,w2_vals,g_2+1,colors = 'k',levels = [-1,1],linewidths = 4.5,zorder = 1,linestyle = '-')
            ax.contour(w1_vals,w2_vals,g_2+1,colors = self.colors[d],levels = [-1,1],linewidths = 2.5,zorder = 1,linestyle = '-')
                
            ax.set_xlabel(r'$x_1$', fontsize = 18,labelpad = 10)
            ax.set_ylabel(r'$x_2$', rotation = 0,fontsize = 18,labelpad = 15)
        
        ##### plot bottom panels ###
        # scatter points in both bottom panels
        for c in range(C):
            ind = np.argwhere(self.y == class_nums[c])
            ind = [v[0] for v in ind]
            ax1.scatter(self.x[ind,0],self.x[ind,1],self.y[ind],s = 50,color = self.colors[c],edgecolor = 'k',linewidth = 1.5)
            ax2.scatter(self.x[ind,0],self.x[ind,1],s = 50,color = self.colors[c],edgecolor = 'k', linewidth = 2)
      
        ax1.plot_surface(w1_vals,w2_vals,g_vals,alpha = 0.1,color = 'w',rstride=45, cstride=45,linewidth=0.25,edgecolor = 'k')

        for c in range(C):
            # plot separator curve in left plot z plane
            ax1.contour(w1_vals,w2_vals,g_vals - c,colors = 'k',levels = [0],linewidths = 3,zorder = 1)

            # color parts of plane with correct colors
            ax1.contourf(w1_vals,w2_vals,g_vals - c +0.5,colors = self.colors[c],alpha = 0.4,levels = [0,1])
             
        # plot separator in right plot
        ax2.contour(w1_vals,w2_vals,g_vals,colors = 'k',levels = range(0,C+1),linewidths = 3,zorder = 1)

        # adjust height of regressor to plot filled contours
        ax2.contourf(w1_vals,w2_vals,g_vals+0.5,colors = self.colors[:],alpha = 0.2,levels = range(0,C+1))

        ### clean up panels
        # set viewing limits on vertical dimension for 3d plot 
        minz = 0
        maxz = max(copy.deepcopy(self.y))
        gapz = (maxz - minz)*0.1
        minz -= gapz
        maxz += gapz
        ax1.set_zlim([minz,maxz])

        ax1.view_init(view[0],view[1]) 

        # clean up panel
        ax1.xaxis.pane.fill = False
        ax1.yaxis.pane.fill = False
        ax1.zaxis.pane.fill = False

        ax1.xaxis.pane.set_edgecolor('white')
        ax1.yaxis.pane.set_edgecolor('white')
        ax1.zaxis.pane.set_edgecolor('white')

        ax1.xaxis._axinfo["grid"]['color'] =  (1,1,1,0)
        ax1.yaxis._axinfo["grid"]['color'] =  (1,1,1,0)
        ax1.zaxis._axinfo["grid"]['color'] =  (1,1,1,0)

        self.move_axis_left(ax1)
        ax1.set_xlabel(r'$x_1$', fontsize = 16,labelpad = 5)
        ax1.set_ylabel(r'$x_2$', rotation = 0,fontsize = 16,labelpad = 5)
        ax1.set_zlabel(r'$y$', rotation = 0,fontsize = 16,labelpad = 5)

        ax2.set_xlabel(r'$x_1$', fontsize = 18,labelpad = 10)
        ax2.set_ylabel(r'$x_2$', rotation = 0,fontsize = 18,labelpad = 15)