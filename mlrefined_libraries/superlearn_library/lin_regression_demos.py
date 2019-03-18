# import standard plotting and animation
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from mpl_toolkits.mplot3d import Axes3D
from IPython.display import clear_output

# import autograd functionality
from autograd import grad as compute_grad   # The only autograd function you may ever need
import autograd.numpy as np
from autograd import hessian as compute_hess
import math
import time
from matplotlib import gridspec
import copy

class Visualizer:
    '''
    Visualize linear regression in 2 and 3 dimensions.  For single input cases (2 dimensions) the path of gradient descent on the cost function can be animated.
    '''
    #### initialize ####
    def __init__(self,data):
        # grab input
        self.x = data[:-1,:].T
        self.y = data[-1:,:].T
        
    def center_data(self):
        # center data
        self.x = self.x - np.mean(self.x)
        self.y = self.y - np.mean(self.y)
        
    ######## linear regression functions ########    
    def least_squares(self,w):
        cost = 0
        for p in range(0,len(self.y)):
            x_p = copy.deepcopy(self.x[p,:])
            x_p.shape = (len(x_p),1)
            y_p = self.y[p]
            cost +=(w[0] + np.dot(w[1:].T,x_p) - y_p)**2
        return cost/float(np.size(self.y))
    
     ######## 3d animation function ########
    # animate gradient descent or newton's method
    def animate_it_3d(self,savepath,w_hist,**kwargs):         
        self.w_hist = w_hist 
        
        ##### setup figure to plot #####
        # initialize figure
        fig = plt.figure(figsize = (8,3))
        artist = fig
        
        # create subplot with 3 panels, plot input function in center plot
        gs = gridspec.GridSpec(1, 2, width_ratios=[2,1]) 
        ax1 = plt.subplot(gs[0],projection='3d'); 
        ax2 = plt.subplot(gs[1]);

        # produce color scheme
        s = np.linspace(0,1,len(self.w_hist[:round(len(self.w_hist)/2)]))
        s.shape = (len(s),1)
        t = np.ones(len(self.w_hist[round(len(self.w_hist)/2):]))
        t.shape = (len(t),1)
        s = np.vstack((s,t))
        self.colorspec = []
        self.colorspec = np.concatenate((s,np.flipud(s)),1)
        self.colorspec = np.concatenate((self.colorspec,np.zeros((len(s),1))),1)
        
        # seed left panel plotting range
        viewmax = 3
        if 'viewmax' in kwargs:
            viewmax = kwargs['viewmax']
        r = np.linspace(-viewmax,viewmax,200)

        # create grid from plotting range
        x1_vals,x2_vals = np.meshgrid(r,r)
        x1_vals.shape = (len(r)**2,1)
        x2_vals.shape = (len(r)**2,1)
        
        x1_vals.shape = (np.size(r),np.size(r))
        x2_vals.shape = (np.size(r),np.size(r))

        # seed left panel view 
        view = [20,50]
        if 'view' in kwargs:
            view = kwargs['view']
        
        # set zaxis to the left
        self.move_axis_left(ax1)
            
        # start animation
        num_frames = len(self.w_hist)
        print ('starting animation rendering...')
        def animate(k):
            # clear panels
            ax1.cla()
            
            # set axis in left panel
            self.move_axis_left(ax1)
            
            # current color
            color = self.colorspec[k]

            # print rendering update
            if np.mod(k+1,25) == 0:
                print ('rendering animation frame ' + str(k+1) + ' of ' + str(num_frames))
            if k == num_frames - 1:
                print ('animation rendering complete!')
                time.sleep(1.5)
                clear_output()
            
            ###### make left panel - plot data and fit ######
            # initialize fit
            w = self.w_hist[k]
                    
            # reshape and plot the surface, as well as where the zero-plane is        
            y_fit = w[0] + w[1]*x1_vals + w[2]*x2_vals

            # plot cost surface
            ax1.plot_surface(x1_vals,x2_vals,y_fit,alpha = 0.1,color = color,rstride=25, cstride=25,linewidth=0.25,edgecolor = 'k',zorder = 2)  
            
            # scatter data
            self.scatter_pts(ax1)
            #ax1.view_init(view[0],view[1])
            
            # plot connector between points for visualization purposes
            if k == 0:
                w_new = self.w_hist[k]
                g_new = self.least_squares(w_new)[0]
                ax2.scatter(k,g_new,s = 0.1,color = 'w',linewidth = 2.5,alpha = 0,zorder = 1)      # plot approx
                
            if k > 0:
                w_old = self.w_hist[k-1]
                w_new = self.w_hist[k]
                g_old = self.least_squares(w_old)[0]
                g_new = self.least_squares(w_new)[0]
     
                ax2.plot([k-1,k],[g_old,g_new],color = color,linewidth = 2.5,alpha = 1,zorder = 2)      # plot approx
                ax2.plot([k-1,k],[g_old,g_new],color = 'k',linewidth = 3.5,alpha = 1,zorder = 1)      # plot approx
            
            # set viewing limits for second panel
            ax2.axhline(y=0, color='k',zorder = 0,linewidth = 0.5)
            ax2.set_xlabel('iteration',fontsize = 12)
            ax2.set_ylabel(r'$g(\mathbf{w})$',fontsize = 12,rotation = 0,labelpad = 25)
            ax2.set_xlim([-0.5,len(self.w_hist)])
            
            # set axis in left panel
            self.move_axis_left(ax1)
                
            return artist,

        anim = animation.FuncAnimation(fig, animate ,frames=num_frames, interval=num_frames, blit=True)
        
        # produce animation and save
        fps = 50
        if 'fps' in kwargs:
            fps = kwargs['fps']
        anim.save(savepath, fps=fps, extra_args=['-vcodec', 'libx264'])
        clear_output()

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
    
    ######## 2d animation function ########
    # animate gradient descent or newton's method
    def animate_it_2d(self,savepath,w_hist,**kwargs):       
        self.w_hist = w_hist
        
        ##### setup figure to plot #####
        # initialize figure
        fig = plt.figure(figsize = (8,3))
        artist = fig
        
        # create subplot with 3 panels, plot input function in center plot
        gs = gridspec.GridSpec(1, 2, width_ratios=[1,1]) 
        ax1 = plt.subplot(gs[0]); 
        ax2 = plt.subplot(gs[1]);

        # produce color scheme
        s = np.linspace(0,1,len(self.w_hist[:round(len(self.w_hist)/2)]))
        s.shape = (len(s),1)
        t = np.ones(len(self.w_hist[round(len(self.w_hist)/2):]))
        t.shape = (len(t),1)
        s = np.vstack((s,t))
        self.colorspec = []
        self.colorspec = np.concatenate((s,np.flipud(s)),1)
        self.colorspec = np.concatenate((self.colorspec,np.zeros((len(s),1))),1)
        
        # seed left panel plotting range
        xmin = np.min(copy.deepcopy(self.x))
        xmax = np.max(copy.deepcopy(self.x))
        xgap = (xmax - xmin)*0.1
        xmin-=xgap
        xmax+=xgap
        x_fit = np.linspace(xmin,xmax,300)
        
        # seed right panel contour plot
        viewmax = 3
        if 'viewmax' in kwargs:
            viewmax = kwargs['viewmax']
        view = [20,100]
        if 'view' in kwargs:
            view = kwargs['view']
        num_contours = 15
        if 'num_contours' in kwargs:
            num_contours = kwargs['num_contours']        
        self.contour_plot(ax2,viewmax,num_contours)
        
        # start animation
        num_frames = len(self.w_hist)
        print ('starting animation rendering...')
        def animate(k):
            # clear panels
            ax1.cla()
            
            # current color
            color = self.colorspec[k]

            # print rendering update
            if np.mod(k+1,25) == 0:
                print ('rendering animation frame ' + str(k+1) + ' of ' + str(num_frames))
            if k == num_frames - 1:
                print ('animation rendering complete!')
                time.sleep(1.5)
                clear_output()
            
            ###### make left panel - plot data and fit ######
            # initialize fit
            w = self.w_hist[k]
            y_fit = w[0] + x_fit*w[1]
            
            # scatter data
            self.scatter_pts(ax1)
            
            # plot fit to data
            ax1.plot(x_fit,y_fit,color = color,linewidth = 3) 

            ###### make right panel - plot contour and steps ######
            if k == 0:
                ax2.scatter(w[0],w[1],s = 90,facecolor = color,edgecolor = 'k',linewidth = 0.5, zorder = 3)
            if k > 0 and k < num_frames:
                self.plot_pts_on_contour(ax2,k,color)
            if k == num_frames -1:
                ax2.scatter(w[0],w[1],s = 90,facecolor = color,edgecolor = 'k',linewidth = 0.5, zorder = 3)
               
            return artist,

        anim = animation.FuncAnimation(fig, animate ,frames=num_frames, interval=num_frames, blit=True)
        
        # produce animation and save
        fps = 50
        if 'fps' in kwargs:
            fps = kwargs['fps']
        anim.save(savepath, fps=fps, extra_args=['-vcodec', 'libx264'])
        clear_output()

    ### animate only the fit ###
    def animate_it_2d_fit_only(self,savepath,w_hist,**kwargs):       
        self.w_hist = w_hist
        
        ##### setup figure to plot #####
        # initialize figure
        fig = plt.figure(figsize = (4,4))
        artist = fig
        
        # create subplot with 3 panels, plot input function in center plot
        gs = gridspec.GridSpec(1, 1) 
        ax1 = plt.subplot(gs[0]); 

        # produce color scheme
        s = np.linspace(0,1,len(self.w_hist[:round(len(self.w_hist)/2)]))
        s.shape = (len(s),1)
        t = np.ones(len(self.w_hist[round(len(self.w_hist)/2):]))
        t.shape = (len(t),1)
        s = np.vstack((s,t))
        self.colorspec = []
        self.colorspec = np.concatenate((s,np.flipud(s)),1)
        self.colorspec = np.concatenate((self.colorspec,np.zeros((len(s),1))),1)
        
        # seed left panel plotting range
        xmin = np.min(copy.deepcopy(self.x))
        xmax = np.max(copy.deepcopy(self.x))
        xgap = (xmax - xmin)*0.1
        xmin-=xgap
        xmax+=xgap
        x_fit = np.linspace(xmin,xmax,300)
        
        # seed right panel contour plot
        viewmax = 3
        if 'viewmax' in kwargs:
            viewmax = kwargs['viewmax']
        view = [20,100]
        if 'view' in kwargs:
            view = kwargs['view']
        
        # start animation
        num_frames = len(self.w_hist)
        print ('starting animation rendering...')
        def animate(k):
            # clear panels
            ax1.cla()
            
            # current color
            color = self.colorspec[k]

            # print rendering update
            if np.mod(k+1,25) == 0:
                print ('rendering animation frame ' + str(k+1) + ' of ' + str(num_frames))
            if k == num_frames - 1:
                print ('animation rendering complete!')
                time.sleep(1.5)
                clear_output()
            
            ###### make left panel - plot data and fit ######
            # initialize fit
            w = self.w_hist[k]
            y_fit = w[0] + x_fit*w[1]
            
            # scatter data
            self.scatter_pts(ax1)
            
            # plot fit to data
            ax1.plot(x_fit,y_fit,color = color,linewidth = 3) 

            return artist,

        anim = animation.FuncAnimation(fig, animate ,frames=num_frames, interval=num_frames, blit=True)
        
        # produce animation and save
        fps = 50
        if 'fps' in kwargs:
            fps = kwargs['fps']
        anim.save(savepath, fps=fps, extra_args=['-vcodec', 'libx264'])
        clear_output()
         
    ###### plot plotting functions ######
    def plot_data(self):
        # construct figure
        fig, axs = plt.subplots(1, 3, figsize=(9,4))

        # create subplot with 2 panels
        gs = gridspec.GridSpec(1, 3, width_ratios=[1,5,1]) 
        ax1 = plt.subplot(gs[0]); ax1.axis('off') 
        ax2 = plt.subplot(gs[1]); 
        ax3 = plt.subplot(gs[2]); ax3.axis('off')
        
        if np.shape(self.x)[1] == 2:
            ax2 = plt.subplot(gs[1],projection='3d'); 

        # scatter points
        self.scatter_pts(ax2)
        
    def plot_regression_fits(self,final_weights):
        # construct figure
        fig, axs = plt.subplots(1, 3, figsize=(8,3))

        # create subplot with 2 panels
        gs = gridspec.GridSpec(1, 3, width_ratios=[1,2,1]) 
        ax1 = plt.subplot(gs[0]); ax1.axis('off') 
        ax2 = plt.subplot(gs[1]); 
        ax3 = plt.subplot(gs[2]); ax3.axis('off')
 
        # scatter points
        self.scatter_pts(ax2)      
        
        # print regression fits
        for weights in final_weights:
            ax2.plot_fit(ax2,weights)
        
    # plot regression fits
    def plot_fit(self,plotting_weights,**kwargs):
        # construct figure
        fig, axs = plt.subplots(1, 3, figsize=(9,4))

        # create subplot with 2 panels
        gs = gridspec.GridSpec(1, 3, width_ratios=[1,5,1]) 
        ax1 = plt.subplot(gs[0]); ax1.axis('off') 
        ax = plt.subplot(gs[1]); 
        ax3 = plt.subplot(gs[2]); ax3.axis('off')
        
        # set plotting limits
        xmin = np.min(copy.deepcopy(self.x))
        xmax = np.max(copy.deepcopy(self.x))
        xgap = (xmax - xmin)*0.25
        xmin -= xgap
        xmax += xgap

        ymin = np.min(copy.deepcopy(self.y))
        ymax = np.max(copy.deepcopy(self.y))
        ygap = (ymax - ymin)*0.25
        ymin -= ygap
        ymax += ygap    

        # initialize points
        ax.scatter(self.x,self.y,color = 'k', edgecolor = 'w',linewidth = 0.9,s = 40,zorder = 0)

        # clean up panel
        ax.set_xlim([xmin,xmax])
        ax.set_ylim([ymin,ymax])

        # label axes
        ax.set_xlabel(r'$x$', fontsize = 12)
        ax.set_ylabel(r'$y$', rotation = 0,fontsize = 12)
        
        # create fit
        s = np.linspace(xmin,xmax,300)
        colors = ['k','magenta']
        if 'colors' in kwargs:
            colors = kwargs['colors']
 
        transformers = [lambda a: a for i in range(len(plotting_weights))]
        if 'transformers' in kwargs:
            transformers = kwargs['transformers']

        for i in range(len(plotting_weights)):
            weights = plotting_weights[i]
            transformer = transformers[i]
            t = weights[0] + weights[1]*transformer(s).flatten()
            ax.plot(s,t,linewidth = 2,color = colors[i],zorder = 3)
            c+=1
    
    # scatter points
    def scatter_pts(self,ax):
        if np.shape(self.x)[1] == 1:
            # set plotting limits
            xmin = np.min(copy.deepcopy(self.x))
            xmax = np.max(copy.deepcopy(self.x))
            xgap = (xmax - xmin)*0.2
            xmin -= xgap
            xmax += xgap
            
            ymin = np.min(copy.deepcopy(self.y))
            ymax = np.max(copy.deepcopy(self.y))
            ygap = (ymax - ymin)*0.2
            ymin -= ygap
            ymax += ygap    

            # initialize points
            ax.scatter(self.x,self.y,color = 'k', edgecolor = 'w',linewidth = 0.9,s = 40)

            # clean up panel
            ax.set_xlim([xmin,xmax])
            ax.set_ylim([ymin,ymax])
            
            # label axes
            ax.set_xlabel(r'$x$', fontsize = 16)
            ax.set_ylabel(r'$y$', rotation = 0,fontsize = 16,labelpad = 15)
            
        if np.shape(self.x)[1] == 2:
            # set plotting limits
            xmin1 = np.min(copy.deepcopy(self.x[:,0]))
            xmax1 = np.max(copy.deepcopy(self.x[:,0])) 
            xgap1 = (xmax1 - xmin1)*0.35
            xmin1 -= xgap1
            xmax1 += xgap1
            
            xmin2 = np.min(copy.deepcopy(self.x[:,1]))
            xmax2 = np.max(copy.deepcopy(self.x[:,1])) 
            xgap2 = (xmax2 - xmin2)*0.35
            xmin2 -= xgap2
            xmax2 += xgap2
            
            ymin = np.min(copy.deepcopy(self.y))
            ymax = np.max(copy.deepcopy(self.y))
            ygap = (ymax - ymin)*0.2
            ymin -= ygap
            ymax += ygap    

            # initialize points
            ax.scatter(self.x[:,0],self.x[:,1],self.y,s = 40,color = 'k', edgecolor = 'w',linewidth = 0.9)

            # clean up panel
            ax.set_xlim([xmin1,xmax1])
            ax.set_ylim([xmin2,xmax2])
            ax.set_zlim([ymin,ymax])
            
            ax.set_xticks(np.arange(round(xmin1) +1, round(xmax1), 1.0))
            ax.set_yticks(np.arange(round(xmin2) +1, round(xmax2), 1.0))

            # label axes
            ax.set_xlabel(r'$x_1$', fontsize = 12,labelpad = 5)
            ax.set_ylabel(r'$x_2$', rotation = 0,fontsize = 12,labelpad = 5)
            ax.set_zlabel(r'$y$', rotation = 0,fontsize = 12,labelpad = -3)

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
           
    # plot points on contour
    def plot_pts_on_contour(self,ax,j,color):
        # plot connector between points for visualization purposes
        w_old = self.w_hist[j-1]
        w_new = self.w_hist[j]
        g_old = self.least_squares(w_old)
        g_new = self.least_squares(w_new)
     
        ax.plot([w_old[0],w_new[0]],[w_old[1],w_new[1]],color = color,linewidth = 3,alpha = 1,zorder = 2)      # plot approx
        ax.plot([w_old[0],w_new[0]],[w_old[1],w_new[1]],color = 'k',linewidth = 3 + 1,alpha = 1,zorder = 1)      # plot approx
    
    ###### function plotting functions #######
    def plot_ls_cost(self,**kwargs):
        # construct figure
        fig, axs = plt.subplots(1, 2, figsize=(8,3))

        # create subplot with 2 panels
        gs = gridspec.GridSpec(1, 2, width_ratios=[1,1]) 
        ax1 = plt.subplot(gs[0],aspect = 'equal'); 
        ax2 = plt.subplot(gs[1],projection='3d'); 
        
        # pull user-defined args
        viewmax = 3
        if 'viewmax' in kwargs:
            viewmax = kwargs['viewmax']
        view = [20,100]
        if 'view' in kwargs:
            view = kwargs['view']
        num_contours = 15
        if 'num_contours' in kwargs:
            num_contours = kwargs['num_contours']
        
        # make contour plot in left panel
        self.contour_plot(ax1,viewmax,num_contours)
        
        # make contour plot in right panel
        self.surface_plot(ax2,viewmax,view)
        
        plt.show()
        
    ### visualize the surface plot of cost function ###
    def surface_plot(self,ax,wmax,view):
        ##### Produce cost function surface #####
        wmax += wmax*0.1
        r = np.linspace(-wmax,wmax,200)

        # create grid from plotting range
        w1_vals,w2_vals = np.meshgrid(r,r)
        w1_vals.shape = (len(r)**2,1)
        w2_vals.shape = (len(r)**2,1)
        w_ = np.concatenate((w1_vals,w2_vals),axis = 1)
        g_vals = []
        for i in range(len(r)**2):
            g_vals.append(self.least_squares(w_[i,:]))
        g_vals = np.asarray(g_vals)

        # reshape and plot the surface, as well as where the zero-plane is
        w1_vals.shape = (np.size(r),np.size(r))
        w2_vals.shape = (np.size(r),np.size(r))
        g_vals.shape = (np.size(r),np.size(r))
        
        # plot cost surface
        ax.plot_surface(w1_vals,w2_vals,g_vals,alpha = 0.1,color = 'w',rstride=25, cstride=25,linewidth=1,edgecolor = 'k',zorder = 2)  
        
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

        ax.set_xlabel(r'$w_0$',fontsize = 12)
        ax.set_ylabel(r'$w_1$',fontsize = 12,rotation = 0)
        ax.set_title(r'$g\left(w_0,w_1\right)$',fontsize = 13)

        ax.view_init(view[0],view[1])
        
    ### visualize contour plot of cost function ###
    def contour_plot(self,ax,wmax,num_contours):
        #### define input space for function and evaluate ####
        w1 = np.linspace(-wmax,wmax,100)
        w2 = np.linspace(-wmax,wmax,100)
        w1_vals, w2_vals = np.meshgrid(w1,w2)
        w1_vals.shape = (len(w1)**2,1)
        w2_vals.shape = (len(w2)**2,1)
        h = np.concatenate((w1_vals,w2_vals),axis=1)
        func_vals = np.asarray([self.least_squares(s) for s in h])
        w1_vals.shape = (len(w1),len(w1))
        w2_vals.shape = (len(w2),len(w2))
        func_vals.shape = (len(w1),len(w2)) 

        ### make contour right plot - as well as horizontal and vertical axes ###
        # set level ridges
        levelmin = min(func_vals.flatten())
        levelmax = max(func_vals.flatten())
        cutoff = 0.5
        cutoff = (levelmax - levelmin)*cutoff
        numper = 3
        levels1 = np.linspace(cutoff,levelmax,numper)
        num_contours -= numper

        levels2 = np.linspace(levelmin,cutoff,min(num_contours,numper))
        levels = np.unique(np.append(levels1,levels2))
        num_contours -= numper
        while num_contours > 0:
            cutoff = levels[1]
            levels2 = np.linspace(levelmin,cutoff,min(num_contours,numper))
            levels = np.unique(np.append(levels2,levels))
            num_contours -= numper

        a = ax.contour(w1_vals, w2_vals, func_vals,levels = levels,colors = 'k')
        ax.contourf(w1_vals, w2_vals, func_vals,levels = levels,cmap = 'Blues')
                
        # clean up panel
        ax.set_xlabel('$w_0$',fontsize = 12)
        ax.set_ylabel('$w_1$',fontsize = 12,rotation = 0)
        ax.set_title(r'$g\left(w_0,w_1\right)$',fontsize = 13)

        ax.axhline(y=0, color='k',zorder = 0,linewidth = 0.5)
        ax.axvline(x=0, color='k',zorder = 0,linewidth = 0.5)
        ax.set_xlim([-wmax,wmax])
        ax.set_ylim([-wmax,wmax])