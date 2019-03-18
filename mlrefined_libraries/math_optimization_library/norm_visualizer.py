# import standard plotting and animation
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from matplotlib import gridspec
from IPython.display import clear_output
from mpl_toolkits.mplot3d import proj3d
from matplotlib.patches import FancyArrowPatch
from matplotlib.text import Annotation
from mpl_toolkits.mplot3d.proj3d import proj_transform

# import autograd functionality
from autograd import grad as compute_grad   # The only autograd function you may ever need
import autograd.numpy as np
import math
import time
import copy

class visualizer:
    '''
    Visualizes norm ball from L1 to Linf, as well as how the distance between two user defined points is defined in that norm.
    ''' 
            
    ##### draws still figure of L1, L2, and Linf ball with visualized distance between 2 points ####       
    def draw_it(self,pt1,pt2,**kwargs):
        ### initialize figure and panels ###
        fig = plt.figure(figsize = (9,4))
       
        # create subplot with 3 panels, plot input function in center plot
        gs = gridspec.GridSpec(1, 2, width_ratios=[1,1]) 
        ax1 = plt.subplot(gs[0],aspect = 'equal')
        ax2 = plt.subplot(gs[1],aspect = 'equal');         
        
        ### make unit balls ###
        w = np.linspace(0,1,2000)
        
        # make l2 ball
        x = np.cos(2*np.pi*w)
        x.shape = (len(x),1)
        y = np.sin(2*np.pi*w)
        y.shape = (len(y),1)
        l2 = np.concatenate((x,y),axis=1)
        
        # make l1 ball 
        l1 = []
        for s in l2:
            s = s/np.sum(abs(s))
            l1.append(s)
        l1 = np.asarray(l1)
        
        # make Linf ball
        linf = []
        for s in l2:
            s = s/np.max(abs(s))
            linf.append(s)
        linf = np.asarray(linf) 
        
        ### plot norm balls ###
        ax1.plot(l2[:,0],l2[:,1],color = 'k',zorder = 1)                           # plot function
        ax1.plot(l1[:,0],l1[:,1],color = 'r',zorder = 1)                           # plot function
        ax1.plot(linf[:,0],linf[:,1],color = 'b',zorder = 1)                           # plot function
        
        ### clean up panel ###
        ax1.axhline(y=0, color='k',zorder = 1,linewidth = 0.25)
        ax1.axvline(x=0, color='k',zorder = 1,linewidth = 0.25)
        ax1.set_xticks([-1,-0.5,0,0.5,1])
        ax1.set_yticks([-1,-0.5,0,0.5,1])
        ax1.set_xlim([-1.2,1.2])
        ax1.set_ylim([-1.2,1.2])
        
        ### scatter two points, show length in norm for l1, l2, and linf ###
        # create l2 distance curve
        pt1 = np.asarray(pt1)
        pt2 = np.asarray(pt2)
        l2_dist = [(1-s)*pt1 + s*pt2 for s in w]
        l2_dist = np.asarray(l2_dist)
        l2_dist.shape = (len(l2_dist),2)
        
        # create l1 distance curve
        pt1_temp = np.asarray([pt1[0],pt1[1]])
        pt2_temp = np.asarray([pt2[0],pt1[1]])
        l1_dist = [(1-s)*pt1_temp + s*pt2_temp for s in w]
        l1_dist = np.asarray(l1_dist)
        l1_dist.shape = (len(l1_dist),2)

        pt1_temp = np.asarray([pt2[0],pt1[1]])
        pt2_temp = np.asarray([pt2[0],pt2[1]])
        l1_dist2 = [(1-s)*pt1_temp + s*pt2_temp for s in w]
        l1_dist2 = np.asarray(l1_dist2)
        l1_dist2.shape = (len(l1_dist2),2)
        l1_dist = np.concatenate((l1_dist,l1_dist2),axis=0)        

        # create linf distance curve
        linf_dist = abs(pt1 - pt2)
        ind = np.argmax(linf_dist)  
        pt1_temp = 0
        pt2_temp = 0
        if ind == 0:
            pt1_temp = np.asarray([pt1[0],pt2[1]])
            pt2_temp = np.asarray([pt2[0],pt2[1]]) 
        elif ind == 1:
            pt1_temp = np.asarray([pt1[0],pt1[1]])
            pt2_temp = np.asarray([pt1[0],pt2[1]]) 
            
        linf_dist = [(1-s)*pt1_temp + s*pt2_temp for s in w]
        linf_dist = np.asarray(linf_dist)
        linf_dist.shape = (len(linf_dist),2)
        
        ### plot points and distances ###
        ax2.scatter(pt1[0],pt1[1],s = 60,linewidth = 1,edgecolor = 'w',color = 'k',zorder = 2)
        ax2.scatter(pt2[0],pt2[1],s = 60,linewidth = 1,edgecolor = 'w',color = 'k',zorder = 2)
                
        ### plot distances ###
        ax2.plot(l2_dist[:,0],l2_dist[:,1],color = 'k',linestyle = '--',linewidth=2,zorder = 1)                           
        ax2.plot(l1_dist[:,0],l1_dist[:,1],color = 'r',linestyle = '--',linewidth=2,zorder = 1)       
        ax2.plot(linf_dist[:,0],linf_dist[:,1],color = 'b',linestyle = '--',linewidth=2,zorder = 1)               
       
        ### clean up panel ###
        ax2.axhline(y=0, color='k',zorder = 1,linewidth = 0.25)
        ax2.axvline(x=0, color='k',zorder = 1,linewidth = 0.25)
        ax2.set_xticks([-1,-0.5,0,0.5,1])
        ax2.set_yticks([-1,-0.5,0,0.5,1])
        ax2.set_xlim([-1.2,1.2])
        ax2.set_ylim([-1.2,1.2])
        ax2.legend(['l2','l1','linf'])

        
        
    ##### animate gradient descent method using single-input function #####
    def animate_2d(self,**kwargs):
        self.g = kwargs['g']                          # input function
        self.grad = compute_grad(self.g)              # gradient of input function
        self.w_init =float( -2)                       # user-defined initial point (adjustable when calling each algorithm)
        self.alpha = 10**-4                           # user-defined step length for gradient descent (adjustable when calling gradient descent)
        self.max_its = 20                             # max iterations to run for each algorithm
        self.w_hist = []                              # container for algorithm path
        
        wmin = -3.1
        wmax = 3.1
        if 'wmin' in kwargs:            
            wmin = kwargs['wmin']
        if 'wmax' in kwargs:
            wmax = kwargs['wmax']
            
        # version of gradient descent to use (normalized or unnormalized)
        self.version = 'unnormalized'
        if 'version' in kwargs:
            self.version = kwargs['version']
            
        # get new initial point if desired
        if 'w_init' in kwargs:
            self.w_init = float(kwargs['w_init'])
            
        # take in user defined step length
        if 'steplength' in kwargs:
            self.steplength = kwargs['steplength']
            
        # take in user defined maximum number of iterations
        if 'max_its' in kwargs:
            self.max_its = float(kwargs['max_its'])
            
        # initialize figure
        fig = plt.figure(figsize = (9,4))
        artist = fig
        
        # remove whitespace from figure
        #fig.subplots_adjust(left=0, right=1, bottom=0, top=1) # remove whitespace
        #fig.subplots_adjust(wspace=0.01,hspace=0.01)

        # create subplot with 3 panels, plot input function in center plot
        gs = gridspec.GridSpec(1, 3, width_ratios=[1,4,1]) 

        ax1 = plt.subplot(gs[0]); ax1.axis('off')
        ax3 = plt.subplot(gs[2]); ax3.axis('off')
        ax = plt.subplot(gs[1]); 

        # generate function for plotting on each slide
        w_plot = np.linspace(wmin,wmax,200)
        g_plot = self.g(w_plot)
        g_range = max(g_plot) - min(g_plot)
        ggap = g_range*0.1
        width = 30
        
        # run gradient descent method
        self.w_hist = []
        self.run_gradient_descent()
        
        # colors for points --> green as the algorithm begins, yellow as it converges, red at final point
        s = np.linspace(0,1,len(self.w_hist[:round(len(self.w_hist)/2)]))
        s.shape = (len(s),1)
        t = np.ones(len(self.w_hist[round(len(self.w_hist)/2):]))
        t.shape = (len(t),1)
        s = np.vstack((s,t))
        self.colorspec = []
        self.colorspec = np.concatenate((s,np.flipud(s)),1)
        self.colorspec = np.concatenate((self.colorspec,np.zeros((len(s),1))),1)
        
        # animation sub-function
        num_frames = 2*len(self.w_hist)+2
        print ('starting animation rendering...')
        def animate(t):
            ax.cla()
            k = math.floor((t+1)/float(2))
            
            # print rendering update            
            if np.mod(t+1,25) == 0:
                print ('rendering animation frame ' + str(t+1) + ' of ' + str(num_frames))
            if t == num_frames - 1:
                print ('animation rendering complete!')
                time.sleep(1.5)
                clear_output()
            
            # plot function
            ax.plot(w_plot,g_plot,color = 'k',zorder = 2)                           # plot function
            
            # plot initial point and evaluation
            if k == 0:
                w_val = self.w_init
                g_val = self.g(w_val)
                ax.scatter(w_val,g_val,s = 90,c = self.colorspec[k],edgecolor = 'k',linewidth = 0.5*((1/(float(k) + 1)))**(0.4),zorder = 3,marker = 'X')            # evaluation on function
                ax.scatter(w_val,0,s = 90,facecolor = self.colorspec[k],edgecolor = 'k',linewidth = 0.5*((1/(float(k) + 1)))**(0.4), zorder = 3)
                
                # draw dashed line connecting w axis to point on cost function
                s = np.linspace(0,g_val)
                o = np.ones((len(s)))
                ax.plot(o*w_val,s,'k--',linewidth=1)

            # plot all input/output pairs generated by algorithm thus far
            if k > 0:
                # plot all points up to this point
                for j in range(min(k-1,len(self.w_hist))):  
                    w_val = self.w_hist[j]
                    g_val = self.g(w_val)
                    ax.scatter(w_val,g_val,s = 90,c = self.colorspec[j],edgecolor = 'k',linewidth = 0.5*((1/(float(j) + 1)))**(0.4),zorder = 3,marker = 'X')            # plot point of tangency
                    ax.scatter(w_val,0,s = 90,facecolor = self.colorspec[j],edgecolor = 'k',linewidth =  0.5*((1/(float(j) + 1)))**(0.4), zorder = 2)
                    
            # plot surrogate function and travel-to point
            if k > 0 and k < len(self.w_hist) + 1:          
                # grab historical weight, compute function and derivative evaluations
                w = self.w_hist[k-1]
                g_eval = self.g(w)
                grad_eval = float(self.grad(w))
            
                # determine width to plot the approximation -- so its length == width defined above
                div = float(1 + grad_eval**2)
                w1 = w - math.sqrt(width/div)
                w2 = w + math.sqrt(width/div)

                # use point-slope form of line to plot
                wrange = np.linspace(w1,w2, 100)
                h = g_eval + grad_eval*(wrange - w)

                # plot tangent line
                ax.plot(wrange,h,color = 'lime',linewidth = 2,zorder = 1)      # plot approx

                # plot tangent point
                ax.scatter(w,g_eval,s = 100,c = 'm',edgecolor = 'k',linewidth = 0.7,zorder = 3,marker = 'X')            # plot point of tangency
            
                # plot next point learned from surrogate
                if np.mod(t,2) == 0:
                    # create next point information
                    w_zero = w - self.alpha*grad_eval
                    g_zero = self.g(w_zero)
                    h_zero = g_eval + grad_eval*(w_zero - w)

                    # draw dashed line connecting the three
                    vals = [0,h_zero,g_zero]
                    vals = np.sort(vals)

                    s = np.linspace(vals[0],vals[2])
                    o = np.ones((len(s)))
                    ax.plot(o*w_zero,s,'k--',linewidth=1)

                    # draw intersection at zero and associated point on cost function you hop back too
                    ax.scatter(w_zero,h_zero,s = 100,c = 'k', zorder = 3,marker = 'X')
                    ax.scatter(w_zero,0,s = 100,c = 'm',edgecolor = 'k',linewidth = 0.7, zorder = 3)
                    ax.scatter(w_zero,g_zero,s = 100,c = 'm',edgecolor = 'k',linewidth = 0.7,zorder = 3, marker = 'X')            # plot point of tangency
                 
            # fix viewing limits
            ax.set_xlim([wmin-0.1,wmax+0.1])
            ax.set_ylim([min(g_plot) - ggap,max(g_plot) + ggap])
            ax.axhline(y=0, color='k',zorder = 0,linewidth = 0.5)
            
            # place title
            ax.set_xlabel(r'$w$',fontsize = 14)
            ax.set_ylabel(r'$g(w)$',fontsize = 14,rotation = 0,labelpad = 25)

            return artist,

        anim = animation.FuncAnimation(fig, animate ,frames=num_frames, interval=num_frames, blit=True)
        
        return(anim)

    # visualize descent on multi-input function
    def visualize3d(self,g,w_init,steplength,max_its,**kwargs):
        ### input arguments ###        
        self.g = g
        self.steplength = steplength
        self.max_its = max_its
        self.grad = compute_grad(self.g)              # gradient of input function

        wmax = 1
        if 'wmax' in kwargs:
            wmax = kwargs['wmax'] + 0.5

        view = [20,-50]
        if 'view' in kwargs:
            view = kwargs['view']

        axes = False
        if 'axes' in kwargs:
            axes = kwargs['axes']

        plot_final = False
        if 'plot_final' in kwargs:
            plot_final = kwargs['plot_final']

        num_contours = 10
        if 'num_contours' in kwargs:
            num_contours = kwargs['num_contours']

        # version of gradient descent to use (normalized or unnormalized)
        self.version = 'unnormalized'
        if 'version' in kwargs:
            self.version = kwargs['version']
            
        # get initial point 
        self.w_init = np.asarray([float(s) for s in w_init])
                    
        # take in user defined step length
        self.steplength = steplength
            
        # take in user defined maximum number of iterations
        self.max_its = max_its
            
        ##### construct figure with panels #####
        # construct figure
        fig = plt.figure(figsize = (9,3))

        # remove whitespace from figure
        fig.subplots_adjust(left=0, right=1, bottom=0, top=1) # remove whitespace

        # create subplot with 3 panels, plot input function in center plot
        gs = gridspec.GridSpec(1, 2, width_ratios=[1,2]) 
        ax = plt.subplot(gs[0],projection='3d'); 
        ax2 = plt.subplot(gs[1],aspect='equal'); 

        #### define input space for function and evaluate ####
        w = np.linspace(-wmax,wmax,200)
        w1_vals, w2_vals = np.meshgrid(w,w)
        w1_vals.shape = (len(w)**2,1)
        w2_vals.shape = (len(w)**2,1)
        h = np.concatenate((w1_vals,w2_vals),axis=1)
        func_vals = np.asarray([g(s) for s in h])
        w1_vals.shape = (len(w),len(w))
        w2_vals.shape = (len(w),len(w))
        func_vals.shape = (len(w),len(w))

        # plot function 
        ax.plot_surface(w1_vals, w2_vals, func_vals, alpha = 0.1,color = 'w',rstride=25, cstride=25,linewidth=1,edgecolor = 'k',zorder = 2)

        # plot z=0 plane 
        ax.plot_surface(w1_vals, w2_vals, func_vals*0, alpha = 0.1,color = 'w',zorder = 1,rstride=25, cstride=25,linewidth=0.3,edgecolor = 'k') 

        ### make contour right plot - as well as horizontal and vertical axes ###
        ax2.contour(w1_vals, w2_vals, func_vals,num_contours,colors = 'k')
        if axes == True:
            ax2.axhline(linestyle = '--', color = 'k',linewidth = 1)
            ax2.axvline(linestyle = '--', color = 'k',linewidth = 1)

        #### run local random search algorithm ####
        self.w_hist = []
        self.run_gradient_descent()

        # colors for points
        s = np.linspace(0,1,len(self.w_hist[:round(len(self.w_hist)/2)]))
        s.shape = (len(s),1)
        t = np.ones(len(self.w_hist[round(len(self.w_hist)/2):]))
        t.shape = (len(t),1)
        s = np.vstack((s,t))
        colorspec = []
        colorspec = np.concatenate((s,np.flipud(s)),1)
        colorspec = np.concatenate((colorspec,np.zeros((len(s),1))),1)

        #### scatter path points ####
        for k in range(len(self.w_hist)):
            w_now = self.w_hist[k]
            ax.scatter(w_now[0],w_now[1],0,s = 60,c = colorspec[k],edgecolor = 'k',linewidth = 0.5*math.sqrt((1/(float(k) + 1))),zorder = 3)

            ax2.scatter(w_now[0],w_now[1],s = 60,c = colorspec[k],edgecolor = 'k',linewidth = 1.5*math.sqrt((1/(float(k) + 1))),zorder = 3)

        #### connect points with arrows ####
        if len(self.w_hist) < 10:
            for i in range(len(self.w_hist)-1):
                pt1 = self.w_hist[i]
                pt2 = self.w_hist[i+1]

                # draw arrow in left plot
                a = Arrow3D([pt1[0],pt2[0]], [pt1[1],pt2[1]], [0, 0], mutation_scale=10, lw=2, arrowstyle="-|>", color="k")
                ax.add_artist(a)

                # draw 2d arrow in right plot
                ax2.arrow(pt1[0],pt1[1],(pt2[0] - pt1[0])*0.78,(pt2[1] - pt1[1])*0.78, head_width=0.1, head_length=0.1, fc='k', ec='k',linewidth=3,zorder = 2,length_includes_head=True)

        ### cleanup panels ###
        ax.set_xlabel('$w_1$',fontsize = 12)
        ax.set_ylabel('$w_2$',fontsize = 12,rotation = 0)
        ax.set_title('$g(w_1,w_2)$',fontsize = 12)
        ax.view_init(view[0],view[1])

        ax2.set_xlabel('$w_1$',fontsize = 12)
        ax2.set_ylabel('$w_2$',fontsize = 12,rotation = 0)
        ax2.axhline(y=0, color='k',zorder = 0,linewidth = 0.5)
        ax2.axvline(x=0, color='k',zorder = 0,linewidth = 0.5)

        # clean up axis
        ax.xaxis.pane.fill = False
        ax.yaxis.pane.fill = False
        ax.zaxis.pane.fill = False

        ax.xaxis.pane.set_edgecolor('white')
        ax.yaxis.pane.set_edgecolor('white')
        ax.zaxis.pane.set_edgecolor('white')

        ax.xaxis._axinfo["grid"]['color'] =  (1,1,1,0)
        ax.yaxis._axinfo["grid"]['color'] =  (1,1,1,0)
        ax.zaxis._axinfo["grid"]['color'] =  (1,1,1,0)

        # plot
        plt.show()
        
    # compare normalized and unnormalized grad descent on 3d example
    def compare_versions_3d(self,g,w_init,steplength,max_its,**kwargs):
        ### input arguments ###        
        self.g = g
        self.steplength = steplength
        self.max_its = max_its
        self.grad = compute_grad(self.g)              # gradient of input function

        wmax = 1
        if 'wmax' in kwargs:
            wmax = kwargs['wmax'] + 0.5

        view = [20,-50]
        if 'view' in kwargs:
            view = kwargs['view']

        axes = False
        if 'axes' in kwargs:
            axes = kwargs['axes']

        plot_final = False
        if 'plot_final' in kwargs:
            plot_final = kwargs['plot_final']

        num_contours = 10
        if 'num_contours' in kwargs:
            num_contours = kwargs['num_contours']
            
        # get initial point 
        self.w_init = np.asarray([float(s) for s in w_init])
                    
        # take in user defined step length
        self.steplength = steplength
            
        # take in user defined maximum number of iterations
        self.max_its = max_its
            
        ##### construct figure with panels #####
        # construct figure
        fig = plt.figure(figsize = (12,6))

        # create subplot with 3 panels, plot input function in center plot
        gs = gridspec.GridSpec(2, 2, width_ratios=[1,4]) 
        ax3 = plt.subplot(gs[0],projection='3d'); 
        ax4 = plt.subplot(gs[1],aspect='equal'); 
        ax5 = plt.subplot(gs[2],projection='3d'); 
        ax6 = plt.subplot(gs[3],aspect='equal'); 
        
        # remove whitespace from figure
        fig.subplots_adjust(left=0, right=1, bottom=0, top=1) # remove whitespace
        
        #### define input space for function and evaluate ####
        w = np.linspace(-wmax,wmax,200)
        w1_vals, w2_vals = np.meshgrid(w,w)
        w1_vals.shape = (len(w)**2,1)
        w2_vals.shape = (len(w)**2,1)
        h = np.concatenate((w1_vals,w2_vals),axis=1)
        func_vals = np.asarray([g(s) for s in h])
        w1_vals.shape = (len(w),len(w))
        w2_vals.shape = (len(w),len(w))
        func_vals.shape = (len(w),len(w))

        #### run local random search algorithms ####
        for algo in ['normalized','unnormalized']:
            # switch normalized / unnormalized
            self.version = algo
            title = ''
            if self.version == 'normalized':
                ax = ax3
                ax2 = ax4
                title = 'normalized gradient descent'
            else:
                ax = ax5
                ax2 = ax6
                title = 'unnormalized gradient descent'
            
           # plot function 
            ax.plot_surface(w1_vals, w2_vals, func_vals, alpha = 0.1,color = 'w',rstride=25, cstride=25,linewidth=1,edgecolor = 'k',zorder = 2)

            # plot z=0 plane 
            ax.plot_surface(w1_vals, w2_vals, func_vals*0, alpha = 0.1,color = 'w',zorder = 1,rstride=25, cstride=25,linewidth=0.3,edgecolor = 'k') 

            ### make contour right plot - as well as horizontal and vertical axes ###
            ax2.contour(w1_vals, w2_vals, func_vals,num_contours,colors = 'k')
            if axes == True:
                ax2.axhline(linestyle = '--', color = 'k',linewidth = 1)
                ax2.axvline(linestyle = '--', color = 'k',linewidth = 1)
            
            self.w_hist = []
            self.run_gradient_descent()

            # colors for points
            s = np.linspace(0,1,len(self.w_hist[:round(len(self.w_hist)/2)]))
            s.shape = (len(s),1)
            t = np.ones(len(self.w_hist[round(len(self.w_hist)/2):]))
            t.shape = (len(t),1)
            s = np.vstack((s,t))
            colorspec = []
            colorspec = np.concatenate((s,np.flipud(s)),1)
            colorspec = np.concatenate((colorspec,np.zeros((len(s),1))),1)

            #### scatter path points ####
            for k in range(len(self.w_hist)):
                w_now = self.w_hist[k]
                ax.scatter(w_now[0],w_now[1],0,s = 60,c = colorspec[k],edgecolor = 'k',linewidth = 0.5*math.sqrt((1/(float(k) + 1))),zorder = 3)

                ax2.scatter(w_now[0],w_now[1],s = 60,c = colorspec[k],edgecolor = 'k',linewidth = 1.5*math.sqrt((1/(float(k) + 1))),zorder = 3)

            #### connect points with arrows ####
            if len(self.w_hist) < 10:
                for i in range(len(self.w_hist)-1):
                    pt1 = self.w_hist[i]
                    pt2 = self.w_hist[i+1]
        
                    # draw arrow in left plot
                    a = Arrow3D([pt1[0],pt2[0]], [pt1[1],pt2[1]], [0, 0], mutation_scale=10, lw=2, arrowstyle="-|>", color="k")
                    ax.add_artist(a)

                    # draw 2d arrow in right plot
                    ax2.arrow(pt1[0],pt1[1],(pt2[0] - pt1[0])*0.78,(pt2[1] - pt1[1])*0.78, head_width=0.1, head_length=0.1, fc='k', ec='k',linewidth=3,zorder = 2,length_includes_head=True)

            ### cleanup panels ###
            ax.set_xlabel('$w_1$',fontsize = 12)
            ax.set_ylabel('$w_2$',fontsize = 12,rotation = 0)
            ax.set_title(title,fontsize = 12)
            ax.view_init(view[0],view[1])
    
            ax2.set_xlabel('$w_1$',fontsize = 12)
            ax2.set_ylabel('$w_2$',fontsize = 12,rotation = 0)
            ax2.axhline(y=0, color='k',zorder = 0,linewidth = 0.5)
            ax2.axvline(x=0, color='k',zorder = 0,linewidth = 0.5)

            # clean up axis
            ax.xaxis.pane.fill = False
            ax.yaxis.pane.fill = False
            ax.zaxis.pane.fill = False

            ax.xaxis.pane.set_edgecolor('white')
            ax.yaxis.pane.set_edgecolor('white')
            ax.zaxis.pane.set_edgecolor('white')

            ax.xaxis._axinfo["grid"]['color'] =  (1,1,1,0)
            ax.yaxis._axinfo["grid"]['color'] =  (1,1,1,0)
            ax.zaxis._axinfo["grid"]['color'] =  (1,1,1,0)

        # plot
        plt.show()      
        
        
#### custom 3d arrow and annotator functions ###    
# nice arrow maker from https://stackoverflow.com/questions/11140163/python-matplotlib-plotting-a-3d-cube-a-sphere-and-a-vector
class Arrow3D(FancyArrowPatch):

    def __init__(self, xs, ys, zs, *args, **kwargs):
        FancyArrowPatch.__init__(self, (0, 0), (0, 0), *args, **kwargs)
        self._verts3d = xs, ys, zs

    def draw(self, renderer):
        xs3d, ys3d, zs3d = self._verts3d
        xs, ys, zs = proj3d.proj_transform(xs3d, ys3d, zs3d, renderer.M)
        self.set_positions((xs[0], ys[0]), (xs[1], ys[1]))
        FancyArrowPatch.draw(self, renderer)