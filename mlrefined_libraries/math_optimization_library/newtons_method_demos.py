# import standard plotting and animation
import matplotlib.pyplot as plt
import matplotlib.animation as animation

# import autograd functionality
from autograd import grad as compute_grad   
from autograd import hessian as compute_hess

import autograd.numpy as np
import math
from IPython.display import clear_output
import time
from matplotlib import gridspec

# simple first order taylor series visualizer
class visualizer:
    '''
    Illustrate gradient descent, Newton method, and Secant method for minimizing an input function, illustrating
    surrogate functions at each step.  A custom slider mechanism is used to progress each algorithm, and points are
    colored from green at the start of an algorithm, to yellow as it converges, and red as the final point.
    ''' 
    def __init__(self,**args):
        self.g = args['g']                            # input function
        self.grad = compute_grad(self.g)              # gradient of input function
        self.hess = compute_hess(self.g)           # hessian of input function
        self.w_init =float( -2)                       # user-defined initial point (adjustable when calling each algorithm)
        self.alpha = 10**-4                           # user-defined step length for gradient descent (adjustable when calling gradient descent)
        self.max_its = 20                             # max iterations to run for each algorithm
        self.w_hist = []                              # container for algorithm path
        self.colors = [[0,1,0.25],[0,0.75,1]]    # set of custom colors used for plotting
        self.beta = 0

    ######## newton's method ########
    # run newton's method
    def run_newtons_method(self):
        w = self.w_init
        self.w_hist = []
        self.w_hist.append(w)
        for k in range(self.max_its):
            # plug in value into func and derivative
            grad_eval = self.grad(w)
            hess_eval = self.hess(w)
            
            # reshape for numpy linalg functionality
            hess_eval.shape = (int((np.size(hess_eval))**(0.5)),int((np.size(hess_eval))**(0.5)))

            # solve linear system for weights
            w = w - np.dot(np.linalg.pinv(hess_eval + self.beta*np.eye(np.size(w))),grad_eval)
                                
            # record
            self.w_hist.append(w)

    # animate the method
    def animate_it(self,**kwargs):
        # get new initial point if desired
        if 'w_init' in kwargs:
            self.w_init = float(kwargs['w_init'])
            
        # take in user defined maximum number of iterations
        if 'max_its' in kwargs:
            self.max_its = int(kwargs['max_its'])
            
        wmax = 3
        if 'wmax' in kwargs:
            wmax = kwargs['wmax']
        wmin = -wmax
        if 'wmin' in kwargs:
            wmin = kwargs['wmin']
            
        # initialize figure
        fig = plt.figure(figsize = (10,5))
        artist = fig

        # create subplot with 3 panels, plot input function in center plot
        gs = gridspec.GridSpec(1, 3, width_ratios=[1,5, 1]) 
        ax1 = plt.subplot(gs[0]); ax1.axis('off');
        ax3 = plt.subplot(gs[2]); ax3.axis('off');

        # plot input function
        ax = plt.subplot(gs[1],aspect = 'equal')
        
        # generate function for plotting on each slide
        w_plot = np.linspace(wmin,wmax,1000)
        g_plot = self.g(w_plot)
        g_range = max(g_plot) - min(g_plot)
        ggap = g_range*0.1
        w_vals = np.linspace(-2.5,2.5,50)
        width = 1
        
        # run newtons method
        self.w_hist = []
        self.run_newtons_method()
        
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
        print ('starting animation rendering...')
        num_frames = 2*len(self.w_hist)+2
        def animate(t):
            ax.cla()
            k = math.floor((t+1)/float(2))
            
            # print rendering update
            if np.mod(k+1,25) == 0:
                print ('rendering animation frame ' + str(k+1) + ' of ' + str(num_frames))
            if t == num_frames - 1:
                print ('animation rendering complete!')
                time.sleep(1.5)
                clear_output()

            # plot function
            ax.plot(w_plot,g_plot,color = 'k',zorder = 1)                           # plot function
            
            # plot initial point and evaluation
            if k == 0:
                w_val = self.w_init
                g_val = self.g(w_val)
                ax.scatter(w_val,g_val,s = 100,c = 'm',edgecolor = 'k',linewidth = 0.7, marker = 'X',zorder = 2)            # plot point of tangency
                ax.scatter(w_val,0,s = 100,c = 'm',edgecolor = 'k',linewidth = 0.7, zorder = 2)
                # draw dashed line connecting w axis to point on cost function
                s = np.linspace(0,g_val)
                o = np.ones((len(s)))
                ax.plot(o*w_val,s,'k--',linewidth=1,zorder = 0)
                
            # plot all input/output pairs generated by algorithm thus far
            if k > 0:
                # plot all points up to this point
                for j in range(min(k-1,len(self.w_hist))):  
                    w_val = self.w_hist[j]
                    g_val = self.g(w_val)
                    ax.scatter(w_val,g_val,s = 90,c = self.colorspec[j],edgecolor = 'k',marker = 'X',linewidth = 0.7,zorder = 3)            # plot point of tangency
                    ax.scatter(w_val,0,s = 90,facecolor = self.colorspec[j],edgecolor = 'k',linewidth = 0.7, zorder = 2)
                          
            # plot surrogate function and travel-to point
            if k > 0 and k < len(self.w_hist) + 1:          
                # grab historical weight, compute function and derivative evaluations    
                w_eval = self.w_hist[k-1]
                if type(w_eval) != float:
                    w_eval = float(w_eval[0][0])

                # plug in value into func and derivative
                g_eval = self.g(w_eval)
                g_grad_eval = self.grad(w_eval)
                g_hess_eval = self.hess(w_eval)

                # determine width of plotting area for second order approximator
                width = 0.5
                if g_hess_eval < 0:
                    width = - width

                # setup quadratic formula params
                a = 0.5*g_hess_eval
                b = g_grad_eval - 2*0.5*g_hess_eval*w_eval
                c = 0.5*g_hess_eval*w_eval**2 - g_grad_eval*w_eval - width

                # solve for zero points
                w1 = (-b + math.sqrt(b**2 - 4*a*c))/float(2*a + 0.00001)
                w2 = (-b - math.sqrt(b**2 - 4*a*c))/float(2*a + 0.00001)

                # compute second order approximation
                wrange = np.linspace(w1,w2, 100)
                h = g_eval + g_grad_eval*(wrange - w_eval) + 0.5*g_hess_eval*(wrange - w_eval)**2 

                # plot tangent line
                ax.plot(wrange,h,color = self.colors[1],linewidth = 2,zorder = 2)      # plot approx

                # plot tangent point
                ax.scatter(w_eval,g_eval,s = 100,c = 'm',edgecolor = 'k', marker = 'X',linewidth = 0.7,zorder = 3)            # plot point of tangency
            
                # plot next point learned from surrogate
                if np.mod(t,2) == 0:
                    # create next point information
                    w_zero = w_eval - g_grad_eval/(g_hess_eval + 10**-5)
                    g_zero = self.g(w_zero)
                    h_zero = g_eval + g_grad_eval*(w_zero - w_eval) + 0.5*g_hess_eval*(w_zero - w_eval)**2

                    # draw dashed line connecting the three
                    vals = [0,h_zero,g_zero]
                    vals = np.sort(vals)

                    s = np.linspace(vals[0],vals[2])
                    o = np.ones((len(s)))
                    ax.plot(o*w_zero,s,'k--',linewidth=1)

                    # draw intersection at zero and associated point on cost function you hop back too
                    ax.scatter(w_zero,h_zero,s = 100,c = 'b',linewidth=0.7, marker = 'X',edgecolor = 'k',zorder = 3)
                    ax.scatter(w_zero,0,s = 100,c = 'm',edgecolor = 'k',linewidth = 0.7, zorder = 3)
                    ax.scatter(w_zero,g_zero,s = 100,c = 'm',edgecolor = 'k',linewidth = 0.7, marker = 'X',zorder = 3)            # plot point of tangency
            
            # fix viewing limits on panel
            ax.set_xlim([wmin,wmax])
            ax.set_ylim([min(-0.3,min(g_plot) - ggap),max(max(g_plot) + ggap,0.3)])
            
            # add horizontal axis
            ax.axhline(y=0, color='k',zorder = 0,linewidth = 0.5)
            
            # label axes
            ax.set_xlabel('$w$',fontsize = 12)
            ax.set_ylabel('$g(w)$',fontsize = 12,rotation = 0,labelpad = 12)
            
            # set tickmarks
            ax.set_xticks(np.arange(round(wmin), round(wmax) + 1, 1.0))
            ax.set_yticks(np.arange(round(min(g_plot) - ggap), round(max(g_plot) + ggap) + 1, 1.0))

            return artist,

        anim = animation.FuncAnimation(fig, animate ,frames=num_frames, interval=num_frames, blit=True)

        return(anim)
    
    # visualize descent on multi-input function
    def draw_it(self,w_init,max_its,**kwargs):
        ### input arguments ###        
        self.max_its = max_its
        self.grad = compute_grad(self.g)              # gradient of input function
        self.w_init = w_init
        
        if 'beta' in kwargs:
            self.beta = kwargs['beta']
            
        pts = 'off'
        if 'pts' in kwargs:
            pts = 'off'
            
        linewidth = 2.5
        if 'linewidth' in kwargs:
            linewidth = kwargs['linewidth']
            
        view = [20,-50]
        if 'view' in kwargs:
            view = kwargs['view']

        axes = False
        if 'axes' in kwargs:
            axes = kwargs['axes']

        plot_final = False
        if 'plot_final' in kwargs:
            plot_final = kwargs['plot_final']

        num_contours = 15
        if 'num_contours' in kwargs:
            num_contours = kwargs['num_contours']
            
        # get initial point 
        self.w_init = w_init
        if np.size(self.w_init) == 2:
            self.w_init = np.asarray([float(s) for s in self.w_init])
        else:
            self.w_init = np.asarray([float(self.w_init)])
        
        # take in user defined maximum number of iterations
        self.max_its = max_its
            
        # construct figure
        fig, axs = plt.subplots(1, 2, figsize=(9,4))

        # create subplot with 3 panels, plot input function in center plot
        gs = gridspec.GridSpec(1, 2, width_ratios=[2,1]) 
        ax = plt.subplot(gs[0],aspect = 'equal'); 
        ax2 = plt.subplot(gs[1]) #  ,sharey = ax); 

        #### run local random search algorithm ####
        self.w_hist = []
        self.run_newtons_method()

        # colors for points
        s = np.linspace(0,1,len(self.w_hist[:round(len(self.w_hist)/2)]))
        s.shape = (len(s),1)
        t = np.ones(len(self.w_hist[round(len(self.w_hist)/2):]))
        t.shape = (len(t),1)
        s = np.vstack((s,t))
        colorspec = []
        colorspec = np.concatenate((s,np.flipud(s)),1)
        colorspec = np.concatenate((colorspec,np.zeros((len(s),1))),1)
    
        #### define input space for function and evaluate ####
        if np.size(self.w_init) == 2:           # function is multi-input, plot 3d function contour
            # set viewing limits on contour plot
            xvals = [self.w_hist[s][0] for s in range(len(self.w_hist))]
            xvals.append(self.w_init[0])
            yvals = [self.w_hist[s][1] for s in range(len(self.w_hist))]
            yvals.append(self.w_init[1])
            xmax = max(xvals)
            xmin = min(xvals)
            xgap = (xmax - xmin)*0.1
            ymax = max(yvals)
            ymin = min(yvals)
            ygap = (ymax - ymin)*0.1
            xmin -= xgap
            xmax += xgap
            ymin -= ygap
            ymax += ygap

            if 'xmin' in kwargs:
                xmin = kwargs['xmin']
            if 'xmax' in kwargs:
                xmax = kwargs['xmax']
            if 'ymin' in kwargs:
                ymin = kwargs['ymin']
            if 'ymax' in kwargs:
                ymax = kwargs['ymax']  

            w1 = np.linspace(xmin,xmax,400)
            w2 = np.linspace(ymin,ymax,400)
            w1_vals, w2_vals = np.meshgrid(w1,w2)
            w1_vals.shape = (len(w1)**2,1)
            w2_vals.shape = (len(w2)**2,1)
            h = np.concatenate((w1_vals,w2_vals),axis=1)
            func_vals = np.asarray([self.g(s) for s in h])
            w1_vals.shape = (len(w1),len(w1))
            w2_vals.shape = (len(w2),len(w2))
            func_vals.shape = (len(w1),len(w2)) 

            ### make contour right plot - as well as horizontal and vertical axes ###
            # set level ridges
            num_contours = kwargs['num_contours']
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
                
            # plot points on contour
            for j in range(len(self.w_hist)):  
                w_val = self.w_hist[j]
                g_val = self.g(w_val)

                # plot in left panel
                if pts == 'on':
                    ax.scatter(w_val[0],w_val[1],s = 30,c = colorspec[j],edgecolor = 'k',linewidth = 1.5*math.sqrt((1/(float(j) + 1))),zorder = 3)

                    ax2.scatter(j,g_val,s = 30,c = colorspec[j],edgecolor = 'k',linewidth = 0.7,zorder = 3)            # plot point of tangency

                # plot connector between points for visualization purposes
                if j > 0:
                    w_old = self.w_hist[j-1]
                    w_new = self.w_hist[j]
                    g_old = self.g(w_old)
                    g_new = self.g(w_new)

                    ax.plot([w_old[0],w_new[0]],[w_old[1],w_new[1]],color = colorspec[j],linewidth = linewidth,alpha = 1,zorder = 2)      # plot approx
                    ax.plot([w_old[0],w_new[0]],[w_old[1],w_new[1]],color = 'k',linewidth = linewidth + 0.4,alpha = 1,zorder = 1)      # plot approx
                    ax2.plot([j-1,j],[g_old,g_new],color = colorspec[j],linewidth = 2,alpha = 1,zorder = 2)      # plot approx
                    ax2.plot([j-1,j],[g_old,g_new],color = 'k',linewidth = 2.5,alpha = 1,zorder = 1)      # plot approx
            
            # clean up panel
            ax.set_xlabel('$w_1$',fontsize = 12)
            ax.set_ylabel('$w_2$',fontsize = 12,rotation = 0,labelpad = 15)
            ax.axhline(y=0, color='k',zorder = 0,linewidth = 0.5)
            ax.axvline(x=0, color='k',zorder = 0,linewidth = 0.5)
            ax.set_xlim([xmin,xmax])
            ax.set_ylim([ymin,ymax])
            
            # set tickmarks
            ax.set_xticks(np.arange(round(xmin), round(xmax) + 1, 1.0))
            ax.set_yticks(np.arange(round(ymin), round(ymax) + 1, 1.0))
            
        else:    # function is single input, plot curve
            xmin = -2
            xmax = 2
            if 'xmin' in kwargs:
                xmin = kwargs['xmin']
            if 'xmax' in kwargs:
                xmax = kwargs['xmax']
                    
            w_plot = np.linspace(xmin,xmax,500)
            g_plot = np.asarray([self.g(s) for s in w_plot])
            ax.plot(w_plot,g_plot,color = 'k',linewidth = 2,zorder = 2)
                
            # set viewing limits
            ymin = min(g_plot)
            ymax = max(g_plot)
            ygap = (ymax - ymin)*0.2
            ymin -= ygap
            ymax += ygap
            ax.set_ylim([ymin,ymax])
                
            # clean up panel
            ax.axhline(y=0, color='k',zorder = 1,linewidth = 0.25)
            ax.axvline(x=0, color='k',zorder = 1,linewidth = 0.25)
            ax.set_xlabel(r'$w$',fontsize = 13)
            ax.set_ylabel(r'$g(w)$',fontsize = 13,rotation = 0,labelpad = 25)   
                
            # function single-input, plot input and evaluation points on function
            for j in range(len(self.w_hist)):  
                w_val = self.w_hist[j]
                g_val = self.g(w_val)
            
                ax.scatter(w_val,g_val,s = 90,c = colorspec[j],edgecolor = 'k',linewidth = 0.5*((1/(float(j) + 1)))**(0.4),zorder = 3,marker = 'X')            # evaluation on function
                ax.scatter(w_val,0,s = 90,facecolor = colorspec[j],edgecolor = 'k',linewidth = 0.5*((1/(float(j) + 1)))**(0.4), zorder = 3)
                    
                ax2.scatter(j,g_val,s = 30,c = colorspec[j],edgecolor = 'k',linewidth = 0.7,zorder = 3)            # plot point of tangency
                    
                # plot connector between points for visualization purposes
                if j > 0:
                    w_old = self.w_hist[j-1][0]
                    w_new = self.w_hist[j][0]
                    g_old = self.g(w_old)
                    g_new = self.g(w_new)
     
                    ax2.plot([j-1,j],[g_old,g_new],color = colorspec[j],linewidth = 2,alpha = 1,zorder = 2)      # plot approx
                    ax2.plot([j-1,j],[g_old,g_new],color = 'k',linewidth = 2.5,alpha = 1,zorder = 1)      # plot approx
      

        # clean panels
        ax2.axhline(y=0, color='k',zorder = 0,linewidth = 0.5)
        ax2.set_xlabel('iteration',fontsize = 12)
        ax2.set_ylabel(r'$g(w)$',fontsize = 12,rotation = 0,labelpad = 25)
            
        ax.set(aspect = 'equal')
        a = ax.get_position()
        yr = ax.get_position().y1 - ax.get_position().y0
        xr = ax.get_position().x1 - ax.get_position().x0
        aspectratio=1.25*xr/yr# + min(xr,yr)
        ratio_default=(ax2.get_xlim()[1]-ax2.get_xlim()[0])/(ax2.get_ylim()[1]-ax2.get_ylim()[0])
        ax2.set_aspect(ratio_default*aspectratio)
            
        # plot
        plt.show()    