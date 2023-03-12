import autograd.numpy as np
from autograd import grad as compute_grad   # The only autograd function you may ever need
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from matplotlib import gridspec
from IPython.display import clear_output
from mpl_toolkits.mplot3d import proj3d
from matplotlib.patches import FancyArrowPatch
from matplotlib.text import Annotation
from mpl_toolkits.mplot3d.proj3d import proj_transform
import time
import math


class convservative_steplength_visualizer:
    '''
    Illustrates how conservative steplength rules work in general.
    ''' 
     
    ######## gradient descent ########
    # run gradient descent 
    def run_gradient_descent(self):
        w = self.w_init
        self.w_hist = []
        self.w_hist.append(w)
        w_old = np.inf
        j = 0
        for j in range(int(self.max_its)):
            # update old w and index
            w_old = w
            
            # plug in value into func and derivative
            grad_eval = self.grad(w)
            
            ### normalized or unnormalized? ###
            if self.version == 'normalized':
                grad_norm = np.linalg.norm(grad_eval)
                if grad_norm == 0:
                    grad_norm += 10**-6*np.sign(2*np.random.rand(1) - 1)
                grad_eval /= grad_norm
                
            ### check what sort of steplength rule to employ ###
            alpha = 0
            if self.steplength == 'diminishing':
                alpha = 1/(1 + j)
            elif self.steplength == 'backtracking':
                alpha = self.backtracking(w,grad_eval)
            elif self.steplength == 'exact': 
                alpha = self.exact(w,grad_eval)
            else:
                alpha = float(self.steplength)            
            
            # take gradient descent step
            w = w - alpha*grad_eval
            
            # record
            self.w_hist.append(w)

    # backtracking linesearch module
    def backtracking(self,w,grad_eval):
        # set input parameters
        alpha = 1
        t = 0.8
        
        # compute initial function and gradient values
        func_eval = self.g(w)
        grad_norm = np.linalg.norm(grad_eval)**2
        
        # loop over and tune steplength
        while self.g(w - alpha*grad_eval) > func_eval - alpha*0.5*grad_norm:
            alpha = t*alpha
        return alpha

    # exact linesearch module
    def exact(self,w,grad_eval):
        # set parameters of linesearch at each step
        valmax = 10
        num_evals = 3000
        
        # set alpha range
        alpha_range = np.linspace(0,valmax,num_evals)
        
        # evaluate function over direction and alpha range, grab alpha giving lowest eval
        steps = [(w - alpha*grad_eval) for alpha in alpha_range]
        func_evals = np.array([self.g(s) for s in steps])
        ind = np.argmin(func_evals)
        best_alpha = alpha_range[ind]
        
        return best_alpha
    
    # visualize descent on multi-input function
    def run(self,g,w_init,steplength_vals,max_its,**kwargs):
        # count up steplength vals
        step_count = len(steplength_vals)
        
        ### input arguments ###        
        self.g = g
        self.max_its = max_its
        self.grad = compute_grad(self.g)              # gradient of input function
        self.w_init = w_init
        
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

        # version of gradient descent to use (normalized or unnormalized)
        self.version = 'unnormalized'
        if 'version' in kwargs:
            self.version = kwargs['version']
            
        # get initial point 
        if np.size(self.w_init) == 2:
            self.w_init = np.asarray([float(s) for s in self.w_init])
        else:
            self.w_init = float(self.w_init)
            
        # take in user defined maximum number of iterations
        self.max_its = max_its
            
        ##### construct figure with panels #####
        # loop over steplengths, plot panels for each
        count = 0
        for step in steplength_vals:
            # construct figure
            fig, axs = plt.subplots(1, 2, figsize=(9,4))

            # create subplot with 3 panels, plot input function in center plot
            gs = gridspec.GridSpec(1, 2, width_ratios=[2,1]) 
            ax = plt.subplot(gs[0],aspect = 'equal'); 
            ax2 = plt.subplot(gs[1]) #  ,sharey = ax); 

            #### run local random search algorithm ####
            self.w_hist = []
            self.steplength = steplength_vals[count]
            self.run_gradient_descent()
            count+=1
            
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
                func_vals = np.asarray([g(s) for s in h])
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
                        ax.scatter(w_val[0],w_val[1],s = 30,color = colorspec[j],edgecolor = 'k',linewidth = 1.5*math.sqrt((1/(float(j) + 1))),zorder = 3)

                        ax2.scatter(j,g_val,s = 30,color = colorspec[j],edgecolor = 'k',linewidth = 0.7,zorder = 3)            # plot point of tangency

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
                ax.set_ylabel('$w_2$',fontsize = 12,rotation = 0)
                ax.axhline(y=0, color='k',zorder = 0,linewidth = 0.5)
                ax.axvline(x=0, color='k',zorder = 0,linewidth = 0.5)
                ax.set_xlim([xmin,xmax])
                ax.set_ylim([ymin,ymax])
                
                
            else:    # function is single input, plot curve
                if 'xmin' in kwargs:
                    xmin = kwargs['xmin']
                if 'xmax' in kwargs:
                    xmax = kwargs['xmax']
                    
                w_plot = np.linspace(xmin,xmax,500)
                g_plot = self.g(w_plot)
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
            
                    ax.scatter(w_val,g_val,s = 90,color = colorspec[j],edgecolor = 'k',linewidth = 0.5*((1/(float(j) + 1)))**(0.4),zorder = 3,marker = 'X')            # evaluation on function
                    ax.scatter(w_val,0,s = 90,facecolor = colorspec[j],edgecolor = 'k',linewidth = 0.5*((1/(float(j) + 1)))**(0.4), zorder = 3)
                    
                    ax2.scatter(j,g_val,s = 30,color = colorspec[j],edgecolor = 'k',linewidth = 0.7,zorder = 3)            # plot point of tangency
                    
                    # plot connector between points for visualization purposes
                    if j > 0:
                        w_old = self.w_hist[j-1]
                        w_new = self.w_hist[j]
                        g_old = self.g(w_old)
                        g_new = self.g(w_new)
     
                        ax2.plot([j-1,j],[g_old,g_new],color = colorspec[j],linewidth = 2,alpha = 1,zorder = 2)      # plot approx
                        ax2.plot([j-1,j],[g_old,g_new],color = 'k',linewidth = 2.5,alpha = 1,zorder = 1)      # plot approx
            
            if axes == True:
                ax.axhline(linestyle = '--', color = 'k',linewidth = 1)
                ax.axvline(linestyle = '--', color = 'k',linewidth = 1)

            # clean panels
            title = self.steplength
            if type(self.steplength) == float or type(self.steplength) == int:
                title = r'$\alpha = $' + str(self.steplength)
            ax.set_title(title,fontsize = 12)

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
   

# simple first order taylor series visualizer
class grad_descent_as_quadratic_visualizer:
    '''
    Illustrate gradient descent as a minimization technique using simple quadratic surrogates.  
    Visual comparison with the standard linear surrogate view.
    ''' 
    def __init__(self,**args):
        self.g = args['g']                            # input function
        self.grad = compute_grad(self.g)              # gradient of input function
        self.hess = compute_grad(self.grad)           # hessian of input function
        self.w_init =float( -2)                       # user-defined initial point (adjustable when calling each algorithm)
        self.alpha = 10**-4                           # user-defined step length for gradient descent (adjustable when calling gradient descent)
        self.max_its = 20                             # max iterations to run for each algorithm
        self.colors = [[0,1,0.25],[0,0.75,1],[1,0.75,0],[1,0,0.75]]       # custom colors for plotting
        
    ######## gradient descent ########
    # run gradient descent 
    def run_gradient_descent(self):
        w = self.w_init
        self.w_hist = []
        self.w_hist.append(w)
        w_old = np.inf
        j = 0
        while (w_old - w)**2 > 10**-5 and j < self.max_its:
            # update old w and index
            w_old = w
            j+=1
            
            # plug in value into func and derivative
            grad_eval = float(self.grad(w))
            
            # take gradient descent step
            w = w - self.alpha*grad_eval
            
            # record
            self.w_hist.append(w)

    # animate the gradient descent method
    def animate_it(self,savepath,**kwargs):
        # get new initial point if desired
        if 'w_init' in kwargs:
            self.w_init = float(kwargs['w_init'])
            
        # take in user defined step length
        if 'alpha' in kwargs:
            self.alpha = float(kwargs['alpha'])
            
        # take in user defined maximum number of iterations
        if 'max_its' in kwargs:
            self.max_its = float(kwargs['max_its'])
            
        # viewing range
        wmax = 5
        if 'wmax' in kwargs:
            wmax = kwargs['wmax']
            
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
        w_plot = np.linspace(-wmax,wmax,1000)
        g_plot = self.g(w_plot)
        g_range = max(g_plot) - min(g_plot)
        ggap = g_range*0.25
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
        print ('starting animation rendering...')
        num_frames = len(self.w_hist)
        def animate(k):
            ax.cla()
            
            # print rendering update            
            if np.mod(k+1,25) == 0:
                print ('rendering animation frame ' + str(k+1) + ' of ' + str(num_frames))
            if k == num_frames - 1:
                print ('animation rendering complete!')
                time.sleep(1.5)
                clear_output()
            
            # plot function
            ax.plot(w_plot,g_plot,color = 'k',zorder = 2)                           # plot function
            
            # plot initial point and evaluation
            if k == 0:
                w_val = self.w_init
                g_val = self.g(w_val)
                ax.scatter(w_val,g_val,s = 100,color = 'r',edgecolor = 'k',linewidth = 0.7,zorder = 3)            # plot point of tangency
                ax.scatter(w_val,0,s = 100,color = 'r',edgecolor = 'k',linewidth = 0.7, zorder = 3, marker = 'X')
                # draw dashed line connecting w axis to point on cost function
                s = np.linspace(0,g_val)
                o = np.ones((len(s)))
                ax.plot(o*w_val,s,'k--',linewidth=1)

            # plot all input/output pairs generated by algorithm thus far
            if k > 0 and k < len(self.w_hist) + 1:
                # plot all points up to this point
                for j in range(min(k,len(self.w_hist))):  
                    alpha_val = 1
                    if j < k-1:
                        alpha_val = 0.1
                        
                    # get next value of weight, function and gradient evaluation
                    w_val = self.w_hist[j]
                    g_val = self.g(w_val)
                    grad_val = float(self.grad(w_val))
                    
                    # plot current point
                    ax.scatter(w_val,g_val,s = 90,color = 'r',edgecolor = 'k',linewidth = 0.7,zorder = 3,alpha = alpha_val)            # plot point of tangency
                    ax.scatter(w_val,0,s = 90,facecolor = 'r',marker = 'X',edgecolor = 'k',linewidth = 0.7, zorder = 3,alpha = alpha_val)
                    
                    #### plot linear surrogate ####
                    # determine width to plot the approximation -- so its length == width defined above
                    div = float(1 + grad_val**2)
                    w1 = w_val - math.sqrt(width/div)
                    w2 = w_val + math.sqrt(width/div)

                    # use point-slope form of line to plot
                    wrange = np.linspace(w1,w2, 100)
                    h = g_val + grad_val*(wrange - w_val)

                    # plot tangent line
                    ax.plot(wrange,h,color = self.colors[0],linewidth = 2,zorder = 1,alpha = alpha_val)      # plot approx

                    # plot tangent point
                    ax.scatter(w_val,g_val,s = 100,color = 'r',edgecolor = 'k',linewidth = 0.7,zorder = 3,alpha = alpha_val)            # plot point of tangency

                    # plot next point learned from surrogate
                    # create next point information
                    w_zero = w_val - self.alpha*grad_val
                    g_zero = self.g(w_zero)
                    h_zero = g_val + grad_val*(w_zero - w_val)

                    # draw intersection at zero and associated point on cost function you hop back too
                    ax.scatter(w_zero,h_zero,s = 100,color = self.colors[0],edgecolor = 'k',linewidth = 0.7, zorder = 3, marker = 'X',alpha = alpha_val)
                    ax.scatter(w_zero,0,s = 100,color = 'r',edgecolor = 'k',linewidth = 0.7, zorder = 3, marker = 'X',alpha = alpha_val)
                    ax.scatter(w_zero,g_zero,s = 100,color = 'r',edgecolor = 'k',linewidth = 0.7,zorder = 3,alpha = alpha_val)            # plot point of tangency
                    
                    ### draw simple quadratic surrogate ###
                    # decide on range for quadratic so it looks nice
                    quad_term = 1/float(2*self.alpha)
                    a = 0.5*quad_term
                    b = grad_val - 2*0.5*quad_term*w_val
                    c = 0.5*quad_term*w_val**2 - grad_val*w_val - width

                    # solve for zero points
                    w1 = (-b + math.sqrt(b**2 - 4*a*c))/float(2*a + 0.00001)
                    w2 = (-b - math.sqrt(b**2 - 4*a*c))/float(2*a + 0.00001)
                    wrange = np.linspace(w1,w2, 100)

                    # create simple quadratic surrogate
                    h = g_val + grad_val*(wrange - w_val) + quad_term*(wrange - w_val)**2
                    
                    # plot simple quadratic surrogate
                    ax.plot(wrange,h,color = self.colors[1],linewidth = 2,zorder = 1,alpha = alpha_val)      # plot approx

                    # plot point of intersection - next gradient descent step - on simple quadratic surrogate
                    h_zero_2 = g_val + grad_val*(w_zero - w_val) + 1/float(2*self.alpha)*(w_zero - w_val)**2

                    ax.scatter(w_zero,h_zero_2,s = 100,color = self.colors[1],edgecolor = 'k',linewidth = 0.7, zorder = 3, marker = 'X',alpha = alpha_val)
                    
                    # draw dashed line connecting w axis to point on cost function
                    s = np.linspace(0,g_val)
                    o = np.ones((len(s)))
                    ax.plot(o*w_val,s,'k--',linewidth=1,alpha = alpha_val)

                    vals = [0,h_zero,h_zero_2,g_zero]
                    vals = np.sort(vals)

                    s = np.linspace(vals[0],vals[3])
                    o = np.ones((len(s)))
                    w_val = self.w_hist[j+1]
                    ax.plot(o*w_val,s,'k--',linewidth=1,alpha = alpha_val)
                 
            # fix viewing limits
            ax.set_xlim([-wmax,wmax])
            ax.set_ylim([min(g_plot) - ggap,max(g_plot) + ggap])
            ax.axhline(y=0, color='k',zorder = 0,linewidth = 0.5)
            ax.set_xlabel(r'$w$',fontsize=12)
            ax.set_ylabel(r'$g(w)$',fontsize=12,rotation = 0,labelpad = 20)
            
            return artist,

        anim = animation.FuncAnimation(fig, animate ,frames=num_frames, interval=num_frames, blit=True)
        
        # produce animation and save
        fps = 50
        if 'fps' in kwargs:
            fps = kwargs['fps']
        anim.save(savepath, fps=fps, extra_args=['-vcodec', 'libx264'])
        clear_output()


class lipschitz_majorizer_visualizer:
    '''
    Illustrate majorization of lipschitz gradient-based quadratic majorizer of an input function
    '''
    def __init__(self,**args):
        self.g = args['g']                       # input function
        self.grad = compute_grad(self.g)         # gradient of input function
        self.hess = compute_grad(self.grad)      # hessian of input function
        self.colors = [[0,1,0.25],[0,0.75,1]]    # set of custom colors used for plotting

    # compute first order approximation
    def animate_it(self,savepath,**kwargs):
        num_frames = 100                          # number of slides to create - the input range [-3,3] is divided evenly by this number
        if 'num_frames' in kwargs:
            num_frames = kwargs['num_frames']
            
        # initialize figure
        fig = plt.figure(figsize = (10,5))
        artist = fig

        # create subplot with 3 panels, plot input function in center plot
        gs = gridspec.GridSpec(1, 3, width_ratios=[1,5, 1]) 
        ax1 = plt.subplot(gs[0]); ax1.axis('off');
        ax3 = plt.subplot(gs[2]); ax3.axis('off');

        # plot input function
        ax = plt.subplot(gs[1],aspect = 'equal')
        
        max_val = 2.5
        if 'max_val' in kwargs:
            max_val = kwargs['max_val']
        w_vals = np.linspace(-max_val,max_val,num_frames)       # range of values over which to plot first / second order approximations
        
        # generate a range of values over which to plot input function, and derivatives
        w_plot = np.linspace(-max_val-0.5,max_val+0.5,200)                  # input range for original function
        
        g_plot = self.g(w_plot)
        g_range = max(g_plot) - min(g_plot)             # used for cleaning up final plot
        ggap = g_range*0.1
        
        # estimate Lipschitz constant over input range
        w_temp = np.linspace(-max_val-0.5,max_val+0.5,2000)
        hess_vals = [abs(self.hess(s)) for s in w_temp]
        L = max(hess_vals)
        alpha = 1/float(L)
     
        # animation sub-function
        print ('starting animation rendering...')
        def animate(k):
            # clear the panel
            ax.cla()
            
            # print rendering update
            if np.mod(k+1,25) == 0:
                print ('rendering animation frame ' + str(k+1) + ' of ' + str(num_frames))
            if k == num_frames - 1:
                print ('animation rendering complete!')
                time.sleep(1.5)
                clear_output()
            
            # grab the next input/output tangency pair, minimum of gradient quadratic
            w_val = w_vals[k]
            g_val = self.g(w_val)
            grad_val = self.grad(w_val)
            w_step = w_val - alpha*grad_val
            g_step = self.g(w_step)

            # plot original function
            ax.plot(w_plot,g_plot,color = 'k',zorder = 0,linewidth=1)                           # plot function
       
            # create and Lipschitz majorizer centered on w_val
            h = lambda w: g_val + grad_val*(w - w_val) + 1/(2*alpha)*(w - w_val)**2
            width = 2*max_val
            w_major = np.linspace(w_step - width,w_step + width,200)
            h_major = h(w_major)
            h_step = h(w_step)
            
            # plot majorizer
            ax.plot(w_major,h_major,color = self.colors[1],zorder = 1,linewidth=2)   
            
            # plot all points
            ax.scatter(w_step,h_step,s = 60,color = 'blue',edgecolor = 'k',linewidth = 0.7,marker = 'X',zorder = 3)           
            ax.scatter(w_step,g_step,s = 60,color = 'lime',edgecolor = 'k',linewidth = 0.7,marker = 'X',zorder = 3)            # plot point of tangency
            ax.scatter(w_step,0,s = 80,color = 'lime',edgecolor = 'k',linewidth = 0.7,zorder = 3)            # plot point of tangency
            ax.scatter(w_val,g_val,s = 60,color = 'r',edgecolor = 'k',linewidth = 0.7,marker = 'X',zorder = 3)            # plot point of tangency
            ax.scatter(w_val,0,s = 80,color = 'r',edgecolor = 'k',linewidth = 0.7,zorder = 3)            # plot point of tangency
            
            # plot visual aid for old point
            tempy = np.linspace(0,g_val,100)
            tempx = w_val*np.ones((100))
            ax.plot(tempx,tempy,linewidth = 0.7,color = 'k',linestyle = '--',zorder = 1)
            
            # plot visual aid for new point
            tempy = np.linspace(0,h_step,100)
            tempx = w_step*np.ones((100))
            ax.plot(tempx,tempy,linewidth = 0.7,color = 'k',linestyle = '--',zorder = 1)
            
            # label axes
            ax.set_xlabel('$w$',fontsize = 12)
            ax.set_ylabel('$g(w)$',fontsize = 12,rotation = 0,labelpad = 12)

            # fix viewing limits on panel
            ax.set_xlim([-max_val,max_val])
            ax.set_ylim([min(-0.3,min(g_plot) - ggap),max(max(g_plot) + ggap,0.3)])
            
            # set tickmarks
            ax.set_xticks(-np.arange(-round(max_val), round(max_val) + 1, 1.0))
            ax.set_yticks(np.arange(round(min(g_plot) - ggap), round(max(g_plot) + ggap) + 1, 1.0))
                
            # set axis 
            ax.axhline(y=0, color='k',zorder = 0,linewidth = 0.5)
            
            return artist,
        
        anim = animation.FuncAnimation(fig, animate,frames=len(w_vals), interval=len(w_vals), blit=True)
        
        # produce animation and save
        fps = 50
        if 'fps' in kwargs:
            fps = kwargs['fps']
        anim.save(savepath, fps=fps, extra_args=['-vcodec', 'libx264'])
        clear_output()




