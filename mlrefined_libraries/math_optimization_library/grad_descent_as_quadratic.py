# import standard plotting and animation
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from IPython.display import clear_output
import time

# import autograd functionality
from autograd import grad as compute_grad   # The only autograd function you may ever need
import autograd.numpy as np
import math
from matplotlib import gridspec

# simple first order taylor series visualizer
class visualizer:
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
                ax.scatter(w_val,g_val,s = 100,c = 'r',edgecolor = 'k',linewidth = 0.7,zorder = 3)            # plot point of tangency
                ax.scatter(w_val,0,s = 100,c = 'r',edgecolor = 'k',linewidth = 0.7, zorder = 3, marker = 'X')
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
                    ax.scatter(w_val,g_val,s = 90,c = 'r',edgecolor = 'k',linewidth = 0.7,zorder = 3,alpha = alpha_val)            # plot point of tangency
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
                    ax.scatter(w_val,g_val,s = 100,c = 'r',edgecolor = 'k',linewidth = 0.7,zorder = 3,alpha = alpha_val)            # plot point of tangency

                    # plot next point learned from surrogate
                    # create next point information
                    w_zero = w_val - self.alpha*grad_val
                    g_zero = self.g(w_zero)
                    h_zero = g_val + grad_val*(w_zero - w_val)

                    # draw intersection at zero and associated point on cost function you hop back too
                    ax.scatter(w_zero,h_zero,s = 100,c = self.colors[0],edgecolor = 'k',linewidth = 0.7, zorder = 3, marker = 'X',alpha = alpha_val)
                    ax.scatter(w_zero,0,s = 100,c = 'r',edgecolor = 'k',linewidth = 0.7, zorder = 3, marker = 'X',alpha = alpha_val)
                    ax.scatter(w_zero,g_zero,s = 100,c = 'r',edgecolor = 'k',linewidth = 0.7,zorder = 3,alpha = alpha_val)            # plot point of tangency
                    
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

                    ax.scatter(w_zero,h_zero_2,s = 100,c = self.colors[1],edgecolor = 'k',linewidth = 0.7, zorder = 3, marker = 'X',alpha = alpha_val)
                    
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