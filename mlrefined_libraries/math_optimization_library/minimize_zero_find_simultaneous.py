# import standard plotting and animation
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from matplotlib import gridspec

# import autograd functionality
from autograd import grad as compute_grad   # The only autograd function you may ever need
import autograd.numpy as np
import math
from IPython.display import clear_output
import time
import copy


# simple first order taylor series visualizer
class visualizer:
    '''
    Illustrate Newton's and Secant method for zero-finding with a customized slider mechanism
    to let user control progression of algorithms.  Both function minimization and derivative
    zero-finding side-by-side simultaneously.
    ''' 
    def __init__(self,**args):
        self.g = args['g']                             # input function
        self.grad = compute_grad(self.g)               # first derivative of input function
        self.hess = compute_grad(self.grad)            # second derivative of input function
        self.w_init =float( -3)                        # user-defined initial point
        self.max_its = 20
     
    ######## newton's method ########
    # run newton's method
    def run_newtons_method(self):
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
            hess_eval = float(self.hess(w))

            # take newtons step
            w = w - grad_eval/(hess_eval + 10**-5)
            
            # record
            self.w_hist.append(w)

    # animate the method
    def draw_it_newtons(self,savepath,**kwargs):
        # user-defined input point
        if 'w_init' in kwargs:
            self.w_init = float(kwargs['w_init'])
            
        # user-defined max_its
        if 'max_its' in kwargs:
            self.max_its = float(kwargs['max_its'])
            
        # initialize figure
        fig = plt.figure(figsize = (12,4))
        artist = fig
        
        # create subplot with 3 panels, plot input function in center plot
        gs = gridspec.GridSpec(1, 2, width_ratios=[1,1]) 
        ax1 = plt.subplot(gs[0]); 
        ax2 = plt.subplot(gs[1]);       
        artist = fig

        # generate function for plotting on each slide
        w_plot = np.linspace(-3.1,3.1,200)
        g_plot = self.g(w_plot)
        grad_plot = [self.grad(v) for v in w_plot]
        g_range = max(g_plot) - min(g_plot)
        ggap = g_range*0.1
        w_vals = np.linspace(-2.5,2.5,50)
        
        # run newtons method
        self.w_hist = []
        self.run_newtons_method()
        
        # colors for points
        s = np.linspace(0,1,len(self.w_hist[:round(len(self.w_hist)/2)]))
        s.shape = (len(s),1)
        t = np.ones(len(self.w_hist[round(len(self.w_hist)/2):]))
        t.shape = (len(t),1)
        s = np.vstack((s,t))
        self.colorspec = []
        self.colorspec = np.concatenate((s,np.flipud(s)),1)
        self.colorspec = np.concatenate((self.colorspec,np.zeros((len(s),1))),1)
        
        # animation sub-function
        print ('beginning animation rendering...')
        def animate(k):
            ax1.cla()
            ax2.cla()
                        
            # print rendering update
            if k == len(self.w_hist):
                print ('animation rendering complete!')
                time.sleep(1.5)
                clear_output()
            
            # plot functions
            ax1.plot(w_plot,g_plot,color = 'k',zorder = 0)               # plot function
            ax2.plot(w_plot,grad_plot,color = 'k',zorder = 2)                           # plot function
            
            # plot initial point and evaluation
            if k == 0:
                w_val = self.w_init
                g_val = self.g(w_val)
                ax1.scatter(w_val,g_val,s = 120,c = 'w',edgecolor = 'k',linewidth = 0.7,zorder = 3)            # plot point of tangency
                ax1.scatter(w_val,0,s = 120,c = 'w',edgecolor = 'k',linewidth = 0.7, zorder = 2, marker = 'X')

                g_val = self.grad(w_val)
                ax2.scatter(w_val,g_val,s = 120,c = 'w',edgecolor = 'k',linewidth = 0.7,zorder = 3)            # plot point of tangency
                ax2.scatter(w_val,0,s = 120,c = 'w',edgecolor = 'k',linewidth = 0.7, zorder = 2, marker = 'X')
            
            # draw functions first, then start animating process
            if k > 0:
                #### cost function (minimizing) view ####
                w_val = self.w_hist[k-1]

                # plug in value into func and derivative
                g_val = self.g(w_val)
                g_grad_val = self.grad(w_val)
                g_hess_val = self.hess(w_val)

                # determine width of plotting area for second order approximator
                width = 5
                if g_hess_val < 0:
                    width = - width

                # setup quadratic formula params
                a = 0.5*g_hess_val
                b = g_grad_val - 2*0.5*g_hess_val*w_val
                c = 0.5*g_hess_val*w_val**2 - g_grad_val*w_val - width

                # solve for zero points
                w1 = (-b + math.sqrt(b**2 - 4*a*c))/float(2*a + 0.00001)
                w2 = (-b - math.sqrt(b**2 - 4*a*c))/float(2*a + 0.00001)

                # compute second order approximation
                wrange = np.linspace(w1,w2, 100)
                h = g_val + g_grad_val*(wrange - w_val) + 0.5*g_hess_val*(wrange - w_val)**2 

                # create next point information
                w_zero = w_val - g_grad_val/(g_hess_val + 10**-5)
                g_zero = self.g(w_zero)
                h_zero = g_val + g_grad_val*(w_zero - w_val) + 0.5*g_hess_val*(w_zero - w_val)**2

                # draw dashed linen connecting the three
                vals = [0,h_zero,g_zero]
                vals = np.sort(vals)
                s = np.linspace(vals[0],vals[2])
                o = np.ones((len(s)))

                # plot all
                ax1.plot(wrange,h,color = self.colorspec[k-1],linewidth = 2,zorder = 1)      # plot approx

                # plot tangent point
                ax1.scatter(w_val, g_val, s = 120, c='w',edgecolor = 'k',linewidth = 1,zorder = 3)

                # created dashed linen connecting the three            
                ax1.plot(o*w_zero,s,'k--',linewidth=1)

                # draw intersection at zero and associated point on cost function you hop back too
                # ax1.scatter(w_zero,h_zero,s = 120,c = 'k', zorder = 2)
                ax1.scatter(w_zero,g_zero,s = 120,c = self.colorspec[k-1],edgecolor = 'k',linewidth = 1,zorder = 3)            # plot point of tangency
                ax1.scatter(w_zero,0,s = 120,facecolor = self.colorspec[k-1],marker = 'X',edgecolor = 'k',linewidth = 1, zorder = 2)

                #### derivative (zero-crossing) view ####
                # grab historical weight, compute function and derivative evaluations
                g_val = float(self.grad(w_val))
                grad_val = float(self.hess(w_val))
                h = g_val + grad_val*(wrange - w_val)

                # draw points
                w_zero = -g_val/grad_val + w_val
                g_zero = self.grad(w_zero)
                s = np.linspace(0,g_zero)
                o = np.ones((len(s)))

                # plot tangent line
                ax2.plot(wrange,h,color = self.colorspec[k-1],linewidth = 2,zorder = 1)      # plot approx

                # plot tangent point
                ax2.scatter(w_val, g_val, s = 120, c='w',edgecolor = 'k',linewidth = 1,zorder = 3)

                # draw dashed lines to highlight zero crossing point
                ax2.plot(o*w_zero,s,'k--',linewidth=1)

                # draw intersection at zero and associated point on cost function you hop back too
                ax2.scatter(w_zero,g_zero,s = 120,c = self.colorspec[k-1],edgecolor = 'k',linewidth = 0.7,zorder = 3)            # plot point of tangency
                ax2.scatter(w_zero,0,s = 120,facecolor = self.colorspec[k-1],marker = 'X',edgecolor = 'k',linewidth = 0.7, zorder = 2)
             
            # fix viewing limits
            ax1.set_xlim([-3,3])
            ax1.set_ylim([min(g_plot) - ggap,max(g_plot) + ggap])
                
            # fix viewing limits
            ax2.set_xlim([-3,3])
            ax2.set_ylim([min(g_plot) - ggap,max(g_plot) + ggap])

            # set titles
            ax1.set_title('cost function (minimizing) view',fontsize = 15)
            ax2.set_title('gradient (zero-crossing) view',fontsize = 15)
   
            # draw axes
            ax1.axhline(y=0, color='k',zorder = 0,linewidth = 0.5)
            ax2.axhline(y=0, color='k',zorder = 0,linewidth = 0.5)

            return artist,

        anim = animation.FuncAnimation(fig, animate,frames=len(self.w_hist)+1, interval=len(self.w_hist)+1, blit=True)

        # produce animation and save
        fps = 50
        if 'fps' in kwargs:
            fps = kwargs['fps']
        anim.save(savepath, fps=fps, extra_args=['-vcodec', 'libx264'])
        clear_output()

    ######## secant method #########
    # run secant method
    def run_secant_method(self):
        # get initial point
        w2 = self.w_init
        
        # create second point nearby w_old
        w1 = w2 - 0.5
        g2 = self.g(w2)
        g1 = self.g(w1)
        if g1 > g2:
            w1 = w2 + 0.5
        
        # setup container for history
        self.w_hist = []
        self.w_hist.append(w2)
        self.w_hist.append(w1)
        
        # start loop
        w_old = np.inf
        j = 0
        while abs(w1 - w2) > 10**-5 and j < self.max_its:  
            # plug in value into func and derivative
            g1 = float(self.grad(w1))
            g2 = float(self.grad(w2))
                        
            # take newtons step
            w = w1 - g1*(w1 - w2)/(g1 - g2 + 10**-6)
            
            # record
            self.w_hist.append(w)
            
            # update old w and index
            j+=1
            w2 = w1
            w1 = w
    
    # animate the method
    def draw_it_secant(self,savepath,**kwargs):
        # user-defined input point
        if 'w_init' in kwargs:
            self.w_init = float(kwargs['w_init'])
        
        # user-defined max_its
        if 'max_its' in kwargs:
            self.max_its = float(kwargs['max_its'])
            
        # initialize figure
        fig = plt.figure(figsize = (12,4))
        
        # create subplot with 3 panels, plot input function in center plot
        gs = gridspec.GridSpec(1, 2, width_ratios=[1,1]) 
        ax1 = plt.subplot(gs[0]); 
        ax2 = plt.subplot(gs[1]); 

        artist = fig

        # generate function for plotting on each slide
        w_plot = np.linspace(-3.1,3.1,200)
        g_plot = self.g(w_plot)
        grad_plot = [self.grad(v) for v in w_plot]
        g_range = max(g_plot) - min(g_plot)
        ggap = g_range*0.1
        w_vals = np.linspace(-2.5,2.5,50)
        width = 5
        
        # run newtons method
        self.w_hist = []
        self.run_secant_method()
    
        # colors for points
        s = np.linspace(0,1,len(self.w_hist[:round(len(self.w_hist)/2)]))
        s.shape = (len(s),1)
        t = np.ones(len(self.w_hist[round(len(self.w_hist)/2):]))
        t.shape = (len(t),1)
        s = np.vstack((s,t))
        self.colorspec = []
        self.colorspec = np.concatenate((s,np.flipud(s)),1)
        self.colorspec = np.concatenate((self.colorspec,np.zeros((len(s),1))),1)
        
        # animation sub-function
        print ('beginning animation rendering...')
        def animate(k):
            ax1.cla()
            ax2.cla()
            
            # print rendering update
            if k == len(self.w_hist)-1:
                print ('animation rendering complete!')
                time.sleep(1.5)
                clear_output()
            
            # plot functions
            ax1.plot(w_plot,g_plot,color = 'k',zorder = 0)               # plot function
            ax2.plot(w_plot,grad_plot,color = 'k',zorder = 2)                           # plot function

            # plot initial point and evaluation
            if k == 0:
                w_val = self.w_init
                g_val = self.g(w_val)
                ax1.scatter(w_val,g_val,s = 120,c = 'w',edgecolor = 'k',linewidth = 1.0,zorder = 3)            # plot point of tangency
                ax1.scatter(w_val,0,s = 120,c = 'w',edgecolor = 'k',linewidth = 1.0, zorder = 2, marker = 'X')

                g_val = self.grad(w_val)
                ax2.scatter(w_val,g_val,s = 120,c = 'w',edgecolor = 'k',linewidth = 1.0,zorder = 3)            # plot point of tangency
                ax2.scatter(w_val,0,s = 120,c = 'w',edgecolor = 'k',linewidth = 1.0, zorder = 2, marker = 'X')
            
            # plot functions first for one slide
            if k > 0:
                #### cost function (minimizing) view ####
                # grab historical weights, form associated secant line
                w2 = self.w_hist[k-1]
                w1 = self.w_hist[k]
                g2 = self.g(w2)
                g1 = self.g(w1)
                grad2 = self.grad(w2)
                grad1 = self.grad(w1)

                # determine width of plotting area for second order approximator
                width = 5
                g_hess_val = (grad1 - grad2)/(w1 - w2)
                if g_hess_val < 0:
                    width = - width

                # setup quadratic formula params
                a = 0.5*g_hess_val
                b = grad1 - 2*0.5*g_hess_val*w1
                c = 0.5*g_hess_val*w1**2 - grad1*w1 - width

                # solve for zero points
                wa = (-b + math.sqrt(b**2 - 4*a*c))/float(2*a + 0.00001)
                wb = (-b - math.sqrt(b**2 - 4*a*c))/float(2*a + 0.00001)

                # compute second order approximation
                wrange = np.linspace(wa,wb, 100)
                h = g1 + grad1*(wrange - w1) + 0.5*g_hess_val*(wrange - w1)**2 

                # create next point information
                w_zero = w1 - grad1/(g_hess_val + 10**-5)
                g_zero = self.g(w_zero)
                h_zero = g1 + grad1*(w_zero - w1) + 0.5*g_hess_val*(w_zero - w1)**2

                # draw dashed linen connecting the three
                vals = [0,h_zero,g_zero]
                vals = np.sort(vals)
                s = np.linspace(vals[0],vals[2])
                o = np.ones((len(s)))

                # plot all
                ax1.plot(wrange,h,color =  self.colorspec[k-1] ,linewidth = 2,zorder = 1)      # plot approx
                ax1.scatter(w1,g1,s = 120,c = 'w',edgecolor = 'k',linewidth = 1.0,zorder = 3)           # plot point of tangency
                ax1.scatter(w2,g2,s = 120,c = 'w',edgecolor = 'k',linewidth = 1.0,zorder = 3)           # plot point of tangency
                ax1.plot(o*w_zero,s,'k--',linewidth=1)

                # draw intersection at zero and associated point on cost function you hop back too
                # ax1.scatter(w_zero,h_zero,s = 120,c = 'k', zorder = 2)
                ax1.scatter(w_zero,g_zero,s = 120,c = self.colorspec[k-1],edgecolor = 'k',linewidth = 0.7,zorder = 3)            # plot point of tangency
                ax1.scatter(w_zero,0,s = 120,facecolor = self.colorspec[k-1],marker = 'X',edgecolor = 'k',linewidth = 0.7, zorder = 2)

                #### derivative (zero-crossing) view ####
                # grab historical weights, form associated secant line
                w2 = self.w_hist[k-1]
                w1 = self.w_hist[k]
                g2 = self.grad(w2)
                g1 = self.grad(w1)
                m = (g1 - g2)/(w1 - w2)

                # use point-slope form of line to plot
                h = g1 + m*(wrange - w1)

                # create dashed line and associated points
                w_zero = -g1/m + w1
                g_zero = self.grad(w_zero)
                s = np.linspace(0,g_zero)
                o = np.ones((len(s)))

                # plot secant line 
                ax2.plot(wrange,h,color =  self.colorspec[k-1],linewidth = 2,zorder = 1)      # plot approx

                # plot intersection points
                ax2.scatter(w1,g1,s = 120,c = 'w',edgecolor = 'k',linewidth = 1.0,zorder = 3)           # plot point of tangency
                ax2.scatter(w2,g2,s = 120,c = 'w',edgecolor = 'k',linewidth = 1.0,zorder = 3)           # plot point of tangency

                # plot dashed line
                ax2.plot(o*w_zero,s,'k--',linewidth=1)

                # draw zero intersection, and associated point on cost function you hop back too
                ax2.scatter(w_zero,g_zero,s = 120,c = self.colorspec[k-1],edgecolor = 'k',linewidth = 0.7,zorder = 3)            # plot point of tangency
                ax2.scatter(w_zero,0,s = 120,facecolor = self.colorspec[k-1],marker = 'X',edgecolor = 'k',linewidth = 0.7, zorder = 2)

            # fix viewing limits
            ax1.set_xlim([-3,3])
            ax1.set_ylim([min(g_plot) - ggap,max(g_plot) + ggap])
            
            ax2.set_xlim([-3.1,3.1])
            ax2.set_ylim([min(g_plot) - ggap,max(g_plot) + ggap])
            
            # draw axes
            ax1.axhline(y=0, color='k',zorder = 0,linewidth = 0.5)
            ax2.axhline(y=0, color='k',zorder = 0,linewidth = 0.5)
                
            # place title
            ax1.set_title('cost function (minimizing) view',fontsize = 15)
            ax2.set_title('gradient (zero-crossing) view',fontsize = 15)

            return artist,

        anim = animation.FuncAnimation(fig, animate,frames=len(self.w_hist), interval=len(self.w_hist), blit=True)

        # produce animation and save
        fps = 50
        if 'fps' in kwargs:
            fps = kwargs['fps']
        anim.save(savepath, fps=fps, extra_args=['-vcodec', 'libx264'])
        clear_output()