import matplotlib.pyplot as plt
from matplotlib import gridspec
import matplotlib.animation as animation
from mpl_toolkits.mplot3d import Axes3D
from IPython.display import clear_output

from autograd import grad as compute_grad 
import autograd.numpy as np
from autograd import hessian as compute_hess
from autograd.misc.flatten import flatten_func
import math
import time
import copy


class MyOptimizers:
    '''
    A list of current optimizers.  In each case - since these are used for educational purposes - the weights at each step are recorded and returned.
    '''
    ### gradient descent ###
    def gradient_descent(self,g,w,**kwargs):                
        # create gradient function
        self.g = g
        self.grad = compute_grad(self.g)
        
        # parse optional arguments        
        max_its = 100
        if 'max_its' in kwargs:
            max_its = kwargs['max_its']
        version = 'unnormalized'
        if 'version' in kwargs:
            version = kwargs['version']
        alpha = 10**-4
        if 'alpha' in kwargs:
            alpha = kwargs['alpha']
        steplength_rule = 'none'    
        if 'steplength_rule' in kwargs:
            steplength_rule = kwargs['steplength_rule']
        projection = 'None'
        if 'projection' in kwargs:
            projection = kwargs['projection']
        verbose = False
        if 'verbose' in kwargs:
            verbose = kwargs['verbose']
       
        # create container for weight history 
        w_hist = []
        w_hist.append(w)
        
        # start gradient descent loop
        if verbose == True:
            print ('starting optimization...')
        for k in range(max_its):   
            # plug in value into func and derivative
            grad_eval = self.grad(w)
            grad_eval.shape = np.shape(w)
            
            ### normalized or unnormalized descent step? ###
            if version == 'normalized':
                grad_norm = np.linalg.norm(grad_eval)
                if grad_norm == 0:
                    grad_norm += 10**-6*np.sign(2*np.random.rand(1) - 1)
                grad_eval /= grad_norm
            
            # use backtracking line search?
            if steplength_rule == 'backtracking':
                alpha = self.backtracking(w,grad_eval)
                
            # use a pre-set diminishing steplength parameter?
            if steplength_rule == 'diminishing':
                alpha = 1/(float(k + 1))
            
            ### take gradient descent step ###
            w = w - alpha*grad_eval
            
            # record
            w_hist.append(w)     
         
        if verbose == True:
            print ('...optimization complete!')
            time.sleep(1.5)
            clear_output()
        
        return w_hist

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
            
    #### newton's method ####            
    def newtons_method(self,g,w,**kwargs):        
        # create gradient and hessian functions
        self.g = g
        
        # flatten gradient for simpler-written descent loop
        flat_g, unflatten, w = flatten_func(self.g, w)
        
        self.grad = compute_grad(flat_g)
        self.hess = compute_hess(flat_g)  
        
        # parse optional arguments        
        max_its = 20
        if 'max_its' in kwargs:
            max_its = kwargs['max_its']
        self.epsilon = 10**(-5)
        if 'epsilon' in kwargs:
            self.epsilon = kwargs['epsilon']
        verbose = False
        if 'verbose' in kwargs:
            verbose = kwargs['verbose']
        
        # create container for weight history 
        w_hist = []
        w_hist.append(unflatten(w))
        
        # start newton's method loop  
        if verbose == True:
            print ('starting optimization...')
            
        geval_old = flat_g(w)
        for k in range(max_its):
            # compute gradient and hessian
            grad_val = self.grad(w)
            hess_val = self.hess(w)
            hess_val.shape = (np.size(w),np.size(w))

            # solve linear system for weights
            w = w - np.dot(np.linalg.pinv(hess_val + self.epsilon*np.eye(np.size(w))),grad_val)
                    
            # eject from process if reaching singular system
            geval_new = flat_g(w)
            if k > 2 and geval_new > geval_old:
                print ('singular system reached')
                time.sleep(1.5)
                clear_output()
                return w_hist
            else:
                geval_old = geval_new
                
            # record current weights
            w_hist.append(unflatten(w))
            
        if verbose == True:
            print ('...optimization complete!')
            time.sleep(1.5)
            clear_output()
        
        return w_hist


class visualizer:
    '''
    Visualize classification on a 2-class dataset with N = 2
    '''
    #### initialize ####
    def __init__(self,data):
        # grab input
        self.data = data
        self.x = data[:,:-1]
        self.y = data[:,-1]
        
        # colors for viewing classification data 'from above'
        self.colors = ['cornflowerblue','salmon','lime','bisque','mediumaquamarine','b','m','g']

    def center_data(self):
        # center data
        self.x = self.x - np.mean(self.x)
        self.y = self.y - np.mean(self.y)
        
    # the counting cost function - for determining best weights from input weight history
    def counting_cost(self,w):
        cost = 0
        for p in range(0,len(self.y)):
            x_p = self.x[p]
            y_p = self.y[p]
            a_p = w[0] + sum([a*b for a,b in zip(w[1:],x_p)])
            cost += (np.sign(a_p) - y_p)**2
        return 0.25*cost
    
    # softmargin svm with softmax cost
    def softmargin(self,w):
        cost = 0
        for p in range(0,len(self.y)):
            x_p = self.x[p]
            y_p = self.y[p]
            a_p = w[0] + sum([a*b for a,b in zip(w[1:],x_p)])                
            cost += np.log(1 + np.exp(-y_p*a_p))+ self.lam*np.dot(w[1:].T,w[1:])
        return cost
    
    # margin-perceptron
    def margin_perceptron(self,w):
        cost = 0
        for p in range(0,len(self.y)):
            x_p = self.x[p]
            y_p = self.y[p]
            a_p = w[0] + sum([a*b for a,b in zip(w[1:],x_p)])
            cost += np.maximum(0,1-y_p*a_p)
        return cost
    
     ######## softmargin vs other method ########
     # produce static image of gradient descent or newton's method run
    def svm_comparison(self):
        # declare an instance of our current our optimizers
        opt = MyOptimizers()
        
        ### run all algs ###
        self.lam = 0
        self.big_whist = []
        for i in range(3):
            # run newton's method
            w_hist = opt.gradient_descent(g = self.margin_perceptron,w = np.random.randn(np.shape(self.x)[1]+1,1),max_its = 50,steplength_rule = 'diminishing')
            
            # find best weights
            w = w_hist[-1]
            
            # store
            self.big_whist.append(w)
            
        # run svm     
        self.lam = 10**(-3)
        
        # run newton's method
        w_hist = opt.newtons_method(g = self.softmargin,w = np.random.randn(np.shape(self.x)[1]+1,1),max_its = 10,epsilon = 10**-8)
        w = w_hist[-1]
        self.big_whist.append(w)
    
    # plot comparison figure
    def svm_comparison_fig(self):      
        #### left panel - multiple runs ####
        fig, axs = plt.subplots(1, 2, figsize=(8,4))
        gs = gridspec.GridSpec(1, 2, width_ratios = [1,1]) 
        ax1 = plt.subplot(gs[0],aspect = 'equal'); 
        ax2 = plt.subplot(gs[1],aspect = 'equal'); 
        
        # plot points - first in 3d, then from above
        self.separator_view(ax1)
               
        # plot separator
        minx = min(min(self.x[:,0]),min(self.x[:,1]))
        maxx = max(max(self.x[:,0]),max(self.x[:,1]))
        gapx = (maxx - minx)*0.25
        minx -= gapx
        maxx += gapx
        
        s = np.linspace(minx,maxx,400)
        for i in range(3):
            w = self.big_whist[i]

            # plot 
            t = - ((w[0])/float(w[2]) + w[1]/float(w[2])*s ) 
            ax1.plot(s,t,linewidth = 2,zorder = 1)
        
        #### right panel - svm runs ####
        self.separator_view(ax2)
        w = self.big_whist[-1]
        
        # create surface
        r = np.linspace(minx,maxx,400)
        x1_vals,x2_vals = np.meshgrid(r,r)
        x1_vals.shape = (len(r)**2,1)
        x2_vals.shape = (len(r)**2,1)
        g_vals = np.tanh( w[0] + w[1]*x1_vals + w[2]*x2_vals )
        g_vals = np.asarray(g_vals)

        # vals for cost surface
        x1_vals.shape = (len(r),len(r))
        x2_vals.shape = (len(r),len(r))
        g_vals.shape = (len(r),len(r))
        
        # plot color filled contour based on separator
        g_vals = np.sign(g_vals) + 1
        ax2.contourf(x1_vals,x2_vals,g_vals,colors = self.colors[:],alpha = 0.1,levels = range(0,2+1))
    
        # plot separator
        s = np.linspace(minx,maxx,400)
        t = - ((w[0])/float(w[2]) + w[1]/float(w[2])*s ) 
        ax2.plot(s,t,color = 'k',linewidth = 3,zorder = 1)
        
        ### determine margin ###
        # determine margin
        margin = self.proj_onto_line(w)
        
        # plot margin planes
        s = np.linspace(minx,maxx,400)
        t = - ((w[0])/float(w[2]) + w[1]/float(w[2])*s ) + margin
        ax2.plot(s,t,color = 'k',linewidth = 1,zorder = 1)
        
        t = - ((w[0])/float(w[2]) + w[1]/float(w[2])*s ) - margin
        ax2.plot(s,t,color = 'k',linewidth = 1,zorder = 1)
            
        plt.show()  
    
                    
     ######## softmargin static figure ########
    # produce static image of gradient descent or newton's method run
    def softmargin_fig(self,w_hist,**kwargs):      
        # determine best weights based on number of misclassifications
        g_count = []
        for j in range(len(w_hist)):
            w = w_hist[j]
            count = self.counting_cost(w)
            g_count.append(count)
        ind = np.argmin(g_count)
        if np.size(ind) > 1:
            w = w_hist[ind[-1]]    
        else:
            w = w_hist[ind]
            
        w = w_hist[-1]
        
        # optional arguments
        cost_plot = 'off'
        if 'cost_plot' in kwargs:
            cost_plot = kwargs['cost_plot']  
            
        g = 0
        if 'g' in kwargs:
            g = kwargs['g']              
            
        ### plot all input data ###
        # generate input range for functions
        minx = min(min(self.x[:,0]),min(self.x[:,1]))
        maxx = max(max(self.x[:,0]),max(self.x[:,1]))
        gapx = (maxx - minx)*0.25
        minx -= gapx
        maxx += gapx

        r = np.linspace(minx,maxx,400)
        x1_vals,x2_vals = np.meshgrid(r,r)
        x1_vals.shape = (len(r)**2,1)
        x2_vals.shape = (len(r)**2,1)
        g_vals = np.tanh( w[0] + w[1]*x1_vals + w[2]*x2_vals )
        g_vals = np.asarray(g_vals)

        # vals for cost surface
        x1_vals.shape = (len(r),len(r))
        x2_vals.shape = (len(r),len(r))
        g_vals.shape = (len(r),len(r))
        
        # create figure to plot
        num_panels = 1
        widths = [1]
        if cost_plot == 'on':
            num_panels = 2
            widths = [2,1]
        fig, axs = plt.subplots(1, num_panels, figsize=(8,4))
        gs = gridspec.GridSpec(1, num_panels, width_ratios=widths) 
        ax1 = plt.subplot(gs[0],aspect = 'equal'); 
        if cost_plot == 'on':
            ax2 = plt.subplot(gs[1]); 
            
        # plot points - first in 3d, then from above
        self.separator_view(ax1)
               
        # plot color filled contour based on separator
        g_vals = np.sign(g_vals) + 1
        ax1.contourf(x1_vals,x2_vals,g_vals,colors = self.colors[:],alpha = 0.1,levels = range(0,2+1))
    
        # plot separator
        s = np.linspace(minx,maxx,400)
        t = - ((w[0])/float(w[2]) + w[1]/float(w[2])*s ) 
        ax1.plot(s,t,color = 'k',linewidth = 3,zorder = 1)
        
        ### determine margin ###
        # determine margin
        margin = self.proj_onto_line(w)
        
        # plot margin planes
        s = np.linspace(minx,maxx,400)
        t = - ((w[0])/float(w[2]) + w[1]/float(w[2])*s ) + margin
        ax1.plot(s,t,color = 'k',linewidth = 1,zorder = 1)
        
        t = - ((w[0])/float(w[2]) + w[1]/float(w[2])*s ) - margin
        ax1.plot(s,t,color = 'k',linewidth = 1,zorder = 1)
 
        # plot cost function value
        if cost_plot == 'on':
            # plot cost function history
            g_hist = []
            for j in range(len(w_hist)):
                w = w_hist[j]
                w = np.asarray(w)
                w.shape = (len(w),1)
                g_eval = g(w)
                g_hist.append(g_eval)
                
            g_hist = np.asarray(g_hist).flatten()
            
            # plot cost function history
            ax2.plot(np.arange(len(g_hist)),g_hist,linewidth = 2)
            ax2.set_xlabel('iteration',fontsize = 13)
            ax2.set_title('cost value',fontsize = 12)
            
        plt.show()
 
    # project onto line
    def proj_onto_line(self,w):
        w_c = copy.deepcopy(w)
        w_0 = -w_c[0]/w_c[2]  # amount to subtract from the vertical of each point
        
        # setup line to project onto
        w_1 = -w_c[1]/w_c[2]
        line_pt = np.asarray([1,w_1])
        line_pt.shape = (2,1)
        line_hat = line_pt / np.linalg.norm(line_pt)
        line_hat.shape = (2,1)

        # loop over points, compute distance of projections                     
        dists = []
        for j in range(len(self.y)):
            pt = copy.deepcopy(self.x[j])
            pt[1]-= w_0
            pt.shape = (2,1)
            proj = np.dot(line_hat.T,pt)*line_hat  
            proj.shape = (2,1)
            d = np.linalg.norm(proj - pt)
            dists.append(d)                  
        
        # find smallest distance to class point
        ind = np.argmin(dists)
        pt_min = copy.deepcopy(self.x[ind])
        
        # create new intercept coeff
        pt_min[1] -= w_0
        w_new = -w_1*pt_min[0] + pt_min[1] 

        return w_new
        
    # plot data 'from above' in seperator view
    def separator_view(self,ax):
        # set plotting limits
        xmax1 = copy.deepcopy(max(self.x[:,0]))
        xmin1 = copy.deepcopy(min(self.x[:,0]))
        xgap1 = (xmax1 - xmin1)*0.05
        xmin1 -= xgap1
        xmax1 += xgap1
            
        xmax2 = copy.deepcopy(max(self.x[:,0]))
        xmin2 = copy.deepcopy(min(self.x[:,0]))
        xgap2 = (xmax2 - xmin2)*0.05
        xmin2 -= xgap2
        xmax2 += xgap2
            
        ymax = max(self.y)
        ymin = min(self.y)
        ygap = (ymax - ymin)*0.2
        ymin -= ygap
        ymax += ygap    

        # scatter points
        classes = np.unique(self.y)
        count = 0
        for num in classes:
            inds = np.argwhere(self.y == num)
            inds = [s[0] for s in inds]
            ax.scatter(self.data[inds,0],self.data[inds,1],color = self.colors[int(count)],linewidth = 1,marker = 'o',edgecolor = 'k',s = 50)
            count+=1
            
        # clean up panel
        ax.set_xlim([round(xmin1)-1,round(xmax1)+1])
        ax.set_ylim([round(xmin2)-1,round(xmax2)+1])

        ax.set_xticks(np.arange(round(xmin1)-1, round(xmax1) + 2, 1.0))
        ax.set_yticks(np.arange(round(xmin2)-1, round(xmax2) + 2, 1.0))

        # label axes
        ax.set_xlabel(r'$x_1$', fontsize = 12,labelpad = 0)
        ax.set_ylabel(r'$x_2$', rotation = 0,fontsize = 12,labelpad = 5)
            