# import standard and plotting 
import math, time, copy
import matplotlib.pyplot as plt
from matplotlib import gridspec
from IPython.display import clear_output
from matplotlib.axes._axes import _log as matplotlib_axes_logger
matplotlib_axes_logger.setLevel('ERROR')
import warnings
warnings.filterwarnings('ignore')

# patch / convex hull libraries
from matplotlib.patches import Circle, Wedge, Polygon
from matplotlib.collections import PatchCollection
from scipy.spatial import ConvexHull
from mpl_toolkits.mplot3d import Axes3D

# import autograd functionality
import autograd.numpy as np
from autograd import grad as compute_grad   
from autograd import hessian as compute_hess
from autograd.misc.flatten import flatten_func


class BasicOptimizer:
    '''
    A list of fundamental optimizers - including
    - gradient_descent (with options for diminishing steplength / backtracking line search)
    - newton's method (with numerical stability parameter)
    
    In each case - since these are implementations are for educational purposes - the weights at each step are recorded and returned.
    '''

    ### gradient descent ###
    def gradient_descent(self,g,w,**kwargs):   
        '''
        basic gradient descent function.  controls include
        
        - max_its: max iterations (int), e.g., 100 (default)
        - version: form of gradient, options include
            - unnormalized (default) 
            - normalized
        - alpha: step length / learning rate, 10**-4 (default)
        - steplength_rule: rule for adaptive steplength - options include 
            - None (default)
            - diminshing
            - backtracking 
        - verbose: options include
            - False (default)
            - True 
            
        '''
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
        '''
        basic newton's method function.  controls include
        
        - max_its: max iterations (int), e.g., 20 (default)
        - epsilon: numerical stability hyperparameter e.g., 10**(-3) (default)
        - verbose: options include
            - False (default)
            - True 
        
        '''
        
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
        self.epsilon = 10**(-3)
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

    
class OvaVsualizer:
    '''
    Demonstrate one-versus-all classification
    
    '''
    
    #### initialize ####
    def __init__(self,data):        
        # grab input
        data = data.T
        self.data = data
        self.x = data[:,:-1]
        if self.x.ndim == 1:
            self.x.shape = (len(self.x),1)
        self.y = data[:,-1]
        self.y.shape = (len(self.y),1)
        
        # colors for viewing classification data 'from above'
        self.colors = [[ 0, 0.4, 1],[1,0,0.4],[0, 1, 0.5],[1, 0.7, 0.5],'violet','mediumaquamarine']

        #self.colors = ['cornflowerblue','salmon','lime','bisque','mediumaquamarine','b','m','g']
        
        # create instance of optimizers
        self.opt = BasicOptimizer()
        
    ### cost functions ###
    # the counting cost function
    def counting_cost(self,w):
        cost = 0
        for p in range(0,len(self.y)):
            x_p = self.x[p]
            y_p = self.y[p]
            a_p = w[0] + sum([a*b for a,b in zip(w[1:],x_p)])
            cost += (np.sign(a_p) - y_p)**2
        return 0.25*cost
    
    # the perceptron relu cost
    def relu(self,w):
        cost = 0
        for p in range(0,len(self.y_temp)):
            x_p = self.x[p]
            y_p = self.y_temp[p]
            a_p = w[0] + sum([a*b for a,b in zip(w[1:],x_p)])
            cost += np.maximum(0,-y_p*a_p)
        return cost

    # the convex softmax cost function
    def softmax(self,w):
        cost = 0
        for p in range(0,len(self.y_temp)):
            x_p = self.x[p]
            y_p = self.y_temp[p]
            a_p = w[0] + sum([a*b for a,b in zip(w[1:],x_p)])
            cost += np.log(1 + np.exp(-y_p*a_p))
        return cost
                   
    ### compare grad descent runs - given cost to counting cost ###
    def solve_2class_subproblems(self,**kwargs):
        # parse args
        max_its = 5
        if 'max_its' in kwargs:
            max_its = kwargs['max_its']
        alpha = 10**-3
        if 'alpha' in kwargs:
            alpha = kwargs['alpha']  
        steplength_rule = 'none'
        if 'steplength_rule' in kwargs:
            steplength_rule = kwargs['steplength_rule']
        version = 'unnormalized'
        if 'version' in kwargs:
            version = kwargs['version'] 
        algo = 'newtons_method'
        if 'algo' in kwargs:
            algo = kwargs['algo']
         
        #### perform all optimizations ###
        self.g = self.softmax
        if 'cost' in kwargs:
            cost = kwargs['cost']
            if cost == 'softmax':
                self.g = self.softmax
            if cost == 'relu':
                self.g = self.relu

        # loop over subproblems and solve
        self.W = []
        num_classes = np.size(np.unique(self.y))
        for i in range(0,num_classes):
            #print ('solving sub-problem number ' + str(i+1))
            # prepare temporary C vs notC sub-probem labels
            self.y_temp = copy.deepcopy(self.y)
            ind = np.argwhere(self.y_temp == (i))
            ind = ind[:,0]
            ind2 = np.argwhere(self.y_temp != (i))
            ind2 = ind2[:,0]
            self.y_temp[ind] = 1
            self.y_temp[ind2] = -1

            # solve the current subproblem
            if algo == 'gradient_descent':# run gradient descent
                w_hist = self.opt.gradient_descent(g = self.g,w = np.random.randn(np.shape(self.x)[1]+1,1),version = version,max_its = max_its, alpha = alpha,steplength_rule = steplength_rule)
            elif algo == 'newtons_method':
                w_hist = self.opt.newtons_method(g = self.g,w = np.random.randn(np.shape(self.x)[1]+1,1),max_its = max_its,epsilon = 10**(-2))
            
            # store best weight for final classification 
            g_count = []
            for j in range(len(w_hist)):
                w = w_hist[j]
                gval = self.g(w)
                g_count.append(gval)
            ind = np.argmin(g_count)
            w = w_hist[ind]
            
            # normalize normal vectors for each classifier
            w_norm = sum([v**2 for v in w[1:]])**(0.5)
            w_1N = [v/w_norm for v in w]
            self.W.append(w_1N)
            
        # reshape
        self.W = np.asarray(self.W)
        self.W.shape = (num_classes,np.shape(self.x)[1] + 1)
    
    # plotting function for the data and individual separators
    def plot_data_and_subproblem_separators(self):
        # determine plotting ranges
        minx = min(min(self.x[:,0]),min(self.x[:,1]))
        maxx = max(max(self.x[:,0]),max(self.x[:,1]))
        gapx = (maxx - minx)*0.1
        minx -= gapx
        maxx += gapx
        
        # initialize figure, plot data, and dress up panels with axes labels etc.
        num_classes = np.size(np.unique(self.y))
        
        ##### setup figure to plot #####
        # initialize figure
        fig = plt.figure(figsize = (9,5))
        gs = gridspec.GridSpec(2, num_classes) 
        
        # create subplots for each sub-problem
        r = np.linspace(minx,maxx,400)
        for a in range(0,num_classes):
            # setup current axis
            ax = plt.subplot(gs[a],aspect = 'equal'); 

            # get current weights
            w = self.W[a]
                
            # color current class
            ax.scatter(self.x[:,0], self.x[:,1], s = 30,color = '0.75')
            t = np.argwhere(self.y == a)
            t = t[:,0]
            ax.scatter(self.x[t,0],self.x[t,1], s = 50,color = self.colors[a],edgecolor = 'k',linewidth = 1.5)

            # draw subproblem separator
            z = - w[0]/w[2] - w[1]/w[2]*r
            ax.plot(r,z,linewidth = 2,color = self.colors[a],zorder = 3)
            ax.plot(r,z,linewidth = 2.75,color = 'k',zorder = 2)

            # dress panel correctly
            ax.set_xlim(minx,maxx)
            ax.set_ylim(minx,maxx)
            ax.axis('off')
         
        # plot final panel with all data and separators
        ax4 = plt.subplot(gs[num_classes + 1],aspect = 'equal'); 
        self.plot_data(ax4)
        self.plot_all_separators(ax4)

        # dress panel
        ax4.set_xlim(minx,maxx)
        ax4.set_ylim(minx,maxx)
        ax4.axis('off')
            
        plt.show()
           
    # show data
    def show_dataset(self):
        # initialize figure
        fig = plt.figure(figsize = (17,6))
        artist = fig
        gs = gridspec.GridSpec(1, 3,width_ratios = [1,3,1]) 

        # setup current axis
        ax = plt.subplot(gs[1],aspect = 'equal'); 
        
        # run axis through data plotter
        self.plot_data(ax)
        
        # determine plotting ranges
        minx = min(min(self.x[:,0]),min(self.x[:,1]))
        maxx = max(max(self.x[:,0]),max(self.x[:,1]))
        gapx = (maxx - minx)*0.1
        minx -= gapx
        maxx += gapx
        
        # dress panel
        ax.set_xlim(minx,maxx)
        ax.set_ylim(minx,maxx)
        
        plt.show()
        
    # color indnividual region using fusion rule
    def show_fusion(self,region):
        # generate input range for viewing range
        minx = min(min(self.x[:,0]),min(self.x[:,1]))
        maxx = max(max(self.x[:,0]),max(self.x[:,1]))
        gapx = (maxx - minx)*0.1
        minx -= gapx
        maxx += gapx
        
        # initialize figure
        fig = plt.figure(figsize = (17,6))
        artist = fig
        gs = gridspec.GridSpec(1, 3,width_ratios = [1,3,1]) 

        # setup current axis
        ax = plt.subplot(gs[1],aspect = 'equal');   
        
        # plot panel with all data and separators
        self.plot_data(ax)
        self.plot_all_separators(ax)
        
        # color region
        self.region_coloring(region = region,ax = ax)
        
        # dress panel
        ax.set_xlim(minx,maxx)
        ax.set_ylim(minx,maxx)
        ax.axis('off')
        
    # show coloring of entire space
    def show_complete_coloring(self):
        # generate input range for viewing range
        minx = min(min(self.x[:,0]),min(self.x[:,1]))
        maxx = max(max(self.x[:,0]),max(self.x[:,1]))
        gapx = (maxx - minx)*0.1
        minx -= gapx
        maxx += gapx
        
        # initialize figure
        fig = plt.figure(figsize = (17,6))
        gs = gridspec.GridSpec(1, 2,width_ratios = [1,1]) 

        # setup current axis
        ax = plt.subplot(gs[0],aspect = 'equal');
        ax2 = plt.subplot(gs[1],aspect = 'equal');
        
        # plot panel with all data and separators
        self.plot_data(ax)
        self.plot_data(ax2)
        self.plot_all_separators(ax)
                
        ### draw multiclass boundary on right panel
        r = np.linspace(minx,maxx,2000)
        w1_vals,w2_vals = np.meshgrid(r,r)
        w1_vals.shape = (len(r)**2,1)
        w2_vals.shape = (len(r)**2,1)
        o = np.ones((len(r)**2,1))
        h = np.concatenate([o,w1_vals,w2_vals],axis = 1)
        pts = np.dot(self.W,h.T)
        g_vals = np.argmax(pts,axis = 0)

        # vals for cost surface
        w1_vals.shape = (len(r),len(r))
        w2_vals.shape = (len(r),len(r))
        g_vals.shape = (len(r),len(r))
        
        # plot contour
        C = len(np.unique(self.y))
        ax2.contour(w1_vals,w2_vals,g_vals,colors = 'k',levels = range(0,C+1),linewidths = 2.75,zorder = 4)
        ax2.contourf(w1_vals,w2_vals,g_vals+1,colors = self.colors[:],alpha = 0.2,levels = range(0,C+1))
        ax.contourf(w1_vals,w2_vals,g_vals+1,colors = self.colors[:],alpha = 0.2,levels = range(0,C+1))

        # dress panel
        ax.set_xlim(minx,maxx)
        ax.set_ylim(minx,maxx)
        ax.axis('off')
        
        ax2.set_xlim(minx,maxx)
        ax2.set_ylim(minx,maxx)
        ax2.axis('off')     
    
    # point and projection illustration
    def point_and_projection(self,point1,point2):
        # generate range for viewing limits
        minx = min(min(self.x[:,0]),min(self.x[:,1]))
        maxx = max(max(self.x[:,0]),max(self.x[:,1]))
        gapx = (maxx - minx)*0.1
        minx -= gapx
        maxx += gapx
        
        # initialize figure
        fig = plt.figure(figsize = (17,6))
        gs = gridspec.GridSpec(1, 2,width_ratios = [1,1]) 

        # setup current axis
        ax = plt.subplot(gs[0],aspect = 'equal');
        ax2 = plt.subplot(gs[1],aspect = 'equal');
        
        ### plot left panel - data, separators, and region coloring
        self.plot_data(ax)
        self.plot_all_separators(ax)        
        
        ### determine projections etc.,
        point = [1] + point1
        point = np.asarray(point)
        point.shape = (len(point),1)
        y = np.dot(self.W,point)
        ind = np.argwhere(y > 0)
        if np.size(ind) == 0:
            num_classes = len(np.unique(self.y))
            ind = np.arange(num_classes).tolist()
        else:
            ind = [v[0] for v in ind]
        point = point[1:]
        ax.scatter(point[0],point[1],c = 'k',edgecolor = 'w',linewidth = 1,s = 90)

        # loop over classifiers and project
        for i in ind:
            # get weights
            w = self.W[i]
            w = np.asarray(w)
            w.shape = (len(w),1)
            w_norm = sum([v**2 for v in w[1:]])

            # make projected point
            add_on = w[0] + sum([v*a for v,a in zip(point,w[1:])])
            add_on /= w_norm
            proj_point = copy.deepcopy(point)
            proj_point -= add_on*w[1:]

            # projected point
            ax.scatter(proj_point[0],proj_point[1],c = self.colors[i],edgecolor = 'k',linewidth = 1,s = 60,zorder = 4,marker = 'X')
                
            # dashed line
            l = np.linspace(proj_point[0],point[0],200)
            b = np.linspace(proj_point[1],point[1],200)
            ax.plot(l,b,linewidth = 1,linestyle = '--',color = 'k',zorder = 3)
            
        # dress panels
        ax.set_xlim(minx,maxx)
        ax.set_ylim(minx,maxx)
        ax.axis('off')

        ### plot left panel - data, separators, and region coloring
        self.plot_data(ax2)
        self.plot_all_separators(ax2)        
        
        ### determine projections etc.,
        point = [1] + point2
        point = np.asarray(point)
        point.shape = (len(point),1)
        y = np.dot(self.W,point)
        ind = np.argwhere(y > 0)
        if np.size(ind) == 0:
            num_classes = len(np.unique(self.y))
            ind = np.arange(num_classes).tolist()
        else:
            ind = [v[0] for v in ind]
        point = point[1:]
        ax2.scatter(point[0],point[1],c = 'k',edgecolor = 'w',linewidth = 1,s = 90)

        # loop over classifiers and project
        for i in ind:
            # get weights
            w = self.W[i]
            w = np.asarray(w)
            w.shape = (len(w),1)
            w_norm = sum([v**2 for v in w[1:]])

            # make projected point
            add_on = w[0] + sum([v*a for v,a in zip(point,w[1:])])
            add_on /= w_norm
            proj_point = copy.deepcopy(point)
            proj_point -= add_on*w[1:]

            # projected point
            ax2.scatter(proj_point[0],proj_point[1],c = self.colors[i],edgecolor = 'k',linewidth = 1,s = 60,zorder = 4,marker = 'X')
                
            # dashed line
            l = np.linspace(proj_point[0],point[0],200)
            b = np.linspace(proj_point[1],point[1],200)
            ax2.plot(l,b,linewidth = 1,linestyle = '--',color = 'k',zorder = 3)
            
        # dress panels
        ax2.set_xlim(minx,maxx)
        ax2.set_ylim(minx,maxx)
        ax2.axis('off')

    ###### utility functions - individual data/separators plotters ###### 
    # plot regions colored by classification
    def region_coloring(self,region,ax):        
        #### color first regions  ####
        # generate input range for functions
        minx = min(min(self.x[:,0]),min(self.x[:,1]))
        maxx = max(max(self.x[:,0]),max(self.x[:,1]))
        gapx = (maxx - minx)*0.1
        minx -= gapx
        maxx += gapx
        
        # plot over range
        r = np.linspace(minx,maxx,200)
        x1_vals,x2_vals = np.meshgrid(r,r)
        x1_vals.shape = (len(r)**2,1)
        x2_vals.shape = (len(r)**2,1)
        o = np.ones((len(r)**2,1))
        x = np.concatenate([o,x1_vals,x2_vals],axis = 1)
        
        ### for region 1, determine points that are uniquely positive for each classifier ###
        ind_set = []
        y = np.dot(self.W,x.T)
        num_classes = np.size(np.unique(self.y))
        
        if region == 1 or region == 'all':
            for i in range(0,num_classes):       
                class_inds = np.arange(num_classes)
                class_inds = np.delete(class_inds,(i),axis = 0)

                # loop over non-current classifier
                ind = np.argwhere(y[class_inds[0]] < 0).tolist()
                ind = [s[0] for s in ind]
                for j in range(1,len(class_inds)):
                    c_ind = class_inds[j]
                    ind2 = np.argwhere(y[c_ind] < 0).tolist()
                    ind2 = [s[0] for s in ind2]
                    ind = [s for s in ind if s in ind2]                

                ind2 = np.argwhere(y[i] > 0).tolist()
                ind2 = [s[0] for s in ind2]
                ind = [s for s in ind if s in ind2]

                # plot polygon over region defined by ind
                x1_ins = np.asarray([x1_vals[s] for s in ind])
                x1_ins.shape = (len(x1_ins),1)
                x2_ins = np.asarray([x2_vals[s] for s in ind])
                x2_ins.shape = (len(x2_ins),1)
                h = np.concatenate((x1_ins,x2_ins),axis = 1)
                vertices = ConvexHull(h).vertices
                poly = [h[v] for v in vertices]
                polygon = Polygon(poly, True)   
                patches = []
                patches.append(polygon)

                p = PatchCollection(patches, alpha=0.2,color = self.colors[i])
                ax.add_collection(p)
                
        if region == 2 or region == 'all':
            for i in range(0,num_classes):       
                class_inds = np.arange(num_classes)
                class_inds = np.delete(class_inds,(i),axis = 0)

                # loop over non-current classifier
                ind = np.argwhere(y[class_inds[0]] > 0).tolist()
                ind = [s[0] for s in ind]
                for j in range(1,len(class_inds)):
                    c_ind = class_inds[j]
                    ind2 = np.argwhere(y[c_ind] > 0).tolist()
                    ind2 = [s[0] for s in ind2]
                    ind = [s for s in ind if s in ind2]                

                ind2 = np.argwhere(y[i] < 0).tolist()
                ind2 = [s[0] for s in ind2]
                ind = [s for s in ind if s in ind2]

                # plot polygon over region defined by ind
                x1_ins = np.asarray([x1_vals[s] for s in ind])
                x1_ins.shape = (len(x1_ins),1)
                x2_ins = np.asarray([x2_vals[s] for s in ind])
                x2_ins.shape = (len(x2_ins),1)
                o = np.ones((len(x2_ins),1))
                h = np.concatenate((o,x1_ins,x2_ins),axis = 1)
                
                # determine regions dominated by one classifier or the other
                vals = []
                for c in class_inds:
                    w = self.W[int(c)]
                    nv = np.dot(w,h.T)
                    vals.append(nv)
                vals = np.asarray(vals)
                vals.shape = (len(class_inds),len(h))
                ind = np.argmax(vals,axis = 0)

                for j in range(len(class_inds)):
                    # make polygon for each subregion
                    ind1 = np.argwhere(ind == j)
                    x1_ins2 = np.asarray([x1_ins[s] for s in ind1])
                    x1_ins2.shape = (len(x1_ins2),1)
                    x2_ins2 = np.asarray([x2_ins[s] for s in ind1])
                    x2_ins2.shape = (len(x2_ins2),1)
                    h = np.concatenate((x1_ins2,x2_ins2),axis = 1)
                    
                    # find convex hull of points
                    vertices = ConvexHull(h).vertices
                    poly = [h[v] for v in vertices]
                    polygon = Polygon(poly, True)   
                    patches = []
                    patches.append(polygon)
                    c = class_inds[j]
                    p = PatchCollection(patches, alpha=0.2,color = self.colors[c])
                    ax.add_collection(p)
                    
        if region == 3 or region == 'all':
            # find negative zone of all classifiers
            ind = np.argwhere(y[0] < 0).tolist()
            ind = [s[0] for s in ind]
            for i in range(1,num_classes):
                ind2 = np.argwhere(y[i] < 0).tolist()
                ind2 = [s[0] for s in ind2]
                ind = [s for s in ind if s in ind2]                

            # loop over negative zone, find max area of each classifier
            x1_ins = np.asarray([x1_vals[s] for s in ind])
            x1_ins.shape = (len(x1_ins),1)
            x2_ins = np.asarray([x2_vals[s] for s in ind])
            x2_ins.shape = (len(x2_ins),1)
            o = np.ones((len(x2_ins),1))
            h = np.concatenate((o,x1_ins,x2_ins),axis = 1)
                
            # determine regions dominated by one classifier or the other
            vals = []
            for c in range(num_classes):
                w = self.W[c]
                nv = np.dot(w,h.T)
                vals.append(nv)
            vals = np.asarray(vals)
            vals.shape = (num_classes,len(h))
            ind = np.argmax(vals,axis = 0)

            # loop over each class, construct polygon region for each
            for c in range(num_classes):
                # make polygon for each subregion
                ind1 = np.argwhere(ind == c)
                x1_ins2 = np.asarray([x1_ins[s] for s in ind1])
                x1_ins2.shape = (len(x1_ins2),1)
                x2_ins2 = np.asarray([x2_ins[s] for s in ind1])
                x2_ins2.shape = (len(x2_ins2),1)
                h = np.concatenate((x1_ins2,x2_ins2),axis = 1)
                    
                # find convex hull of points
                vertices = ConvexHull(h).vertices
                poly = [h[v] for v in vertices]
                polygon = Polygon(poly, True)   
                patches = []
                patches.append(polygon)
                p = PatchCollection(patches, alpha=0.2,color = self.colors[c])
                ax.add_collection(p)    
                
                
    # plot data
    def plot_data(self,ax):
        # initialize figure, plot data, and dress up panels with axes labels etc.
        num_classes = np.size(np.unique(self.y))
                
        # color current class
        for a in range(0,num_classes):
            t = np.argwhere(self.y == a)
            t = t[:,0]
            ax.scatter(self.x[t,0],self.x[t,1], s = 50,color = self.colors[a],edgecolor = 'k',linewidth = 1.5)
        
    # plot separators
    def plot_all_separators(self,ax):
        # determine plotting ranges
        minx = min(min(self.x[:,0]),min(self.x[:,1]))
        maxx = max(max(self.x[:,0]),max(self.x[:,1]))
        gapx = (maxx - minx)*0.1
        minx -= gapx
        maxx += gapx
        
        # initialize figure, plot data, and dress up panels with axes labels etc.
        num_classes = np.size(np.unique(self.y))
                
        # color current class
        r = np.linspace(minx,maxx,400)
        for a in range(0,num_classes):
            # get current weights
            w = self.W[a]
            
            # draw subproblem separator
            z = - w[0]/w[2] - w[1]/w[2]*r
            r = np.linspace(minx,maxx,400)
            ax.plot(r,z,linewidth = 2,color = self.colors[a],zorder = 3)
            ax.plot(r,z,linewidth = 2.75,color = 'k',zorder = 2)
            
            
    # show coloring of entire space
    def show_discrete_step(self,w_hist,view,**kwargs):
        '''
        # determine best set of weights from history
        cost_evals = []
        for i in range(len(w_hist)):
            W = w_hist[i]
            cost = self.counting_cost(W)
            cost_evals.append(cost)
        ind = np.argmin(cost_evals)
        '''
        # reshape data for plotting
        self.x = np.asarray(self.x)
        self.y.shape = (len(self.y),1)
        
        # or just take last weights
        self.W = w_hist[-1]
        
        # initialize figure
        fig = plt.figure(figsize = (17,6))
        gs = gridspec.GridSpec(1, 2,width_ratios = [1.5,1]) 
            
        # setup current axis
        ax = plt.subplot(gs[1],projection = '3d');
        ax2 = plt.subplot(gs[0],aspect = 'equal');
        
        # load in args
        zplane = 'on'
        if 'zplane' in kwargs:
            zplane = kwargs['zplane']
       
        # generate input range for viewing range
        minx = min(min(self.x[:,0]),min(self.x[:,1]))
        maxx = max(max(self.x[:,0]),max(self.x[:,1]))
        gapx = (maxx - minx)*0.1
        minx -= gapx
        maxx += gapx
        
        # plot panel with all data and separators
        # scatter points in both panels
        class_nums = np.unique(self.y)
        C = len(class_nums)

        # plot data in right panel from above
        self.plot_data(ax2)
        self.x = self.x.T

        ### draw multiclass boundary on right panel
        r = np.linspace(minx,maxx,4000)
        w1_vals,w2_vals = np.meshgrid(r,r)
        w1_vals.shape = (len(r)**2,1)
        w2_vals.shape = (len(r)**2,1)
        o = np.ones((len(r)**2,1))
        h = np.concatenate([o,w1_vals,w2_vals],axis = 1)
        pts = np.dot(h,self.W)
        g_vals = np.argmax(pts,axis = 1) 

        # vals for cost surface
        w1_vals.shape = (len(r),len(r))
        w2_vals.shape = (len(r),len(r))
        g_vals.shape = (len(r),len(r))
        
        # plot contour in right panel
        C = len(np.unique(self.y))
        ax2.contour(w1_vals,w2_vals,g_vals,colors = 'k',levels = range(-1,C-1),linewidths = 2.75,zorder = 4);
        ax2.contourf(w1_vals,w2_vals,g_vals,colors = self.colors[:],alpha = 0.2,levels = range(-1,C));
        
        ### plot discrete step function ###
        # to plot the step function, plot the bottom and top steps separately - z1 and z2
        steps = np.unique(g_vals)
        num_steps = np.arange(len(steps))
        
        # loop over each step and plot
        g_vals_copy = copy.deepcopy(g_vals)
        g_vals_copy.shape = (len(r)**2,1)
        for step in steps:
            # copy surface            
            g_copy = np.zeros((len(r)**2,1))
            g_copy.fill(np.nan)
            
            # find step in copy, nan out all else
            ind = np.argwhere(g_vals_copy == step)
            ind = [v[0] for v in ind]
            for i in ind:
                g_copy[i] = step
            
            # reshape and plot
            try:
                g_copy.shape = (len(r),len(r))
                ax.plot_surface(w1_vals,w2_vals,g_copy,alpha = 0.25,color = 'w',zorder = 0,edgecolor = 'k',linewidth=0.25,cstride = 200, rstride = 200,shade=10,antialiased=True);
            except Exception as e: # hack for plotting - exception useful for debug mode
                pass

        # plot zplane = 0 in left 3d panel - showing intersection of regressor with z = 0 (i.e., its contour, the separator, in the 3d plot too)?
        if zplane == 'on':
            g_vals +=1
            #ax.plot_surface(w1_vals,w2_vals,g_vals*0-1,alpha = 0.1) 
            
            # loop over each class and color in z-plane
            for c in class_nums:
                try:
                    # plot separator curve in left plot z plane
                    ax.contour(w1_vals,w2_vals,g_vals - (c+1),colors = 'k',levels = [0],linewidths = 3,zorder = 1);

                    # color parts of plane with correct colors
                    ax.contourf(w1_vals,w2_vals, g_vals - 0.5 - c ,colors = self.colors[(int(c)):],alpha = 0.1,levels = range(0,2));
                except Exception as e: # hack for plotting - exception useful for debug mode
                    pass
                
        # scatter points in 3d
        for c in range(C):
            ind = np.argwhere(self.y == class_nums[c])
            ind = [v[0] for v in ind]
            ax.scatter(self.x[0,ind],self.x[1,ind],self.y[ind],s = 80,color = self.colors[c],edgecolor = 'k',linewidth = 1.5,zorder = 3)
            
        # dress panel
        ax.view_init(view[0],view[1])
        #ax.axis('off')
        ax.set_xlim(minx,maxx)
        ax.set_ylim(minx,maxx)
        ax.set_zlim(-0.1,C - 1+0.1)
        
        ax2.set_xticks([])
        ax2.set_yticks([])
        ax2.set_yticklabels([])
        ax2.set_xticklabels([])
        ax2.set_xlim(minx,maxx)
        ax2.set_ylim(minx,maxx)
        #ax2.set_ylabel(r'$x_2$',rotation = 0,fontsize = 12,labelpad = 10)
        #ax2.set_xlabel(r'$x_1$',fontsize = 12)
        
        
            
            
