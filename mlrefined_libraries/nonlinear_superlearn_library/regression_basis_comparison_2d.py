# import standard plotting and animation
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from mpl_toolkits.mplot3d import Axes3D
from IPython.display import clear_output
from matplotlib import gridspec
import autograd.numpy as np
import copy
import time
import bisect


class Visualizer:
    '''
    Class for visualizing nonlinear regression fits to N = 1 dimensional input datasets
    '''

    # load target function
    def load_data(self,csvname):
        data = np.loadtxt(csvname,delimiter = ',').T
        self.x = data[:,0]
        self.y = data[:,1]
        self.y.shape = (len(self.y),1)
        
    # initialize after animation call
    def dial_settings(self):        
        #### initialize split points for trees ####
        # sort data by values of input
        self.x_t = copy.deepcopy(self.x)
        self.y_t = copy.deepcopy(self.y)
        sorted_inds = np.argsort(self.x_t,axis = 0)
        self.x_t = self.x_t[sorted_inds]
        self.y_t = self.y_t[sorted_inds]

        # create temp copy of data, sort from smallest to largest
        splits = []
        levels = []
        residual = copy.deepcopy(self.y_t)

        ## create simple 'weak learner' between each consecutive pair of points ##
        for p in range(len(self.x_t) - 1):
            # determine points on each side of split
            split = (self.x_t[p] + self.x_t[p+1])/float(2)
            splits.append(split)

            # gather points to left and right of split
            pts_left  = [t for t in self.x_t if t <= split]
            resid_left = residual[:len(pts_left)]
            resid_right = residual[len(pts_left):]

            # compute average on each side
            ave_left = np.mean(resid_left)
            ave_right = np.mean(resid_right)
            levels.append([ave_left,ave_right])
                
        # randomize splits for this experiment
        self.splits = splits
        self.levels = levels
                
        # generate features
        self.F_tree = self.tree_feats()
        
    # least squares
    def least_squares(self,w):
        cost = 0
        for p in range(0,len(self.y)):
            x_p = self.x[p]
            y_p = self.y[p]
            cost +=(self.predict(x_p,w) - y_p)**2
        return cost
    
    ##### transformation functions #####
    # poly features
    def poly_feats(self,D):
        F = []
        for deg in range(D+1):
            F.append(self.x**deg)
        F = np.asarray(F)
        F.shape = (D+1,len(self.x))
        return F.T
    
    # tanh features
    def tanh_feats(self,D):
        F = [np.ones((len(self.x)))]
        for deg in range(D):
            F.append(np.tanh(self.R[deg,0] + self.R[deg,1]*self.x))
        F = np.asarray(F)
        F.shape = (D+1,len(self.x))
        return F.T
    
    # stump-tree feats
    def tree_feats(self):
        # feat matrix container
        F = []

        # loop over points and create feature vector based on stump for each
        for pt in self.x:
            f = [1]
            for i in range(len(self.splits)):
                # get current stump
                split = self.splits[i]
                level = self.levels[i]
                
                # check - which side of this split does the pt lie?
                if pt <= split:  # lies to the left - so evaluate at left level
                    f.append(level[0])
                else:
                    f.append(level[1])
            
            # save stump evaluations - this is our feature vector for pt
            F.append(f)
                    
        # make numpy array
        F = np.asarray(F)
        return F

    ##### prediction functions #####
    # standard polys
    def poly_predict(self,pt,w):
        # linear combo
        val = w[0] + sum([w[i]*pt**i for i in range(1,self.D+1)])
        return val
    
    # single hidden layer tanh network with fixed random weights
    def tanh_predict(self,pt,w):
        # linear combo
        val = w[0] + sum([w[i]*np.tanh(self.R[i-1,0] + self.R[i-1,1]*pt)  for i in range(1,self.D+1)])
        return val
    
    # tree prediction
    def tree_predict(self,pt,w): 
        # our return prediction
        val = copy.deepcopy(w[0])
        
        # loop over current stumps and collect weighted evaluation
        for i in range(len(self.splits)):
            # get current stump
            split = self.splits[i]
            levels = self.levels[i]
                
            # check - which side of this split does the pt lie?
            if pt <= split:  # lies to the left - so evaluate at left level
                val += w[i+1]*levels[0]
            else:
                val += w[i+1]*levels[1]
        return val

    
    ###### optimizer ######
    def boosting(self,F,y,its):
        '''
        Alternating descent wrapper for general Least Squares function
        '''
        g = lambda w: np.linalg.norm(np.dot(F,w) - y)

        # settings 
        tol = 10**(-8)                  # tolerance to between sweeps to stop (optional)
        N = np.shape(F)[1]                        # length of weights
        w = np.zeros((N,1))        # initialization
        w_history = [copy.deepcopy(w)]              # record each weight for plotting

        # outer loop - each is a sweep through every variable once
        i = 0
        g_change = np.inf; gval1 = g(w);
        r = np.copy(y)
        r.shape = (len(r),1)
        for i in range(its):
            # what value do we get?
            vals = np.dot(F.T,r)

            # determine best ind
            n = np.argmax(np.abs(vals))
            f_n = np.asarray(F[:,n])
            num = sum([a*b for a,b in zip(f_n,r)])[0]
            den = sum([a**2 for a in f_n])
            w_n = num/den  
            r = np.asarray([a - w_n*b for a,b in zip(r,f_n)])
            w[n] += w_n

            # record weights at each step for kicks
            w_history.append(copy.deepcopy(w))

            i+=1
        return w_history
    
    
    ###### fit and compare ######
    def brows_fits(self,savepath,**kwargs):
        self.colors = [[1,0,0.4], [ 0, 0.4, 1],[0, 1, 0.5],[1, 0.7, 0.5],[0.7, 0.6, 0.5],'mediumaquamarine']

        # parse input args
        num_elements = [1,10,len(self.y)]
        if 'num_elements' in kwargs:
            num_elements = kwargs['num_elements']
        scatter = 'off'
        if 'scatter' in kwargs:
            scatter = kwargs['scatter']
        
        # set dials for tanh network and trees
        num_elements = [v+1 for v in num_elements]
        self.num_elements = max(num_elements)
        self.dial_settings()

        ### run through each feature type, boost, collect associated weight histories
        # make full poly and tanh feature matrices
        self.F_poly = self.poly_feats(self.num_elements)
           
        # random weights for tanh network, tanh transform 
        scale = 1
        self.R = scale*np.random.randn(self.num_elements,2)
        self.F_tanh = self.tanh_feats(self.num_elements)
            
        # collect poly and tanh weights over each desired level
        poly_weight_history = []
        tanh_weight_history = []
        for element in num_elements:
            # fit weights to data
            w = np.linalg.lstsq(self.F_poly[:,:element], self.y)[0]
            
            # store weights
            poly_weight_history.append(w)
            
            # fit weights to data
            w = np.linalg.lstsq(self.F_tanh[:,:element], self.y)[0]
            
            # store weights
            tanh_weight_history.append(w)          
            
        ### create tree weight history via boosting - then filter out for desired levels
        stump_weight_history = self.boosting(self.F_tree,self.y,its = 3000)
        
        # compute number of non-zeros per weight in history
        nonzs = [len(np.argwhere(w != 0)) for w in stump_weight_history]
            
        # find unique additions
        huh = np.asarray([np.sign(abs(nonzs[p] - nonzs[p+1])) for p in range(len(nonzs)-1)])
        inds = np.argwhere(huh == 1)
        inds = [v[0] for v in inds]

        # sift through, make sure to pick the best fit
        new_inds = []
        for j in range(len(inds)-1):
            val = inds[j+1] - inds[j]
            if val > 2:
                new_inds.append(inds[j+1] - 1)
            else:
                new_inds.append(inds[j])
        new_inds.append(inds[-1])
        stump_weight_history = [stump_weight_history[ind] for ind  in new_inds]
            
        ### plot it
        # construct figure
        fig = plt.figure(figsize = (10,4))
        artist = fig

        # create subplot with 3 panels, plot input function in center plot
        gs = gridspec.GridSpec(1, 3, width_ratios=[1,1, 1]) 
        ax1 = plt.subplot(gs[0]); ax1.axis('off');
        ax2 = plt.subplot(gs[1]); ax2.axis('off');
        ax3 = plt.subplot(gs[2]); ax3.axis('off');

        # set viewing range for all 3 panels
        xmax = max(copy.deepcopy(self.x))
        xmin = min(copy.deepcopy(self.x))
        xgap = (xmax - xmin)*0.0
        xmax += xgap
        xmin -= xgap
        ymax = max(copy.deepcopy(self.y))[0]
        ymin = min(copy.deepcopy(self.y))[0]
        ygap = (ymax - ymin)*0.4
        ymax += ygap
        ymin -= ygap
        
        # animate
        print ('beginning animation rendering...')
        num_frames = len(num_elements)
        def animate(k):
            # clear the panel
            ax1.cla()
            ax2.cla()
            ax3.cla()

            # print rendering update
            if np.mod(k+1,5) == 0:
                print ('rendering animation frame ' + str(k+1) + ' of ' + str(len(num_elements)))
            if k == len(num_elements) - 1:
                print ('animation rendering complete!')
                time.sleep(1)
                clear_output()
                
            # loop over panels, produce plots
            self.D = num_elements[k] 
            cs = 0
            for ax in [ax1,ax2,ax3]:
                # fit to data
                F = 0
                predict = 0
                w = 0
                if ax == ax1: 
                    w = poly_weight_history[k]
                    self.D = len(w) - 1
                    ax.set_title(str(self.D) + ' poly units',fontsize = 14)
                    self.predict = self.poly_predict
                                       
                elif ax == ax2:
                    w = tanh_weight_history[k]
                    self.D = len(w) - 1
                    ax.set_title(str(self.D) + ' tanh units',fontsize = 14)
                    self.predict = self.tanh_predict
                    
                elif ax == ax3:
                    item = min(len(self.y)-1, num_elements[k]-1,len(stump_weight_history)-1) 
                    w = stump_weight_history[item]
                    ax.set_title(str(item) + ' tree units',fontsize = 14)
                    self.predict = self.tree_predict

                ####### plot all and dress panel ######
                # produce learned predictor
                s = np.linspace(xmin,xmax,400)
                t = [self.predict(np.asarray([v]),w) for v in s]
    
                # plot approximation and data in panel
                if scatter == 'off':
                    ax.plot(self.x,self.y,c = 'k',linewidth = 2)
                elif scatter == 'on':
                    ax.scatter(self.x,self.y,c = 'k',edgecolor = 'w',s = 30,zorder = 1)

                ax.plot(s,t,linewidth = 2.75,color = self.colors[cs],zorder = 3)
                cs += 1
                
                # cleanup panel
                ax.set_xlim([xmin,xmax])
                ax.set_ylim([ymin,ymax])
                ax.set_xlabel(r'$x$', fontsize = 14,labelpad = 10)
                ax.set_ylabel(r'$y$', rotation = 0,fontsize = 14,labelpad = 10)
                ax.set_xticks(np.arange(round(xmin), round(xmax)+1, 1.0))
                ax.set_yticks(np.arange(round(ymin), round(ymax)+1, 1.0))

            return artist,
            
        anim = animation.FuncAnimation(fig, animate ,frames=num_frames, interval=num_frames, blit=True)
        
        # produce animation and save
        fps = 50
        if 'fps' in kwargs:
            fps = kwargs['fps']
        anim.save(savepath, fps=fps, extra_args=['-vcodec', 'libx264'])
        clear_output()
 

 