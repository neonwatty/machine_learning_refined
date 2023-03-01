import matplotlib.pyplot as plt
import matplotlib.animation as animation
from mpl_toolkits.mplot3d import Axes3D
from IPython.display import clear_output
from matplotlib import gridspec

import copy, time, bisect
import autograd.numpy as np

class Visualizer:
    '''
    Class for visualizing nonlinear regression fits to N = 1 dimensional input datasets
    '''

    # load target function
    def load_data(self,csvname):
        data = np.loadtxt(csvname,delimiter = ',')
        if data.shape[0] < data.shape[1]:
            self.x = data[0,:]
            self.y = data[1,:]
        else:
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
 

 
class SingleVisualizer:
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
    def browse_single_fit(self,savepath,**kwargs):
        # parse input args
        num_elements = [1,10,len(self.y)]
        if 'num_units' in kwargs:
            num_elements = kwargs['num_units']
            
        basis = kwargs['basis']
       
        self.colors = [[1,0,0.4], [ 0, 0.4, 1],[0, 1, 0.5],[1, 0.7, 0.5],[0.7, 0.6, 0.5],'mediumaquamarine']
        
        # construct figure
        fig = plt.figure(figsize = (9,4))
        artist = fig

        # create subplot with 3 panels, plot input function in center plot
        gs = gridspec.GridSpec(1, 3, width_ratios=[2,1,0.25]) 
        ax1 = plt.subplot(gs[0]); ax1.axis('off');
        ax2 = plt.subplot(gs[1]); ax2.axis('off');
        ax3 = plt.subplot(gs[2]); ax3.axis('off');

        # set dials for tanh network and trees
        num_elements = [v+1 for v in num_elements]
        self.num_elements = max(num_elements)

        # choose basis type
        self.F = []
        weight_history = []
        if basis == 'poly':
            self.F = self.poly_feats(self.num_elements)
            
            # collect poly and tanh weights over each desired level
            for element in num_elements:
                # fit weights to data
                w = np.linalg.lstsq(self.F[:,:element], self.y,rcond = 10**(-15))[0]

                # store weights
                weight_history.append(w)
                
            self.predict = self.poly_predict

        if basis == 'tanh':
            # random weights for tanh network, tanh transform 
            scale = 1
            self.R = scale*np.random.randn(self.num_elements,2)
            self.F = self.tanh_feats(self.num_elements)
            
            # collect poly and tanh weights over each desired level
            for element in num_elements:
                # fit weights to data
                w = np.linalg.lstsq(self.F[:,:element], self.y)[0]

                # store weights
                weight_history.append(w)
            
            self.predict = self.tanh_predict

        if basis == 'tree':
            self.dial_settings()
            self.F = self.F_tree
            weight_history = self.boosting(self.F,self.y,its = 3000)

            # compute number of non-zeros per weight in history
            nonzs = [len(np.argwhere(w != 0)) for w in weight_history]

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
            weight_history = [weight_history[ind] for ind  in new_inds]
            weight_history = [weight_history[ind - 2] for ind in num_elements]
            self.predict = self.tree_predict
            
            # generate three panels - one to show current basis element being fit
            gs = gridspec.GridSpec(1, 3, width_ratios=[1,1,1]) 
            ax = plt.subplot(gs[0]); ax1.axis('off');
            ax1 = plt.subplot(gs[1]); ax2.axis('off');
            ax2 = plt.subplot(gs[2]); ax2.axis('off');

        # compute cost eval history
        cost_evals = []
        for i in range(len(weight_history)):
            item = copy.deepcopy(i)
            if basis == 'tree':
                item = min(len(self.y)-1, num_elements[i]-1,len(weight_history)-1) 
            w = weight_history[item]
            self.D = len(w) - 1

            cost = self.least_squares(w)
            cost_evals.append(cost)
     
        # plot cost path - scale to fit inside same aspect as classification plots
        cost_evals = [v/float(np.size(self.y)) for v in cost_evals]
        num_iterations = len(weight_history)
        minxc = min(num_elements)-1
        maxxc = max(num_elements)+1
        gapxc = (maxxc - minxc)*0.1
        minxc -= gapxc
        maxxc += gapxc
        minc = min(copy.deepcopy(cost_evals))
        maxc = max(copy.deepcopy(cost_evals))
        gapc = (maxc - minc)*0.1
        minc -= gapc
        maxc += gapc

        ### plot it
        # set viewing range for all 3 panels
        xmax = max(copy.deepcopy(self.x))
        xmin = min(copy.deepcopy(self.x))
        xgap = (xmax - xmin)*0.05
        xmax += xgap
        xmin -= xgap
        ymax = max(copy.deepcopy(self.y))[0]
        ymin = min(copy.deepcopy(self.y))[0]
        ygap = (ymax - ymin)*0.1
        ymax += ygap
        ymin -= ygap

        # animate
        print ('beginning animation rendering...')
        def animate(k):
            # clear the panel
            ax1.cla()
            ax2.cla()
            
            # print rendering update
            if np.mod(k+1,5) == 0:
                print ('rendering animation frame ' + str(k+1) + ' of ' + str(len(num_elements)))
            if k == len(num_elements):
                print ('animation rendering complete!')
                time.sleep(1)
                clear_output()
                
            # loop over panels, produce plots
            if k > 0:
                self.D = num_elements[k-1] + 1
                cs = 0

                # fit to data
                F = 0
                predict = 0
                w = 0
                if basis == 'poly': 
                    w = weight_history[k-1]
                    self.D = len(w) - 1
                    ax1.set_title(str(self.D) + ' poly units',fontsize = 14)
                    self.predict = self.poly_predict

                elif basis == 'tanh':
                    w = weight_history[k-1]
                    self.D = len(w) - 1
                    ax1.set_title(str(self.D) + ' tanh units',fontsize = 14)
                    self.predict = self.tanh_predict

                elif basis == 'tree':
                    item = min(len(self.y)-1, num_elements[k]-1,len(weight_history)-1) 
                    w = weight_history[item]
                    ax1.set_title(str(item) + ' tree units',fontsize = 14)
                    self.predict = self.tree_predict

                ####### plot all and dress panel ######
                # produce learned predictor
                s = np.linspace(xmin,xmax,400)
                t = [self.predict(np.asarray([v]),w) for v in s]
                ax1.plot(s,t,linewidth = 2.75,color = self.colors[2],zorder = 3)
                            
                # cost function value
                ax2.plot(num_elements,cost_evals,color = 'k',linewidth = 2.5,zorder = 1)
                ax2.scatter(num_elements[k-1],cost_evals[k-1],color = self.colors[2],s = 70,edgecolor = 'w',linewidth = 1.5,zorder = 3)

                ax2.set_xlabel('model',fontsize = 12)
                ax2.set_title('cost function plot',fontsize = 12)

                # cleanp panel
                ax2.set_xlim([minxc,maxxc])
                ax2.set_ylim([minc,maxc])
                #ax2.xaxis.set_major_locator(MaxNLocator(integer=True))

            # plot approximation and data in panel
            ax1.scatter(self.x,self.y,c = 'k',edgecolor = 'w',s = 50,zorder = 1)
            #cs += 1

            # cleanup panel
            ax1.set_xlim([xmin,xmax])
            ax1.set_ylim([ymin,ymax])
            ax1.set_xlabel(r'$x$', fontsize = 14,labelpad = 10)
            ax1.set_ylabel(r'$y$', rotation = 0,fontsize = 14,labelpad = 10)
            ax1.set_xticks(np.arange(round(xmin), round(xmax)+1, 1.0))
            #ax1.set_yticks(np.arange(round(ymin), round(ymax)+1, 1.0))
            
            ### if basis == tree, show the most recently added element as well
            if basis == 'tree':
                ax.cla()
                
                # plot data
                ax.scatter(self.x,self.y,c = 'k',edgecolor = 'w',s = 50,zorder = 1)

                # plot tree
                item = min(len(self.y)-1, num_elements[k]-1,len(weight_history)-1)
                w = 0
                if k == 0:   # on the first slide just show first stump
                    w = np.sign(weight_history[item])
                else:        # show most recently added
                    w1 = weight_history[item]
                    w2 = weight_history[item-1]
                    w = w1 - w2
                    ind = np.argmax(np.abs(w))
                    w2 = np.zeros((len(w),1))
                    w2[ind] = 1
                    w = w2
                ax.set_title('best fit tree unit',fontsize = 14)
                self.predict = self.tree_predict
                s = np.linspace(xmin,xmax,400)
                t = [self.predict(np.asarray([v]),w) for v in s]
                ax.plot(s,t,linewidth = 2.75,color = self.colors[0],zorder = 3)
                
                # cleanup panel
                ax.set_xlim([xmin,xmax])
                ax.set_ylim([ymin,ymax])
                ax.set_xlabel(r'$x$', fontsize = 14,labelpad = 10)
                ax.set_ylabel(r'$y$', rotation = 0,fontsize = 14,labelpad = 10)
                ax.set_xticks(np.arange(round(xmin), round(xmax)+1, 1.0))
                ax.set_yticks(np.arange(round(ymin), round(ymax)+1, 1.0))
            return artist,
                
        anim = animation.FuncAnimation(fig, animate,frames = len(num_elements)+1, interval = len(num_elements)+1, blit=True)
        
        # produce animation and save
        fps = 50
        if 'fps' in kwargs:
            fps = kwargs['fps']
        anim.save(savepath, fps=fps, extra_args=['-vcodec', 'libx264'])
        clear_output()    
    
    ########### cross-validation functionality ###########
    # function for splitting dataset into k folds
    def split_data(self,folds):
        # split data into k equal (as possible) sized sets
        L = np.size(self.y)
        order = np.random.permutation(L)
        c = np.ones((L,1))
        L = int(np.round((1/folds)*L))
        for s in np.arange(0,folds-2):
            c[order[s*L:(s+1)*L]] = s + 2
        c[order[(folds-1)*L:]] = folds
        return c
    
    ###### fit and compare ######
    def brows_single_cross_val(self,savepath,**kwargs):
        # parse input args
        num_elements = [1,10,len(self.y)]
        if 'num_elements' in kwargs:
            num_elements = kwargs['num_elements']
        basis = kwargs['basis']
        self.colors = [[1,0,0.4], [ 0, 0.4, 1],[0, 1, 0.5],[1, 0.7, 0.5],[0.7, 0.6, 0.5],'mediumaquamarine']
        folds = kwargs['folds']
        
        # make indices for split --> keep first fold for test, last k-1 for training
        c = self.split_data(folds)
        train_inds = np.argwhere(c > 1)
        train_inds = [v[0] for v in train_inds]
       
        test_inds = np.argwhere(c == 1)
        test_inds = [v[0] for v in test_inds]
        
        # split up points this way
        self.x_train = copy.deepcopy(self.x[train_inds])
        self.x_test = copy.deepcopy(self.x[test_inds])
        self.y_train = copy.deepcopy(self.y[train_inds])
        self.y_test = copy.deepcopy(self.y[test_inds])
        
        # set dials for tanh network and trees
        num_elements = [v+1 for v in num_elements]
        self.num_elements = max(num_elements)

        # choose basis type
        self.F = []
        weight_history = []
        if basis == 'poly':
            self.F = self.poly_feats(self.num_elements)
            self.F_train = self.F[train_inds,:]
            self.F_test = self.F[test_inds,:]
            
            # collect poly and tanh weights over each desired level
            for element in num_elements:
                # fit weights to data
                w = np.linalg.lstsq(self.F_train[:,:element], self.y_train)[0]

                # store weights
                weight_history.append(w)
                
            self.predict = self.poly_predict

        if basis == 'tanh':
            # random weights for tanh network, tanh transform 
            scale = 1
            self.R = scale*np.random.randn(self.num_elements,2)
            self.F = self.tanh_feats(self.num_elements)
            self.F_train = self.F[train_inds,:]
            self.F_test = self.F[test_inds,:]
            
            # collect poly and tanh weights over each desired level
            for element in num_elements:
                # fit weights to data
                w = np.linalg.lstsq(self.F_train[:,:element], self.y_train)[0]

                # store weights
                weight_history.append(w)
            
            self.predict = self.tanh_predict

        if basis == 'tree':
            self.dial_settings()
            self.F = self.F_tree
            self.F_train = self.F[train_inds,:]
            self.F_test = self.F[test_inds,:]
            weight_history = self.boosting(self.F_train,self.y_train,its = 3000)

            # compute number of non-zeros per weight in history
            nonzs = [len(np.argwhere(w != 0)) for w in weight_history]

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
            weight_history = [weight_history[ind] for ind  in new_inds]
            weight_history = [weight_history[ind - 2] for ind in num_elements]
            self.predict = self.tree_predict

        
        ### compute training and testing cost eval history ###
        train_errors = []
        test_errors = []
        for i in range(len(weight_history)):
            item = copy.deepcopy(i)
            if basis == 'tree':
                item = min(len(self.y)-1, num_elements[i]-1,len(weight_history)-1) 
            w = weight_history[item]
            self.D = len(w) - 1

            # compute training error 
            self.x_orig = copy.deepcopy(self.x)
            self.x = self.x_train
            self.y_orig = copy.deepcopy(self.y)
            self.y = self.y_train           
            train_error = (self.least_squares(w)/float(len(self.y_train)))**(0.5)
            train_errors.append(train_error)
            
            # compute testing error
            self.x = copy.deepcopy(self.x_orig)
            self.x_orig = copy.deepcopy(self.x)
            self.x = self.x_test
            self.y = copy.deepcopy(self.y_orig)
            self.y_orig = copy.deepcopy(self.y)
            self.y = self.y_test          
            test_error = (self.least_squares(w)/float(len(self.y_test)))**(0.5)
            
            self.y = copy.deepcopy(self.y_orig)
            self.x = copy.deepcopy(self.x_orig)

            # store training and testing errors
            test_error = self.least_squares(w)
            test_errors.append(test_error)
     
        # normalize training and validation errors
        train_errors = [v/float(np.size(self.y_train)) for v in train_errors]
        test_errors = [v/float(np.size(self.y_test)) for v in test_errors]

        # plot cost path - scale to fit inside same aspect as classification plots
        num_iterations = len(weight_history)
        minxc = min(num_elements)-1
        maxxc = max(num_elements)-1
        gapxc = (maxxc - minxc)*0.1
        minxc -= gapxc
        maxxc += gapxc
        minc = min(min(copy.deepcopy(train_errors)),min(copy.deepcopy(test_errors)))
        maxc = max(max(copy.deepcopy(train_errors[:])),max(copy.deepcopy(test_errors[:5])))
        gapc = (maxc - minc)*0.1
        minc -= gapc
        maxc += gapc

        ### plot it
        # construct figure
        fig = plt.figure(figsize = (5,5))
        artist = fig

        # create subplot with 3 panels, plot input function in center plot
        gs = gridspec.GridSpec(2, 2)#, width_ratios=[1,1,1,1]) 
        ax = plt.subplot(gs[0]); ax.axis('off');
        ax1 = plt.subplot(gs[1]); ax1.axis('off');
        ax2 = plt.subplot(gs[2]); ax2.axis('off');
        ax3 = plt.subplot(gs[3]); ax2.axis('off');

        # set viewing range for all 3 panels
        xmax = max(copy.deepcopy(self.x))
        xmin = min(copy.deepcopy(self.x))
        xgap = (xmax - xmin)*0.05
        xmax += xgap
        xmin -= xgap
        ymax = max(copy.deepcopy(self.y))[0]
        ymin = min(copy.deepcopy(self.y))[0]
        ygap = (ymax - ymin)*0.4
        ymax += ygap
        ymin -= ygap
        
        # animate
        print ('beginning animation rendering...')
        def animate(k):
            # clear the panel
            ax.cla()
            ax1.cla()
            ax2.cla()
            ax3.cla()
            
            # print rendering update
            if np.mod(k+1,5) == 0:
                print ('rendering animation frame ' + str(k+1) + ' of ' + str(len(num_elements)))
            if k == len(num_elements):
                print ('animation rendering complete!')
                time.sleep(1)
                clear_output()
                
                
            #### plot data and clean up panels ####
            # scatter data
            ax.scatter(self.x,self.y,color = 'k',edgecolor = 'w',s = 50,zorder = 1)
            ax1.scatter(self.x_train,self.y_train,color = [0,0.7,1],edgecolor = 'k',s = 60,zorder = 1)
            ax2.scatter(self.x_test,self.y_test,color = [1,0.8,0.5],edgecolor = 'k',s = 60,zorder = 1)

            # cleanup panels
            ax.set_xlim([xmin,xmax])
            ax.set_ylim([ymin,ymax])
            ax.set_xlabel(r'$x$', fontsize = 14,labelpad = 0)
            ax.set_ylabel(r'$y$', rotation = 0,fontsize = 14,labelpad = 5)
            ax.set_xticks(np.arange(round(xmin), round(xmax)+1, 1.0))
            ax.set_yticks(np.arange(round(ymin), round(ymax)+1, 1.0))
            ax.set_title('original data',fontsize = 15)

            ax1.set_xlim([xmin,xmax])
            ax1.set_ylim([ymin,ymax])
            ax1.set_xlabel(r'$x$', fontsize = 14,labelpad = 0)
            ax1.set_ylabel(r'$y$', rotation = 0,fontsize = 14,labelpad = 5)
            ax1.set_xticks(np.arange(round(xmin), round(xmax)+1, 1.0))
            ax1.set_yticks(np.arange(round(ymin), round(ymax)+1, 1.0))
            ax1.set_title('training data',fontsize = 15)

            ax2.set_xlim([xmin,xmax])
            ax2.set_ylim([ymin,ymax])
            ax2.set_xlabel(r'$x$', fontsize = 14,labelpad = 0)
            ax2.set_ylabel(r'$y$', rotation = 0,fontsize = 14,labelpad = 5)
            ax2.set_xticks(np.arange(round(xmin), round(xmax)+1, 1.0))
            ax2.set_yticks(np.arange(round(ymin), round(ymax)+1, 1.0))
            ax2.set_title('validation data',fontsize = 15)
                          
             # cleanup
            ax3.set_xlabel('model',fontsize = 12)
            ax3.set_title('errors',fontsize = 15)
           
            # cleanp panel
            ax3.set_xlim([minxc,maxxc])
            ax3.set_ylim([minc,maxc])
            ax3.xaxis.set_major_locator(MaxNLocator(integer=True))
                     
            if k > 0:
                # loop over panels, produce plots
                self.D = num_elements[k-1] 
                cs = 0

                # fit to data
                F = 0
                predict = 0
                w = 0
                if basis == 'poly': 
                    w = weight_history[k-1]
                    self.D = len(w) - 1
                    #ax1.set_title(str(self.D) + ' poly units',fontsize = 14)
                    self.predict = self.poly_predict

                elif basis == 'tanh':
                    w = weight_history[k-1]
                    self.D = len(w) - 1
                    #ax1.set_title(str(self.D) + ' tanh units',fontsize = 14)
                    self.predict = self.tanh_predict

                elif basis == 'tree':
                    item = min(len(self.y)-1, num_elements[k-1]-1,len(weight_history)-1) 
                    w = weight_history[item]
                    #ax1.set_title(str(item) + ' tree units',fontsize = 14)
                    self.predict = self.tree_predict


                # produce learned predictor
                s = np.linspace(xmin,xmax,400)
                t = [self.predict(np.asarray([v]),w) for v in s]

                # plot approximation and data in panel
                ax.plot(s,t,linewidth = 2.75,color = self.colors[cs],zorder = 3)
                ax1.plot(s,t,linewidth = 2.75,color = self.colors[cs],zorder = 3)
                ax2.plot(s,t,linewidth = 2.75,color = self.colors[cs],zorder = 3)
                cs += 1

                ### plot training and testing errors  
                ax3.plot([v-1 for v in num_elements[:k]],train_errors[:k],color = [0,0.7,1],linewidth = 1.5,zorder = 1,label = 'training')
                ax3.scatter([v-1 for v in num_elements[:k]],train_errors[:k],color = [0,0.7,1],s = 70,edgecolor = 'w',linewidth = 1.5,zorder = 3)

                ax3.plot([v-1 for v in num_elements[:k]],test_errors[:k],color = [1,0.8,0.5],linewidth = 1.5,zorder = 1,label = 'validation')
                ax3.scatter([v-1 for v in num_elements[:k]],test_errors[:k],color= [1,0.8,0.5],s = 70,edgecolor = 'w',linewidth = 1.5,zorder = 3)
                #legend = ax3.legend(loc='upper right')

            return artist,
            
        anim = animation.FuncAnimation(fig, animate,frames = len(num_elements)+1, interval = len(num_elements)+1, blit=True)
        
        # produce animation and save
        fps = 50
        if 'fps' in kwargs:
            fps = kwargs['fps']
        anim.save(savepath, fps=fps, extra_args=['-vcodec', 'libx264'])
        clear_output()   
        
        
from intro_general_library import superlearn_setup as general_superlearn_setup
from intro_boost_library import stump_booster 

class ClassificationVisualizer:
    



    '''
    Visualize linear regression in 2 and 3 dimensions.  For single input cases (2 dimensions) the path of gradient descent on the cost function can be animated.
    '''
    #### initialize ####
    def __init__(self,csvname):
        # grab input
        data = np.loadtxt(csvname,delimiter = ',')
        self.x = data[:-1,:]
        self.y = data[-1,:]
        self.y.shape = (1,len(self.y))
        self.colors = ['salmon','cornflowerblue','lime','bisque','mediumaquamarine','b','m','g']
        
    ####### fit each type of universal approximator #######
    def run_approximators(self,**kwargs):
        # define number of units per approximator
        poly_units = 5
        net_units = 8
        tree_units = 8
        
        if 'poly_units' in kwargs:
            poly_units = kwargs['poly_units']
        if 'net_units' in kwargs:
            net_units = kwargs['net_units']
        if 'tree_units' in kwargs:
            tree_units = kwargs['tree_units']
            
        # run polys
        self.runs1 = self.run_poly(poly_units)
        
        # run nets
        self.runs2 = self.run_net(net_units)
        
        # run trees
        self.runs3 = self.run_trees(tree_units) 

    ### run classification with poly features ###
    def run_poly(self,num_units):
        runs = []
        for j in range(num_units):
            print ('fitting ' + str(j + 1) + ' poly units')

            # import the v1 library
            mylib = general_superlearn_setup.Setup(self.x,self.y)

            # choose features
            mylib.choose_features(name = 'polys',degree = j+1)

            # choose normalizer
            mylib.choose_normalizer(name = 'none')

            # pick training set
            mylib.make_train_valid_split(train_portion=1)

            # choose cost
            mylib.choose_cost(name = 'softmax')

            # fit an optimization
            mylib.fit(max_its = 5,optimizer = 'newtons_method',epsilon = 10**(-5))

            # add model to list
            runs.append(copy.deepcopy(mylib))
            
        print ('poly run complete')
        time.sleep(1.5)
        clear_output()
        return runs 

    ### run classification with net features ###
    def run_net(self,num_units):
        runs = []
        for j in range(num_units):
            print ('fitting ' + str(j + 1) + ' net units')

            # import the v1 library
            mylib = general_superlearn_setup.Setup(self.x,self.y)

            # choose features
            mylib.choose_features(name = 'multilayer_perceptron',layer_sizes = [2,j+1,1],activation = 'tanh')

            # choose normalizer
            mylib.choose_normalizer(name = 'standard')

            # pick training set
            mylib.make_train_valid_split(train_portion=1)

            # choose cost
            mylib.choose_cost(name = 'softmax')

            # fit an optimization
            mylib.fit(max_its = 1000,alpha_choice = 10**(0),optimizer = 'gradient_descent')

            # add model to list
            runs.append(copy.deepcopy(mylib))
            
        print ('net run complete')
        time.sleep(1.5)
        clear_output()
        return runs 
    
    ### run classification with tree features ###
    def run_trees(self,num_rounds):
        # import booster
        mylib = stump_booster.Setup(self.x,self.y)

        # choose normalizer
        mylib.choose_normalizer(name = 'none')

        # pick training set
        mylib.make_train_valid_split(train_portion=1)

        # choose cost|
        mylib.choose_cost(name = 'softmax')

        # choose optimizer
        mylib.choose_optimizer('newtons_method',max_its=1)

        # run boosting
        mylib.boost(num_rounds)

        return mylib
  
    ########## show classification results ##########
    def animate_comparisons(self,savepath,frames,**kwargs):
        pt_size = 55
        if 'pt_size' in kwargs:
            pt_size = kwargs['pt_size']
            
        ### get inds for each run ###
        runs1 = self.runs1; runs2 = self.runs2; runs3 = self.runs3;
        inds1 = np.arange(0,len(runs1),int(len(runs1)/float(frames)))
        inds2 = np.arange(0,len(runs2),int(len(runs2)/float(frames)))
        inds3 = np.arange(0,len(runs3.models),int(len(runs3.models)/float(frames)))
            
        # select inds of history to plot
        num_runs = frames

        # construct figure
        fig = plt.figure(figsize=(9,4))
        artist = fig

        # create subplot with 1 active panel
        gs = gridspec.GridSpec(1, 3, width_ratios = [1,1,1]) 
        ax1 = plt.subplot(gs[0]); ax1.set_aspect('equal'); 
        ax1.axis('off');
        ax1.xaxis.set_visible(False) # Hide only x axis
        ax1.yaxis.set_visible(False) # Hide only x axis
        
        ax2 = plt.subplot(gs[1]); ax2.set_aspect('equal'); 
        ax2.axis('off');
        ax2.xaxis.set_visible(False) # Hide only x axis
        ax2.yaxis.set_visible(False) # Hide only x axis
        
        ax3 = plt.subplot(gs[2]); ax3.set_aspect('equal'); 
        ax3.axis('off');
        ax3.xaxis.set_visible(False) # Hide only x axis
        ax3.yaxis.set_visible(False) # Hide only x axis
        
        # viewing ranges
        xmin1 = min(copy.deepcopy(self.x[0,:]))
        xmax1 = max(copy.deepcopy(self.x[0,:]))
        xgap1 = (xmax1 - xmin1)*0.05
        xmin1 -= xgap1
        xmax1 += xgap1

        xmin2 = min(copy.deepcopy(self.x[1,:]))
        xmax2 = max(copy.deepcopy(self.x[1,:]))
        xgap2 = (xmax2 - xmin2)*0.05
        xmin2 -= xgap2
        xmax2 += xgap2

        # start animation
        num_frames = num_runs
        print ('starting animation rendering...')
        def animate(k):
            # clear panels
            ax1.cla()
            ax2.cla()
            ax3.cla()

            # print rendering update
            if np.mod(k+1,25) == 0:
                print ('rendering animation frame ' + str(k+1) + ' of ' + str(num_frames))
            if k == num_frames - 1:
                print ('animation rendering complete!')
                time.sleep(1.5)
                clear_output()
            
            # scatter data
            ind0 = np.argwhere(self.y == +1)
            ind0 = [e[1] for e in ind0]
            ind1 = np.argwhere(self.y == -1)
            ind1 = [e[1] for e in ind1]
            for ax in [ax1,ax2,ax3]:
                ax.scatter(self.x[0,ind0],self.x[1,ind0],s = pt_size, color = self.colors[0], edgecolor = 'k',antialiased=True)
                ax.scatter(self.x[0,ind1],self.x[1,ind1],s = pt_size, color = self.colors[1], edgecolor = 'k',antialiased=True)
                
            if k == 0:
                ax1.set_title(str(0) + ' units fit to data',fontsize = 14,color = 'w')
                ax1.set_title(str(0) + ' units fit to data',fontsize = 14,color = 'w')
                ax1.set_title(str(0) + ' units fit to data',fontsize = 14,color = 'w')
                
                ax1.set_xlim([xmin1,xmax1])
                ax1.set_ylim([xmin2,xmax2])
                ax2.set_xlim([xmin1,xmax1])
                ax2.set_ylim([xmin2,xmax2])                
                ax3.set_xlim([xmin1,xmax1])
                ax3.set_ylim([xmin2,xmax2])
                
            # plot fit
            if k > 0:
                # get current run
                a1 = inds1[k-1] 
                a2 = inds2[k-1] 
                a3 = inds3[k-1] 

                run1 = runs1[a1]
                a1 = len(run1.w_init) - 1

                run2 = runs2[a2]
                model3 = runs3.models[a3]
                steps = runs3.best_steps[:a3+1]
                
                # plot models to data
                self.draw_fit(ax1,run1,a1)
                self.draw_fit(ax2,run2,a2 + 1)
                self.draw_boosting_fit(ax3,steps,a3)
                
            return artist,

        anim = animation.FuncAnimation(fig, animate ,frames=num_frames+1, interval=num_frames+1, blit=True)
        
        # produce animation and save
        fps = 50
        if 'fps' in kwargs:
            fps = kwargs['fps']
        anim.save(savepath, fps=fps, extra_args=['-vcodec', 'libx264'])
        clear_output()


    def draw_fit(self,ax,run,num_units):
        # viewing ranges
        xmin1 = min(copy.deepcopy(self.x[0,:]))
        xmax1 = max(copy.deepcopy(self.x[0,:]))
        xgap1 = (xmax1 - xmin1)*0.05
        xmin1 -= xgap1
        xmax1 += xgap1

        xmin2 = min(copy.deepcopy(self.x[1,:]))
        xmax2 = max(copy.deepcopy(self.x[1,:]))
        xgap2 = (xmax2 - xmin2)*0.05
        xmin2 -= xgap2
        xmax2 += xgap2

        ymin = min(copy.deepcopy(self.y))
        ymax = max(copy.deepcopy(self.y))
        ygap = (ymax - ymin)*0.05
        ymin -= ygap
        ymax += ygap
             
        # plot boundary for 2d plot
        r1 = np.linspace(xmin1,xmax1,300)
        r2 = np.linspace(xmin2,xmax2,300)
        s,t = np.meshgrid(r1,r2)
        s = np.reshape(s,(np.size(s),1))
        t = np.reshape(t,(np.size(t),1))
        h = np.concatenate((s,t),axis = 1).T
        
        # plot total fit
        cost = run.cost
        model = run.model
        feat = run.feature_transforms
        normalizer = run.normalizer
        cost_history = run.train_cost_histories[0]
        weight_history = run.weight_histories[0]

        # get best weights                
        win = np.argmin(cost_history)
        w = weight_history[win]        
        
        model = lambda b: run.model(normalizer(b),w)
        z = model(h)
        z = np.sign(z)

        # reshape it
        s.shape = (np.size(r1),np.size(r2))
        t.shape = (np.size(r1),np.size(r2))     
        z.shape = (np.size(r1),np.size(r2))

        #### plot contour, color regions ####
        ax.contour(s,t,z,colors='k', linewidths=2.5,levels = [0],zorder = 2)
        ax.contourf(s,t,z,colors = [self.colors[1],self.colors[0]],alpha = 0.15,levels = range(-1,2))
                
        ### cleanup left plots, create max view ranges ###
        ax.set_xlim([xmin1,xmax1])
        ax.set_ylim([xmin2,xmax2])
        ax.set_title(str(num_units) + ' units fit to data',fontsize = 14)
    
    def draw_boosting_fit(self,ax,steps,ind):
        # viewing ranges
        xmin1 = min(copy.deepcopy(self.x[0,:]))
        xmax1 = max(copy.deepcopy(self.x[0,:]))
        xgap1 = (xmax1 - xmin1)*0.05
        xmin1 -= xgap1
        xmax1 += xgap1

        xmin2 = min(copy.deepcopy(self.x[1,:]))
        xmax2 = max(copy.deepcopy(self.x[1,:]))
        xgap2 = (xmax2 - xmin2)*0.05
        xmin2 -= xgap2
        xmax2 += xgap2

        ymin = min(copy.deepcopy(self.y))
        ymax = max(copy.deepcopy(self.y))
        ygap = (ymax - ymin)*0.05
        ymin -= ygap
        ymax += ygap

        # plot boundary for 2d plot
        r1 = np.linspace(xmin1,xmax1,30)
        r2 = np.linspace(xmin2,xmax2,30)
        s,t = np.meshgrid(r1,r2)
        s = np.reshape(s,(np.size(s),1))
        t = np.reshape(t,(np.size(t),1))
        h = np.concatenate((s,t),axis = 1).T

        model = lambda x: np.sum([v(x) for v in steps],axis=0)
        z = model(h)
        z = np.sign(z)

        # reshape it
        s.shape = (np.size(r1),np.size(r2))
        t.shape = (np.size(r1),np.size(r2))     
        z.shape = (np.size(r1),np.size(r2))

        #### plot contour, color regions ####
        ax.contour(s,t,z,colors='k', linewidths=2.5,levels = [0],zorder = 2)
        ax.contourf(s,t,z,colors = [self.colors[1],self.colors[0]],alpha = 0.15,levels = range(-1,2))

        ### cleanup left plots, create max view ranges ###
        ax.set_xlim([xmin1,xmax1])
        ax.set_ylim([xmin2,xmax2])
        ax.set_title(str(ind+1) + ' units fit to data',fontsize = 14)