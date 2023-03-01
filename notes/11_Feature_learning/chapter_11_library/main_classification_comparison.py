# import standard plotting and animation
import matplotlib.pyplot as plt
from matplotlib import gridspec
from matplotlib.ticker import FormatStrFormatter
import matplotlib.animation as animation
from IPython.display import clear_output

# import autograd functionality
import autograd.numpy as np

# import standard libraries
import math
import time
import copy
from inspect import signature

# custom fit libraries
from . import intro_boost_library
from . import intro_general_library

class Visualizer:
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
            mylib = intro_general_library.superlearn_setup.Setup(self.x,self.y)

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
            mylib = intro_general_library.superlearn_setup.Setup(self.x,self.y)

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
        mylib = intro_boost_library.stump_booster.Setup(self.x,self.y)

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