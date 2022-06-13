# import standard plotting and animation
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from mpl_toolkits.mplot3d import Axes3D
from IPython.display import clear_output
from matplotlib import gridspec
import matplotlib.patches as mpatches

# import autograd functionality
import autograd.numpy as np
from autograd.misc.flatten import flatten_func
from autograd import value_and_grad 

# import some basic libs
import math
import copy
import time

# import mlrefined libraries
from mlrefined_libraries import math_optimization_library as optlib

class Visualizer:
    '''
    animations for visualizing sparse feature selection for regression and 
    classification
    '''
    def __init__(self,x,y,**kwargs):
        # get input/output pairs
        self.x_orig = x
        self.y_orig = y
       
        # normalize input for optimization
        normalize = False
        normalize_out = False
        if 'normalize' in kwargs:
            normalize = kwargs['normalize']
        if 'normalize_out' in kwargs:
            normalize_out = kwargs['normalize_out']
        if normalize == True:
            # normalize input?
            normalizer,inverse_normalizer = self.standard_normalizer(self.x_orig)

            # normalize input by subtracting off mean and dividing by standard deviation
            self.x = normalizer(self.x_orig)
        else:
            self.x = x_orig
            
        if normalize_out == True:
            # normalize input?
            normalizer,inverse_normalizer = self.standard_normalizer(self.y_orig)

            # normalize input by subtracting off mean and dividing by standard deviation
            self.y = normalizer(self.y_orig)
        else:
            self.y = self.y_orig
        
    # standard normalization function 
    def standard_normalizer(self,x):
        # compute the mean and standard deviation of the input
        x_means = np.mean(x,axis = 1)[:,np.newaxis]
        x_stds = np.std(x,axis = 1)[:,np.newaxis]   

        # check to make sure thta x_stds > small threshold, for those not
        # divide by 1 instead of original standard deviation
        ind = np.argwhere(x_stds < 10**(-2))
        if len(ind) > 0:
            ind = [v[0] for v in ind]
            adjust = np.zeros((x_stds.shape))
            adjust[ind] = 1.0
            x_stds += adjust

        # create standard normalizer function
        normalizer = lambda data: (data - x_means)/x_stds

        # create inverse standard normalizer
        inverse_normalizer = lambda data: data*x_stds + x_means

        # return normalizer 
        return normalizer,inverse_normalizer

    #### main animator #####
    # compare multiple l1 regularized runs
    def compare_lams(self,savepath,g,lams,**kwargs):       
        if 'counter' in kwargs:
            counter = kwargs['counter']
            
        # initialize figure
        fig = plt.figure(figsize = (9,3))
        artist = fig
        num_lams = len(lams)
        gs = gridspec.GridSpec(1,1) 
        ax = plt.subplot(gs[0])
        
        ### run over all input lamdas ###
        # setup optimizations
        alpha_choice = 10**(-1)
        max_its = 1000
        batch_size = self.y.size
        algo = 'gradient_descent'
        w = 0.1*np.random.randn(self.x.shape[0]+1,1)
        if 'max_its' in kwargs:
            max_its = kwargs['max_its']
        if 'alpha' in kwargs:
            alpha = kwargs['alpha']
        if 'batch_size' in kwargs:
            batch_size = kwargs['batch_size']
        if 'algo' in kwargs:
            algo = kwargs['algo']
            
        # start animation
        num_frames = len(lams)
        print ('starting animation rendering...')
        def animate(k):            
            # clear panels
            ax.cla()
            lam = lams[k]
            
            # print rendering update
            if np.mod(k+1,25) == 0:
                print ('rendering animation frame ' + str(k+1) + ' of ' + str(num_frames))
            if k == num_frames - 1:
                print ('animation rendering complete!')
                time.sleep(1.5)
                clear_output()

            # run optimization
            if algo == 'gradient_descent':
                weight_history,cost_history = self.gradient_descent(g,w,self.x,self.y,lam,alpha_choice,max_its,batch_size)
            if algo == 'RMSprop':
                weight_history,cost_history = self.RMSprop(g,w,self.x,self.y,lam,alpha_choice,max_its,batch_size)

            # choose set of weights to plot based on lowest cost val
            ind = np.argmin(cost_history)
            
            # classification? then base on accuracy
            if 'counter' in kwargs:
                # create counting cost history as well
                counts = [counter(v,self.x,self.y,lam) for v in weight_history]
                if k == 0:
                    ind = np.argmin(counts)
                count = counts[ind]
                acc = 1 - count/self.y.size
                acc = np.round(acc,2)
                
            # save lowest misclass weights
            w_best = weight_history[ind][1:]
            
            # plot
            ax.axhline(c='k',zorder = 2)
            
            # make bar plot
            ax.bar(np.arange(0,len(w_best)), w_best, color='k', alpha=0.5)
                
            # dress panel
            title1 = r'$\lambda = ' + str(np.round(lam,2)) + '$' 
            costval = cost_history[ind][0]
            title2 = ', cost val = ' + str(np.round(costval,2))
            if 'counter' in kwargs:
                title2 = ', accuracy = ' + str(acc)
            title = title1 + title2
            ax.set_title(title)
            ax.set_xlabel('learned weights')
            return artist,

        anim = animation.FuncAnimation(fig, animate ,frames=num_frames, interval=num_frames, blit=True)
        
        # produce animation and save
        fps = 50
        if 'fps' in kwargs:
            fps = kwargs['fps']
        anim.save(savepath, fps=fps, extra_args=['-vcodec', 'libx264'])
        clear_output()

    
    # static graphics
    def plot_regress(self,id1,labels):
        fig = plt.figure(figsize = (10,3))
        plt.xlabel(labels[0])
        plt.ylabel(labels[1])
        plt.scatter(self.x_orig[id1,:],self.y_orig,color = 'k',edgecolor = 'w',s = 30)
        #plt.axes().set_aspect('equal', 'datalim')
        plt.show()


    def plot_classif(self, id_1, id_2,labels):
        # create figure for plotting
        fig = plt.figure(figsize = (5,5))

        # setup colors / labels for plot
        red_patch = mpatches.Patch(color='red', label=labels[0])
        blue_patch = mpatches.Patch(color='blue', label=labels[1])
        plt.legend(handles=[red_patch, blue_patch])
        plt.legend(handles=[red_patch, blue_patch], loc = 2)
        
        # scatter plot data
        ind = np.argwhere(self.y == -1)
        ind = [v[1] for v in ind]
        plt.scatter(self.x_orig[id_1,ind],self.x_orig[id_2,ind], color='r', s=30) #plotting the data
        
        ind = np.argwhere(self.y == +1)
        ind = [v[1] for v in ind]

        plt.scatter(self.x_orig[id_1,ind],self.x_orig[id_2,ind], color='b', s=30) #plotting the data        
        plt.show()
    
    #### optimizers ####
    # minibatch gradient descent
    def gradient_descent(self,g,w,x_train,y_train,lam,alpha_choice,max_its,batch_size): 
        # flatten the input function, create gradient based on flat function
        g_flat, unflatten, w = flatten_func(g, w)
        grad = value_and_grad(g_flat)

        # record history
        num_train = y_train.shape[1]
        w_hist = [unflatten(w)]
        train_hist = [g_flat(w,x_train,y_train,lam,np.arange(num_train))]
        
        # how many mini-batches equal the entire dataset?
        num_batches = int(np.ceil(np.divide(num_train, batch_size)))

        # over the line
        alpha = 0
        for k in range(max_its):             
            # check if diminishing steplength rule used
            if alpha_choice == 'diminishing':
                alpha = 1/float(k)
            else:
                alpha = alpha_choice
            
            for b in range(num_batches):
                # collect indices of current mini-batch
                batch_inds = np.arange(b*batch_size, min((b+1)*batch_size, num_train))
            
                # plug in value into func and derivative
                cost_eval,grad_eval = grad(w,x_train,y_train,lam,batch_inds)
                grad_eval.shape = np.shape(w)

                # take descent step with momentum
                w = w - alpha*grad_eval

            # update training and validation cost
            train_cost = g_flat(w,x_train,y_train,lam,np.arange(num_train))

            # record weight update, train and val costs
            w_hist.append(unflatten(w))
            train_hist.append(train_cost)
        return w_hist,train_hist

    # minibatch gradient descent
    def RMSprop(self,g,w,x_train,y_train,lam,alpha,max_its,batch_size,**kwargs): 
        # rmsprop params
        gamma=0.9
        eps=10**-8
        if 'gamma' in kwargs:
            gamma = kwargs['gamma']
        if 'eps' in kwargs:
            eps = kwargs['eps']

        # flatten the input function, create gradient based on flat function
        g_flat, unflatten, w = flatten_func(g, w)
        grad = value_and_grad(g_flat)

        # initialize average gradient
        avg_sq_grad = np.ones(np.size(w))

        # record history
        num_train = y_train.size
        w_hist = [unflatten(w)]
        train_hist = [g_flat(w,x_train,y_train,lam,np.arange(num_train))]

        # how many mini-batches equal the entire dataset?
        num_batches = int(np.ceil(np.divide(num_train, batch_size)))

        # over the line
        for k in range(max_its):                   
            # loop over each minibatch
            for b in range(num_batches):
                # collect indices of current mini-batch
                batch_inds = np.arange(b*batch_size, min((b+1)*batch_size, num_train))

                # plug in value into func and derivative
                cost_eval,grad_eval = grad(w,x_train,y_train,lam,batch_inds)
                grad_eval.shape = np.shape(w)

                # update exponential average of past gradients
                avg_sq_grad = gamma*avg_sq_grad + (1 - gamma)*grad_eval**2 

                # take descent step 
                w = w - alpha*grad_eval / (avg_sq_grad**(0.5) + eps)

            # update training and validation cost
            train_cost = g_flat(w,x_train,y_train,lam,np.arange(num_train))

            # record weight update, train and val costs
            w_hist.append(unflatten(w))
            train_hist.append(train_cost)

        return w_hist,train_hist