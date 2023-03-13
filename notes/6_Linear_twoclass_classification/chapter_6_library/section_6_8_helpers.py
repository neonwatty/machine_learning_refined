import matplotlib.pyplot as plt
from matplotlib import gridspec
from IPython.display import clear_output

import autograd.numpy as np
from autograd import value_and_grad 
from autograd.misc.flatten import flatten_func
import math
import time
import copy


# gradient descent function - inputs: g (input function), alpha (steplength parameter), max_its (maximum number of iterations), w (initialization)
def gradient_descent(g,alpha_choice,max_its,w):
    # flatten the input function to more easily deal with costs that have layers of parameters
    g_flat, unflatten, w = flatten_func(g, w) # note here the output 'w' is also flattened

    # compute the gradient function of our input function - note this is a function too
    # that - when evaluated - returns both the gradient and function evaluations (remember
    # as discussed in Chapter 3 we always ge the function evaluation 'for free' when we use
    # an Automatic Differntiator to evaluate the gradient)
    gradient = value_and_grad(g_flat)

    # run the gradient descent loop
    weight_history = []      # container for weight history
    cost_history = []        # container for corresponding cost function history
    alpha = 0
    for k in range(1,max_its+1):
        # check if diminishing steplength rule used
        if alpha_choice == 'diminishing':
            alpha = 1/float(k)
        else:
            alpha = alpha_choice
        
        # evaluate the gradient, store current (unflattened) weights and cost function value
        cost_eval,grad_eval = gradient(w)
        weight_history.append(unflatten(w))
        cost_history.append(cost_eval)

        # take gradient descent step
        w = w - alpha*grad_eval
            
    # collect final weights
    weight_history.append(unflatten(w))
    # compute final cost function value via g itself (since we aren't computing 
    # the gradient at the final step we don't get the final cost function value 
    # via the Automatic Differentiatoor) 
    cost_history.append(g_flat(w))  
    return weight_history,cost_history

class visualizer:
    '''
    Compare cost functions for two-class classification
    
    '''
    
    #### initialize ####
    def __init__(self,data):        
        # grab input
        self.data = data
        self.x = data[:-1,:]
        self.y = data[-1:,:] 
        
    ### cost functions ###
    def counting_cost(self,w):
        # compute predicted labels
        y_hat = np.sign(self.model(self.x,w))
                
        # compare to true labels
        ind = np.argwhere(self.y != y_hat)
        ind = [v[1] for v in ind]
       
        cost = np.sum(len(ind))
        return cost
    
    # compute stats for F1 score
    def confusion_matrix(self,w):
        # compute predicted labels
        y_hat = np.sign(self.model(self.x,w))
        
        # determine indices of real and predicted label values
        ind1 = np.argwhere(self.y == +1)
        ind1 = [v[1] for v in ind1]

        ind2 = np.argwhere(self.y == -1)
        ind2 = [v[1] for v in ind2]
        
        ind3 = np.argwhere(y_hat == +1)
        ind3 = [v[1] for v in ind3]

        ind4 = np.argwhere(y_hat == -1)
        ind4 = [v[1] for v in ind4]    
        
        # compute elements of confusion matrix
        A = len(list(set.intersection(*[set(ind1), set(ind3)])))
        B = len(list(set.intersection(*[set(ind1), set(ind4)])))
        C = len(list(set.intersection(*[set(ind2), set(ind3)])))
        D = len(list(set.intersection(*[set(ind2), set(ind4)])))
        return A,B,C,D
        
    # compute balanced accuracy
    def compute_balanced_accuracy(self,w):
        # compute confusion matrix
        A,B,C,D = self.confusion_matrix(w)
        
        # compute precision and recall
        precision = 0
        if A > 0:
            precision = A/(A + B)
            
        specif = 0
        if D > 0:
            specif = D/(C + D)
        
        # compute balanced accuracy
        balanced_accuracy = (precision + specif)/2
        return balanced_accuracy
    
    # compute linear combination of input point
    def model(self,x,w):
        a = w[0] + np.dot(x.T,w[1:])
        return a.T
    
    # the perceptron relu cost
    def relu(self,w):
        cost = np.sum(np.maximum(0,-self.y*self.model(self.x,w)))
        return cost/float(np.size(y))

    # the convex softmax cost function
    def softmax(self,w):
        cost = np.sum(np.log(1 + np.exp(-self.y*self.model(self.x,w))))
        return cost/float(np.size(self.y))
                   
    ### compare grad descent runs - given cost to counting cost ###
    def compare_to_counting(self,cost,**kwargs):
        # parse args
        num_runs = 1
        if 'num_runs' in kwargs:
            num_runs = kwargs['num_runs']
        max_its = 200
        if 'max_its' in kwargs:
            max_its = kwargs['max_its']
        alpha = 10**-3
        if 'alpha' in kwargs:
            alpha = kwargs['alpha']  
         
        #### perform all optimizations ###
        g = self.softmax
        if cost == 'softmax':
            g = self.softmax
        if cost == 'relu':
            g = self.relu
        g_count = self.counting_cost

        big_w_hist = []
        for j in range(num_runs):
            # construct random init
            w_init = np.random.randn(np.shape(self.x)[0]+1,1)
            
            # run optimizer
            w_hist,g_hist = gradient_descent(g = g, alpha_choice = alpha,max_its = max_its,w = w_init)
            
            # store history
            big_w_hist.append(w_hist)
            
        ##### setup figure to plot #####
        # initialize figure
        fig = plt.figure(figsize = (8,4))
        artist = fig
        
        # create subplot with 3 panels, plot input function in center plot
        gs = gridspec.GridSpec(1, 2, width_ratios=[1,1]) 
        ax1 = plt.subplot(gs[0]); 
        ax2 = plt.subplot(gs[1]);
        
        #### start runs and plotting ####
        for j in range(num_runs):
            w_hist = big_w_hist[j]
            
            # evaluate counting cost / other cost for each weight in history, then plot
            count_evals = []
            cost_evals = []
            for k in range(len(w_hist)):
                w = w_hist[k]
                g_eval = g(w)
                cost_evals.append(g_eval)
                
                count_eval = g_count(w)
                count_evals.append(count_eval)
                
            # plot each 
            ax1.plot(np.arange(0,len(w_hist)),count_evals[:len(w_hist)],linewidth = 2)
            ax2.plot(np.arange(0,len(w_hist)),cost_evals[:len(w_hist)],linewidth = 2)
                
        #### cleanup plots ####
        # label axes
        ax1.set_xlabel('iteration',fontsize = 13)
        ax1.set_ylabel('num misclassifications',rotation = 90,fontsize = 13)
        ax1.set_title('number of misclassifications',fontsize = 14)
        ax1.axhline(y=0, color='k',zorder = 0,linewidth = 0.5)
        
        ax2.set_xlabel('iteration',fontsize = 13)
        ax2.set_ylabel('cost value',rotation = 90,fontsize = 13)
        title = cost + ' cost'
        ax2.set_title(title,fontsize = 14)
        ax2.axhline(y=0, color='k',zorder = 0,linewidth = 0.5)
        
        plt.show()
        
    ### compare grad descent runs - given cost to counting cost ###
    def compare_to_balanced_accuracy(self,cost,**kwargs):
        # parse args
        num_runs = 1
        if 'num_runs' in kwargs:
            num_runs = kwargs['num_runs']
        max_its = 200
        if 'max_its' in kwargs:
            max_its = kwargs['max_its']
        alpha = 10**-3
        if 'alpha' in kwargs:
            alpha = kwargs['alpha']  
         
        #### perform all optimizations ###
        g = self.softmax
        if cost == 'softmax':
            g = self.softmax
        if cost == 'relu':
            g = self.relu
        computer = self.compute_balanced_accuracy
        g_count = self.counting_cost

        self.big_w_hist = []
        for j in range(num_runs):
            # construct random init
            w_init = np.random.randn(np.shape(self.x)[0]+1,1)
            
            # run optimizer
            w_hist,g_hist = gradient_descent(g = g, alpha_choice = alpha,max_its = max_its,w = w_init)
            
            # store history
            self.big_w_hist.append(w_hist)
            
        ##### setup figure to plot #####
        # initialize figure
        fig = plt.figure(figsize = (9,3))
        artist = fig
        
        # create subplot with 3 panels, plot input function in center plot
        gs = gridspec.GridSpec(1, 2, width_ratios=[1,1]) 
        ax1 = plt.subplot(gs[0]); 
        ax2 = plt.subplot(gs[1]);
        
        #### start runs and plotting ####
        for j in range(num_runs):
            w_hist = self.big_w_hist[j]
            
            # evaluate counting cost / other cost for each weight in history, then plot
            self.balanced_vals = []
            cost_evals = []
            self.count_evals = []
            for k in range(len(w_hist)):
                w = w_hist[k]
                g_eval = g(w)
                cost_evals.append(g_eval)
                
                count_eval = 1 - g_count(w)/self.y.size
                self.count_evals.append(count_eval)
                
                balanced_accuracy = computer(w)
                self.balanced_vals.append(balanced_accuracy)
                
            # plot each             
            ax1.plot(np.arange(0,len(w_hist)),self.count_evals[:len(w_hist)],linewidth = 2,label = 'accuracy')
            ax1.plot(np.arange(0,len(w_hist)),self.balanced_vals[:len(w_hist)],linewidth = 2,label = 'balanced accuracy')
            ax1.legend(loc = 4)
            
            ax2.plot(np.arange(0,len(w_hist)),cost_evals[:len(w_hist)],linewidth = 2)
                
        #### cleanup plots ####
        # label axes      
        ax1.set_xlabel('iteration',fontsize = 13)
        ax1.set_title('metrics',fontsize = 14)
        ax1.axhline(y=0, color='k',zorder = 0,linewidth = 0.5)
        
        ax2.set_xlabel('iteration',fontsize = 13)
        ax2.set_ylabel('cost value',rotation = 90,fontsize = 13)
        title = cost + ' cost'
        ax2.set_title(title,fontsize = 14)
        ax2.axhline(y=0, color='k',zorder = 0,linewidth = 0.5)
        
        plt.show()   