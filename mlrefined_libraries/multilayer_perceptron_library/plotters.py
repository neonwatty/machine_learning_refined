# import standard plotting and animation
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from matplotlib import gridspec
from IPython.display import clear_output
from mpl_toolkits.mplot3d import proj3d
from matplotlib.text import Annotation
from mpl_toolkits.mplot3d.proj3d import proj_transform
from matplotlib import gridspec
from autograd.misc.flatten import flatten_func
from autograd import grad as compute_grad   


# import autograd functionality
import numpy as np
import math
import time
import copy

class Visualizer:
    '''
    Various plotting and visualization functions for illustrating training / fitting of nonlinear regression and classification
    '''             
    
    # compare regression cost histories from multiple runs
    def compare_regression_histories(self, histories, start,**kwargs):
        # initialize figure
        fig = plt.figure(figsize = (8,3))

        # create subplot with 1 panel
        gs = gridspec.GridSpec(1, 1) 
        ax = plt.subplot(gs[0]); 
        
        # any labels to add?        
        labels = [' ',' ']
        if 'labels' in kwargs:
            labels = kwargs['labels']

        # run through input histories, plotting each beginning at 'start' iteration
        for c in range(len(histories)):
            history = histories[c]
            
            label = 0
            if c == 0:
                label = labels[0]
            else:
                label = labels[1]
                
            # check if a label exists, if so add it to the plot
            if np.size(label) == 0:
                ax.plot(np.arange(start,len(history),1),history[start:],linewidth = 3*(0.8)**(c)) 
            else:               
                ax.plot(np.arange(start,len(history),1),history[start:],linewidth = 3*(0.8)**(c),label = label) 

        # clean up panel
        ax.set_xlabel('iteration',fontsize = 12)
        ax.set_ylabel('cost function value',fontsize = 12)
        ax.set_title('cost function value at each step of gradient descent',fontsize = 15)
        if np.size(label) > 0:
            plt.legend(loc='upper right')
        ax.set_xlim([start - 1,len(history)+1])
        plt.show()
    
    
    def compare_classification_histories(self, g, x, y, **kwargs):
        '''
        A module for computing / plotting the cost and misclassification histories for a given run of gradient descent.
        '''
        
        num_pts = len(y)
        
        max_its = 100
        if 'max_its' in kwargs:
            max_its = kwargs['max_its']

        alpha = 1e-4
        if 'alpha' in kwargs:
            alpha = kwargs['alpha']     
            
        batch_size = 50
        if 'batch_size' in kwargs:
            batch_size = kwargs['batch_size']  
            
        version = 'unnormalized'
        if 'version' in kwargs:
            version = kwargs['version']       
        
        multiclass = False
        # initialize w based on number of classes
        if len(np.unique(y))>2:
            # multiclass
            multiclass = True 
            w = 0.1*np.random.randn(np.shape(x)[0]+1,len(np.unique(y)))
        else:
            # binary
            w = .1*np.random.randn(np.shape(x)[0]+1,1)
            
        
        # compute linear combination of input point
        def model(x,w):
            # tack a 1 onto the top of each input point all at once
            o = np.ones((1,np.shape(x)[1]))
            x = np.vstack((o,x))

            # compute linear combination and return
            a = np.dot(x.T,w)
            return a    
       
        
        def minibatch_gradient_descent(g, w, alpha, num_pts, batch_size, max_its, version):    
            # flatten the input function, create gradient based on flat function
            g_flat, unflatten, w = flatten_func(g, w)
            grad = compute_grad(g_flat)

            # record history
            w_hist = []
            w_hist.append(unflatten(w))

            # how many mini-batches equal the entire dataset?
            num_batches = int(np.ceil(np.divide(num_pts, batch_size)))
            # over the line
            for k in range(max_its):   
                # loop over each minibatch
                for b in range(num_batches):
                    # collect indices of current mini-batch
                    batch_inds = np.arange(b*batch_size, min((b+1)*batch_size, num_pts))

                    # plug in value into func and derivative
                    grad_eval = grad(w, batch_inds)
                    grad_eval.shape = np.shape(w)

                    ### normalized or unnormalized descent step? ###
                    if version == 'normalized':
                        grad_norm = np.linalg.norm(grad_eval)
                        if grad_norm == 0:
                            grad_norm += 10**-6*np.sign(2*np.random.rand(1) - 1)
                        grad_eval /= grad_norm

                    # take descent step with momentum
                    w = w - alpha*grad_eval

                # record weight update
                w_hist.append(unflatten(w))

            return w_hist
        
        
        
        # local copies of the softmax cost function written more compactly, for scoping issues
        softmax = lambda w: np.sum(np.log(1 + np.exp((-y)*(w[0] + np.dot(x.T,w[1:])))))
        binary_count = lambda w: 0.25*np.sum((np.sign(w[0] + np.dot(x.T,w[1:])) - y)**2)
        
        
        # multiclass counting cost
        def multiclass_count(w):                
            # pre-compute predictions on all points
            all_evals = model(x,w)

            # compute predictions of each input point
            y_predict = (np.argmax(all_evals,axis = 1))[:,np.newaxis]

            # compare predicted label to actual label
            count = np.sum(np.abs(np.sign(y - y_predict)))

            # return number of misclassifications
            return count

        def multiclass_perceptron(w):

            lam = 10**-3
            # pre-compute predictions on all points
            all_evals = model(x,w)

            # compute maximum across data points
            a =  np.max(all_evals, axis = 1)        

            # compute cost in compact form using numpy broadcasting
            b = all_evals[np.arange(len(y)), y.astype(int).flatten()]
            cost = np.sum(a - b)

            # add regularizer
            cost = cost + lam*np.linalg.norm(w[1:,:],'fro')**2

            # return average
            return cost/float(len(y))
        
        
        # specify cost and count functions
        cost_function = softmax
        count_function = binary_count
        if multiclass == True:
            cost_function = multiclass_perceptron
            count_function = multiclass_count

        
        # run batch gradient descent
        batch = num_pts
        batch_weights = minibatch_gradient_descent(g, w, alpha, num_pts, batch, max_its, version)

        # run mini-batch gradient descent
        batch = batch_size
        minibatch_weights = minibatch_gradient_descent(g, w, alpha, num_pts, batch, max_its, version)

        # collect the weights 
        weight_histories = [batch_weights, minibatch_weights]


        # initialize figure
        fig = plt.figure(figsize = (9,3))

        # create subplot with 3 panels, plot input function in center plot
        gs = gridspec.GridSpec(1, 2) 
        ax1 = plt.subplot(gs[0]); 
        ax2 = plt.subplot(gs[1]);

        # loop over histories and plot all
        c = 1
        for weight_history in weight_histories:
            # loop over input weight history and create associated cost and misclassification histories
            cost_history = []
            count_history = []
            for weight in weight_history:
                cost_val = cost_function(weight)
                cost_history.append(cost_val)

                count_val = count_function(weight)
                count_history.append(count_val)

            # now plot each, one per panel
            ax1.plot(cost_history)  
            label = 'batch'
            if c == 2:
                label = 'mini-batch'
            if c == 3:
                label = 'stochastic'
            ax2.plot(count_history,label = label)
            c+=1

        # label each panel
        ax1.set_xlabel('iteration')
        ax1.set_ylabel('cost function value')
        ax1.set_title('cost function history')

        ax2.set_xlabel('iteration')
        ax2.set_ylabel('misclassifications')
        ax2.set_title('number of misclassificaions')

        ax2.legend()

        plt.show()
        
       
    
    
    
  