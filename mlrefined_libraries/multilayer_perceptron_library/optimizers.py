# clear display
from IPython.display import clear_output

# import autograd functionality
from autograd import grad as compute_grad  
import autograd.numpy as np
import math
import time
import copy
from autograd.misc.flatten import flatten_func

class Setup:
    '''
    Optimizer(s) for multilayer perceptron function
    '''    
        
    ########## optimizer ##########
    # gradient descent function
    def gradient_descent(self,g,w,alpha,max_its,beta,version,**kwargs):
        verbose = False
        if 'verbose' in kwargs:
            verbose = kwargs['verbose']
        
        # flatten the input function, create gradient based on flat function
        g_flat, unflatten, w = flatten_func(g, w)
        grad = compute_grad(g_flat)

        # record history
        w_hist = []
        w_hist.append(unflatten(w))

        # start gradient descent loop
        z = np.zeros((np.shape(w)))      # momentum term

        if verbose == True:
            print ('starting optimization...')
            
        # over the line
        for k in range(max_its):   
            # plug in value into func and derivative
            grad_eval = grad(w)
            grad_eval.shape = np.shape(w)

            ### normalized or unnormalized descent step? ###
            if version == 'normalized':
                grad_norm = np.linalg.norm(grad_eval)
                if grad_norm == 0:
                    grad_norm += 10**-6*np.sign(2*np.random.rand(1) - 1)
                grad_eval /= grad_norm

            # take descent step with momentum
            z = beta*z + grad_eval
            w = w - alpha*z

            # record weight update
            w_hist.append(unflatten(w))

        if verbose == True:
            print ('...optimization complete!')
            time.sleep(1.5)
            clear_output()
            
        return w_hist