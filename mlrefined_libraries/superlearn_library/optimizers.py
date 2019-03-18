# clear display
from IPython.display import clear_output

# import autograd functionality
from autograd import grad as compute_grad   # The only autograd function you may ever need
import autograd.numpy as np
from autograd import hessian as compute_hess
import math
import time
import copy
from autograd.misc.flatten import flatten_func

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