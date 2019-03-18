import autograd.numpy as np
from autograd import value_and_grad 
from autograd import hessian
from autograd.misc.flatten import flatten_func

# minibatch gradient descent
def gradient_descent(g, alpha, max_its, w, num_pts, batch_size,**kwargs):   
    # pluck out args
    beta = 0
    if 'beta' in kwargs:
        beta = kwargs['beta']
    normalize = False
    if 'normalize' in kwargs:
        normalize = kwargs['normalize']
    
    # flatten the input function, create gradient based on flat function
    g_flat, unflatten, w = flatten_func(g, w)
    grad = value_and_grad(g_flat)

    # record history
    w_hist = []
    w_hist.append(unflatten(w))
   
    # how many mini-batches equal the entire dataset?
    num_batches = int(np.ceil(np.divide(num_pts, batch_size)))
    
    # initialization for momentum direction
    h = np.zeros((w.shape))
    
    # over the line
    for k in range(max_its):   
        # loop over each minibatch
        for b in range(num_batches):
            # collect indices of current mini-batch
            batch_inds = np.arange(b*batch_size, min((b+1)*batch_size, num_pts))

            # plug in value into func and derivative
            cost_eval,grad_eval = grad(w,batch_inds)
            grad_eval.shape = np.shape(w)
            
            # normalize?
            if normalize == True:
                grad_eval = np.sign(grad_eval)
                
            # momentum step 
            # h = beta*h - (1 - beta)*grad_eval    
            
            # take descent step with momentum
            w = w - alpha*grad_eval

        # record weight update
        w_hist.append(unflatten(w))

    return w_hist


# RMSprop advanced first order optimizer
def RMSprop(g, alpha, max_its, w, num_pts, batch_size,**kwargs):    
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
    w_hist = [unflatten(w)]
    
    # how many mini-batches equal the entire dataset?
    num_batches = int(np.ceil(np.divide(num_pts, batch_size)))

    # over the line
    for k in range(max_its):                   
        # loop over each minibatch
        for b in range(num_batches):
            # collect indices of current mini-batch
            batch_inds = np.arange(b*batch_size, min((b+1)*batch_size, num_pts))
            
            # plug in value into func and derivative
            cost_eval,grad_eval = grad(w,batch_inds)
            grad_eval.shape = np.shape(w)
            
            # update exponential average of past gradients
            avg_sq_grad = gamma*avg_sq_grad + (1 - gamma)*grad_eval**2 
    
            # take descent step 
            w = w - alpha*grad_eval / (avg_sq_grad**(0.5) + eps)

        # record weight update, train and val costs
        w_hist.append(unflatten(w))
        
    return w_hist


# newtons method function - inputs: g (input function), max_its (maximum number of iterations), w (initialization)
def newtons_method(g, epsilon, max_its, w, num_pts, batch_size,**kwargs):      
    
    # flatten the input function, create gradient based on flat function
    g_flat, unflatten, w = flatten_func(g, w)
    grad = value_and_grad(g_flat)
    hess = hessian(g_flat)

    # record history
    w_hist = []
    w_hist.append(unflatten(w))
   
    # how many mini-batches equal the entire dataset?
    num_batches = int(np.ceil(np.divide(num_pts, batch_size)))
    
    # over the line
    for k in range(max_its):   
        for b in range(num_batches):
            # collect indices of current mini-batch
            batch_inds = np.arange(b*batch_size, min((b+1)*batch_size, num_pts))
            
            # plug in value into func and derivative
            cost_eval,grad_eval = grad(w,batch_inds)
            grad_eval.shape = np.shape(w)

            # evaluate the hessian
            hess_eval = hess(w,batch_inds)

            # reshape for numpy linalg functionality
            hess_eval.shape = (int((np.size(hess_eval))**(0.5)),int((np.size(hess_eval))**(0.5)))
            hess_eval += epsilon*np.eye(np.size(w))

            # solve second order system system for weight update
            A = hess_eval 
            b = grad_eval
            w = np.linalg.lstsq(A,np.dot(A,w) - b)[0]            
        
        # record weight update, train and val costs
        w_hist.append(unflatten(w))
        
        if np.linalg.norm(w) > 100:
            return w_hist

    return w_hist
