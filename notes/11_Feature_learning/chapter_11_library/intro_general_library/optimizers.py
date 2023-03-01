import autograd.numpy as np
from autograd import value_and_grad 
from autograd import hessian
from autograd.misc.flatten import flatten_func

# minibatch gradient descent
def gradient_descent(g, alpha, max_its, w, num_pts, batch_size,**kwargs):    
    # flatten the input function, create gradient based on flat function    
    g_flat, unflatten, w = flatten_func(g, w)
    grad = value_and_grad(g_flat)
    
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
            cost_eval,grad_eval = grad(w,batch_inds)
            grad_eval.shape = np.shape(w)
            
            # take descent step with momentum
            w = w - alpha*grad_eval

        # record weight update
        w_hist.append(unflatten(w))

    return w_hist

# newtons method function - inputs: g (input function), max_its (maximum number of iterations), w (initialization)
def newtons_method(g,max_its,w,num_pts,batch_size,**kwargs):
    # flatten input funciton, in case it takes in matrices of weights
    flat_g, unflatten, w = flatten_func(g, w)
    
    # compute the gradient / hessian functions of our input function -
    # note these are themselves functions.  In particular the gradient - 
    # - when evaluated - returns both the gradient and function evaluations (remember
    # as discussed in Chapter 3 we always ge the function evaluation 'for free' when we use
    # an Automatic Differntiator to evaluate the gradient)
    gradient = value_and_grad(flat_g)
    hess = hessian(flat_g)
    
    # set numericxal stability parameter / regularization parameter
    epsilon = 10**(-7)
    if 'epsilon' in kwargs:
        epsilon = kwargs['epsilon']

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
            
            # evaluate the gradient, store current weights and cost function value
            cost_eval,grad_eval = gradient(w,batch_inds)

            # evaluate the hessian
            hess_eval = hess(w,batch_inds)

            # reshape for numpy linalg functionality
            hess_eval.shape = (int((np.size(hess_eval))**(0.5)),int((np.size(hess_eval))**(0.5)))

            # solve second order system system for weight update
            A = hess_eval + epsilon*np.eye(np.size(w))
            b = grad_eval
            w = np.linalg.lstsq(A,np.dot(A,w) - b)[0]

            #w = w - np.dot(np.linalg.pinv(hess_eval + epsilon*np.eye(np.size(w))),grad_eval)
            
        # record weights after each epoch
        w_hist.append(unflatten(w))

    # collect final weights
    w_hist.append(unflatten(w))
    
    return w_hist