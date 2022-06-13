import autograd.numpy as np
from autograd import value_and_grad 
from autograd import hessian
from autograd.misc.flatten import flatten_func

# minibatch gradient descent
def RMSprop(g,alpha,max_its,w,num_pts,batch_size,**kwargs): 
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
    train_hist = [g_flat(w,np.arange(num_pts))]
    
    # how many mini-batches equal the entire dataset?
    num_batches = int(np.ceil(np.divide(num_pts, batch_size)))

    # over the line
    for k in range(max_its):                   
        # loop over each minibatch
        train_cost = 0
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
        
        # update training and validation cost
        train_cost = g_flat(w,np.arange(num_pts))

        # record weight update, train and val costs
        w_hist.append(unflatten(w))
        train_hist.append(train_cost)
        
    return w_hist,train_hist

# minibatch gradient descent
def gradient_descent(g, alpha, max_its, w, num_pts, batch_size,**kwargs):    
    # flatten the input function, create gradient based on flat function
    g_flat, unflatten, w = flatten_func(g, w)
    grad = value_and_grad(g_flat)

    # record history
    w_hist = []
    w_hist.append(unflatten(w))
    cost_hist = [g_flat(w,np.arange(num_pts))]
   
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
        cost_hist.append(g_flat(w,np.arange(num_pts)))
    return w_hist,cost_hist


# newtons method function - inputs: g (input function), max_its (maximum number of iterations), w (initialization)
def newtons_method(g,max_its,w,num_pts,batch_size,**kwargs):
    # flatten input funciton, in case it takes in matrices of weights
    g_flat, unflatten, w = flatten_func(g, w)
    
    # compute the gradient / hessian functions of our input function -
    gradient = value_and_grad(g_flat)
    hess = hessian(g_flat)
    
    # set numericxal stability parameter / regularization parameter
    epsilon = 10**(-7)
    if 'epsilon' in kwargs:
        epsilon = kwargs['epsilon']

    # record history
    w_hist = []
    w_hist.append(unflatten(w))
    cost_hist = [g_flat(w,np.arange(num_pts))]

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
            
            '''
            # compute minimum eigenvalue of hessian matrix 
            eigs, vecs = np.linalg.eig(hess_eval)
            smallest_eig = np.min(eigs)
            adjust = 0
            if smallest_eig < 0:
                adjust = np.abs(smallest_eig)
            '''

            # solve second order system system for weight update
            A = hess_eval + (epsilon)*np.eye(np.size(w))
            b = grad_eval
            w = np.linalg.lstsq(A,np.dot(A,w) - b)[0]

            #w = w - np.dot(np.linalg.pinv(hess_eval + epsilon*np.eye(np.size(w))),grad_eval)
            
        # record weights after each epoch
        w_hist.append(unflatten(w))
        cost_hist.append(g_flat(w,np.arange(num_pts)))

    return w_hist,cost_hist