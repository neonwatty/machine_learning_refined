import autograd.numpy as np
from autograd import value_and_grad 
from autograd import hessian
from autograd.misc.flatten import flatten_func

#### optimizers ####
# minibatch gradient descent
def gradient_descent(g,w,x,y,alpha_choice,max_its,batch_size): 
    # flatten the input function, create gradient based on flat function
    g_flat, unflatten, w = flatten_func(g, w)
    grad = value_and_grad(g_flat)

    # record history
    num_train = y.size
    w_hist = [unflatten(w)]
    train_hist = [g_flat(w,x,y,np.arange(num_train))]

    # how many mini-batches equal the entire dataset?
    num_batches = int(np.ceil(np.divide(num_train, batch_size)))

    # over the line
    alpha = 0
    print ('grads')

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
            cost_eval,grad_eval = grad(w,x,y,batch_inds)
            grad_eval.shape = np.shape(w)

            # take descent step with momentum
            w = w - alpha*grad_eval

        # update training and validation cost
        train_cost = g_flat(w,x,y,np.arange(num_train))

        # record weight update, train and val costs
        w_hist.append(unflatten(w))
        train_hist.append(train_cost)
    return w_hist,train_hist

# newtons method function
def newtons_method(g,w,x,y,max_its,**kwargs): 
    # flatten input funciton, in case it takes in matrices of weights
    g_flat, unflatten, w = flatten_func(g, w)
    
    # compute the gradient / hessian functions of our input function
    grad = value_and_grad(g_flat)
    hess = hessian(g_flat)
    
    # set numericxal stability parameter / regularization parameter
    epsilon = 10**(-7)
    if 'epsilon' in kwargs:
        epsilon = kwargs['epsilon']
    
    # record history
    num_train = y.size
    w_hist = [unflatten(w)]
    train_hist = [g_flat(w,x,y,np.arange(num_train))]

    # over the line
    for k in range(max_its):   
        # evaluate the gradient, store current weights and cost function value
        cost_eval,grad_eval = grad(w,x,y,np.arange(num_train))

        # evaluate the hessian
        hess_eval = hess(w,x,y,np.arange(num_train))

        # reshape for numpy linalg functionality
        hess_eval.shape = (int((np.size(hess_eval))**(0.5)),int((np.size(hess_eval))**(0.5)))

        # solve second order system system for weight update
        A = hess_eval + epsilon*np.eye(np.size(w))
        b = grad_eval
        w = np.linalg.lstsq(A,np.dot(A,w) - b)[0]

        #w = w - np.dot(np.linalg.pinv(hess_eval + epsilon*np.eye(np.size(w))),grad_eval)
            
        # update training and validation cost
        train_cost = g_flat(w,x,y,np.arange(num_train))

        # record weight update, train and val costs
        w_hist.append(unflatten(w))
        train_hist.append(train_cost)
    
    return w_hist,train_hist