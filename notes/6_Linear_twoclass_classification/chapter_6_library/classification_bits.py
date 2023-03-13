# import autograd functionality
from autograd import grad as compute_grad   # The only autograd function you may ever need
import autograd.numpy as np
from autograd import hessian as compute_hess
import math
import time
import copy
from autograd.misc.flatten import flatten_func
    
#### newton's method ####            
def newtons_method(g,w,x,y,beta,max_its):        
    # flatten gradient for simpler-written descent loop
    flat_g, unflatten, w = flatten_func(g, w)

    grad = compute_grad(flat_g)
    hess = compute_hess(flat_g)  

    # create container for weight history 
    w_hist = []
    w_hist.append(unflatten(w))
    
    g_hist = []
    geval_old = flat_g(w,x,y,beta)
    g_hist.append(geval_old)

    # main loop
    epsilon = 10**(-7)
    for k in range(max_its):
        # compute gradient and hessian
        grad_val = grad(w,x,y,beta)
        hess_val = hess(w,x,y,beta)
        hess_val.shape = (np.size(w),np.size(w))

        # solve linear system for weights
        w = w - np.dot(np.linalg.pinv(hess_val + epsilon*np.eye(np.size(w))),grad_val)

        # eject from process if reaching singular system
        geval_new = flat_g(w,x,y,beta)
        if k > 2 and geval_new > geval_old:
            print ('singular system reached')
            time.sleep(1.5)
            clear_output()
            return w_hist
        else:
            geval_old = geval_new

        # record current weights
        w_hist.append(unflatten(w))
        g_hist.append(geval_new)

    return w_hist,g_hist

# compute linear combination of input points
def model(x,w):
    a = w[0] + np.dot(x.T,w[1:])
    return a.T
   
# softmax cost
def softmax(w,x,y,beta):
    # compute cost over batch        
    cost = np.sum(beta*np.log(1 + np.exp(-y*model(x,w))))
    return cost/float(np.size(y))