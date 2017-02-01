# test gradient descent
import numpy as np

# fitting function itself
def y_function(t,w):
    yt = w[0] + w[1]*t + w[2]*np.sin(w[3] + t*w[4])
    return yt

# compute cost function at given iteration
def compute_cost(x,y,w):
    P = len(y)
    cost = 0
    for p in range(0,P):
        xp = x[p]
        c0 = (w[0] + w[1]*xp + w[2]*np.sin(w[3] + xp*w[4]) - y[p])**2
        cost+= c0
    return cost/float(P)

# compute gradient evaluated at single point
def compute_gradient(xp,yp,w):
    # container for partial gradient
    grad = np.zeros((len(w),1))

    # first layer constants
    c0 = (w[0] + w[1]*xp + w[2]*np.sin(w[3] + xp*w[4]) - yp)
    
    # second layer constants
    c1 = np.sin(w[3] + xp*w[4])
    
    # third layer constants 
    c2 = w[2]*np.cos(w[3] + xp*w[4])
    
    # fill up gradient
    grad[0] = c0
    grad[1] = c0*xp
    grad[2] = c0*c1
    grad[3] = c0*c2
    grad[4] = c0*c2*xp
    return grad
    
# perform gradient descent
def grad_descent(x,y,maxits):
    P = len(y)
    
    # make container for weights and cost evaluations
    w = 1/float(P)*np.random.randn(5,1)
    current_cost = compute_cost(x,y,w)
    costs = np.zeros((maxits-1,1))
    
    # make placeholder for best w
    best_cost_ever = np.inf
    best_w = w
    
    for k in range(1,maxits):
        alpha = 1/float(k+1)
        
        for p in range(0,P):
            xp = x[p]
            yp = y[p]
            current_grad = compute_gradient(xp,yp,w)
            w = w - alpha*current_grad       
        current_cost = compute_cost(x,y,w)
        costs[k-1] = current_cost
        
        # make best w selection
        if current_cost < best_cost_ever:
            current_cost = best_cost_ever
            best_w = w
       
    # return the best w and the full cost function evals
    return best_w,costs

            
        
            
    