import numpy as np

class demo_logistic_regression:
    
    # inits
    def __init__(self):
        self.solver = 0
        self.max_its = 0
        self.kernel = 0
        self.w = 0
        self.cost_history = []
        self.X = 0
        self.X_orig = 0
        self.y = 0
        self.K = 0
        self.kwargs = []
    
    #### kernel functions ####
    # kernel function for converting training data
    def kernelize_train(self):
        # initialize kernel matrix
        num_pts = np.shape(self.X)[1]
        K = np.zeros((num_pts,num_pts))

        # shorthand for input data
        X = self.X
        y = self.y

        # define poly kernel on training data
        if self.kwargs['kernel'] == 'poly':
            if 'degree' not in self.kwargs:
                print 'to use the polynomial kernel you must define a degree'
                return

            deg = self.kwargs['degree']
            for i in range(num_pts):
                for j in range(i,num_pts):
                    temp = (1 + np.dot(X[:,i].T,X[:,j]))**deg - 1
                    K[i,j] = temp
                    if i != j:
                        K[j,i] = temp

        # define rbf kernel on training data
        if self.kwargs['kernel'] == 'rbf':
            if 'gamma' not in self.kwargs:
                print 'to use the rbf kernel you must give a value for gamma'
                return

            gamma = self.kwargs['gamma']
            for i in range(num_pts):
                for j in range(i,num_pts):
                    temp = np.exp(-gamma*np.linalg.norm(X[:,i] - X[:,j])**2)
                    K[i,j] = temp
                    if i != j:
                        K[j,i] = temp   
    
        return K
    
    # kernel function for converting training data
    def kernelize_test(self,X_test):
        # rotate test data if necessary to match dimension of original
        if np.shape(X_test)[0] != np.shape(self.X_orig)[0]:
            X_test = X_test.T
            
        k = np.zeros((np.shape(X_test)[1],np.shape(self.X_orig)[1]))

        # define poly kernel on new data
        if self.kwargs['kernel'] == 'poly':
            deg = self.kwargs['degree']
            k = (1 + np.dot(X_test.T,self.X_orig))**deg - 1

        # define rbf kernel on new data
        if self.kwargs['kernel'] == 'rbf':
            gamma = self.kwargs['gamma']

            # a fancy quick way to compute the distance between each row of X_test and self.X_orig
            d = (X_test.T**2).sum(axis=-1)[:, np.newaxis] + (self.X_orig.T**2).sum(axis=-1)
            d-= 2 * np.squeeze(X_test.T.dot(self.X_orig.T[..., np.newaxis]), axis=-1)
            
            # exponentiate
            k = np.exp(-gamma*d)

        return k
        
    #### logistic regression functions ####
    # function for computing logistic regression cost function value
    def compute_cost(self,X,y,w):
        cost = 0
        for p in range(0,len(y)):
            x_p = X[:,p]
            y_p = y[p]
            cost += np.log(1 + np.exp(-y_p*(np.dot(x_p.T,w))))
        return cost[0]
    
    # function for computing the softmax cost gradient
    def compute_gradient(self,X,y,w):
        # produce gradient for each class weights
        grad = 0
        for p in range(0,len(y)):
            x_p = X[:,p]
            y_p = y[p]
            grad+= -1/(1 + np.exp(y_p*np.dot(x_p.T,w)))*y_p*x_p

        grad.shape = (len(grad),1)
        return grad

    # gradient descent function for softmax cost/logistic regression 
    def grad_descent(self,w,max_its,alpha):
        # compute cost function value 
        cost_val = self.compute_cost(self.X,self.y,w)
        temp = [v for v in w]
        temp.append(cost_val)
        self.cost_history.append(temp) 
        
        # run grad descent
        for k in range(max_its):
            # compute gradient
            grad = self.compute_gradient(self.X,self.y,w)
            
            # compute step length
            if not alpha:
                alpha = self.line_search(self.X,self.y,w,grad,self.compute_cost)
            
            # take descent step
            w = w - alpha*grad
        
            # compute cost function value 
            cost_val = self.compute_cost(self.X,self.y,w)
            temp = [v for v in w]
            temp.append(cost_val)
            self.cost_history.append(temp) 
            
        self.w = w
            
    #### line search module - used for with both linear regression and logistic regression grad descent functions ####
    def line_search(self,X,y,w,grad,cost_fun):
        alpha = 1
        t = 0.5
        g_w = cost_fun(X,y,w)
        norm_w = np.linalg.norm(grad)**2
        
        # determine step length
        while cost_fun(X,y,w - alpha*grad) > g_w - alpha*0.5*norm_w:
            alpha = t*alpha
        
        # return line-fitting step length
        return alpha
    
    ### Newton's method functions ###
    # make your own exponential function that ignores cases where exp = inf
    def my_exp(self,val):
        newval = 0
        if val > 100:
            newval = np.inf
        if val < -100:
            newval = 0
        if val < 100 and val > -100:
            newval = np.exp(val)
        return newval

    # calculate grad and Hessian for newton's method
    def compute_grad_and_hess(self,w):
        hess = 0
        grad = 0
        for p in range(0,len(self.y)):
            # precompute
            x_p = self.X[:,p]
            y_p = self.y[p]
            s = 1/(1 + self.my_exp(y_p*np.dot(x_p.T,w)))
            g = s*(1-s)

            # update grad and hessian
            grad+= -s*y_p*x_p
            hess+= np.outer(x_p,x_p)*g

        # add small positive value to diagonal of Hessian to prevent singular systems
        hess += 1e-6*np.eye(np.shape(hess)[0])
        grad.shape = (len(grad),1)
        return grad,hess

    # note here you are loading *transformed features* as X
    def newtons_method(self,w,max_its):
        # compute cost function value 
        cost_val = self.compute_cost(self.X,self.y,w)
        temp = [v for v in w]
        temp.append(cost_val)
        self.cost_history.append(temp) 
        
        # outer descent loop
        for k in range(1,max_its+1):
            # compute gradient and Hessian
            grad,hess = self.compute_grad_and_hess(w)

            # take Newton method step
            temp = np.dot(hess,w) - grad
            w = np.dot(np.linalg.pinv(hess),temp)

            # compute cost function value 
            cost_val = self.compute_cost(self.X,self.y,w)
            temp = [v for v in w]
            temp.append(cost_val)
            self.cost_history.append(temp) 
            
        self.w = w

    
    ### main fitting and prediction functions ###
    # fit the chosen solver
    def fit(self,X,y,**kwargs):
        # store data internally
        self.X = X
        self.y = y
        self.kwargs = kwargs
        
        # transform training data via kernel if requested
        if "kernel" in self.kwargs:
            self.X = self.X.T
            self.X_orig = np.copy(self.X)
            self.X = self.kernelize_train()
        else:
            # pad data with ones for more compact gradient computation
            o = np.ones((np.shape(X)[0],1))
            X = np.concatenate((o,X),axis = 1)
            self.X = X.T 
            
        # default solver is gradient descent with line-search for step size
        self.solver = 'grad descent'
        self.max_its = 30
        self.init = np.zeros((np.shape(self.X)[0],1))
        
        # switch solver if defined in fitting, change
        if "solver" in self.kwargs:
            self.solver = self.kwargs["solver"]
            
        # if max_its defined in fitting, change
        if "max_its" in self.kwargs:
            self.max_its = self.kwargs["max_its"]
            
        # set initialization if defined
        if "init" in self.kwargs:
            self.inits = self.kwargs["init"]
            
        # if step length is defined
        alpha = []
        if "alpha" in self.kwargs:
            alpha = kwargs["alpha"]
        
        # run fitting
        cost_history = 0
        w = 0
        if self.solver == 'grad descent':
            self.grad_descent(self.init,self.max_its,alpha)
        if self.solver == 'newtons method':
            self.newtons_method(self.init,self.max_its)

        return self.w ,self.cost_history
    
    # predict - input arg can be kernels
    def predict(self,X_test,**kwargs):        
        # if kernel used transform test data
        if "kernel" in self.kwargs:
            X_test = self.kernelize_test(X_test)
        else:
            o = np.ones((np.shape(X_test)[0],1))
            X_test = np.concatenate((o,X_test),axis = 1)
        
        # set weight to test
        w_test = self.w
        if 'test_weight' in kwargs:
            w_test = kwargs['test_weight']

        f = np.tanh(np.dot(X_test,w_test))
        
        return f