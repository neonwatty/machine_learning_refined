import autograd.numpy as np
from inspect import signature

class Setup:
    def __init__(self,cost_name,reg_name):             
        ### make cost function choice ###
        # for regression
        if cost_name == 'least_squares':
            self.cost = self.least_squares
        if cost_name == 'least_absolute_deviations':
            self.cost = self.least_absolute_deviations
            
        # for two-class classification
        if cost_name == 'softmax':
            self.cost = self.softmax
        if cost_name == 'perceptron':
            self.cost = self.perceptron
        if cost_name == 'twoclass_counter':
            self.cost = self.counting_cost
            
        # for multiclass classification
        if cost_name == 'multiclass_perceptron':
            self.cost = self.multiclass_perceptron
        if cost_name == 'multiclass_softmax':
            self.cost = self.multiclass_softmax
        if cost_name == 'multiclass_counter':
            self.cost = self.multiclass_counting_cost
            
        # choose regularizer
        self.lam = 0
        if reg_name == 'L2':
            self.reg = self.L2
        if reg_name == 'L1':
            self.reg = self.L1
            
    ### regularizers ###
    def L1(self,w):
         return self.lam*np.sum(np.abs(w[1:]))
  
    def L2(self,w):
         return self.lam*np.sum((w[1:])**2)
        
    # set lambda value (regularization penalty)
    def set_lambda(self,lam):
        self.lam = lam
        
    ### setup model ###
    def model(self,x,w):
        a = w[0] + np.dot(x.T,w[1:])
        return a.T
    
    ###### regression costs #######
    # an implementation of the least squares cost function for linear regression
    def least_squares(self,w,x,y,iter):
        # get batch of points
        x_p = x[:,iter]
        y_p = y[:,iter]
        
        # compute cost
        cost = np.sum((self.model(x_p,w) - y_p)**2)
        
        # add regularizer 
        cost += self.reg(w)
        
        # return average
        return cost/float(np.size(y_p))

    # a compact least absolute deviations cost function
    def least_absolute_deviations(self,w,x,y,iter):
        # get batch of points
        x_p = x[:,iter]
        y_p = y[:,iter]
        
        # compute cost
        cost = np.sum(np.abs(self.model(x_p,w) - y_p))
        
        # add regularizer 
        cost += self.reg(w)
        
        # return average
        return cost/float(np.size(y_p))

    ###### two-class classification costs #######
    # the convex softmax cost function
    def softmax(self,w,x,y,iter):
        # get batch of points
        x_p = x[:,iter]
        y_p = y[:,iter]
        
        # compute cost over batch
        cost = np.sum(np.log(1 + np.exp(-y_p*(self.model(x_p,w)))))
        
        # add regularizer 
        cost += self.reg(w)
        
        # return average
        return cost/float(np.size(y_p))

    # the convex relu cost function
    def relu(self,w,x,y,iter):
        # get batch of points
        x_p = x[:,iter]
        y_p = y[:,iter]
        
        # compute cost over batch
        cost = np.sum(np.maximum(0,-y_p*self.model(x_p,w)))
        
        # add regularizer 
        cost += self.reg(w)
        
        # return average
        return cost/float(np.size(y_p))
    
    # the counting cost function
    def counting_cost(self,w,x,y,iter):
        cost = np.sum(np.abs(np.sign(self.model(x,w)) - self.y))
        return cost 

    ###### multiclass classification costs #######
    # multiclass perceptron
    def multiclass_perceptron(self,w,x,y,iter):
        # get subset of points
        x_p = x[:,iter]
        y_p = y[:,iter]

        # pre-compute predictions on all points
        all_evals = self.model(x_p,w)

        # compute maximum across data points
        a =  np.max(all_evals,axis = 0)        

        # compute cost in compact form using numpy broadcasting
        b = all_evals[y_p.astype(int).flatten(),np.arange(np.size(y_p))]
        cost = np.sum(a - b)

        # add regularizer 
        cost += self.reg(w)
        
        # return average
        return cost/float(np.size(y_p))

    # multiclass softmax
    def multiclass_softmax(self,w,x,y,iter):
        # get subset of points
        x_p = x[:,iter]
        y_p = y[:,iter]
        
        # pre-compute predictions on all points
        all_evals = self.model(x_p,w)

        # compute softmax across data points
        a = np.log(np.sum(np.exp(all_evals),axis = 0)) 

        # compute cost in compact form using numpy broadcasting
        b = all_evals[y_p.astype(int).flatten(),np.arange(np.size(y_p))]
        cost = np.sum(a - b)

        # add regularizer 
        cost += self.reg(w)
        
        # return average
        return cost/float(np.size(y_p))

    # multiclass misclassification cost function - aka the fusion rule
    def multiclass_counting_cost(self,w,x,y,iter):            
        # pre-compute predictions on all points
        all_evals = self.model(x,w)

        # compute predictions of each input point
        y_predict = (np.argmax(all_evals,axis = 0))[np.newaxis,:]

        # compare predicted label to actual label
        count = np.sum(np.abs(np.sign(y - y_predict)))

        # return number of misclassifications
        return count