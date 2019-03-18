import autograd.numpy as np
from inspect import signature

class Setup:
    def __init__(self,name,x,y,feature_transforms,**kwargs):
        # point to input/output for cost functions
        self.x = x
        self.y = y
       
        # make copy of feature transformation
        self.feature_transforms = feature_transforms
        
        # count parameter layers of input to feature transform
        self.sig = signature(self.feature_transforms)
        
        ### make cost function choice ###
        # for regression
        if name == 'least_squares':
            self.cost = self.least_squares
        if name == 'least_absolute_deviations':
            self.cost = self.least_absolute_deviations
            
        # for two-class classification
        if name == 'softmax':
            self.cost = self.softmax
        if name == 'perceptron':
            self.cost = self.perceptron
        if name == 'twoclass_counter':
            self.cost = self.counting_cost
            
        # for multiclass classification
        if name == 'multiclass_perceptron':
            self.cost = self.multiclass_perceptron
        if name == 'multiclass_softmax':
            self.cost = self.multiclass_softmax
        if name == 'multiclass_counter':
            self.cost = self.multiclass_counting_cost
            
        # for autoencoder
        if name == 'autoencoder':
            self.feature_transforms = feature_transforms
            self.feature_transforms_2 = kwargs['feature_transforms_2']
            self.cost = self.autoencoder

            
    ###### cost functions #####
    # compute linear combination of input point
    def model(self,x,w):   
        # feature transformation - switch for dealing
        # with feature transforms that either do or do
        # not have internal parameters
        f = 0
        if len(self.sig.parameters) == 2:
            f = self.feature_transforms(x,w[0])
        else: 
            f = self.feature_transforms(x)    

        # tack a 1 onto the top of each input point all at once
        o = np.ones((1,np.shape(f)[1]))
        f = np.vstack((o,f))

        # compute linear combination and return
        # switch for dealing with feature transforms that either 
        # do or do not have internal parameters
        a = 0
        if len(self.sig.parameters) == 2:
            a = np.dot(f.T,w[1])
        else:
            a = np.dot(f.T,w)
        return a.T
    
    ###### regression costs #######
    # an implementation of the least squares cost function for linear regression
    def least_squares(self,w,iter):
        # get batch of points
        x_p = self.x[:,iter]
        y_p = self.y[:,iter]
        
        # compute cost over batch
        cost = np.sum((self.model(x_p,w) - y_p)**2)
        return cost/float(np.size(y_p))

    # a compact least absolute deviations cost function
    def least_absolute_deviations(self,w,iter):
        # get batch of points
        x_p = self.x[:,iter]
        y_p = self.y[:,iter]
        
        # compute cost over batch
        cost = np.sum(np.abs(self.model(x_p,w) - y_p))
        return cost/float(np.size(y_p))

    ###### two-class classification costs #######
    # the convex softmax cost function
    def softmax(self,w,iter):
        # get batch of points
        x_p = self.x[:,iter]
        y_p = self.y[:,iter]
        
        # compute cost over batch
        cost = np.sum(np.log(1 + np.exp(-y_p*self.model(x_p,w))))
        return cost/float(np.size(y_p))

    # the convex relu cost function
    def relu(self,w,iter):
        # get batch of points
        x_p = self.x[:,iter]
        y_p = self.y[:,iter]
        
        # compute cost over batch
        cost = np.sum(np.maximum(0,-y_p*self.model(x_p,w)))
        return cost/float(np.size(y_p))

    # the counting cost function
    def counting_cost(self,w):
        cost = np.sum((np.sign(self.model(self.x,w)) - self.y)**2)
        return 0.25*cost 

    ###### multiclass classification costs #######
    # multiclass perceptron
    def multiclass_perceptron(self,w,iter):
        # get subset of points
        x_p = self.x[:,iter]
        y_p = self.y[:,iter]

        # pre-compute predictions on all points
        all_evals = self.model(x_p,w)

        # compute maximum across data points
        a =  np.max(all_evals,axis = 0)        

        # compute cost in compact form using numpy broadcasting
        b = all_evals[y_p.astype(int).flatten(),np.arange(np.size(y_p))]
        cost = np.sum(a - b)

        # return average
        return cost/float(np.size(y_p))

    # multiclass softmax
    def multiclass_softmax(self,w,iter):     
        # get subset of points
        x_p = self.x[:,iter]
        y_p = self.y[:,iter]
        
        # pre-compute predictions on all points
        all_evals = self.model(x_p,w)

        # compute softmax across data points
        a = np.log(np.sum(np.exp(all_evals),axis = 0)) 

        # compute cost in compact form using numpy broadcasting
        b = all_evals[y_p.astype(int).flatten(),np.arange(np.size(y_p))]
        cost = np.sum(a - b)

        # return average
        return cost/float(np.size(y_p))

    # multiclass misclassification cost function - aka the fusion rule
    def multiclass_counting_cost(self,w):                
        # pre-compute predictions on all points
        all_evals = self.model(self.x,w)

        # compute predictions of each input point
        y_predict = (np.argmax(all_evals,axis = 0))[np.newaxis,:]

        # compare predicted label to actual label
        count = np.sum(np.abs(np.sign(self.y - y_predict)))

        # return number of misclassifications
        return count
    
    ### for autoencoder ###
    def encoder(self,x,w):    
        # feature transformation 
        f = self.feature_transforms(x,w[0])

        # tack a 1 onto the top of each input point all at once
        o = np.ones((1,np.shape(f)[1]))
        f = np.vstack((o,f))

        # compute linear combination and return
        a = np.dot(f.T,w[1])
        return a.T

    def decoder(self,v,w):
        # feature transformation 
        f = self.feature_transforms_2(v,w[0])

        # tack a 1 onto the top of each input point all at once
        o = np.ones((1,np.shape(f)[1]))
        f = np.vstack((o,f))

        # compute linear combination and return
        a = np.dot(f.T,w[1])
        return a.T
    
    def autoencoder(self,w):
        # encode input
        a = self.encoder(self.x,w[0])
        
        # decode result
        b = self.decoder(a,w[1])
        
        # compute Least Squares error
        cost = np.sum((b - self.x)**2)
        return cost/float(self.x.shape[1])