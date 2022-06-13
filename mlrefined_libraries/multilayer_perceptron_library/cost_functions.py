# import autograd functionality
from autograd import grad as compute_grad   # The only autograd function you may ever need
import autograd.numpy as np
import copy

class Setup:
    '''
    Cost functinos
    '''
    def choose_cost(self,cost_name,predict,x,y,**kwargs):
        self.predict = predict
        self.x = x
        self.y = y
        if cost_name == 'least_squares':
            self.cost = self.least_squares
        if cost_name == 'twoclass_softmax':
            self.cost = self.twoclass_softmax
        if cost_name == 'multiclass_softmax':
            self.cost = self.multiclass_softmax
         
        if cost_name == 'twoclass_counter':
            self.cost = self.twoclass_counter
        if cost_name == 'multiclass_counter':
            self.cost = self.multiclass_counter

    ########## cost functions ##########
    # least squares cost
    def least_squares(self,w):
        cost = np.sum((self.predict(self.x,w) - self.y)**2)
        return cost
    
    # two-class softmax / logistic regression cost
    def twoclass_softmax(self,w):
        cost = np.sum(np.log(1 + np.exp((-self.y)*(self.predict(self.x,w)))))
        return cost

    # multiclass softmaax regularized by the summed length of all normal vectors
    def multiclass_softmax(self,W):        
        # pre-compute predictions on all points
        all_evals = self.predict(self.x,W)

        # compute cost in compact form using numpy broadcasting
        a = np.log(np.sum(np.exp(all_evals),axis = 1)) 
        b = all_evals[np.arange(len(self.y)),self.y]
        cost = np.sum(a - b)
        return cost
    
    ### misclassification counters ###
    # two-class
    def twoclass_counter(self,w):
        misclassifications = 0.25*np.sum((np.sign(self.predict(self.x,w)) - self.y)**2)
        return misclassifications

    # multiclass
    def multiclass_counter(self,W):
        '''
        fusion rule for counting number of misclassifications on an input multiclass dataset
        '''

        # create predicted labels
        y_predict = np.argmax(self.predict(self.x,W),axis = 1) 

        # compare to actual labels
        misclassifications = int(np.sum([abs(np.sign(a - b)) for a,b in zip(self.y,y_predict)]))
        return misclassifications