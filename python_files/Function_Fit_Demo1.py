import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn import ensemble
from sklearn.preprocessing import PolynomialFeatures
from sklearn import linear_model
from sklearn.neural_network import MLPRegressor
from sklearn.ensemble import GradientBoostingRegressor
from ipywidgets import interact
from ipywidgets import widgets
import math

class Fit_Bases:
    
    def __init__(self):
        self.x = 0
        self.y = 0

    # load target function
    def load_target(self,csvname):
        data = np.asarray(pd.read_csv(csvname,header = None))
        self.x = data[:,0][:, np.newaxis]
        self.y = data[:,1]
        
    # plot target function
    def plot_target(self):
        plt.plot(self.x,self.y,color = 'r',linewidth = 2.5)
        
        # dress panel correctly with axis labels etc.
        plt.xlim(min(self.x),max(self.x))
        plt.ylim(min(self.y)-0.1,max(self.y)+0.1)
        plt.yticks([],[])
        plt.axis('off')

    # plot approximation
    def plot_approx(self,clf,domain,transform):
        # use regressor to make predictions
        z = clf.predict(transform)

        # plot regressor
        plt.plot(domain,z,linewidth = 3,color = 'b')

    ### demo with animation or sliders - showing function approximation with polynoimal, neural network, and stumps/trees - should zip up into one function but laziness overtaketh me
    # polys
    def browse_poly_fit(self):
        def show_fit(num_elements):
            # plot our points
            self.plot_target()
            
            # Create linear regression object
            poly = PolynomialFeatures(degree=num_elements)
            f = poly.fit_transform(self.x)
    
            clf = linear_model.LinearRegression()
            clf.fit(f, self.y)        

            # plot classification boundary and color regions appropriately
            r = np.linspace(-0.1,1.1,300)[:, np.newaxis]
            pr = poly.fit_transform(r)

            # plot approximation
            self.plot_approx(clf,r,pr)

        interact(show_fit, num_elements=widgets.IntSlider(min=1,max=20,step=1,value=1))


    # demo with animation or sliders - showing function approximation with polynoimal, neural network, and stumps/trees
    def browse_tree_fit(self):
        def show_fit(num_elements):
            # plot our points
            self.plot_target()

            # load in gradient booster
            clf = GradientBoostingRegressor(n_estimators=num_elements, learning_rate=1,max_depth=2, random_state=0, loss='ls')
         
            # fit classifier
            self.y.shape = (len(self.y),)
            clf.fit(self.x,self.y)

            # plot classification boundary and color regions appropriately
            r = np.linspace(-0.1,1.1,300)[:, np.newaxis]

            # plot approximation
            self.plot_approx(clf,r,r)
            
        interact(show_fit, num_elements=widgets.IntSlider(min=1,max=50,step=1,value=1))
        
      
    # demo with animation or sliders - showing function approximation with polynoimal, neural network, and stumps/trees
    def browse_net_fit(self):
        def show_fit(num_elements):
            # plot our points
            self.plot_target()
            
            # create the decision tree classifier with appropriate 
            clf = MLPRegressor(solver = 'lbgfs',alpha = 0,activation = 'tanh',random_state = 1,hidden_layer_sizes = (num_elements,num_elements))

            # fit classifier
            self.y.shape = (len(self.y),)
            clf.fit(self.x,self.y)

            # plot classification boundary and color regions appropriately
            r = np.linspace(-0.1,1.1,300)[:, np.newaxis]

            # plot approximation
            self.plot_approx(clf,r,r)
            
        interact(show_fit, num_elements=widgets.IntSlider(min=1,max=5,step=1,value=1))


