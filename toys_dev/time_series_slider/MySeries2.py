import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import math
from ipywidgets import interact
from ipywidgets import widgets

class MySeries():
    '''
    Time series prediction.  User need only input time series csv
    (note: this should consist of a single row/column without header)
    '''
    def __init__(self):
        # variables for training / predictions
        self.w = 0
        self.y = 0
        self.y_predictions = []
        
    # load data
    def load_data(self,csvname):
        self.y = np.asarray(pd.read_csv(csvname,header = None))
    
    # trains linear time series model - outputing weights
    def train_model(self):
        # determine training period - preset
        P = min(24,int(math.floor(len(self.y)/float(4))))

        # initialize A and b
        A = 0
        b = 0

        # loop over vectors to build A and b
        for n in range(0,len(self.y)-P):
            x_n = np.fliplr(self.y[n:n+P])
            y_n = self.y[n+P]
            A += np.outer(x_n,x_n)
            b += x_n*y_n

        # solve linear system Ax = b for properly tuned weights w using pinv (for stability)
        self.w = np.dot(np.linalg.pinv(A),b)
    
    # make predictions - after training - returns predictions
    def make_predictions(self,num_periods):
        P = len(self.w)    # training window length
        self.y_predictions = []
        
        # loop over most recent part of series to make prediction
        y_input = list(self.y[-P:])
        for p in range(num_periods):
            # compute and store prediction
            pred = list(sum([s*t for s,t in zip(y_input,self.w)]))
            self.y_predictions.append(pred[0])

            # kick out last entry in y_input and insert most recent prediction at front
            del y_input[-1]
            y_input = pred + y_input

        
    # plot input series as well as prediction
    def plot_all(self):

        # plot series
        plt.plot(np.arange(len(self.y)),self.y,color = 'b',linewidth = 3)

        # plot fit 
        plt.plot(len(self.y) + np.arange(len(self.y_predictions)),self.y_predictions,color = 'r',linewidth = 3)
        plt.plot([len(self.y)-1,len(self.y)],[self.y[-1],self.y_predictions[0]],color = 'r',linewidth = 3)
        plt.xlabel('time period',fontsize = 13)
        plt.ylabel('value',fontsize = 13)
        plt.xticks([])
        plt.yticks([])
        plt.title('simple time series prediction (in red)',fontsize = 15)
	plt.savefig('foo.png')


    # a general purpose function for running and plotting the result of a user-defined input classification algorithm
    def browse_vals(self):
        # params for slider
        slider_min = 1
        slider_max = 200
        slider_start = 1

        def show_fit(num_periods):
            # set parameter value of classifier
            self.make_predictions(num_periods)               # make predictions using the trained model
            self.plot_all() 

        interact(show_fit,num_periods=widgets.IntSlider(min=slider_min,max=slider_max,step=0,value=slider_start))