import math
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from mpl_toolkits.mplot3d import Axes3D
from IPython import display
import time
from matplotlib import pyplot as plt

class Regression_Datamaker:
    def __init__(self):
        self.xs = []
        self.ys = []
        
        fig = plt.figure(num=None, figsize=(10, 5), dpi=80, facecolor='w', edgecolor='k')
        ax1 = fig.add_subplot(111,aspect='equal')
        ax1.set_xlim([0,1])
        ax1.set_ylim([0,1])  
        self.ax_to_plot = ax1
        self.cid = fig.canvas.mpl_connect('button_press_event', self)
    
    # function to capture point-and-click data
    def __call__(self, event):
        if not event.inaxes:
            return
        self.xs.append(event.xdata)
        self.ys.append(event.ydata)
        self.ax_to_plot.scatter(self.xs,self.ys)
        
    # function to save the data
    def save_data(self,csvname):
        # save the dataset you made above
        x = np.asarray(self.xs)
        x.shape = (len(x),1)

        y = np.asarray(self.ys)
        y.shape = (len(y),1)

        c = np.concatenate([x,y],axis = 1)
        d = pd.DataFrame(c)
        d.to_csv(csvname,index = False,header = None)