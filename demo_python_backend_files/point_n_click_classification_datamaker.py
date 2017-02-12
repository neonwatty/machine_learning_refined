import math
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from IPython import display

class Classification_Datamaker:
    def __init__(self):
        # initialize variables + containers for plotting
        self.label_num = 0
        self.colors = ['salmon','cornflowerblue','lime','bisque','mediumaquamarine','b','m','g']
        
        self.pts = []
        self.labels = []

        # initialize interactive plot
        fig = plt.figure(num=None, figsize=(7, 7), dpi=80, facecolor='w', edgecolor='k')
        self.ax1 = fig.add_subplot(111)
        self.ax_to_plot = self.ax1
        self.clean_plot()
        self.cid = fig.canvas.mpl_connect('button_press_event', self)

    def __call__(self, event):
        if not event.inaxes:
            self.label_num+=1
            return
        
        # plot click
        x = event.xdata
        y = event.ydata
        self.interactive_pt = self.ax1.scatter(x,y,color = self.colors[self.label_num],zorder = 2,linewidth = 1,marker = 'o',edgecolor = 'k',s = 50)
        self.clean_plot()
        
        # save datapoint
        pt = np.asarray([x,y])
        self.pts.append(pt)
        self.labels.append(self.label_num)
        
    # clean up plot
    def clean_plot(self):
        self.ax1.set_xticks([])
        self.ax1.set_yticks([])
        self.ax1.set_xlim([-2,2])
        self.ax1.set_ylim([-2,2])
        
    # save data
    def save_data(self,csvname):
        p = np.asarray(self.pts)
        l = np.asarray(self.labels)
        l.shape = (len(l),1)
        f = np.concatenate((p,l),axis=1)
        f = pd.DataFrame(f)
        f.to_csv(csvname,header = None, index = False)