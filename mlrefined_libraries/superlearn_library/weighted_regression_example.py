# Import required libraries
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

'''
custom scatter plot if there's repreated data, the dots 
representing more frequent points will be larger.
'''
class Visualizer:    
    def my_scatter(self,x, y, c):
        # count number of occurances for each element
        s = np.asarray([sum(x==i) for i in x])
        # plot data using s as size vector
        plt.scatter(x, y, s**2, color=c)    
    
    def plot_it(self,csvname):
        # read data
        data = pd.read_csv(csvname, index_col=0)
        n_row, n_col = np.shape(data)
        
        # plot data
        fig = plt.figure(figsize=(12,5))
        ax = fig.gca()
        colors=['r', 'b', 'g', 'y', 'm']

        # use my_scatter to plot each column of the dataframe
        for i in range(0,n_col):
            self.my_scatter(data[data.columns[i]], float(data.columns[i])*np.ones(n_row), c=colors[i])

        # clean up
        ax.set_xticks(np.arange(3.4, 7.5, .2))
        ax.set_xlabel('time')
        ax.set_yticks([.25,.50,.67,.75,1.0])
        ax.set_ylabel('portion of ramp traveled')
        ax.set_ylim([.15,1.1])
        plt.grid(color='gray', linestyle='-', linewidth=1, alpha=.15)
        plt.show()