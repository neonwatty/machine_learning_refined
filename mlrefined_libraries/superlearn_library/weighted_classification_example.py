# Import required libraries
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

class Visualizer:
    def load_data(self,csvname):
        # Read census data
        census_data = pd.read_csv(csvname, 
                      names = ["age","workclass","education_level","education-num","marital-status","occupation","relationship","race","sex","capital-gain","capital-loss","hours-per-week","native-country","income"])

        # Extract feature columns
        feature_cols = list(census_data.columns[:-1])
        
        # Extract target column 'income'
        target_col = census_data.columns[-1] 

        # Separate the data into feature data and target data (X_all and y_all, respectively)
        X_all = census_data[feature_cols]
        y_all = census_data[target_col]

        # update data with log of capital-gain and capital-loss values
        X_all = np.log(X_all['capital-gain'] + 1)
        
        # convert labels to numerical value
        y_all = np.asarray(y_all)
        y_all.shape = (len(y_all),1)
        X_all = np.asarray(X_all)
        ind1 = np.argwhere(y_all == "<=50K")
        ind1 = [s[0] for s in ind1]
        ind2 = np.argwhere(y_all == ">50K")
        ind2 = [s[0] for s in ind2]
        y_all[ind1] = -1
        y_all[ind2] = +1
        y_all = np.asarray([s[0] for s in y_all])

        # keep only the portion of data where capital gain > 0 
        ind = np.argwhere(X_all > 0)
        ind = [s[0] for s in ind]
        y = y_all[ind]
        x = X_all[ind]
        x = np.asarray(x, dtype=np.float)    

        return x, y

    # quantizes x using values in the bin_centers
    def quantize(self,x):
        # specify bin centers
        self.bin_centers = np.linspace(4.5, 11.5, 15)
        x_q = x
        for i in range(0,len(x)):
            dist = np.abs(self.bin_centers-x[i])
            x_q[i] = self.bin_centers[np.argmin(dist)]   
        return x_q

    def my_scatter(self,x, y, ax, c):
        # count number of occurances for each element
        s = np.asarray([sum(x==i) for i in x])
        # plot data using s as size vector
        ax.scatter(x, y, s, color=c) 
 
    def plot(self,csvname):
        x, y = self.load_data(csvname)

        # quantize x
        x_quantized = self.quantize(x)
        
        # seprate positive class and negative class for plotting 
        x_pos = x_quantized[y>0]
        x_neg = x_quantized[y<0]

        # plot data
        fig = plt.figure(figsize=(9,5))
        ax = fig.gca()

        # use my_scatter to plot each class
        self.my_scatter(x_pos, np.ones(len(x_pos)),ax, c='r')
        self.my_scatter(x_neg, -np.ones(len(x_neg)),ax, c='b')

        # clean up
        ax.set_xticks(self.bin_centers)
        ax.set_xlabel('log capital gain')
        ax.set_yticks([-1,1])
        ax.set_ylabel('class (make > $50k)')
        ax.set_ylim([-2.5,2.5])
        plt.grid(color='gray', linestyle='-', linewidth=1, alpha=.15)
        plt.show()