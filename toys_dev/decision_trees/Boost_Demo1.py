import time
import matplotlib.pyplot as plt
from IPython import display
import numpy as np
import pandas as pd

class Boost_Demo1:
    def __init__(self):
        self.x = 0
        self.y = 0
        
    # load in a dataset via csv file
    def load_data(self,csvname):
        data = np.asarray(pd.read_csv('noisy_sin_sample.csv',header = None))
        self.x = data[:,0]
        self.y = data[:,1]
        
        # sort x values from smallest to largest
        inds = np.argsort(self.x)
        self.x = np.asarray(self.x[inds])
        self.x  = self.x[:, np.newaxis]
        self.y = np.asarray(self.y[inds])
 

    # illustrate the fitting of stumps to a boosted regressor
    def show_splitpoint(self,num_trees):
        # cycle across points, measuring total error with each split
        splits = []
        heights = []
        residual = self.y.copy()

        for d in range(num_trees):
            fig = plt.figure(figsize = (30,10))
            ax1 = fig.add_subplot(131)
            ax2 = fig.add_subplot(132)
            ax3 = fig.add_subplot(133)

            costs = []
            temp_heights = []
            temp_splits = []
            for p in range(1,len(self.x)-1):
                # determine points on each side of split
                split = (self.x[p]+self.x[p+1])/float(2)
                temp_splits.append(split)
                resid_left  = [t[0] for t in self.x if t <= split]
                resid_left = residual[:len(resid_left)]
                resid_right = residual[len(resid_left):]

                # compute average on each side
                ave_left = np.mean(resid_left)
                ave_right = np.mean(resid_right)
                temp_heights.append([ave_left,ave_right])

                # compute least squares error on each side
                cost_left = sum((ave_left - resid_left)**2)
                cost_right = sum((ave_right - resid_right)**2)
                total_cost = cost_left + cost_right
                costs.append(total_cost)

                ## plot everything nicely
                # plot left panel
                ax1.scatter(self.x,self.y,linewidth = 5,color = 'k')
                if d > 0:
                    a = np.linspace(min(self.x),max(self.x),100)
                    model = np.zeros((len(a),1))
                    for s in range(len(splits)):
                        current_split = splits[s]
                        left = [t for t in a if t < current_split]
                        model[0:len(left)] += heights[s][0]
                        model[len(left):] += heights[s][1]
                    ax1.plot(a,model,linewidth = 3,color = 'b')  

                ax1.set_title('dataset with best fit sum',fontsize = 25)
                ax1.set_xlim([min(self.x) - 0.2, max(self.x)+0.2])
                ax1.set_ylim([min(self.y)-0.2,max(self.y)+0.2])

                # plot middle panel
                ax2.scatter(self.x,residual,linewidth = 6,color = [0,0.2,1])
                l = np.linspace(0,split,100)
                hl = ave_left*np.ones((len(l),1))
                r = np.linspace(split,1,100)
                hr = ave_right*np.ones((len(r),1))
                ax2.plot(l,hl,linewidth = 3,color = 'r')
                ax2.plot(r,hr,linewidth = 3,color = 'r')
                ax2.plot([split,split],[-5,5],'-',color = 'k',linestyle = '--')
                ax2.set_xlim([min(self.x) - 0.2, max(self.x)+0.2])
                ax2.set_ylim([min(self.y)-0.2,max(self.y)+0.2])
                ax2.set_title('residual w/split search',fontsize = 25)

                # plot right panel
                ax3.scatter(np.arange(0,len(costs)),costs,color = 'g',linewidth = 5)
                ax3.set_xlim([0,len(self.x)])
                ax3.set_ylim([0,max(costs)+1])
                ax3.set_title('cost values',fontsize = 25)
                ax3.set_xlim([-1,len(self.x)-2])

                # clear current figure
                display.display(plt.gcf()) 
                display.clear_output(wait=True)

                if p < len(self.x):
                    ax1.clear()
                    ax2.clear()
                    ax3.clear()

            # determine lowest cost
            best_cost_ind = np.argmin(costs)
            split_pt = temp_splits[best_cost_ind]
            split_heights = temp_heights[best_cost_ind]
            splits.append(split_pt)
            heights.append(split_heights)

            time.sleep(1)

            # plot left panel
            ax1.scatter(self.x,self.y,linewidth = 5,color = 'k')
            a = np.linspace(min(self.x),max(self.x),100)
            model = np.zeros((len(a),1))
            for s in range(len(splits)):
                current_split = splits[s]
                left = [t for t in a if t < current_split]
                model[0:len(left)] += heights[s][0]
                model[len(left):] += heights[s][1]

            ax1.plot(a,model,linewidth = 3,color = 'b')  
            ax1.set_xlim([min(self.x) - 0.2, max(self.x)+0.2])
            ax1.set_ylim([min(self.y)-0.2,max(self.y)+0.2])
            ax1.set_title('dataset w/best fit',fontsize = 25)

            # plot middle panel
            ax2.scatter(self.x,residual,linewidth = 6,color = [0,0.2,1])
            l = np.linspace(0,split_pt,100)
            hl = heights[-1][0]*np.ones((len(l),1))
            r = np.linspace(split_pt,1,100)
            hr = heights[-1][1]*np.ones((len(r),1))
            ax2.plot(l,hl,linewidth = 3,color = 'r')
            ax2.plot(r,hr,linewidth = 3,color = 'r')
            ax2.plot([split_pt,split_pt],[-5,5],'-',color = 'k',linestyle = '--')
            ax2.set_xlim([min(self.x) - 0.2, max(self.x)+0.2])
            ax2.set_ylim([min(self.y)-0.2,max(self.y)+0.2])
            ax2.set_title('residual w/split search',fontsize = 25)

            # plot right panel
            ax3.scatter(np.arange(0,len(costs)),costs,color = 'g',linewidth = 5)
            ax3.set_xlim([0,len(self.x)])
            ax3.scatter(best_cost_ind,costs[best_cost_ind], s=80, facecolors='none', edgecolors='m',linewidth = 8)
            ax3.set_ylim([0,max(costs)+1])
            ax3.set_xlim([-1,len(self.x)-2])
            ax3.set_title('cost values',fontsize = 25)

            # update residual - compute model prediction and subtract from current residual
            y_hat = []
            for pt in range(len(self.x)):
                if self.x[pt] <= split_pt:
                    y_hat.append(split_heights[0])
                else:
                    y_hat.append(split_heights[1])
            residual -= y_hat

            display.display(plt.gcf()) 
            display.clear_output(wait=True)

            time.sleep(2)