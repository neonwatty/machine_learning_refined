import matplotlib.pyplot as plt
from IPython.display import clear_output
import matplotlib.animation as animation
import classification_bits as bits
import numpy as np
import pandas as pd
import time

class animation_visualizer:
    
    def load_data(self,csvname):
        data = np.loadtxt(csvname,delimiter = ',')
        self.data = data

        x = data[0:2,:]
        y = data[-1,:][np.newaxis,:]
        
        special_class = +1
        return x,y,special_class

    def animate_weightings(self,savepath,csvname,**kwargs):
        self.x,self.y,special_class = self.load_data(csvname)
        self.color_opts = np.array([[1,0,0.4], [ 0, 0.4, 1],[0, 1, 0.5],[1, 0.7, 0.5],[0.7, 0.6, 0.5]])

        # pick out user-defined arguments
        num_slides = 2
        if 'num_slides' in kwargs:
            num_slides = kwargs['num_slides']

        # make range for plot
        base_size = 100
        size_range = np.linspace(base_size, 20*base_size, num_slides)
        weight_range = np.linspace(1,10,num_slides)
        
        # generate figure to plot onto
        fig = plt.figure(figsize=(5,5))
        artist = fig
        ax = plt.subplot(111)
        
        # animation sub-function
        ind1 = np.argwhere(self.y == special_class)
        ind1 = [v[1] for v in ind1]
        
        # run animator
        max_its = 5
        w = 0.1*np.random.randn(3,1)
        g = bits.softmax
        def animate(k):
            ax.cla()
            
            # print rendering update
            if np.mod(k+1,25) == 0:
                print ('rendering animation frame ' + str(k+1) + ' of ' + str(num_slides))
            if k == num_slides - 1:
                print ('animation rendering complete!')
                time.sleep(1.5)
                clear_output()
            
            # define beta
            special_size = size_range[k]
            special_weight = weight_range[k]
            beta = np.ones((1,self.y.size))
            beta[:,ind1] = special_weight
            
            # run optimizer
            w_hist,g_hist = bits.newtons_method(g,w,self.x,self.y,beta,max_its)

            # determine minimum classification weightings


            w_best = w_hist[-1]
            self.model = lambda data: bits.model(data,w_best)
            
            # scatter plot all data
            self.plot_data(ax,special_class,special_size)
            
            # draw decision boundary
            self.draw_decision_boundary(ax)
            return artist,
        
        anim = animation.FuncAnimation(fig, animate ,frames=num_slides, interval=num_slides, blit=True)
        
        # produce animation and save
        fps = 50
        if 'fps' in kwargs:
            fps = kwargs['fps']
        anim.save(savepath, fps=fps, extra_args=['-vcodec', 'libx264'])
        clear_output()
                
    def plot_data(self,ax,special_class,special_size):
        # scatter points in both panels
        class_nums = np.unique(self.y)
        C = len(class_nums)
        z = 3
        for c in range(C):
            ind = np.argwhere(self.y == class_nums[c])
            ind = [v[1] for v in ind]
            s = 80
            if class_nums[c] == special_class:
                s = special_size
                z = 0
            ax.scatter(self.x[0,ind],self.x[1,ind],s = s,color = self.color_opts[c],edgecolor = 'k',linewidth = 1.5,zorder = z)
            
        # control viewing limits
        minx = min(self.x[0,:])
        maxx = max(self.x[0,:])
        gapx = (maxx - minx)*0.1
        minx -= gapx
        maxx += gapx
        
        miny = min(self.x[1,:])
        maxy = max(self.x[1,:])
        gapy = (maxy - miny)*0.1
        miny -= gapy
        maxy += gapy
        
        ax.set_xlim([minx,maxx])
        ax.set_ylim([miny,maxy])
        #ax.axis('equal')
        ax.axis('off')

    # toy plot
    def draw_decision_boundary(self,ax,**kwargs):  
        # control viewing limits
        minx = min(self.x[0,:])
        maxx = max(self.x[0,:])
        gapx = (maxx - minx)*0.1
        minx -= gapx
        maxx += gapx
        
        miny = min(self.x[1,:])
        maxy = max(self.x[1,:])
        gapy = (maxy - miny)*0.1
        miny -= gapy
        maxy += gapy

        r = np.linspace(minx,maxx,200)
        s = np.linspace(miny,maxy,200)
        w1_vals,w2_vals = np.meshgrid(r,s)
        w1_vals.shape = (len(r)**2,1)
        w2_vals.shape = (len(s)**2,1)
        h = np.concatenate([w1_vals,w2_vals],axis = 1)
        g_vals = self.model(h.T)
        g_vals = np.asarray(g_vals)

        # vals for cost surface
        w1_vals.shape = (len(r),len(s))
        w2_vals.shape = (len(r),len(s))
        g_vals.shape = (len(r),len(s))
        
        # plot separator curve in right plot
        ax.contour(w1_vals,w2_vals,g_vals,colors = 'k',levels = [0],linewidths = 3,zorder = 1)
            
        # plot color filled contour based on separator
        g_vals = np.sign(g_vals) + 1
        ax.contourf(w1_vals,w2_vals,g_vals,colors = self.color_opts[:],alpha = 0.1,levels = range(0,2+1))



class static_visualizer:
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