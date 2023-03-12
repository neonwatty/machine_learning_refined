import matplotlib.animation as animation
from IPython.display import clear_output
import numpy as np
import pandas as pd

# class for illustrating regression weighting
class weighted_regression_animation_visualizer:
    ##### a simple data loading function #####
    def load(self,csvname,**args):
        sep = ','
        data = np.loadtxt(csvname,delimiter = ',')
        self.x = data[:,0]
        self.y = data[:,1]
        
    # generate a dataset if desired
    def generate_data(self,num_pts,csvname):
        # load in points (or make them)
        x = np.random.rand(num_pts)
        x = x
        y = x + 0.25*np.random.randn(num_pts)
        y = y
        
        # save 
        x.shape = (len(x),1)
        y.shape = (len(y),1)
        data = np.concatenate((x,y),axis = 1)
        np.savetxt(csvname,data)
        
        # put to global
        x = x.tolist()
        x = [v[0] for v in x]

        y = y.tolist()
        y = [v[0] for v in y]

        self.x = x
        self.y = y

    # weighted linear regression solver
    def weighted_linear_regression(self,inputs,outputs,special_ind,special_weight):        
        # setup linear system 
        A = 0
        b = 0
        for p in range(len(outputs)):
            # get pth point
            x_p = inputs[p]
            x_p = np.asarray([1,x_p])
            x_p.shape = (len(x_p),1)
            y_p = np.asarray(outputs[p])
            
            # make pth outer product matrix and solution vector
            lef = np.outer(x_p,x_p.T)
            rig = y_p*x_p
            
            # weight if special ind
            if p == special_ind:
                lef = lef*special_weight
                rig = rig*special_weight
                
            # add to totals
            A += lef
            b += rig
         
        # solve linear system
        w = np.linalg.solve(A,b)
        
        return w
        
    # animate regression weighting
    def animate_weighting(self,savepath,csvname,**kwargs):
        data = np.loadtxt(csvname,delimiter = ',')
        x = data[:,0]
        y = data[:,1]
        
        # pick out user-defined arguments
        num_slides = kwargs['num_slides']
        special_ind = kwargs['special_ind']

        # make range for plot
        base_size = 100
        size_range = np.linspace(base_size, 20*base_size, num_slides)
        weight_range = np.linspace(1,100,num_slides)
        
        # generate figure to plot onto
        fig = plt.figure(figsize=(6,6))
        artist = fig
        ax = plt.subplot(111)
        
        # animation sub-function
        weights = np.ones((len(x),1))
        weights = weights.tolist()
        weights = [s[0] for s in weights]
        def animate(k):
            ax.cla()
            special_size = size_range[k]
            special_weight = weight_range[k]
 
            # scatter plot all data
            ax.scatter(x,y,s = base_size, c = 'k', edgecolor = 'w',zorder = 0)

            # scatter plot weighted point
            ax.scatter(x[special_ind],y[special_ind],s = special_size,c = 'r',edgecolor = 'w',zorder = 2)
            ax.axis('off')
            
            # compute regression line given weighted data
            w = self.weighted_linear_regression(inputs = x,outputs = y,special_ind = special_ind, special_weight = special_weight)

            # plot regression line ontop of data
            s = np.linspace(0,1,100)
            t = w[0] + w[1]*s
            ax.plot(s,t,color = 'b',linewidth = 5,zorder = 1)
            ax.axhline(0.25,c='k',zorder = 1)
            ax.axvline(0,c='k',zorder = 1)
            
            
            return artist,
        
        anim = animation.FuncAnimation(fig, animate ,frames=num_slides, interval=num_slides, blit=True)
        
        # produce animation and save
        fps = 50
        if 'fps' in kwargs:
            fps = kwargs['fps']
        anim.save(savepath, fps=fps, extra_args=['-vcodec', 'libx264'])
        clear_output()



'''
custom scatter plot if there's repreated data, the dots 
representing more frequent points will be larger.
'''
class weighted_regression_static_visualizer:    
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