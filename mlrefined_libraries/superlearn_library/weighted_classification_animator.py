import matplotlib.pyplot as plt
import autograd.numpy as np                 # Thinly-wrapped numpy

# import custom JS animator
import sys
from IPython.display import clear_output

# import standard plotting and animation
import matplotlib.pyplot as plt
import matplotlib.animation as animation

# import classification bits
from . import classification_bits as bits
import time
    
class Visualizer:
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