
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.animation as animation
# load in subdirectories containing python files
from JSAnimation import IPython_display


class demo():
    def __init__(self):
        a = 0
        self.x = []
        self.y = []
        
    # load in a two-dimensional dataset from csv - input should be in first column, oiutput in second column, no headers 
    def load_data(self,*args):
        # load data
        data = np.asarray(pd.read_csv(args[0],header = None))
    
        # import data and reshape appropriately
        self.x = data[:,0:-1]
        self.x.shape = (len(self.x),1)
        self.y = data[:,-1]
        self.y.shape = (len(self.y),1)

    # compute least squares cost between loaded points and sample line
    def compute_cost(self,b,w):
        y_pred = b + w*self.x
        cost = 0
        for p in range(len(self.y)):
            xp = self.x[p]
            yp = self.y[p]
            y_pred = b + w*xp
            cost += np.log(1 + np.exp(-yp*y_pred))
        return cost
    
    # create random slope and intercept - make the 'random' but so that they are visible in the panel
    def random_line(self):
        # params
        b = 4*np.random.rand(1) - 2
        w = 16*np.random.rand(1) - 8
        return b,w
    
    # animate sampling random lines and plotting squared error
    def animate_sampling(self,num_samples):
        # first generate all of the samples and compute all of the costs
        line_params = []   # container for storing random weights
        costs = []         # container for storing associated cost of random line on data
        for c in range(num_samples):
            # get random b and w - create a random line
            b,w = self.random_line()
            line_params.append([b,w])
            
            # use random line to compute cost on data
            g = self.compute_cost(b,w)
            costs.append(g)
        
        # pre-compute things that will be printed in each frame of the animation - e.g., input of each random line
        view_gap = (max(self.x) - min(self.x))/float(10)
        x_line = np.linspace(min(self.x)-view_gap,max(self.x)+view_gap,100)
        
        ### with all of the lines / costs computed we can animate them!
        # create figure and panels
        fig = plt.figure(num=None, figsize=(15,5), dpi=80, facecolor='w', edgecolor='k')
        ax1 = plt.subplot(121)
        ax2 = plt.subplot(122,projection='3d')
        
        ### plot the true cost surface and a reference
        # create surface over range of b and w
        r = np.linspace(-8.1,8.1,100)   
        r2 = np.linspace(-2.5,2.5,100)
        s,t = np.meshgrid(r,r2)
        shape = np.shape(s)
        s = np.reshape(s,(np.size(s),1))
        t = np.reshape(t,(np.size(t),1))
        cost_surface = np.zeros(np.shape(s))
        for i in range(len(s)):
            cost = self.compute_cost(t[i],s[i])
            cost_surface[i] = cost
                    
        # compute the optimal weights - just take minimum off of surface
        i = np.argmin(cost_surface)
        b = t[i]
        w = s[i]
        g = cost_surface[i]
        line_params.insert(0,[b,w])
        line_params.append([b,w])
        costs.insert(0,g)
        costs.append(g)
        
        # reshape cost surface vector for plotting
        s.shape = (np.size(r),np.size(r))
        t.shape = (np.size(r2),np.size(r2))
        cost_surface.shape = (np.size(r),np.size(r2))      
        
        # plot cost surface and reference
        ax2.plot_surface(s,t,cost_surface,alpha = 0.15)
        ax2.plot_surface(s,t,cost_surface*0,alpha = 0.1)
        
        ## clean up right plot, label axes, etc., - we won't be clearing it every frame, so might as well just take care of this once
        # set viewing angle
        ax2.view_init(0,140)        

        # turn off tick marks - just to keep the panel from being to busy, but you might want to see them so just delete these to get axes values back
        ax2.set_xticks([])
        ax2.set_yticks([])
        ax2.set_zticks([])

        # add axis labels
        ax2.set_xlabel('intercept',fontsize = 14,labelpad = -5)
        ax2.set_ylabel('slope',fontsize = 14,labelpad = -5)
        ax2.zaxis.set_rotate_label(False)  # disable automatic rotation
        ax2.set_zlabel('cost  ',fontsize = 14, rotation = 0,labelpad = 1)
            
        def show_samples(c):
            # get current random line and associated cost
            r = line_params[c]
            b = r[0]
            w = r[1]
            g = costs[c]
 
            ### plot left panel
            # clear panel for next slide in animation
            ax1.cla()
            
            # plot points
            pos_inds = np.argwhere(self.y > 0)
            pos_inds = [v[0] for v in pos_inds]
            ax1.scatter(self.x[pos_inds],self.y[pos_inds],color = 'salmon',linewidth = 1,marker = 'o',edgecolor = 'k',s = 90)

            neg_inds = np.argwhere(self.y < 0)
            neg_inds = [v[0] for v in neg_inds]
            artist = ax1.scatter(self.x[neg_inds],self.y[neg_inds],color = 'cornflowerblue',linewidth = 1,marker = 'o',edgecolor = 'k',s = 90)
                        
            # plot line parameters with associated cost
            y_line = np.tanh(b + w*x_line)       
            if c > 0 and c < num_samples+1:
                ax1.plot(x_line,y_line,color = 'm',linewidth = 4,zorder = 0)
                ax2.scatter(w,b,g,color = 'm',s = 50,alpha = 0.4)
            else:
                ax1.plot(x_line,y_line,color = 'lime',linewidth = 4,zorder = 0)
                ax2.scatter(w,b,g,color = 'lime',s = 180,alpha = 0.8,zorder = 0)
                

            return artist,  # for this animation to work something must be returned to the animator - doesn't seem to matter which object 
        
        anim = animation.FuncAnimation(fig, show_samples,frames=num_samples+2, interval=num_samples+2, blit=True)
        
        # set frames per second in animation
        IPython_display.anim_to_html(anim,fps = 5)
        
        return(anim)