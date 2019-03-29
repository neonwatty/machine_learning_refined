# import standard plotting and animation
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from mpl_toolkits.mplot3d import Axes3D
from IPython.display import clear_output

# import autograd functionality
from autograd import grad as compute_grad   # The only autograd function you may ever need
import autograd.numpy as np
from autograd import hessian as compute_hess
import math
import time
from matplotlib import gridspec
import copy

class Visualizer:
    '''
    Compare various basis units on 3d classification
    '''
    #### initialize ####
    def __init__(self,csvname):
        # grab input
        data = np.loadtxt(csvname,delimiter = ',')
        self.x = data[:-1,:]
        self.y = data[-1,:]
        self.y.shape = (1,len(self.y))
        self.colors = ['salmon','cornflowerblue','lime','bisque','mediumaquamarine','b','m','g']
  
    ########## animate results of nonlinear classification ##########
    def animate_classifications(self,savepath,runs,frames,**kwargs):
        # select subset of runs
        inds = np.arange(0,len(runs),int(len(runs)/float(frames)))
        
        # pull out cost vals
        cost_evals = []
        for run in runs:
            # get current run histories
            cost_history = run.train_cost_histories[0]
            weight_history = run.weight_histories[0]

            # get best weights                
            win = np.argmin(cost_history)
            
            # get best cost val
            cost = cost_history[win]
            cost_evals.append(cost)
                    
        # select inds of history to plot
        num_runs = frames

        # initialize figure
        fig = plt.figure(figsize = (9,4))
        artist = fig

        # create subplot with 3 panels, plot input function in center plot
        gs = gridspec.GridSpec(1, 3, width_ratios=[2,1,0.25]) 
        ax1 = plt.subplot(gs[0]); ax1.set_aspect('equal'); ax1.axis('off');
        ax2 = plt.subplot(gs[1]); ax2.axis('off');
        ax3 = plt.subplot(gs[2]); ax3.axis('off');
        
        # setup viewing range
        num_elements = len(cost_evals)
        minxc = 0.5
        maxxc = num_elements + 0.5

        ymax = max(copy.deepcopy(cost_evals))
        ymin = min(copy.deepcopy(cost_evals))
        ygap = (ymax - ymin)*0.1
        ymax += ygap
        ymin -= ygap
        
        # plot cost path - scale to fit inside same aspect as classification plots        
        # start animation
        num_frames = num_runs
        print ('starting animation rendering...')
        def animate(k):
            # clear panels
            ax1.cla()
            ax2.cla()
            
            # print rendering update
            if np.mod(k+1,25) == 0:
                print ('rendering animation frame ' + str(k+1) + ' of ' + str(num_frames))
            if k == num_frames - 1:
                print ('animation rendering complete!')
                time.sleep(1.5)
                clear_output()
            
            #### scatter data ####
            # plot points in 2d and 3d
            ind0 = np.argwhere(self.y == +1)
            ind0 = [e[1] for e in ind0]
            ax1.scatter(self.x[0,ind0],self.x[1,ind0],s = 55, color = self.colors[0], edgecolor = 'k')
                        
            ind1 = np.argwhere(self.y == -1)
            ind1 = [e[1] for e in ind1]
            ax1.scatter(self.x[0,ind1],self.x[1,ind1],s = 55, color = self.colors[1], edgecolor = 'k')
            
            # plot boundary
            if k > 0:
                # get current run for cost function history plot
                a = inds[k-1]
                run = runs[a]
                
                # pluck out current weights 
                self.draw_fit(ax1,run,a)
                
                # cost function value
                ax2.plot(np.arange(1,num_elements + 1),cost_evals,color = 'k',linewidth = 2.5,zorder = 1)
                ax2.scatter(a + 1,cost_evals[a],color = self.colors[0],s = 70,edgecolor = 'w',linewidth = 1.5,zorder = 3)
            
            # cleanup panels
            ax1.set_yticklabels([])
            ax1.set_xticklabels([])
            ax1.set_xticks([])
            ax1.set_yticks([])
            ax1.set_xlabel(r'$x_1$',fontsize = 15)
            ax1.set_ylabel(r'$x_2$',fontsize = 15,rotation = 0,labelpad = 20)  
            
            ax2.set_xlabel('number of units',fontsize = 12)
            ax2.set_title('cost function plot',fontsize = 14)
            ax2.set_xlim([minxc,maxxc])
            ax2.set_ylim([ymin,ymax])
        
            return artist,

        anim = animation.FuncAnimation(fig, animate ,frames=num_frames+1, interval=num_frames+1, blit=True)
        
        # produce animation and save
        fps = 50
        if 'fps' in kwargs:
            fps = kwargs['fps']
        anim.save(savepath, fps=fps, extra_args=['-vcodec', 'libx264'])
        clear_output()   
            
    def draw_fit(self,ax,run,ind):
        # viewing ranges
        xmin1 = min(copy.deepcopy(self.x[0,:]))
        xmax1 = max(copy.deepcopy(self.x[0,:]))
        xgap1 = (xmax1 - xmin1)*0.05
        xmin1 -= xgap1
        xmax1 += xgap1

        xmin2 = min(copy.deepcopy(self.x[1,:]))
        xmax2 = max(copy.deepcopy(self.x[1,:]))
        xgap2 = (xmax2 - xmin2)*0.05
        xmin2 -= xgap2
        xmax2 += xgap2

        ymin = min(copy.deepcopy(self.y))
        ymax = max(copy.deepcopy(self.y))
        ygap = (ymax - ymin)*0.05
        ymin -= ygap
        ymax += ygap
             
        # plot boundary for 2d plot
        r1 = np.linspace(xmin1,xmax1,300)
        r2 = np.linspace(xmin2,xmax2,300)
        s,t = np.meshgrid(r1,r2)
        s = np.reshape(s,(np.size(s),1))
        t = np.reshape(t,(np.size(t),1))
        h = np.concatenate((s,t),axis = 1).T
        
        # plot total fit
        cost = run.cost
        model = run.model
        feat = run.feature_transforms
        normalizer = run.normalizer
        cost_history = run.train_cost_histories[0]
        weight_history = run.weight_histories[0]

        # get best weights                
        win = np.argmin(cost_history)
        w = weight_history[win]        
        
        model = lambda b: run.model(normalizer(b),w)
        z = model(h)
        z = np.sign(z)

        # reshape it
        s.shape = (np.size(r1),np.size(r2))
        t.shape = (np.size(r1),np.size(r2))     
        z.shape = (np.size(r1),np.size(r2))

        #### plot contour, color regions ####
        ax.contour(s,t,z,colors='k', linewidths=2.5,levels = [0],zorder = 2)
        ax.contourf(s,t,z,colors = [self.colors[1],self.colors[0]],alpha = 0.15,levels = range(-1,2))
                
        ### cleanup left plots, create max view ranges ###
        ax.set_xlim([xmin1,xmax1])
        ax.set_ylim([xmin2,xmax2])
        ax.set_title(str(ind+1) + ' units fit to data',fontsize = 14)
        
        
    ########## animate results of boosting ##########
    def animate_boosting_classifications(self,run,frames,**kwargs):
        # select subset of runs
        inds = np.arange(0,len(run.models),int(len(run.models)/float(frames)))

        # select inds of history to plot
        num_runs = frames

        # initialize figure
        fig = plt.figure(figsize = (9,4))
        artist = fig
        
        # create subplot with 3 panels, plot input function in center plot
        gs = gridspec.GridSpec(1, 1) 
        ax = plt.subplot(gs[0],aspect = 'equal'); ax.axis('off');
        
        # start animation
        num_frames = num_runs
        print ('starting animation rendering...')
        def animate(k):
            # clear panels
            ax.cla()
            
            # print rendering update
            if np.mod(k+1,25) == 0:
                print ('rendering animation frame ' + str(k+1) + ' of ' + str(num_frames))
            if k == num_frames - 1:
                print ('animation rendering complete!')
                time.sleep(1.5)
                clear_output()
            
            #### scatter data ####
            # plot points in 2d and 3d
            ind0 = np.argwhere(self.y == +1)
            ind0 = [e[1] for e in ind0]
            ax.scatter(self.x[0,ind0],self.x[1,ind0],s = 55, color = self.colors[0], edgecolor = 'k')
                        
            ind1 = np.argwhere(self.y == -1)
            ind1 = [e[1] for e in ind1]
            ax.scatter(self.x[0,ind1],self.x[1,ind1],s = 55, color = self.colors[1], edgecolor = 'k')
            
            # plot boundary
            if k > 0:
                # get current run for cost function history plot
                a = inds[k-1]
                model = run.models[a]
                steps = run.best_steps[:a+1]
                
                # pluck out current weights 
                self.draw_boosting_fit(ax,steps,a)
            
            # cleanup panel
            ax.set_yticklabels([])
            ax.set_xticklabels([])
            ax.set_xticks([])
            ax.set_yticks([])
            ax.set_xlabel(r'$x_1$',fontsize = 15)
            ax.set_ylabel(r'$x_2$',fontsize = 15,rotation = 0,labelpad = 20)  
        
            return artist,

        anim = animation.FuncAnimation(fig, animate ,frames=num_frames+1, interval=num_frames+1, blit=True)
        
        return(anim)
    
    def draw_boosting_fit(self,ax,steps,ind):
        # viewing ranges
        xmin1 = min(copy.deepcopy(self.x[0,:]))
        xmax1 = max(copy.deepcopy(self.x[0,:]))
        xgap1 = (xmax1 - xmin1)*0.05
        xmin1 -= xgap1
        xmax1 += xgap1

        xmin2 = min(copy.deepcopy(self.x[1,:]))
        xmax2 = max(copy.deepcopy(self.x[1,:]))
        xgap2 = (xmax2 - xmin2)*0.05
        xmin2 -= xgap2
        xmax2 += xgap2

        ymin = min(copy.deepcopy(self.y))
        ymax = max(copy.deepcopy(self.y))
        ygap = (ymax - ymin)*0.05
        ymin -= ygap
        ymax += ygap
             
        # plot boundary for 2d plot
        r1 = np.linspace(xmin1,xmax1,30)
        r2 = np.linspace(xmin2,xmax2,30)
        s,t = np.meshgrid(r1,r2)
        s = np.reshape(s,(np.size(s),1))
        t = np.reshape(t,(np.size(t),1))
        h = np.concatenate((s,t),axis = 1).T
        
        model = lambda x: np.sum([v(x) for v in steps],axis=0)
        z = model(h)
        z = np.sign(z)

        # reshape it
        s.shape = (np.size(r1),np.size(r2))
        t.shape = (np.size(r1),np.size(r2))     
        z.shape = (np.size(r1),np.size(r2))

        #### plot contour, color regions ####
        ax.contour(s,t,z,colors='k', linewidths=2.5,levels = [0],zorder = 2)
        ax.contourf(s,t,z,colors = [self.colors[1],self.colors[0]],alpha = 0.15,levels = range(-1,2))
                
        ### cleanup left plots, create max view ranges ###
        ax.set_xlim([xmin1,xmax1])
        ax.set_ylim([xmin2,xmax2])
        ax.set_title(str(ind+1) + ' units fit to data',fontsize = 14)
        
 