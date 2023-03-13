# import standard plotting and animation
import matplotlib.pyplot as plt
from matplotlib import gridspec
from matplotlib.ticker import FormatStrFormatter
import matplotlib.animation as animation
from mpl_toolkits.mplot3d import Axes3D
from IPython.display import clear_output

# import autograd functionality
import autograd.numpy as np

# import standard libraries
import math
import time
import copy
from inspect import signature

class Visualizer:
    '''
    Visualize linear regression in 2 and 3 dimensions.  For single input cases (2 dimensions) the path of gradient descent on the cost function can be animated.
    '''
    #### initialize ####
    def __init__(self,csvname):
        # grab input
        data = np.loadtxt(csvname,delimiter = ',')
        self.x = data[:-1,:]
        self.y = data[-1:,:] 
        
        self.colors = [[1,0.8,0.5],[0,0.7,1]]
        
        # if 1-d regression data make sure points are sorted
        if np.shape(self.x)[1] == 1:
            ind = np.argsort(self.x.flatten())
            self.x = self.x[ind,:]
            self.y = self.y[ind,:]
            
    ########## show boosting crossval on 1d regression, with fit to residual ##########
    def animate_trainval_boosting(self,savepath,runner,num_frames,**kwargs):
        # construct figure
        fig = plt.figure(figsize=(9,4))
        artist = fig
        
        ### get inds for each run ###
        inds = np.arange(0,len(runner.models),int(len(runner.models)/float(num_frames)))
        
        # create subplot with 1 active panel
        gs = gridspec.GridSpec(1, 2) 
        ax = plt.subplot(gs[0]); 
        ax2 = plt.subplot(gs[1]); 
        
        # global names for train / valid sets
        train_inds = runner.train_inds
        valid_inds = runner.valid_inds
        
        self.x_train = self.x[:,train_inds]
        self.y_train = self.y[:,train_inds]
        
        self.x_valid = self.x[:,valid_inds]
        self.y_valid = self.y[:,valid_inds]
        
        self.normalizer = runner.normalizer
        train_errors = runner.train_cost_vals
        valid_errors = runner.valid_cost_vals
        num_units = len(runner.models)
        
        # set plotting limits
        xmax = np.max(copy.deepcopy(self.x))
        xmin = np.min(copy.deepcopy(self.x))
        xgap = (xmax - xmin)*0.1
        xmin -= xgap
        xmax += xgap

        ymax = np.max(copy.deepcopy(self.y))
        ymin = np.min(copy.deepcopy(self.y))
        ygap = (ymax - ymin)*0.1
        ymin -= ygap
        ymax += ygap  
        
        # start animation
        print ('starting animation rendering...')
        def animate(k):
            # clear panels
            ax.cla()
            ax2.cla()
            
            # print rendering update
            if np.mod(k+1,25) == 0:
                print ('rendering animation frame ' + str(k+1) + ' of ' + str(num_frames))
            if k == num_frames - 1:
                print ('animation rendering complete!')
                time.sleep(1.5)
                clear_output()
            
            ####### plot total model on original dataset #######      
            ax.scatter(self.x_train,self.y_train,color = 'k',s = 60,edgecolor = self.colors[1] ,linewidth = 1.5)
        
            ax.scatter(self.x_valid,self.y_valid,color = 'k',s = 60,edgecolor = self.colors[0] ,linewidth = 1.5)
                
            # plot fit
            if k > 0:
                # plot current fit
                a = inds[k-1] 
                steps = runner.best_steps[:a+1]
                self.draw_fit(ax,steps,a)
                    
                # plot train / valid errors up to this point
                self.plot_train_valid_errors(ax2,k-1,train_errors,valid_errors,inds)
                
            # make panel size
            ax.set_xlim([xmin,xmax])
            ax.set_ylim([ymin,ymax])
            return artist,

        anim = animation.FuncAnimation(fig, animate ,frames=num_frames+1, interval=num_frames+1, blit=True)
        
        # produce animation and save
        fps = 50
        if 'fps' in kwargs:
            fps = kwargs['fps']
        anim.save(savepath, fps=fps, extra_args=['-vcodec', 'libx264'])
        clear_output()   
             
    # 1d regression demo
    def draw_fit(self,ax,steps,ind):
        # set plotting limits
        xmax = np.max(copy.deepcopy(self.x))
        xmin = np.min(copy.deepcopy(self.x))
        xgap = (xmax - xmin)*0.1
        xmin -= xgap
        xmax += xgap
        
        # plot fit
        s = np.linspace(xmin,xmax,2000)[np.newaxis,:]
        model = lambda x: np.sum([v(x) for v in steps],axis=0)
        t = model(self.normalizer(s))

        ax.plot(s.T,t.T,linewidth = 4,c = 'k')
        ax.plot(s.T,t.T,linewidth = 2,c = 'r')

    # plot training / validation errors
    def plot_train_valid_errors(self,ax,k,train_errors,valid_errors,inds):
        num_elements = np.arange(len(train_errors))

        ax.plot([v+1 for v in inds[:k+1]] ,train_errors[:k+1],color = [0,0.7,1],linewidth = 2.5,zorder = 1,label = 'training')
        #ax.scatter([v+1  for v in inds[:k+1]] ,train_errors[:k+1],color = [0,0.7,1],s = 70,edgecolor = 'w',linewidth = 1.5,zorder = 3)

        ax.plot([v+1  for v in inds[:k+1]] ,valid_errors[:k+1],color = [1,0.8,0.5],linewidth = 2.5,zorder = 1,label = 'validation')
        #ax.scatter([v+1  for v in inds[:k+1]] ,valid_errors[:k+1],color= [1,0.8,0.5],s = 70,edgecolor = 'w',linewidth = 1.5,zorder = 3)
        ax.set_title('cost function history',fontsize = 15)

        # cleanup
        ax.set_xlabel('number of units',fontsize = 12)

        # cleanp panel                
        num_iterations = len(train_errors)
        minxc = 0.5
        maxxc = len(num_elements) + 0.5
        minc = min(min(copy.deepcopy(train_errors)),min(copy.deepcopy(valid_errors)))
        maxc = max(max(copy.deepcopy(train_errors[:10])),max(copy.deepcopy(valid_errors[:10])))
        gapc = (maxc - minc)*0.25
        minc -= gapc
        maxc += gapc
        
        ax.set_xlim([minxc,maxxc])
        ax.set_ylim([minc,maxc])
        
        # ax.xaxis.set_major_locator(MaxNLocator(integer=True))
        #labels = [str(v) for v in num_units]
        #ax.set_xticks(np.arange(1,len(num_elements)+1))
       # ax.set_xticklabels(num_units)
    

class VisualizerV3:
    '''
    Visualize linear regression in 2 and 3 dimensions.  For single input cases (2 dimensions) the path of gradient descent on the cost function can be animated.
    '''
    #### initialize ####
    def __init__(self,csvname):
        # grab input
        data = np.loadtxt(csvname,delimiter = ',')
        self.x = data[:-1,:]
        self.y = data[-1:,:] 
        
        self.colors = [[1,0.8,0.5],[0,0.7,1]]
        
        # if 1-d regression data make sure points are sorted
        if np.shape(self.x)[1] == 1:
            ind = np.argsort(self.x.flatten())
            self.x = self.x[ind,:]
            self.y = self.y[ind,:]
            

    ########## show boosting results on 1d regression, with fit to residual ##########
    def animate_boosting(self,savepath,runs,frames,**kwargs):
        # select subset of runs
        inds = np.arange(0,len(runs),int(len(runs)/float(frames)))
            
        # select inds of history to plot
        num_runs = frames

        # construct figure
        fig = plt.figure(figsize=(9,4))
        artist = fig
        
        # parse any input args
        scatter = 'none'
        if 'scatter' in kwargs:
            scatter = kwargs['scatter']

        # create subplot with 1 active panel
        gs = gridspec.GridSpec(1, 3, width_ratios=[1,0.01,1]) 
        ax = plt.subplot(gs[0]); 
        ax1 = plt.subplot(gs[2]);
        ax3 = plt.subplot(gs[1]); ax3.axis('off')
        
        # start animation
        num_frames = num_runs
        print ('starting animation rendering...')
        def animate(k):
            # clear panels
            ax.cla()
            ax1.cla()
            
            # print rendering update
            if np.mod(k+1,25) == 0:
                print ('rendering animation frame ' + str(k+1) + ' of ' + str(num_frames))
            if k == num_frames - 1:
                print ('animation rendering complete!')
                time.sleep(1.5)
                clear_output()
            
            # get current run for cost function history plot
            a = inds[k]
            run = runs[a]
            
            # pluck out current weights 
            self.draw_fit(ax,ax1,runs,a)
                
            return artist,

        anim = animation.FuncAnimation(fig, animate ,frames=num_frames, interval=num_frames, blit=True)
        
        # produce animation and save
        fps = 50
        if 'fps' in kwargs:
            fps = kwargs['fps']
        anim.save(savepath, fps=fps, extra_args=['-vcodec', 'libx264'])
        clear_output()   
        
    # 1d regression demo
    def draw_fit(self,ax,ax1,runs,ind):
        # set plotting limits
        xmax = np.max(copy.deepcopy(self.x))
        xmin = np.min(copy.deepcopy(self.x))
        xgap = (xmax - xmin)*0.1
        xmin -= xgap
        xmax += xgap

        ymax = np.max(copy.deepcopy(self.y))
        ymin = np.min(copy.deepcopy(self.y))
        ygap = (ymax - ymin)*0.1
        ymin -= ygap
        ymax += ygap    

        ####### plot total model on original dataset #######
        # scatter original data
        ax.scatter(self.x.flatten(),self.y.flatten(),color = 'k',s = 60,edgecolor = 'w',linewidth = 1.5,zorder = 3)
        
        # plot fit on residual
        s = np.linspace(xmin,xmax,2000)[np.newaxis,:]
           
        # plot total fit
        t = 0
        for i in range(ind+1):
            # get current run
            run = runs[i]
            model = run.model
            normalizer = run.normalizer

            # get best weights                     
            t += model(normalizer(s))

        ax.plot(s.T,t.T,linewidth = 4,c = 'k',zorder = 1)
        ax.plot(s.T,t.T,linewidth = 2,c = 'r',zorder = 1)
        
        ##### plot residual info #####
        # get all functions from final step
        run = runs[ind]
        model = run.model
        inverse_normalizer = run.inverse_normalizer
        normalizer = run.normalizer
        x_temp = inverse_normalizer(runs[ind].x)
        y_temp = runs[ind].y
        
        # scatter residual data
        ax1.scatter(x_temp,y_temp,color = 'k',s = 60,edgecolor = 'w',linewidth = 1.5,zorder = 3)

        # make prediction
        t = model(normalizer(s))
        
        # plot fit to residual
        ax1.plot(s.T,t.T,linewidth = 4,c = 'k',zorder = 1)
        ax1.plot(s.T,t.T,linewidth = 2,c = 'r',zorder = 1)
        
        ### clean up panels ###             
        ax.set_xlim([xmin,xmax])
        ax.set_ylim([ymin,ymax])
    
        ax1.set_xlim([xmin,xmax])
        ax1.set_ylim([ymin,ymax])
        
        # label axes
        ax.set_xlabel(r'$x$', fontsize = 14)
        ax.set_ylabel(r'$y$', rotation = 0,fontsize = 14,labelpad = 15)
        ax.set_title('model ' + str(ind+1) + ' fit to original',fontsize = 14)
        
        ax1.set_xlabel(r'$x$', fontsize = 16)
        ax1.set_ylabel(r'$y$', rotation = 0,fontsize = 16,labelpad = 15)
        ax1.set_title('unit ' + str(ind+1) + ' fit to residual',fontsize = 14)