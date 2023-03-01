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
    def animate_trainval_regularization(self,savepath,runs,frames,num_units,**kwargs):
        # get training / validation errors
        train_errors = []
        valid_errors = []
        for run in runs:
            # get histories
            train_costs = run.train_cost_histories[0]
            valid_costs = run.valid_cost_histories[0]
            weights = run.weight_histories[0]
            
            # select based on minimum training
            ind = np.argmin(train_costs)
            train_cost = train_costs[ind]
            valid_cost = valid_costs[ind]
            weight = weights[ind]
            
            # store
            train_errors.append(train_cost)
            valid_errors.append(valid_cost)
            
        # select subset of runs
        inds = np.arange(0,len(runs),int(len(runs)/float(frames)))
        train_errors = [train_errors[v] for v in inds]
        valid_errors = [valid_errors[v] for v in inds]
        labels = []
        for f in range(frames):
            run = runs[f]
            labels.append(np.round(run.lam,2))
        
        # select inds of history to plot
        num_runs = frames

        # construct figure
        fig = plt.figure(figsize=(9,4))
        artist = fig
        
        # parse any input args
        scatter = 'none'
        if 'scatter' in kwargs:
            scatter = kwargs['scatter']
        show_history = False
        if 'show_history' in kwargs:
            show_history = kwargs['show_history']

        # create subplot with 1 active panel
        gs = gridspec.GridSpec(1, 3, width_ratios=[5,5,1]) 
        ax = plt.subplot(gs[0]); 
        ax1 = plt.subplot(gs[1]);
        ax3 = plt.subplot(gs[2]); ax3.axis('off')
        
        if show_history == True:
            # create subplot with 2 active panels
            gs = gridspec.GridSpec(1, 2, width_ratios=[1.3,1]) 
            ax = plt.subplot(gs[0]); 
            ax1 = plt.subplot(gs[1]); 
        
        # start animation
        num_frames = num_runs + 1
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
           
            if k == 0:
                # get current run for cost function history plot
                a = inds[k]
                run = runs[a]

                # pluck out current weights 
                plot_fit = False
                self.draw_fit_trainval(ax,run,plot_fit)
                                
                # show cost function history
                if show_history == True:
                    ax1.cla()
                    plot = False
                    self.plot_train_valid_errors(ax1,k,train_errors,valid_errors,labels,plot)
                    #ax1.set_yscale('log',basey=10)
            else:
                # get current run for cost function history plot
                a = inds[k-1]
                run = runs[a]
           
                # pluck out current weights 
                plot_fit = True
                self.draw_fit_trainval(ax,run,plot_fit)
                
                # show cost function history
                if show_history == True:
                    ax1.cla()
                    plot = True
                    self.plot_train_valid_errors(ax1,k-1,train_errors,valid_errors,labels,plot)
                    #ax1.set_yscale('log',basey=10)
            return artist,

        anim = animation.FuncAnimation(fig, animate ,frames=num_frames, interval=num_frames, blit=True)
        
        # produce animation and save
        fps = 50
        if 'fps' in kwargs:
            fps = kwargs['fps']
        anim.save(savepath, fps=fps, extra_args=['-vcodec', 'libx264'])
        clear_output()   


    def plot_train_valid_errors(self,ax,k,train_errors,valid_errors,labels,plot):      
        if plot == True:
            ax.plot(labels[:k+1] ,train_errors[:k+1],color = [0,0.7,1],linewidth = 2.5,zorder = 1,label = 'training')
            #ax.scatter(labels[:k+1],train_errors[:k+1],color = [0,0.7,1],s = 70,edgecolor = 'w',linewidth = 1.5,zorder = 3)

            ax.plot(labels[:k+1] ,valid_errors[:k+1],color = [1,0.8,0.5],linewidth = 2.5,zorder = 1,label = 'validation')
            #ax.scatter(labels[:k+1] ,valid_errors[:k+1],color= [1,0.8,0.5],s = 70,edgecolor = 'w',linewidth = 1.5,zorder = 3)

        # cleanup
        ax.set_xlabel(r'$\lambda$',fontsize = 12)
        ax.set_title('errors',fontsize = 15)

        # cleanp panel                
        num_iterations = len(train_errors)
        minxc = -0.05
        maxxc = max(labels) + 0.05
        minc = min(min(copy.deepcopy(train_errors)),min(copy.deepcopy(valid_errors)))
        maxc = max(max(copy.deepcopy(train_errors[:])),max(copy.deepcopy(valid_errors[5:])))
        
        
        gapc = (maxc - minc)*0.1
        minc -= gapc
        maxc += gapc
        maxc = min(maxc,35)
        
        ax.set_xlim([minxc,maxxc])
        ax.set_ylim([minc,maxc])
        
    # 1d regression demo
    def draw_fit_trainval(self,ax,run,plot_fit):
        # set plotting limits
        xmax = np.max(copy.deepcopy(self.x))
        xmin = np.min(copy.deepcopy(self.x))
        xgap = (xmax - xmin)*0.1
        xmin -= xgap
        xmax += xgap

        ymax = np.max(copy.deepcopy(self.y))
        ymin = np.min(copy.deepcopy(self.y))
        ygap = (ymax - ymin)*0.3
        ymin -= ygap
        ymax += ygap    

        ####### plot total model on original dataset #######
        # scatter original data - training and validation sets
        train_inds = run.train_inds
        valid_inds = run.valid_inds
        ax.scatter(self.x[:,train_inds],self.y[:,train_inds],color = self.colors[1],s = 40,edgecolor = 'k',linewidth = 0.9)
        ax.scatter(self.x[:,valid_inds],self.y[:,valid_inds],color = self.colors[0],s = 40,edgecolor = 'k',linewidth = 0.9)
        
        if plot_fit == True:
            # plot fit on residual
            s = np.linspace(xmin,xmax,2000)[np.newaxis,:]

            # plot total fit
            t = 0

            # get current run
            cost = run.cost
            model = run.model
            feat = run.feature_transforms
            normalizer = run.normalizer
            cost_history = run.train_cost_histories[0]
            weight_history = run.weight_histories[0]

            # get best weights                
            win = np.argmin(cost_history)
            w = weight_history[win]        
            t = model(normalizer(s),w)

            ax.plot(s.T,t.T,linewidth = 4,c = 'k')
            ax.plot(s.T,t.T,linewidth = 2,c = 'r')
            
            lam = run.lam
            ax.set_title( 'lam = ' + str(np.round(lam,2)) + ' and fit to original',fontsize = 14)

        if plot_fit == False:
            ax.set_title('test',fontsize = 14,color = 'w')

        ### clean up panels ###             
        ax.set_xlim([xmin,xmax])
        ax.set_ylim([ymin,ymax])
        
        # label axes
        ax.set_xlabel(r'$x$', fontsize = 14)
        ax.set_ylabel(r'$y$', rotation = 0,fontsize = 14,labelpad = 15)

    ########## show boosting results on 1d regression, with fit to residual ##########
    def animate_boosting(self,runs,frames,**kwargs):
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
        show_history = False
        if 'show_history' in kwargs:
            show_history = kwargs['show_history']

        # create subplot with 1 active panel
        gs = gridspec.GridSpec(1, 3, width_ratios=[5,5,1]) 
        ax = plt.subplot(gs[0]); 
        ax1 = plt.subplot(gs[1]);
        ax3 = plt.subplot(gs[2]); ax3.axis('off')
        
        if show_history == True:
            # create subplot with 2 active panels
            gs = gridspec.GridSpec(1, 2, width_ratios=[2,1]) 
            ax = plt.subplot(gs[0]); 
            ax1 = plt.subplot(gs[1]); 
        
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
            
            # show cost function history
            if show_history == True:
                ax1.cla()
                ax1.scatter(current_ind,cost_history[current_ind],s = 60,color = 'r',edgecolor = 'k',zorder = 3)
                self.plot_cost_history(ax1,cost_history,start)
                
            return artist,

        anim = animation.FuncAnimation(fig, animate ,frames=num_frames, interval=num_frames, blit=True)
        
        return(anim)

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
        ax.scatter(self.x.flatten(),self.y.flatten(),color = 'k',s = 40,edgecolor = 'w',linewidth = 0.9)
        
        # plot fit on residual
        s = np.linspace(xmin,xmax,2000)[np.newaxis,:]
           
        # plot total fit
        t = 0
        for i in range(ind+1):
            # get current run
            run = runs[i]
            cost = run.cost
            model = run.model
            feat = run.feature_transforms
            normalizer = run.normalizer
            cost_history = run.cost_histories[0]
            weight_history = run.weight_histories[0]

            # get best weights                
            win = np.argmin(cost_history)
            w = weight_history[win]        
            t += model(normalizer(s),w)

        ax.plot(s.T,t.T,linewidth = 4,c = 'k')
        ax.plot(s.T,t.T,linewidth = 2,c = 'r')
        
        ##### plot residual info #####
        # get all functions from final step
        run = runs[ind]
        model = run.model
        inverse_normalizer = run.inverse_normalizer
        normalizer = run.normalizer
        cost_history = run.cost_histories[0]
        weight_history = run.weight_histories[0]
        win = np.argmin(cost_history)
        w = weight_history[win]      
        x_temp = inverse_normalizer(runs[ind].x)
        y_temp = runs[ind].y
        
        # scatter residual data
        ax1.scatter(x_temp,y_temp,color = 'k',s = 40,edgecolor = 'w',linewidth = 0.9)

        # make prediction
        t = model(normalizer(s),w)
        
        # plot fit to residual
        ax1.plot(s.T,t.T,linewidth = 4,c = 'k')
        ax1.plot(s.T,t.T,linewidth = 2,c = 'r')
        
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

     