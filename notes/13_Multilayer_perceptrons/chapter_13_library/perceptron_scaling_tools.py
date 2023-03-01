# import standard plotting and animation
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from matplotlib import gridspec
from IPython.display import clear_output
from mpl_toolkits.mplot3d import proj3d
from matplotlib.patches import FancyArrowPatch
from matplotlib.text import Annotation
from mpl_toolkits.mplot3d.proj3d import proj_transform

# import autograd functionality
from autograd import grad as compute_grad   
from autograd import hessian as compute_hess
import autograd.numpy as np
import math
import time
import copy

class Visualizer:    
    # standard normalization function 
    def standard_normalizer(self,x):
        # compute the mean and standard deviation of the input
        x_means = np.mean(x,axis = 1)[:,np.newaxis]
        x_stds = np.std(x,axis = 1)[:,np.newaxis]   
        
        # check to make sure thta x_stds > small threshold, for those not
        # divide by 1 instead of original standard deviation
        ind = np.argwhere(x_stds < 10**(-2))
        if len(ind) > 0:
            ind = [v[0] for v in ind]
            adjust = np.zeros((x_stds.shape))
            adjust[ind] = 1.0
            x_stds += adjust

        # create standard normalizer function
        normalizer = lambda data: (data - x_means)/x_stds

        # create inverse standard normalizer
        inverse_normalizer = lambda data: data*x_stds + x_means

        # return normalizer 
        return normalizer,inverse_normalizer

    # fully evaluate our network features using the tensor of weights in w
    def feature_transforms(self,a, w):    
        # loop through each layer matrix
        self.all_activations = []
        for W in w:
            #  pad with ones (to compactly take care of bias) for next layer computation        
            o = np.ones((1,np.shape(a)[1]))
            a = np.vstack((o,a))

            # compute inner product with current layer weights
            a = np.dot(a.T, W).T

            # output of layer activation
            a = self.activation(a)
            
            # normalized architecture or not?
            if self.normalize == True:
                normalizer,inverse_normalizer = self.standard_normalizer(a)
                a = normalizer(a)
                
            # store activation for plotting
            self.all_activations.append(a)
        return a
    
    
    def scatter_activations(self,ax):
        g = self.all_activations
        num_layers = len(g)
        for b in range(num_layers):
            f = g[b]
            label = r'($f^{(' + str(b+1) + ')}_1,\,f^{(' + str(b+1) + ')}_2$)'
            ax.scatter(f[0,:],f[1,:], c = self.colors[b],s = 60, edgecolor = 'k',linewidth = 1,label=label)
        
        if num_layers == 1:
            ax.set_xlabel(r'$f^{(1)}_1$', fontsize = 14,labelpad = 10)
            ax.set_ylabel(r'$f^{(1)}_2$', rotation = 0,fontsize = 14,labelpad = 10)
        
    def shifting_distribution(self,savepath,run,frames,x,**kwargs):
        self.colors = ['cyan','magenta','lime','orange']
        
        # select inds of history to plot
        weight_history = run.weight_histories[0]
        cost_history = run.cost_histories[0]
        inds = np.arange(0,len(weight_history),int(len(weight_history)/float(frames)))
        weight_history_sample = [weight_history[v] for v in inds]
        cost_history_sample = [cost_history[v] for v in inds]
        start = inds[0]
        feature_transforms = run.feature_transforms
                
        # define activations
        self.activation = np.tanh
        if 'activation' in kwargs:
            self.activation = kwargs['activation']
        self.normalize = False
        if 'normalize' in kwargs:
            self.normalize = kwargs['normalize']
        xmin = 0
        xmax = 1
        
        ### set viewing limits for scatter plot ###
        xmins = []
        xmaxs = []
        ymins = []
        ymaxs = []
        
        for k in range(len(inds)):
            current_ind = inds[k]
            w_best = weight_history[current_ind]
            bl = self.feature_transforms(x,w_best[0])
            
            # get all activations and look over each layer
            g = self.all_activations
            num_layers = len(g)
            for b in range(num_layers):
                f = g[b]
    
                xmin = np.min(copy.deepcopy(f[0,:]))
                xmax = np.max(copy.deepcopy(f[0,:]))

                ymin = np.min(copy.deepcopy(f[1,:]))
                ymax = np.max(copy.deepcopy(f[1,:]))

                xmins.append(xmin)
                xmaxs.append(xmax)
                ymins.append(ymin)
                ymaxs.append(ymax)

        xmin = min(xmins)
        xmax = max(xmaxs)
        xgap = (xmax - xmin)*0.1
        xmin -= xgap
        xmax += xgap
        ymin = min(ymins)
        ymax = max(ymaxs)
        ygap = (ymax - ymin)*0.1    
        ymin -= ygap
        ymax += ygap
            
         # create figure 
        fig = plt.figure(figsize = (9,4))
        artist = fig

        # create subplots
        show_history = False
        if 'show_history' in kwargs:
            show_history = kwargs['show_history']

        gs = gridspec.GridSpec(1,1)
        ax = plt.subplot(gs[0]); 
        axs = [ax]
        if show_history == True:
            gs = gridspec.GridSpec(1,2)
            ax = plt.subplot(gs[0]); 
            axs = [ax]
            ax1 = plt.subplot(gs[1]); 
            axs.append(ax1)
            c = 0

        # remove whitespace from figure
        fig.subplots_adjust(left=0, right=1, bottom=0, top=1) # remove whitespace
        fig.subplots_adjust(wspace=0.01,hspace=0.01)

        # start animation
        num_frames = len(inds)
        print ('starting animation rendering...')
        def animate(k):        
            # get current index to plot
            current_ind = inds[k]

            if show_history == True:
                ax = axs[-1]
                ax.cla()
                ax.scatter(current_ind,cost_history[current_ind],s = 60,color = 'r',edgecolor = 'k',zorder = 3)
                self.plot_cost_history(ax,cost_history,start=0)

            # print rendering update
            if np.mod(k+1,25) == 0:
                print ('rendering animation frame ' + str(k+1) + ' of ' + str(num_frames))
            if k == num_frames - 1:
                print ('animation rendering complete!')
                time.sleep(1.5)
                clear_output()

            # pluck out current weights 
            w_best = weight_history[current_ind]
            f = self.feature_transforms(x,w_best[0])

            # produce static img
            ax = axs[0]
            ax.cla()
            self.scatter_activations(ax)
            ax.set_xlim([xmin,xmax])
            ax.set_ylim([ymin,ymax])
            #ax.legend(loc=0, borderaxespad=0.)
        
            return artist,

        anim = animation.FuncAnimation(fig, animate,frames=num_frames,interval = 25,blit=False)

        # produce animation and save
        fps = 50
        if 'fps' in kwargs:
            fps = kwargs['fps']
        anim.save(savepath, fps=fps, extra_args=['-vcodec', 'libx264'])
        clear_output()           
            

    # a small Python function for plotting the distributions of input features
    def single_layer_animation(self,savepath,run,frames,x,**kwargs):
        # select inds of history to plot
        weight_history = run.weight_history
        cost_history = run.cost_history
        inds = np.arange(0,len(weight_history),int(len(weight_history)/float(frames)))
        weight_history_sample = [weight_history[v] for v in inds]
        cost_history_sample = [cost_history[v] for v in inds]
        start = inds[0]
        feature_transforms = run.feature_transforms
                
        # define activations
        self.activation = np.tanh
        if 'activation' in kwargs:
            self.activation = kwargs['activation']
        self.normalize = False
        if 'normalize' in kwargs:
            self.normalize = kwargs['normalize']

        # create figure 
        fig = plt.figure(figsize = (9,6))
        artist = fig

        # create subplots
        N = np.shape(feature_transforms(x,weight_history[0][0]))[0]
        layer_sizes = []
        self.feature_transforms(x,weight_history[0][0])
        layer_sizes = [np.shape(v)[0] for v in self.all_activations]
        num_layers = len(layer_sizes)
        max_units = max(layer_sizes)
        
        show_history = False
        if 'show_history' in kwargs:
            show_history = kwargs['show_history']

        gs = gridspec.GridSpec(num_layers,max_units)
        axs = []
        for n in range(num_layers*max_units):
            ax = plt.subplot(gs[n]); 
            axs.append(ax)
        if show_history == True:
            gs = gridspec.GridSpec(num_layers + 1,max_units)
            ax = plt.subplot(gs[0,:]); 
            axs = [ax]
            c = 0
            for n in range(num_layers):
                current_layer = layer_sizes[n]
                current_axs = []
                for m in range(current_layer):
                    ax = plt.subplot(gs[n+1,m]); 
                    current_axs.append(ax)
                axs.append(current_axs)

        # remove whitespace from figure
        fig.subplots_adjust(left=0, right=1, bottom=0, top=1) # remove whitespace
        fig.subplots_adjust(wspace=0.01,hspace=0.01)

        # start animation
        num_frames = len(inds)
        print ('starting animation rendering...')
        def animate(k):        
            # get current index to plot
            current_ind = inds[k]

            if show_history == True:
                ax = axs[0]
                ax.cla()
                ax.scatter(current_ind,cost_history[current_ind],s = 60,color = 'r',edgecolor = 'k',zorder = 3)
                self.plot_cost_history(ax,cost_history,start=0)

            # print rendering update
            if np.mod(k+1,25) == 0:
                print ('rendering animation frame ' + str(k+1) + ' of ' + str(num_frames))
            if k == num_frames - 1:
                print ('animation rendering complete!')
                time.sleep(1.5)
                clear_output()

            # pluck out current weights 
            w_best = weight_history[current_ind]
            f = self.feature_transforms(x,w_best[0])

            # produce static img
            for u in range(num_layers):
                bl = self.all_activations[u]
                local_axs = axs[u+1]
                for ax in local_axs:
                    ax.cla()
                self.single_layer_distributions(u,bl,local_axs)

            return artist,

        anim = animation.FuncAnimation(fig, animate,frames=num_frames,interval = 25,blit=False)

        # produce animation and save
        fps = 50
        if 'fps' in kwargs:
            fps = kwargs['fps']
        anim.save(savepath, fps=fps, extra_args=['-vcodec', 'libx264'])
        clear_output()
        
    def single_layer_distributions(self,u,x,axs):
        # loop over input and plot each individual input dimension value
        all_bins = []
        N = x.shape[0]
        for n in range(N):
            hist, bins = np.histogram(x[n,:], bins=30)
            all_bins.append(bins.ravel())

        # determine range for all subplots
        maxview = np.max(all_bins)
        minview = np.min(all_bins)
        viewrange = (maxview - minview)*0.1
        maxview += viewrange
        minview -= viewrange

        for n in range(N):
            ax = axs[n]
            hist, bins = np.histogram(x[n,:], bins=30)
            width = 0.7 * (bins[1] - bins[0])
            center = (bins[:-1] + bins[1:]) / 2
            ax.barh(center, hist,width)
            ax.set_title(r'$f_' + str(n+1) + '^{(' + str(u+1) + ')}$',fontsize=14)
            ax.set_ylim([minview,maxview])

    #### compare cost function histories ####
    def plot_cost_history(self,ax,history,start):
        # plotting colors
        colors = ['k']

        # plot cost function history
        ax.plot(np.arange(start,len(history),1),history[start:],linewidth = 3,color = 'k') 

        # clean up panel / axes labels
        xlabel = 'step $k$'
        ylabel = r'$g\left(\mathbf{w}^k\right)$'
        ax.set_xlabel(xlabel,fontsize = 14)
        ax.set_ylabel(ylabel,fontsize = 14,rotation = 0,labelpad = 25)
        title = 'cost history'
        ax.set_title(title,fontsize = 18)

        # plotting limits
        xmin = 0; xmax = len(history); xgap = xmax*0.05; 

        xmin -= xgap; xmax += xgap;
        ymin = np.min(history); ymax = np.max(history); ygap = (ymax - ymin)*0.1;
        ymin -= ygap; ymax += ygap;

        ax.set_xlim([xmin,xmax]) 
        ax.set_ylim([ymin,ymax]) 