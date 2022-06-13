### import basic libs ###
import autograd.numpy as np
import matplotlib.pyplot as plt
from matplotlib import gridspec
import copy
import time

### import custom libs ###
from . import optimizers 
from . import cost_functions
from . import normalizers

### animation libs ###
import matplotlib.animation as animation
from IPython.display import clear_output
import matplotlib.patches as mpatches

class Setup:
    def __init__(self,x,y,**kwargs):
        # link in data
        self.x_orig = x
        self.y_orig = y
        
        # make containers for all histories
        self.weight_histories = []
        self.train_cost_histories = []
        self.train_count_histories = []
        self.valid_cost_histories = []
        self.valid_count_histories = []

    #### define normalizer ####
    def choose_normalizer(self,name):       
        # produce normalizer / inverse normalizer
        s = normalizers.Setup(self.x_orig,name)
        self.normalizer = s.normalizer
        self.inverse_normalizer = s.inverse_normalizer
        
        # normalize input 
        self.x = self.normalizer(self.x_orig)
        self.normalizer_name = name
        
        # normalize input 
        self.y = self.y_orig
  
    #### define cost function ####
    def choose_cost(self,cost_name,reg_name,**kwargs):
        # create cost on entire dataset
        self.cost = cost_functions.Setup(cost_name,reg_name)
                
        # if the cost function is a two-class classifier, build a counter too
        if cost_name == 'softmax' or cost_name == 'perceptron':
            funcs = cost_functions.Setup('twoclass_counter',reg_name)
            self.counter = funcs.cost
            
        if cost_name == 'multiclass_softmax' or cost_name == 'multiclass_perceptron':
            funcs = cost_functions.Setup('multiclass_counter',reg_name)
            self.counter = funcs.cost
            
        self.cost_name = cost_name
        self.reg_name = reg_name
            
    #### setup optimization ####
    def choose_optimizer(self,optimizer_name,**kwargs):
        # general params for optimizers
        max_its = 500; 
        alpha_choice = 10**(-1);
        epsilon = 10**(-10)
        
        # set parameters by hand
        if 'max_its' in kwargs:
            max_its = kwargs['max_its']
        if 'alpha_choice' in kwargs:
            alpha_choice = kwargs['alpha_choice']
        if 'epsilon' in kwargs:
            epsilon = kwargs['epsilon']
            
        # batch size for gradient descent?
        self.w = 0.0*np.random.randn(self.x.shape[0] + 1,1)
        num_pts = np.size(self.y)
        batch_size = np.size(self.y)
        if 'batch_size' in kwargs:
            self.batch_size = kwargs['batch_size']
        
        # run gradient descent
        if optimizer_name == 'gradient_descent':
            self.optimizer = lambda cost,x,w: optimizers.gradient_descent(cost,w,x,self.y,alpha_choice,max_its,batch_size)
        
        if optimizer_name == 'newtons_method':
            self.optimizer = lambda cost,x,w: optimizers.newtons_method(cost,w,x,self.y,max_its,epsilon=epsilon)
       
    ### try-out various regularization params ###
    def tryout_lams(self,lams,**kwargs):
        # choose number of rounds
        self.lams = lams
        num_rounds = len(lams)

        # container for costs and weights 
        self.cost_vals = []
        self.weights = []
        
        # reset initialization
        self.w_init = 0.1*np.random.randn(self.x.shape[0] + 1,1)
            
        # loop over lams and try out each
        for i in range(num_rounds):        
            # set lambda
            lam = self.lams[i]
            self.cost.set_lambda(lam)
        
            # load in current model
            w_hist,c_hist = self.optimizer(self.cost.cost,self.x,self.w_init)
            
            # determine smallest cost value attained
            ind = np.argmin(c_hist)            
            weight = w_hist[ind]
            cost_val = c_hist[ind]
            self.weights.append(weight)
            self.cost_vals.append(cost_val)
            
        # determine best value of lamba from the above runs
        ind = np.argmin(self.cost_vals)
        self.best_lam = self.lams[ind]
        self.best_weights = self.weights[ind]
        
    # compare multiple l1 regularized runs
    def animate_lams(self,savepath,**kwargs):                   
        # initialize figure
        fig = plt.figure(figsize = (9,3))
        artist = fig
        gs = gridspec.GridSpec(1,1) 
        ax = plt.subplot(gs[0])
        
        ### run over all input lamdas ###
        # start animation
        num_frames = len(self.lams)
        print ('starting animation rendering...')
        def animate(k):            
            # clear panels
            ax.cla()
            lam = self.lams[k]
            
            # print rendering update
            if np.mod(k+1,25) == 0:
                print ('rendering animation frame ' + str(k+1) + ' of ' + str(num_frames))
            if k == num_frames - 1:
                print ('animation rendering complete!')
                time.sleep(1.5)
                clear_output()
                
            # save lowest misclass weights
            w_best = self.weights[k][1:]
            
            # plot
            ax.axhline(c='k',zorder = 2)
            
            # make bar plot
            ax.bar(np.arange(1,len(w_best)+1).flatten(),np.array(w_best).flatten(), color='k', alpha=0.5)
                
            # dress panel
            title1 = r'$\lambda = ' + str(np.round(lam,2)) + '$' 
            costval = self.cost_vals[k][0]
            title2 = ', cost val = ' + str(np.round(costval,2))
            title = title1 + title2
            ax.set_title(title)
            ax.set_xlabel('learned weights')
            
            # change tick labels to used
            ax.set_xticks(np.arange(1,self.x.shape[0]+2))
            ax.set_xticklabels(np.arange(1,self.x.shape[0]+2))
            
            # clip viewing range
            ax.set_xlim([0.25,self.x.shape[0] + 0.75])
            return artist,

        anim = animation.FuncAnimation(fig, animate ,frames=num_frames, interval=num_frames, blit=True)
        
        # produce animation and save
        fps = 50
        if 'fps' in kwargs:
            fps = kwargs['fps']
        anim.save(savepath, fps=fps, extra_args=['-vcodec', 'libx264'])
        clear_output()

    # static graphics
    def plot_regress(self,id1,labels):
        # initialize figure
        fig = plt.figure(figsize = (9,3))

        # create subplot with 1 panel
        gs = gridspec.GridSpec(1, 1) 
        ax = plt.subplot(gs[0]); 
        
        # scatter plot
        ax.scatter(self.x[id1-1,:],self.y,color = 'k',edgecolor = 'w',s = 30)
    
        # dress panel
        ax.set_xlabel(labels[0])
        ax.set_ylabel(labels[1])