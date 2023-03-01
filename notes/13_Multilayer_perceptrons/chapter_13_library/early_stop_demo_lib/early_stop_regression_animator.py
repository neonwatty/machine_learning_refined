# import standard plotting and animation
import matplotlib.pyplot as plt
from matplotlib import gridspec
from matplotlib.ticker import FormatStrFormatter
import matplotlib.animation as animation
from mpl_toolkits.mplot3d import Axes3D
from IPython.display import clear_output
from matplotlib.ticker import MaxNLocator, FuncFormatter

# import autograd functionality
import autograd.numpy as np
import math
import time
from matplotlib import gridspec
import copy
from matplotlib.ticker import FormatStrFormatter
from inspect import signature

class Visualizer:
    '''
    Visualize cross validation performed on N = 2 dimensional input classification datasets
    '''
    #### initialize ####
    def __init__(self,csvname):
        # grab input
        data = np.loadtxt(csvname,delimiter = ',')
        self.x = data[:-1,:]
        self.y = data[-1:,:] 
        
        self.colors = ['salmon','cornflowerblue','lime','bisque','mediumaquamarine','b','m','g']

    
    #### animate multiple runs on single regression ####
    def animate_trainval_earlystop(self,run,frames,**kwargs):
        train_errors = run.train_cost_histories[0]
        valid_errors = run.valid_cost_histories[0]
        weight_history = run.weight_histories[0]
        num_units = len(weight_history)

        # select subset of runs
        inds = np.arange(0,len(weight_history),int(len(weight_history)/float(frames)))
        weight_history = [weight_history[v] for v in inds]
        train_errors = [train_errors[v] for v in inds]
        valid_errors = [valid_errors[v] for v in inds]
       
        # construct figure
        fig = plt.figure(figsize = (6,6))
        artist = fig

        # create subplot with 4 panels, plot input function in center plot
        gs = gridspec.GridSpec(2, 2)
        ax = plt.subplot(gs[0]); 
        ax1 = plt.subplot(gs[2]); 
        ax2 = plt.subplot(gs[3]); 
        ax3 = plt.subplot(gs[1]); 
        
        # start animation
        num_frames = len(inds)        
        print ('starting animation rendering...')
        def animate(k):
            print (k)
            # clear panels
            ax.cla()
            ax1.cla()
            ax2.cla()
            ax3.cla()
            
            # print rendering update
            if np.mod(k+1,25) == 0:
                print ('rendering animation frame ' + str(k+1) + ' of ' + str(num_frames))
            if k == num_frames - 1:
                print ('animation rendering complete!')
                time.sleep(1.5)
                clear_output()
            
            #### plot training, testing, and full data ####            
            # pluck out current weights 
            w_best = weight_history[k]
            
            # produce static img
            self.draw_data(ax,w_best,run,train_valid = 'original')
            self.draw_fit(ax,run,w_best,train_valid = 'train')
            
            self.draw_data(ax1,w_best,run,train_valid = 'train')
            self.draw_fit(ax1,run,w_best,train_valid = 'train')
            self.draw_data(ax2,w_best,run,train_valid = 'validate')
            self.draw_fit(ax2,run,w_best,train_valid = 'validate')

            #### plot training / validation errors ####
            self.plot_train_valid_errors(ax3,k,train_errors,valid_errors,num_units)
            return artist,

        anim = animation.FuncAnimation(fig, animate ,frames=num_frames, interval=num_frames, blit=True)
        
        # produce animation and save
        fps = 50
        if 'fps' in kwargs:
            fps = kwargs['fps']
        anim.save(savepath, fps=fps, extra_args=['-vcodec', 'libx264'])
        
        clear_output()   
    
    ##### draw boundary #####
    def draw_fit(self,ax,run,w,train_valid):
        ### create boundary data ###
        # get visual boundary
        xmin1 = np.min(copy.deepcopy(self.x))
        xmax1 = np.max(copy.deepcopy(self.x))
        xgap1 = (xmax1 - xmin1)*0.1
        xmin1 -= xgap1
        xmax1 += xgap1 
        
        ymin1 = np.min(copy.deepcopy(self.y))
        ymax1 = np.max(copy.deepcopy(self.y))
        ygap1 = (ymax1 - ymin1)*0.3
        ymin1 -= ygap1
        ymax1 += ygap1 
        
        # plot boundary for 2d plot
        s = np.linspace(xmin1,xmax1,300)[np.newaxis,:]
        
        # plot total fit
        cost = run.cost
        model = run.model
        normalizer = run.normalizer
        t = model(normalizer(s),w)
        
        #### plot contour, color regions ####
        ax.plot(s.flatten(),t.flatten(),c = 'magenta',linewidth = 2.5,zorder = 3)  
        ax.set_xlim([xmin1,xmax1])
        ax.set_ylim([ymin1,ymax1])
        
    ######## show N = 2 static image ########
    # show coloring of entire space
    def draw_data(self,ax,w_best,runner,train_valid):
        cost = runner.cost
        predict = runner.model
        feat = runner.feature_transforms
        normalizer = runner.normalizer
        inverse_nornalizer = runner.inverse_normalizer
      
        # or just take last weights
        self.w = w_best
        
        ### create boundary data ###
        xmin1 = np.min(copy.deepcopy(self.x))
        xmax1 = np.max(copy.deepcopy(self.x))
        xgap1 = (xmax1 - xmin1)*0.05
        xmin1 -= xgap1
        xmax1 += xgap1

        ### loop over two panels plotting each ###
        # plot training points
        if train_valid == 'train':
            # reverse normalize data
            x_train = inverse_nornalizer(runner.x_train).T
            y_train = runner.y_train
            
            # plot data
            ax.scatter(x_train,y_train,s = 45, color = [0,0.7,1], edgecolor = 'k',linewidth = 1,zorder = 3)
            ax.set_title('training data',fontsize = 15)

        if train_valid == 'validate':
            # reverse normalize data
            x_valid = inverse_nornalizer(runner.x_valid).T
            y_valid = runner.y_valid
        
            # plot testing points
            ax.scatter(x_valid,y_valid,s = 45, color = [1,0.8,0.5], edgecolor = 'k',linewidth = 1,zorder = 3)

            ax.set_title('validation data',fontsize = 15)
                
        if train_valid == 'original':
            # plot all points
            ax.scatter(self.x,self.y,s = 55, color = 'k', edgecolor = 'w',linewidth = 1,zorder = 3)
            ax.set_title('original data',fontsize = 15)

        # cleanup panel
        ax.set_xlabel(r'$x_1$',fontsize = 15)
        ax.set_ylabel(r'$x_2$',fontsize = 15,rotation = 0,labelpad = 20)
        ax.xaxis.set_major_formatter(FormatStrFormatter('%.1f'))
        ax.yaxis.set_major_formatter(FormatStrFormatter('%.1f'))
        
    def plot_train_valid_errors(self,ax,k,train_errors,valid_errors,num_units):
        num_elements = np.arange(len(train_errors))

        ax.plot([v+1 for v in num_elements[:k+1]] ,train_errors[:k+1],color = [0,0.7,1],linewidth = 2.5,zorder = 1,label = 'training')
        #ax.scatter([v+1  for v in num_elements[:k+1]] ,train_errors[:k+1],color = [0,0.7,1.0],s = 70,edgecolor = 'w',linewidth = 1.5,zorder = 3)

        ax.plot([v+1  for v in num_elements[:k+1]] ,valid_errors[:k+1],color = [1,0.8,0.5],linewidth = 2.5,zorder = 1,label = 'validation')
        #ax.scatter([v+1  for v in num_elements[:k+1]] ,valid_errors[:k+1],color= [1,0.8,0.5],s = 70,edgecolor = 'w',linewidth = 1.5,zorder = 3)
        ax.set_title('errors',fontsize = 15)

        # cleanup
        ax.set_xlabel('step',fontsize = 12)

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
        
        tics = np.arange(1,len(num_elements)+1 + len(num_elements)/float(5),len(num_elements)/float(5))
        labels = np.arange(1,num_units+1 + num_units/float(5),num_units/float(5))
        labels = [int(np.around(v,decimals=-1)) for v in labels]

        ax.xaxis.set_major_locator(MaxNLocator(integer=True))
        ax.set_xticks(tics)
        ax.set_xticklabels(labels)


        