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
    def animate_trainval_boosting(self,savepath,runner,num_frames,**kwargs):        
        # select subset of runs
        inds = np.arange(0,len(runner.models),int(len(runner.models)/float(num_frames)))
        labels = np.arange(0,len(runner.models),int(len(runner.models)/float(5)))

        # global names for train / valid sets
        train_inds = runner.train_inds
        valid_inds = runner.valid_inds
        
        self.x_train = self.x[:,train_inds]
        self.y_train = self.y[:,train_inds]
        
        self.x_valid = self.x[:,valid_inds]
        self.y_valid = self.y[:,valid_inds]
        
        self.normalizer = runner.normalizer
        train_errors = runner.train_count_vals
        valid_errors = runner.valid_count_vals
        num_units = len(runner.models)
       
        # construct figure
        fig = plt.figure(figsize = (6,6))
        artist = fig

        # create subplot with 3 panels, plot input function in center plot
        gs = gridspec.GridSpec(2, 2)
        ax1 = plt.subplot(gs[0]); 
        ax2 = plt.subplot(gs[1]); 
        ax3 = plt.subplot(gs[2]); 
        ax4 = plt.subplot(gs[3]);
        
        # start animation
        num_frames = len(inds) -2       
        print ('starting animation rendering...')
        def animate(k):
            # clear panels
            ax1.cla()
            ax2.cla()
            ax3.cla()
            ax4.cla()
            
            # print rendering update
            if np.mod(k+1,25) == 0:
                print ('rendering animation frame ' + str(k+1) + ' of ' + str(num_frames))
            if k == num_frames - 1:
                print ('animation rendering complete!')
                time.sleep(1.5)
                clear_output()
            
            #### plot training, testing, and full data ####            
            # pluck out current weights 
            a = inds[k]
            steps = runner.best_steps[:a+1]
            
            # produce static img
            self.draw_boundary(ax1,steps,a)
            self.static_N2_simple(ax1,train_valid = 'original')
            self.draw_boundary(ax2,steps,a)

            self.static_N2_simple(ax2,train_valid = 'train')
            self.draw_boundary(ax3,steps,a)
            self.static_N2_simple(ax3,train_valid = 'validate')

            #### plot training / validation errors ####
            self.plot_train_valid_errors(ax4,a,train_errors,valid_errors,inds)
            return artist,

        anim = animation.FuncAnimation(fig, animate ,frames=num_frames, interval=num_frames, blit=True)
        
        # produce animation and save
        fps = 50
        if 'fps' in kwargs:
            fps = kwargs['fps']
        anim.save(savepath, fps=fps, extra_args=['-vcodec', 'libx264'])
        clear_output()   
        
    ##### draw boundary #####
    def draw_boundary(self,ax,steps,ind):
        ### create boundary data ###
        # get visual boundary
        xmin1 = np.min(self.x[0,:])
        xmax1 = np.max(self.x[0,:])
        xgap1 = (xmax1 - xmin1)*0.05
        xmin1 -= xgap1
        xmax1 += xgap1

        xmin2 = np.min(self.x[1,:])
        xmax2 = np.max(self.x[1,:])
        xgap2 = (xmax2 - xmin2)*0.05
        xmin2 -= xgap2
        xmax2 += xgap2    
        
        # plot boundary for 2d plot
        r1 = np.linspace(xmin1,xmax1,300)
        r2 = np.linspace(xmin2,xmax2,300)
        s,t = np.meshgrid(r1,r2)
        s = np.reshape(s,(np.size(s),1))
        t = np.reshape(t,(np.size(t),1))
        h = np.concatenate((s,t),axis = 1).T
        
        # plot total fit
        model = lambda x: np.sum([v(x) for v in steps],axis=0)
        z = model(self.normalizer(h))
        z1 = np.sign(z)        

        # reshape it
        s.shape = (np.size(r1),np.size(r2))
        t.shape = (np.size(r1),np.size(r2))     
        z1.shape = (np.size(r1),np.size(r2))
        
        #### plot contour, color regions ####
        ax.contour(s,t,z1,colors='k', linewidths=2.5,levels = [0],zorder = 2)
        ax.contourf(s,t,z1,colors = [self.colors[1],self.colors[0]],alpha = 0.15,levels = range(-1,2))
        
    ######## show N = 2 static image ########
    # show coloring of entire space
    def static_N2_simple(self,ax,train_valid):        
        ### create boundary data ###
        xmin1 = np.min(self.x[0,:])
        xmax1 = np.max(self.x[0,:])
        xgap1 = (xmax1 - xmin1)*0.05
        xmin1 -= xgap1
        xmax1 += xgap1

        xmin2 = np.min(self.x[1,:])
        xmax2 = np.max(self.x[1,:])
        xgap2 = (xmax2 - xmin2)*0.05
        xmin2 -= xgap2
        xmax2 += xgap2    

        ### loop over two panels plotting each ###
        # plot training points
        if train_valid == 'train':
            # reverse normalize data
            x_train = self.x_train
            y_train = self.y_train
            
            # plot data
            ind0 = np.argwhere(y_train == +1)
            ind0 = [v[1] for v in ind0]
            ax.scatter(x_train[0,ind0],x_train[1,ind0],s = 45, color = self.colors[0], edgecolor = [0,0.7,1],linewidth = 1,zorder = 3)

            ind1 = np.argwhere(y_train == -1)
            ind1 = [v[1] for v in ind1]
            ax.scatter(x_train[0,ind1],x_train[1,ind1],s = 45, color = self.colors[1], edgecolor = [0,0.7,1],linewidth = 1,zorder = 3)
            ax.set_title('training data',fontsize = 15)

        if train_valid == 'validate':
            # reverse normalize data
            x_valid = self.x_valid
            y_valid = self.y_valid
        
            # plot testing points
            ind0 = np.argwhere(y_valid == +1)
            ind0 = [v[1] for v in ind0]
            ax.scatter(x_valid[0,ind0],x_valid[1,ind0],s = 45, color = self.colors[0], edgecolor = [1,0.8,0.5],linewidth = 1,zorder = 3)

            ind1 = np.argwhere(y_valid == -1)
            ind1 = [v[1] for v in ind1]
            ax.scatter(x_valid[0,ind1],x_valid[1,ind1],s = 45, color = self.colors[1], edgecolor = [1,0.8,0.5],linewidth = 1,zorder = 3)
            ax.set_title('validation data',fontsize = 15)
                
        if train_valid == 'original':
            # plot all points
            ind0 = np.argwhere(self.y == +1)
            ind0 = [v[1] for v in ind0]
            ax.scatter(self.x[0,ind0],self.x[1,ind0],s = 55, color = self.colors[0], edgecolor = 'k',linewidth = 1,zorder = 3)

            ind1 = np.argwhere(self.y == -1)
            ind1 = [v[1] for v in ind1]
            ax.scatter(self.x[0,ind1],self.x[1,ind1],s = 55, color = self.colors[1], edgecolor = 'k',linewidth = 1,zorder = 3)
            ax.set_title('original data',fontsize = 15)

        # cleanup panel
        ax.set_xlabel(r'$x_1$',fontsize = 15)
        ax.set_ylabel(r'$x_2$',fontsize = 15,rotation = 0,labelpad = 20)
        ax.xaxis.set_major_formatter(FormatStrFormatter('%.1f'))
        ax.yaxis.set_major_formatter(FormatStrFormatter('%.1f'))
        
    def plot_train_valid_errors(self,ax,k,train_errors,valid_errors,inds):
        num_elements = np.arange(len(train_errors))

        ax.plot([v+1 for v in inds[:k+1]] ,[train_errors[v] for v in inds[:k+1]],color = [0,0.7,1],linewidth = 2.5,zorder = 1,label = 'training')
        # ax.scatter([v+1  for v in inds[:k+1]] ,train_errors[:k+1],color = [0,0.7,1],s = 70,edgecolor = 'w',linewidth = 1.5,zorder = 3)

        ax.plot([v+1  for v in inds[:k+1]] ,[valid_errors[v] for v in inds[:k+1]],color = [1,0.8,0.5],linewidth = 2.5,zorder = 1,label = 'validation')
        # ax.scatter([v+1  for v in inds[:k+1]] ,valid_errors[:k+1],color= [1,0.8,0.5],s = 70,edgecolor = 'w',linewidth = 1.5,zorder = 3)
        ax.set_title('misclassifications',fontsize = 15)

        # cleanup
        ax.set_xlabel('number of units',fontsize = 12)
        ax.set_ylabel('number of misclassifications',fontsize = 12)

        # cleanp panel                
        num_iterations = len(train_errors)
        minxc = 0.5
        maxxc = len(num_elements) + 0.5
        minc = min(min(copy.deepcopy(train_errors)),min(copy.deepcopy(valid_errors)))
        maxc = max(max(copy.deepcopy(train_errors[:10])),max(copy.deepcopy(valid_errors[:10])))
        gapc = (maxc - minc)*0.25
        minc -= gapc
        maxc += gapc
        
        ax.set_xlim([minxc- 0.5,maxxc + 0.5])
        ax.set_ylim([minc,maxc])
        
        # ax.xaxis.set_major_locator(MaxNLocator(integer=True))
        #labels = [str(v) for v in num_units]
        #ax.set_xticks(np.arange(1,len(num_elements)+1))
       # ax.set_xticklabels(num_units)


        