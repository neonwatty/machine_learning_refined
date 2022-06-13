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
        data = data.T
        self.x = data[:,:-1]
        self.y = data[:,-1]

        self.colors = ['salmon','cornflowerblue','lime','bisque','mediumaquamarine','b','m','g']
        
    # the counting cost function
    def counting_cost(self,run,x,y,w):
        cost = np.sum((np.sign(model(x,w)) -y)**2)
        return 0.25*cost 
    
    #### animate multiple runs on single regression ####
    def animate_crossval_classifications(self,savepath,runs,**kwargs):
        weight_history = []
        train_errors = []
        valid_errors = []
        for run in runs:
            # get histories
            train_counts = run.train_count_histories[0]
            valid_counts = run.valid_count_histories[0]
            weights = run.weight_histories[0]
            
            # select based on minimum training
            ind = np.argmin(train_counts)
            train_count = train_counts[ind]
            valid_count = valid_counts[ind]
            weight = weights[ind]
            
            # store
            train_errors.append(train_count)
            valid_errors.append(valid_count)
            weight_history.append(weight)
       
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
        num_frames = len(runs)        
        print ('starting animation rendering...')
        def animate(k):
            print (k)
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
            w_best = weight_history[k]
            run = runs[k]
            
            # produce static img
            self.static_N2_simple(ax1,w_best,run,train_valid = 'original')
            self.static_N2_simple(ax2,w_best,run,train_valid = 'train')
            self.static_N2_simple(ax3,w_best,run,train_valid = 'validate')

            #### plot training / validation errors ####
            self.plot_train_valid_errors(ax4,k,train_errors,valid_errors)
            return artist,

        anim = animation.FuncAnimation(fig, animate ,frames=num_frames, interval=num_frames, blit=True)
        
        # produce animation and save
        fps = 50
        if 'fps' in kwargs:
            fps = kwargs['fps']
        anim.save(savepath, fps=fps, extra_args=['-vcodec', 'libx264'])
        clear_output()        
    

    def plot_train_valid_errors(self,ax,k,train_errors,valid_errors):
        num_elements = np.arange(len(train_errors))

        ax.plot([v+1 for v in num_elements[:k+1]] ,train_errors[:k+1],color = [0,0.7,1],linewidth = 1.5,zorder = 1,label = 'training')
        ax.scatter([v+1  for v in num_elements[:k+1]] ,train_errors[:k+1],color = [0,0.7,1],s = 70,edgecolor = 'w',linewidth = 1.5,zorder = 3)

        ax.plot([v+1  for v in num_elements[:k+1]] ,valid_errors[:k+1],color = [1,0.8,0.5],linewidth = 1.5,zorder = 1,label = 'validation')
        ax.scatter([v+1  for v in num_elements[:k+1]] ,valid_errors[:k+1],color= [1,0.8,0.5],s = 70,edgecolor = 'w',linewidth = 1.5,zorder = 3)
        ax.set_title('misclassifications',fontsize = 15)

        # cleanup
        ax.set_xlabel('model',fontsize = 12)

        # cleanp panel                
        num_iterations = len(train_errors)
        minxc = 0.5
        maxxc = len(num_elements) + 0.5
        minc = min(min(copy.deepcopy(train_errors)),min(copy.deepcopy(valid_errors)))
        maxc = max(max(copy.deepcopy(train_errors[:])),max(copy.deepcopy(valid_errors[:])))
        gapc = (maxc - minc)*0.1
        minc -= gapc
        maxc += gapc
        
        ax.set_xlim([minxc,maxxc])
        ax.set_ylim([minc,maxc])
        
        #ax.xaxis.set_major_locator(MaxNLocator(integer=True))
        #labels = [str(v) for v in num_elements]
        ax.set_xticks(np.arange(1,len(num_elements)+1))
        #ax.set_xticklabels(num_units)

            
    ######## show N = 2 static image ########
    # show coloring of entire space
    def static_N2_simple(self,ax,w_best,runner,train_valid):
        cost = runner.cost
        predict = runner.model
        feat = runner.feature_transforms
        normalizer = runner.normalizer
        inverse_nornalizer = runner.inverse_normalizer
      
        # or just take last weights
        self.w = w_best
        
        ### create boundary data ###
        xmin1 = np.min(self.x[:,0])
        xmax1 = np.max(self.x[:,0])
        xgap1 = (xmax1 - xmin1)*0.05
        xmin1 -= xgap1
        xmax1 += xgap1

        xmin2 = np.min(self.x[:,1])
        xmax2 = np.max(self.x[:,1])
        xgap2 = (xmax2 - xmin2)*0.05
        xmin2 -= xgap2
        xmax2 += xgap2    

        # plot boundary for 2d plot
        r1 = np.linspace(xmin1,xmax1,300)
        r2 = np.linspace(xmin2,xmax2,300)
        s,t = np.meshgrid(r1,r2)
        s = np.reshape(s,(np.size(s),1))
        t = np.reshape(t,(np.size(t),1))
        h = np.concatenate((s,t),axis = 1)
        
        # compute model on train data
        z1 = predict(normalizer(h.T),self.w)
        z1 = np.sign(z1)

        # reshape it
        s.shape = (np.size(r1),np.size(r2))
        t.shape = (np.size(r1),np.size(r2))     
        z1.shape = (np.size(r1),np.size(r2))

        ### loop over two panels plotting each ###
        # plot training points
        if train_valid == 'train':
            # reverse normalize data
            x_train = inverse_nornalizer(runner.x_train).T
            y_train = runner.y_train
            
            # plot data
            ind0 = np.argwhere(y_train == +1)
            ind0 = [v[1] for v in ind0]
            ax.scatter(x_train[ind0,0],x_train[ind0,1],s = 45, color = self.colors[0], edgecolor = [0,0.7,1],linewidth = 1,zorder = 3)

            ind1 = np.argwhere(y_train == -1)
            ind1 = [v[1] for v in ind1]
            ax.scatter(x_train[ind1,0],x_train[ind1,1],s = 45, color = self.colors[1], edgecolor = [0,0.7,1],linewidth = 1,zorder = 3)
            ax.set_title('training data',fontsize = 15)

        if train_valid == 'validate':
            # reverse normalize data
            x_valid = inverse_nornalizer(runner.x_valid).T
            y_valid = runner.y_valid
        
            # plot testing points
            ind0 = np.argwhere(y_valid == +1)
            ind0 = [v[1] for v in ind0]
            ax.scatter(x_valid[ind0,0],x_valid[ind0,1],s = 45, color = self.colors[0], edgecolor = [1,0.8,0.5],linewidth = 1,zorder = 3)

            ind1 = np.argwhere(y_valid == -1)
            ind1 = [v[1] for v in ind1]
            ax.scatter(x_valid[ind1,0],x_valid[ind1,1],s = 45, color = self.colors[1], edgecolor = [1,0.8,0.5],linewidth = 1,zorder = 3)
            ax.set_title('validation data',fontsize = 15)
                
        if train_valid == 'original':
            # plot all points
            ind0 = np.argwhere(self.y == +1)
            ax.scatter(self.x[ind0,0],self.x[ind0,1],s = 55, color = self.colors[0], edgecolor = 'k',linewidth = 1,zorder = 3)

            ind1 = np.argwhere(self.y == -1)
            ax.scatter(self.x[ind1,0],self.x[ind1,1],s = 55, color = self.colors[1], edgecolor = 'k',linewidth = 1,zorder = 3)
            ax.set_title('original data',fontsize = 15)

        #### plot contour, color regions ####
        ax.contour(s,t,z1,colors='k', linewidths=2.5,levels = [0],zorder = 2)
        ax.contourf(s,t,z1,colors = [self.colors[1],self.colors[0]],alpha = 0.15,levels = range(-1,2))

        # cleanup panel
        ax.set_xlabel(r'$x_1$',fontsize = 15)
        ax.set_ylabel(r'$x_2$',fontsize = 15,rotation = 0,labelpad = 20)
        ax.xaxis.set_major_formatter(FormatStrFormatter('%.1f'))
        ax.yaxis.set_major_formatter(FormatStrFormatter('%.1f'))
        
      
    
    
    
   
    
    # toy plot
    def static_MULTI_simple(self,run,w,**kwargs):
        model = run.model
        normalizer = run.normalizer
        feat = run.feature_transforms
        
        # grab args
        view = [20,-70]
        if 'view' in kwargs:
            view = kwargs['view']
 
        ### plot all input data ###
        # generate input range for functions
        minx = min(min(self.x[:,0]),min(self.x[:,1]))
        maxx = max(max(self.x[:,0]),max(self.x[:,1]))
        gapx = (maxx - minx)*0.1
        minx -= gapx
        maxx += gapx

        r = np.linspace(minx,maxx,600)
        w1_vals,w2_vals = np.meshgrid(r,r)
        w1_vals.shape = (len(r)**2,1)
        w2_vals.shape = (len(r)**2,1)
        h = np.concatenate([w1_vals,w2_vals],axis = 1).T

        g_vals = model(normalizer(h),w)
        g_vals = np.asarray(g_vals)
        g_new = copy.deepcopy(g_vals).T
        g_vals = np.argmax(g_vals,axis = 0)

        # vals for cost surface
        w1_vals.shape = (len(r),len(r))
        w2_vals.shape = (len(r),len(r))
        g_vals.shape = (len(r),len(r))

        # count points
        class_nums = np.unique(self.y)
        C = int(len(class_nums))
        
        fig = plt.figure(figsize = (10,7))
        gs = gridspec.GridSpec(2, C) 

        #### left plot - data and fit in original space ####
        # setup current axis
        #ax1 = plt.subplot(gs[C],projection = '3d');
        ax2 = plt.subplot(gs[C+1],aspect = 'equal');
        fig.subplots_adjust(left=0,right=1,bottom=0,top=1)   # remove whitespace around 3d figure

        ##### plot top panels ####
        for d in range(C):
            # create panel
            ax = plt.subplot(gs[d],aspect = 'equal');
                       
            for c in range(C):
                # plot points
                ind = np.argwhere(self.y == class_nums[c])
                ind = [v[0] for v in ind]
                ax.scatter(self.x[ind,0],self.x[ind,1],s = 50,color = self.colors[c],edgecolor = 'k', linewidth = 2)
            
            g_2 = np.sign(g_new[:,d])
            g_2.shape = (len(r),len(r))

            # plot separator curve 
            ax.contour(w1_vals,w2_vals,g_2+1,colors = 'k',levels = [-1,1],linewidths = 4.5,zorder = 1,linestyle = '-')
            ax.contour(w1_vals,w2_vals,g_2+1,colors = self.colors[d],levels = [-1,1],linewidths = 2.5,zorder = 1,linestyle = '-')
                
            ax.set_xlabel(r'$x_1$', fontsize = 18,labelpad = 10)
            ax.set_ylabel(r'$x_2$', rotation = 0,fontsize = 18,labelpad = 15)
        
        ##### plot bottom panels ###
        # scatter points in both bottom panels
        for c in range(C):
            ind = np.argwhere(self.y == class_nums[c])
            ind = [v[0] for v in ind]
            #ax1.scatter(self.x[ind,0],self.x[ind,1],self.y[ind],s = 50,color = self.colors[c],edgecolor = 'k',linewidth = 1.5)
            ax2.scatter(self.x[ind,0],self.x[ind,1],s = 50,color = self.colors[c],edgecolor = 'k', linewidth = 2)
      
        #ax1.plot_surface(w1_vals,w2_vals,g_vals,alpha = 0.1,color = 'w',rstride=45, cstride=45,linewidth=0.25,edgecolor = 'k')

        #for c in range(C):
            # plot separator curve in left plot z plane
            #ax1.contour(w1_vals,w2_vals,g_vals - c,colors = 'k',levels = [0],linewidths = 3,zorder = 1)

            # color parts of plane with correct colors
            #ax1.contourf(w1_vals,w2_vals,g_vals - c +0.5,colors = self.colors[c],alpha = 0.4,levels = [0,1])
             
        # plot separator in right plot
        ax2.contour(w1_vals,w2_vals,g_vals,colors = 'k',levels = range(0,C+1),linewidths = 3,zorder = 1)

        # adjust height of regressor to plot filled contours
        ax2.contourf(w1_vals,w2_vals,g_vals+0.5,colors = self.colors[:],alpha = 0.2,levels = range(0,C+1))
        

    #### animate multiple runs on single regression ####
    def animate_multiclass_crossval_classifications(self,runs,**kwargs):
        weight_history = []
        train_errors = []
        valid_errors = []
        for run in runs:
            # get histories
            train_counts = run.train_count_histories[0]
            valid_counts = run.valid_count_histories[0]
            weights = run.weight_histories[0]
            
            # select based on minimum training
            ind = np.argmin(train_counts)
            train_count = train_counts[ind]
            valid_count = valid_counts[ind]
            weight = weights[ind]
            
            # store
            train_errors.append(train_count)
            valid_errors.append(valid_count)
            weight_history.append(weight)
       
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
        num_frames = len(runs)        
        print ('starting animation rendering...')
        def animate(k):
            print (k)
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
            w_best = weight_history[k]
            run = runs[k]
            
            # produce static img
            self.static_MULTI_simple(ax1,w_best,run,train_valid = 'original')
            self.static_MULTI_simple(ax2,w_best,run,train_valid = 'train')
            self.static_MULTI_simple(ax3,w_best,run,train_valid = 'validate')

            #### plot training / validation errors ####
            self.plot_train_valid_errors(ax4,k,train_errors,valid_errors)
            return artist,

        anim = animation.FuncAnimation(fig, animate ,frames=num_frames, interval=num_frames, blit=True)
        
        return(anim)

  
     
    
    
    ######## show N = 2 static image ########
    # show coloring of entire space
    def static_MULTI_simple(self,ax,w_best,runner,train_valid):
        cost = runner.cost
        predict = runner.model
        feat = runner.feature_transforms
        normalizer = runner.normalizer
        inverse_nornalizer = runner.inverse_normalizer
      
        # or just take last weights
        self.w = w_best
        
        ### create boundary data ###
        xmin1 = np.min(self.x[:,0])
        xmax1 = np.max(self.x[:,0])
        xgap1 = (xmax1 - xmin1)*0.05
        xmin1 -= xgap1
        xmax1 += xgap1

        xmin2 = np.min(self.x[:,1])
        xmax2 = np.max(self.x[:,1])
        xgap2 = (xmax2 - xmin2)*0.05
        xmin2 -= xgap2
        xmax2 += xgap2    

        # plot boundary for 2d plot
        r1 = np.linspace(xmin1,xmax1,800)
        r2 = np.linspace(xmin2,xmax2,800)
        s,t = np.meshgrid(r1,r2)
        s = np.reshape(s,(np.size(s),1))
        t = np.reshape(t,(np.size(t),1))
        h = np.concatenate((s,t),axis = 1)
        
        # compute model on train data
        C = len(np.unique(self.y))
   
        z1 = predict(normalizer(h.T),self.w)
        z1 = np.asarray(z1)
        z1 = np.argmax(z1,axis = 0)

        # reshape it
        s.shape = (np.size(r1),np.size(r2))
        t.shape = (np.size(r1),np.size(r2))     
        z1.shape = (np.size(r1),np.size(r2))

        ### loop over two panels plotting each ###
        # plot training points
        if train_valid == 'train':
            # reverse normalize data
            x_train = inverse_nornalizer(runner.x_train).T
            y_train = runner.y_train
            
            # scatter data
            for c in range(C):
                # plot points
                ind0 = np.argwhere(y_train == c)
                ind0 = [v[1] for v in ind0]
                ax.scatter(x_train[ind0,0],x_train[ind0,1],s = 45,color = self.colors[c], edgecolor = [0,0.7,1],linewidth = 1,zorder = 3)     
            
            ax.set_title('training data',fontsize = 15)

        if train_valid == 'validate':
            # reverse normalize data
            x_valid = inverse_nornalizer(runner.x_valid).T
            y_valid = runner.y_valid
            
            # scatter data
            for c in range(C):
                # plot points
                ind0 = np.argwhere(y_valid == c)
                ind0 = [v[1] for v in ind0]
                ax.scatter(x_valid[ind0,0],x_valid[ind0,1],s = 45,color = self.colors[c], edgecolor = [1,0.8,0.5],linewidth = 1,zorder = 3)
 
            ax.set_title('validation data',fontsize = 15)
                
        if train_valid == 'original':
            
            # scatter data
            for c in range(C):
                # plot points
                ind0 = np.argwhere(self.y == c)
                ind0 = [v[0] for v in ind0]
                ax.scatter(self.x[ind0,0],self.x[ind0,1],s = 55,color = self.colors[c], edgecolor = 'k',linewidth = 1,zorder = 3)

            ax.set_title('original data',fontsize = 15)
        
        #### plot contour, color regions ####
        for c in range(C):
            # plot separator curve 
            ax.contour(s,t,z1+1,colors = 'k',levels = [-1,1],linewidths = 1.5,zorder = 1,linestyle = '-')
            #ax.contour(s,t,z1+1,colors = self.colors[c],levels = [-1,1],linewidths = 2.5,zorder = 1,linestyle = '-')
                
        # plot separator in right plot
        ax.contour(s,t,z1,colors = 'k',levels = range(0,C+1),linewidths = 3,zorder = 1)
            
        # adjust height of regressor to plot filled contours
        ax.contourf(s,t,z1+0.5,colors = self.colors[:],alpha = 0.2,levels = range(0,C+1))
        
      

        
        
        
        
        
        
        
        # cleanup panel
        ax.set_xlabel(r'$x_1$',fontsize = 15)
        ax.set_ylabel(r'$x_2$',fontsize = 15,rotation = 0,labelpad = 20)
        ax.xaxis.set_major_formatter(FormatStrFormatter('%.1f'))
        ax.yaxis.set_major_formatter(FormatStrFormatter('%.1f'))        