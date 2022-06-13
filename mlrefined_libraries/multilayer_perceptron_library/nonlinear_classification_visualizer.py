# import standard plotting and animation
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from mpl_toolkits.mplot3d import Axes3D
from IPython.display import clear_output

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
    Visualize linear regression in 2 and 3 dimensions.  For single input cases (2 dimensions) the path of gradient descent on the cost function can be animated.
    '''
    #### initialize ####
    def __init__(self,csvname):
        # grab input
        data = np.loadtxt(csvname,delimiter = ',')
        data = data.T
        self.x = data[:,:-1]
        self.y = data[:,-1]

        self.colors = ['salmon','cornflowerblue','lime','bisque','mediumaquamarine','b','m','g']
    
    ######## show N = 1 static image ########
    # show coloring of entire space
    def static_N1_img(self,w_best,cost,predict,**kwargs):
        # or just take last weights
        self.w = w_best
        
        # initialize figure
        fig = plt.figure(figsize = (9,4))
        
        show_cost = False
        if show_cost == True:   
            gs = gridspec.GridSpec(1, 3,width_ratios = [1,1,1],height_ratios = [1]) 
            
            # create third panel for cost values
            ax3 = plt.subplot(gs[2],aspect = 'equal')
            
        else:
            gs = gridspec.GridSpec(1, 2,width_ratios = [1,1]) 

        #### left plot - data and fit in original space ####
        # setup current axis
        ax = plt.subplot(gs[0]);
        ax2 = plt.subplot(gs[1],aspect = 'equal');
        
        # scatter original points
        self.scatter_pts(ax,self.x)
        ax.set_xlabel(r'$x$', fontsize = 14,labelpad = 10)
        ax.set_ylabel(r'$y$', rotation = 0,fontsize = 14,labelpad = 10)
        
        # create fit
        gapx = (max(self.x) - min(self.x))*0.1
        s = np.linspace(min(self.x) - gapx,max(self.x) + gapx,100)
        t = [np.tanh(predict(np.asarray([v]),self.w)) for v in s]
        
        # plot fit
        ax.plot(s,t,c = 'lime')
        ax.axhline(linewidth=0.5, color='k',zorder = 1)

        #### plot data in new space in middle panel (or right panel if cost function decrease plot shown ) #####
        if 'f_x' in kwargs:
            f_x = kwargs['f_x']

            # scatter points
            self.scatter_pts(ax2,f_x)

            # create and plot fit
            s = np.linspace(min(f_x) - 0.1,max(f_x) + 0.1,100)
            t = np.tanh(self.w[0] + self.w[1]*s)
            ax2.plot(s,t,c = 'lime')
            ax2.set_xlabel(r'$f\,(x)$', fontsize = 14,labelpad = 10)
            ax2.set_ylabel(r'$y$', rotation = 0,fontsize = 14,labelpad = 10)
        
        if 'f2_x' in kwargs:
            ax2 = plt.subplot(gs[1],projection = '3d');   
            view = kwargs['view']
            
            # get input
            f1_x = kwargs['f1_x']
            f2_x = kwargs['f2_x']

            # scatter points
            f1_x = np.asarray(f1_x)
            f1_x.shape = (len(f1_x),1)
            f2_x = np.asarray(f2_x)
            f2_x.shape = (len(f2_x),1)
            xtran = np.concatenate((f1_x,f2_x),axis = 1)
            self.scatter_pts(ax2,xtran)

            # create and plot fit
            s1 = np.linspace(min(f1_x) - 0.1,max(f1_x) + 0.1,100)
            s2 = np.linspace(min(f2_x) - 0.1,max(f2_x) + 0.1,100)
            t1,t2 = np.meshgrid(s1,s2)
            
            # compute fitting hyperplane
            t1.shape = (len(s1)**2,1)
            t2.shape = (len(s2)**2,1)
            r = np.tanh(self.w[0] + self.w[1]*t1 + self.w[2]*t2)
            
            # reshape for plotting
            t1.shape = (len(s1),len(s1))
            t2.shape = (len(s2),len(s2))
            r.shape = (len(s1),len(s2))
            ax2.plot_surface(t1,t2,r,alpha = 0.1,color = 'lime',rstride=10, cstride=10,linewidth=0.5,edgecolor = 'k')
                
            # label axes
            self.move_axis_left(ax2)
            ax2.set_xlabel(r'$f_1(x)$', fontsize = 12,labelpad = 5)
            ax2.set_ylabel(r'$f_2(x)$', rotation = 0,fontsize = 12,labelpad = 5)
            ax2.set_zlabel(r'$y$', rotation = 0,fontsize = 12,labelpad = -3)
            ax2.view_init(view[0],view[1])
            
        # plot cost function decrease
        if  show_cost == True: 
            # compute cost eval history
            g = cost
            cost_evals = []
            for i in range(len(w_hist)):
                W = w_hist[i]
                cost = g(W)
                cost_evals.append(cost)
     
            # plot cost path - scale to fit inside same aspect as classification plots
            num_iterations = len(w_hist)
            minx = min(self.x)
            maxx = max(self.x)
            gapx = (maxx - minx)*0.1
            minc = min(cost_evals)
            maxc = max(cost_evals)
            gapc = (maxc - minc)*0.1
            minc -= gapc
            maxc += gapc
            
            s = np.linspace(minx + gapx,maxx - gapx,num_iterations)
            scaled_costs = [c/float(max(cost_evals))*(maxx-gapx) - (minx+gapx) for c in cost_evals]
            ax3.plot(s,scaled_costs,color = 'k',linewidth = 1.5)
            ax3.set_xlabel('iteration',fontsize = 12)
            ax3.set_title('cost function plot',fontsize = 12)
            
            # rescale number of iterations and cost function value to fit same aspect ratio as other two subplots
            ax3.set_xlim(minx,maxx)
            #ax3.set_ylim(minc,maxc)
            
            ### set tickmarks for both axes - requries re-scaling   
            # x axis
            marks = range(0,num_iterations,round(num_iterations/5.0))
            ax3.set_xticks(s[marks])
            labels = [item.get_text() for item in ax3.get_xticklabels()]
            ax3.set_xticklabels(marks)
            
            ### y axis
            r = (max(scaled_costs) - min(scaled_costs))/5.0
            marks = [min(scaled_costs) + m*r for m in range(6)]
            ax3.set_yticks(marks)
            labels = [item.get_text() for item in ax3.get_yticklabels()]
            
            r = (max(cost_evals) - min(cost_evals))/5.0
            marks = [int(min(cost_evals) + m*r) for m in range(6)]
            ax3.set_yticklabels(marks)

    ######## show N = 2 static image ########
    # show coloring of entire space
    def static_N2_img(self,w_best,runner,**kwargs):
        cost = runner.cost
        predict = runner.model
        feat = runner.feature_transforms
        normalizer = runner.normalizer
                
        # count parameter layers of input to feature transform
        sig = signature(feat)
        sig = len(sig.parameters)

        # or just take last weights
        self.w = w_best
        
        zplane = 'on'
        if 'zplane' in kwargs:
            zplane = kwargs['zplane']
        view1 = [20,45]
        if 'view1' in kwargs:
            view1 = kwargs['view1']
        view2 = [20,30]
        if 'view2' in kwargs:
            view2 = kwargs['view2']  
            
        # initialize figure
        fig = plt.figure(figsize = (10,9))
        gs = gridspec.GridSpec(2, 2,width_ratios = [1,1]) 

        #### left plot - data and fit in original space ####
        # setup current axis
        ax = plt.subplot(gs[0],aspect = 'equal');
        ax2 = plt.subplot(gs[1],aspect = 'equal');
        ax3 = plt.subplot(gs[2],projection = '3d');
        ax4 = plt.subplot(gs[3],projection = '3d');
        
        ### cleanup left plots, create max view ranges ###
        xmin1 = np.min(self.x[:,0])
        xmax1 = np.max(self.x[:,0])
        xgap1 = (xmax1 - xmin1)*0.05
        xmin1 -= xgap1
        xmax1 += xgap1
        ax.set_xlim([xmin1,xmax1])
        ax3.set_xlim([xmin1,xmax1])

        xmin2 = np.min(self.x[:,1])
        xmax2 = np.max(self.x[:,1])
        xgap2 = (xmax2 - xmin2)*0.05
        xmin2 -= xgap2
        xmax2 += xgap2
        ax.set_ylim([xmin2,xmax2])
        ax3.set_ylim([xmin2,xmax2])

        ymin = np.min(self.y)
        ymax = np.max(self.y)
        ygap = (ymax - ymin)*0.05
        ymin -= ygap
        ymax += ygap
        ax3.set_zlim([ymin,ymax])
        
        ax3.axis('off')
        ax3.view_init(view1[0],view1[1])

        ax.set_yticklabels([])
        ax.set_xticklabels([])
        ax.set_xticks([])
        ax.set_yticks([])
        ax.set_xlabel(r'$x_1$',fontsize = 15)
        ax.set_ylabel(r'$x_2$',fontsize = 15,rotation = 0,labelpad = 20)
            
        #### plot left panels ####
        # plot points in 2d and 3d
        ind0 = np.argwhere(self.y == +1)
        ax.scatter(self.x[ind0,0],self.x[ind0,1],s = 55, color = self.colors[0], edgecolor = 'k')
        ax3.scatter(self.x[ind0,0],self.x[ind0,1],self.y[ind0],s = 55, color = self.colors[0], edgecolor = 'k')

        ind1 = np.argwhere(self.y == -1)
        ax.scatter(self.x[ind1,0],self.x[ind1,1],s = 55, color = self.colors[1], edgecolor = 'k')
        ax3.scatter(self.x[ind1,0],self.x[ind1,1],self.y[ind1],s = 55, color = self.colors[1], edgecolor = 'k')
       
        # plot boundary for 2d plot
        r1 = np.linspace(xmin1,xmax1,100)
        r2 = np.linspace(xmin2,xmax2,100)
        s,t = np.meshgrid(r1,r2)
        s = np.reshape(s,(np.size(s),1))
        t = np.reshape(t,(np.size(t),1))
        h = np.concatenate((s,t),axis = 1)
        z = predict(normalizer(h.T),self.w)
        z = np.tanh(z)
        
        # reshape it
        s.shape = (np.size(r1),np.size(r2))
        t.shape = (np.size(r1),np.size(r2))     
        z.shape = (np.size(r1),np.size(r2))
        
        #### plot contour, color regions ####
        ax.contour(s,t,z,colors='k', linewidths=2.5,levels = [0],zorder = 2)
        ax.contourf(s,t,z,colors = [self.colors[1],self.colors[0]],alpha = 0.15,levels = range(-1,2))
        ax3.plot_surface(s,t,z,alpha = 0.1,color = 'w',rstride=10, cstride=10,linewidth=0.5,edgecolor = 'k')

        # plot zplane = 0 in left 3d panel - showing intersection of regressor with z = 0 (i.e., its contour, the separator, in the 3d plot too)?
        if zplane == 'on':
            # plot zplane
            ax3.plot_surface(s,t,z*0,alpha = 0.1,rstride=20, cstride=20,linewidth=0.15,color = 'w',edgecolor = 'k') 
            
            # plot separator curve in left plot
            ax3.contour(s,t,z,colors = 'k',levels = [0],linewidths = 3,zorder = 1)
            ax3.contourf(s,t,z,colors = self.colors[1],levels = [0,1],zorder = 1,alpha = 0.1)
            ax3.contourf(s,t,z+1,colors = self.colors[0],levels = [0,1],zorder = 1,alpha = 0.1)
        
        #### plot right panel scatter ####
        # transform data
        f = 0 
        if sig == 1:
            f = feat(normalizer(self.x.T)).T
        else:
            f = feat(normalizer(self.x.T),self.w[0]).T
        x1 = f[:,0]
        x2 = f[:,1]
        #x1 = [f1(e) for e in self.x]
        #x2 = [f2(e) for e in self.x]
        ind0 = [v[0] for v in ind0]
        ind1 = [v[0] for v in ind1]

        # plot points on desired panel
        v1 = [x1[e] for e in ind0]
        v2 = [x2[e] for e in ind0]
        ax2.scatter(v1,v2,s = 55, color = self.colors[0], edgecolor = 'k')
        ax4.scatter(v1,v2,self.y[ind0],s = 55, color = self.colors[0], edgecolor = 'k')

        v1 = [x1[e] for e in ind1]
        v2 = [x2[e] for e in ind1]        
        ax2.scatter(v1,v2,s = 55, color = self.colors[1], edgecolor = 'k')
        ax4.scatter(v1,v2,self.y[ind1],s = 55, color = self.colors[1], edgecolor = 'k')
        
        ### cleanup right panels - making max viewing ranges ###
        
        xmin1 = np.min(x1)
        xmax1 = np.max(x1)
        xgap1 = (xmax1 - xmin1)*0.05
        xmin1 -= xgap1
        xmax1 += xgap1
        ax2.set_xlim([xmin1,xmax1])
        ax4.set_xlim([xmin1,xmax1])

        xmin2 = np.min(x2)
        xmax2 = np.max(x2)
        xgap2 = (xmax2 - xmin2)*0.05
        xmin2 -= xgap2
        xmax2 += xgap2
        
        ax2.set_yticklabels([])
        ax2.set_xticklabels([])
        ax2.set_xticks([])
        ax2.set_yticks([])
        ax2.set_xlabel(r'$f\,_1\left(\mathbf{x}\right)$',fontsize = 15)
        ax2.set_ylabel(r'$f\,_2\left(\mathbf{x}\right)$',fontsize = 15)        
        
        ### plot right panel 3d scatter ###
        #### make right plot contour ####
        r1 = np.linspace(xmin1,xmax1,100)
        r2 = np.linspace(xmin2,xmax2,100)
        s,t = np.meshgrid(r1,r2)
        
        s.shape = (1,len(r1)**2)
        t.shape = (1,len(r2)**2)
       # h = np.vstack((s,t))
       # h = feat(normalizer(h))  
       # s = h[0,:]
       # t = h[1,:]
        z = 0
        if sig == 1:
            z = self.w[0] + self.w[1]*s + self.w[2]*t
        else:
            z = self.w[1][0] + self.w[1][1]*s + self.w[1][2]*t
        z = np.tanh(np.asarray(z))
        
        s.shape = (np.size(r1),np.size(r2))
        t.shape = (np.size(r1),np.size(r2))
        z.shape = (np.size(r1),np.size(r2))

        ax2.contour(s,t,z,colors='k', linewidths=2.5,levels = [0],zorder = 2)
        ax2.contourf(s,t,z,colors = [self.colors[1],self.colors[0]],alpha = 0.15,levels = range(-1,2))

        #### plot right surface plot ####
        # plot regression surface
        ax4.plot_surface(s,t,z,alpha = 0.1,color = 'w',rstride=10, cstride=10,linewidth=0.5,edgecolor = 'k')
            
        # plot zplane = 0 in left 3d panel - showing intersection of regressor with z = 0 (i.e., its contour, the separator, in the 3d plot too)?
        if zplane == 'on':
            # plot zplane
            ax4.plot_surface(s,t,z*0,alpha = 0.1,rstride=20, cstride=20,linewidth=0.15,color = 'w',edgecolor = 'k') 
            
            # plot separator curve in left plot
            ax4.contour(s,t,z,colors = 'k',levels = [0],linewidths = 3,zorder = 1)
            ax4.contourf(s,t,z,colors = self.colors[0],levels = [0,1],zorder = 1,alpha = 0.1)
            ax4.contourf(s,t,z+1,colors = self.colors[1],levels = [0,1],zorder = 1,alpha = 0.1)
   
        ax2.set_ylim([xmin2,xmax2])
        ax4.set_ylim([xmin2,xmax2])

        ax4.axis('off')
        ax4.view_init(view2[0],view2[1])
        ax4.set_zlim([ymin,ymax])

    ######## show N = 2 static image ########
    # show coloring of entire space
    def static_N2_simple(self,w_best,runner,**kwargs):
        cost = runner.cost
        predict = runner.model
        feat = runner.feature_transforms
        normalizer = runner.normalizer
                
        # count parameter layers of input to feature transform
        sig = signature(feat)
        sig = len(sig.parameters)

        # or just take last weights
        self.w = w_best

        # construct figure
        fig, axs = plt.subplots(1, 2, figsize=(9,4))

        # create subplot with 2 panels
        gs = gridspec.GridSpec(1, 2) 
        ax2 = plt.subplot(gs[1],aspect = 'equal'); 
        ax1 = plt.subplot(gs[0],projection='3d'); 

        # scatter points
        self.scatter_pts(ax1,self.x)

        ### from above
        ax2.set_xlabel(r'$x_1$',fontsize = 15)
        ax2.set_ylabel(r'$x_2$',fontsize = 15,rotation = 0,labelpad = 20)
        ax2.xaxis.set_major_formatter(FormatStrFormatter('%.1f'))
        ax2.yaxis.set_major_formatter(FormatStrFormatter('%.1f'))

        # plot points in 2d and 3d
        C = len(np.unique(self.y))
        if C == 2:
            ind0 = np.argwhere(self.y == +1)
            ax2.scatter(self.x[ind0,0],self.x[ind0,1],s = 55, color = self.colors[0], edgecolor = 'k')

            ind1 = np.argwhere(self.y == -1)
            ax2.scatter(self.x[ind1,0],self.x[ind1,1],s = 55, color = self.colors[1], edgecolor = 'k')
        else:
            for c in range(C):
                ind0 = np.argwhere(self.y == c)
                ax2.scatter(self.x[ind0,0],self.x[ind0,1],s = 55, color = self.colors[c], edgecolor = 'k')

        self.move_axis_left(ax1)
        ax1.set_xlabel(r'$x_1$', fontsize = 12,labelpad = 5)
        ax1.set_ylabel(r'$x_2$', rotation = 0,fontsize = 12,labelpad = 5)
        ax1.set_zlabel(r'$y$', rotation = 0,fontsize = 12,labelpad = -3)

        ### create surface and boundary plot ###
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
        if 'view' in kwargs:
            view = kwargs['view']
            ax1.view_init(view[0],view[1])

        # plot boundary for 2d plot
        r1 = np.linspace(xmin1,xmax1,300)
        r2 = np.linspace(xmin2,xmax2,300)
        s,t = np.meshgrid(r1,r2)
        s = np.reshape(s,(np.size(s),1))
        t = np.reshape(t,(np.size(t),1))
        h = np.concatenate((s,t),axis = 1)
        z = predict(normalizer(h.T),self.w)
        z = np.sign(z)
        
        # reshape it
        s.shape = (np.size(r1),np.size(r2))
        t.shape = (np.size(r1),np.size(r2))     
        z.shape = (np.size(r1),np.size(r2))
        
        #### plot contour, color regions ####
        ax2.contour(s,t,z,colors='k', linewidths=2.5,levels = [0],zorder = 2)
        ax2.contourf(s,t,z,colors = [self.colors[1],self.colors[0]],alpha = 0.15,levels = range(-1,2))
        ax1.plot_surface(s,t,z,alpha = 0.1,color = 'w',rstride=30, cstride=30,linewidth=0.5,edgecolor = 'k')

            
    ###### plot plotting functions ######
    def plot_data(self):
        fig = 0
        # plot data in two and one-d
        if np.shape(self.x)[1] < 2:
            # construct figure
            fig, axs = plt.subplots(2,1, figsize=(4,4))
            gs = gridspec.GridSpec(2,1,height_ratios = [6,1]) 
            ax1 = plt.subplot(gs[0],aspect = 'equal');
            ax2 = plt.subplot(gs[1],sharex = ax1); 
            
            # set plotting limits
            xmax = copy.deepcopy(max(self.x))
            xmin = copy.deepcopy(min(self.x))
            xgap = (xmax - xmin)*0.2
            xmin -= xgap
            xmax += xgap
            
            ymax = max(self.y)
            ymin = min(self.y)
            ygap = (ymax - ymin)*0.5
            ymin -= ygap
            ymax += ygap    

            ### plot in 2-d
            ax1.scatter(self.x,self.y,color = 'k', edgecolor = 'w',linewidth = 0.9,s = 40)

            # clean up panel
            ax1.set_xlim([xmin,xmax])
            ax1.set_ylim([ymin,ymax])
            ax1.axhline(linewidth=0.5, color='k',zorder = 1)
            
            ### plot in 1-d
            ind0 = np.argwhere(self.y == +1)
            ax2.scatter(self.x[ind0],np.zeros((len(self.x[ind0]))),s = 55, color = self.colors[0], edgecolor = 'k',zorder = 3)

            ind1 = np.argwhere(self.y == -1)
            ax2.scatter(self.x[ind1],np.zeros((len(self.x[ind1]))),s = 55, color = self.colors[1], edgecolor = 'k',zorder = 3)
            ax2.set_yticks([0])
            ax2.axhline(linewidth=0.5, color='k',zorder = 1)
        
            ax2.xaxis.set_major_formatter(FormatStrFormatter('%.1f'))
            ax2.yaxis.set_major_formatter(FormatStrFormatter('%.1f'))
            
        if np.shape(self.x)[1] == 2:
            # construct figure
            fig, axs = plt.subplots(1, 2, figsize=(9,4))

            # create subplot with 2 panels
            gs = gridspec.GridSpec(1, 2) 
            ax2 = plt.subplot(gs[1],aspect = 'equal'); 
            ax1 = plt.subplot(gs[0],projection='3d'); 

            # scatter points
            self.scatter_pts(ax1,self.x)
            
            ### from above
            ax2.set_xlabel(r'$x_1$',fontsize = 15)
            ax2.set_ylabel(r'$x_2$',fontsize = 15,rotation = 0,labelpad = 20)
            ax2.xaxis.set_major_formatter(FormatStrFormatter('%.1f'))
            ax2.yaxis.set_major_formatter(FormatStrFormatter('%.1f'))
            
            # plot points in 2d and 3d
            C = len(np.unique(self.y))
            if C == 2:
                ind0 = np.argwhere(self.y == +1)
                ax2.scatter(self.x[ind0,0],self.x[ind0,1],s = 55, color = self.colors[0], edgecolor = 'k')

                ind1 = np.argwhere(self.y == -1)
                ax2.scatter(self.x[ind1,0],self.x[ind1,1],s = 55, color = self.colors[1], edgecolor = 'k')
            else:
                for c in range(C):
                    ind0 = np.argwhere(self.y == c)
                    ax2.scatter(self.x[ind0,0],self.x[ind0,1],s = 55, color = self.colors[c], edgecolor = 'k')
                    
        
            self.move_axis_left(ax1)
            ax1.set_xlabel(r'$x_1$', fontsize = 12,labelpad = 5)
            ax1.set_ylabel(r'$x_2$', rotation = 0,fontsize = 12,labelpad = 5)
            ax1.set_zlabel(r'$y$', rotation = 0,fontsize = 12,labelpad = -3)
        
    # scatter points
    def scatter_pts(self,ax,x):
        if np.shape(x)[1] == 1:
            # set plotting limits
            xmax = copy.deepcopy(max(x))
            xmin = copy.deepcopy(min(x))
            xgap = (xmax - xmin)*0.2
            xmin -= xgap
            xmax += xgap
            
            ymax = max(self.y)
            ymin = min(self.y)
            ygap = (ymax - ymin)*0.2
            ymin -= ygap
            ymax += ygap    

            # initialize points
            ax.scatter(x,self.y,color = 'k', edgecolor = 'w',linewidth = 0.9,s = 40)

            # clean up panel
            ax.set_xlim([xmin,xmax])
            ax.set_ylim([ymin,ymax])
            
        if np.shape(x)[1] == 2:
            # set plotting limits
            xmax1 = copy.deepcopy(max(x[:,0]))
            xmin1 = copy.deepcopy(min(x[:,0]))
            xgap1 = (xmax1 - xmin1)*0.1
            xmin1 -= xgap1
            xmax1 += xgap1
            
            xmax2 = copy.deepcopy(max(x[:,1]))
            xmin2 = copy.deepcopy(min(x[:,1]))
            xgap2 = (xmax2 - xmin2)*0.1
            xmin2 -= xgap2
            xmax2 += xgap2
            
            ymax = max(self.y)
            ymin = min(self.y)
            ygap = (ymax - ymin)*0.2
            ymin -= ygap
            ymax += ygap    

            # initialize points
            ax.scatter(x[:,0],x[:,1],self.y.flatten(),s = 40,color = 'k', edgecolor = 'w',linewidth = 0.9)

            # clean up panel
            ax.set_xlim([xmin1,xmax1])
            ax.set_ylim([xmin2,xmax2])
            ax.set_zlim([ymin,ymax])
            
            ax.set_xticks(np.arange(round(xmin1), round(xmax1)+1, 1.0))
            ax.set_yticks(np.arange(round(xmin2), round(xmax2)+1, 1.0))
            ax.set_zticks(np.arange(round(ymin), round(ymax)+1, 1.0))
           
            # clean up panel
            ax.xaxis.pane.fill = False
            ax.yaxis.pane.fill = False
            ax.zaxis.pane.fill = False

            ax.xaxis.pane.set_edgecolor('white')
            ax.yaxis.pane.set_edgecolor('white')
            ax.zaxis.pane.set_edgecolor('white')

            ax.xaxis._axinfo["grid"]['color'] =  (1,1,1,0)
            ax.yaxis._axinfo["grid"]['color'] =  (1,1,1,0)
            ax.zaxis._axinfo["grid"]['color'] =  (1,1,1,0)
           
    # set axis in left panel
    def move_axis_left(self,ax):
        tmp_planes = ax.zaxis._PLANES 
        ax.zaxis._PLANES = ( tmp_planes[2], tmp_planes[3], 
                             tmp_planes[0], tmp_planes[1], 
                             tmp_planes[4], tmp_planes[5])
        view_1 = (25, -135)
        view_2 = (25, -45)
        init_view = view_2
        ax.view_init(*init_view) 
        
        
    # toy plot
    def multiclass_plot(self,run,w,**kwargs):
        model = run.model
        normalizer = run.normalizer
        
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
        g_vals = np.argmax(g_vals,axis = 0)

        # vals for cost surface
        w1_vals.shape = (len(r),len(r))
        w2_vals.shape = (len(r),len(r))
        g_vals.shape = (len(r),len(r))

        # create figure to plot
        fig = plt.figure(num=None, figsize=(12,5), dpi=80, facecolor='w', edgecolor='k')

        ### create 3d plot in left panel
        ax1 = plt.subplot(121,projection = '3d')
        ax2 = plt.subplot(122)

        fig.subplots_adjust(left=0,right=1,bottom=0,top=1)   # remove whitespace around 3d figure

        # scatter points in both panels
        class_nums = np.unique(self.y)
        C = len(class_nums)
        for c in range(C):
            ind = np.argwhere(self.y == class_nums[c])
            ind = [v[0] for v in ind]
            ax1.scatter(self.x[ind,0],self.x[ind,1],self.y[ind],s = 80,color = self.colors[c],edgecolor = 'k',linewidth = 1.5)
            ax2.scatter(self.x[ind,0],self.x[ind,1],s = 110,color = self.colors[c],edgecolor = 'k', linewidth = 2)
            
        # switch for 2class / multiclass view
        if C == 2:
            # plot regression surface
            ax1.plot_surface(w1_vals,w2_vals,g_vals,alpha = 0.1,color = 'k',rstride=20, cstride=20,linewidth=0,edgecolor = 'k') 

            # plot zplane = 0 in left 3d panel - showing intersection of regressor with z = 0 (i.e., its contour, the separator, in the 3d plot too)?
            ax1.plot_surface(w1_vals,w2_vals,g_vals*0,alpha = 0.1,rstride=20, cstride=20,linewidth=0.15,color = 'k',edgecolor = 'k') 
            
            # plot separator in left plot z plane
            ax1.contour(w1_vals,w2_vals,g_vals,colors = 'k',levels = [0],linewidths = 3,zorder = 1)

            # color parts of plane with correct colors
            ax1.contourf(w1_vals,w2_vals,g_vals+1,colors = self.colors[:],alpha = 0.1,levels = range(0,2))
            ax1.contourf(w1_vals,w2_vals,-g_vals+1,colors = self.colors[1:],alpha = 0.1,levels = range(0,2))
    
            # plot separator in right plot
            ax2.contour(w1_vals,w2_vals,g_vals,colors = 'k',levels = [0],linewidths = 3,zorder = 1)

            # adjust height of regressor to plot filled contours
            ax2.contourf(w1_vals,w2_vals,g_vals+1,colors = self.colors[:],alpha = 0.1,levels = range(0,C+1))

            ### clean up panels
            # set viewing limits on vertical dimension for 3d plot           
            minz = min(copy.deepcopy(self.y))
            maxz = max(copy.deepcopy(self.y))

            gapz = (maxz - minz)*0.1
            minz -= gapz
            maxz += gapz

        # multiclass view
        else:   
            ax1.plot_surface(w1_vals,w2_vals,g_vals,alpha = 0.1,color = 'w',rstride=45, cstride=45,linewidth=0.25,edgecolor = 'k')

            for c in range(C):
                # plot separator curve in left plot z plane
                ax1.contour(w1_vals,w2_vals,g_vals - c,colors = 'k',levels = [0],linewidths = 3,zorder = 1)

                # color parts of plane with correct colors
                ax1.contourf(w1_vals,w2_vals,g_vals - c +0.5,colors = self.colors[c],alpha = 0.4,levels = [0,1])
             
                
            # plot separator in right plot
            ax2.contour(w1_vals,w2_vals,g_vals,colors = 'k',levels = range(0,C+1),linewidths = 3,zorder = 1)
            
            # adjust height of regressor to plot filled contours
            ax2.contourf(w1_vals,w2_vals,g_vals+0.5,colors = self.colors[:],alpha = 0.2,levels = range(0,C+1))

            ### clean up panels
            # set viewing limits on vertical dimension for 3d plot 
            minz = 0
            maxz = max(copy.deepcopy(self.y))
            gapz = (maxz - minz)*0.1
            minz -= gapz
            maxz += gapz
            ax1.set_zlim([minz,maxz])

            ax1.view_init(view[0],view[1]) 

        # clean up panel
        ax1.xaxis.pane.fill = False
        ax1.yaxis.pane.fill = False
        ax1.zaxis.pane.fill = False

        ax1.xaxis.pane.set_edgecolor('white')
        ax1.yaxis.pane.set_edgecolor('white')
        ax1.zaxis.pane.set_edgecolor('white')

        ax1.xaxis._axinfo["grid"]['color'] =  (1,1,1,0)
        ax1.yaxis._axinfo["grid"]['color'] =  (1,1,1,0)
        ax1.zaxis._axinfo["grid"]['color'] =  (1,1,1,0)

        self.move_axis_left(ax1)
        ax1.set_xlabel(r'$x_1$', fontsize = 16,labelpad = 5)
        ax1.set_ylabel(r'$x_2$', rotation = 0,fontsize = 16,labelpad = 5)
        ax1.set_zlabel(r'$y$', rotation = 0,fontsize = 16,labelpad = 5)

        ax2.set_xlabel(r'$x_1$', fontsize = 18,labelpad = 10)
        ax2.set_ylabel(r'$x_2$', rotation = 0,fontsize = 18,labelpad = 15)
        
        
    # toy plot
    def show_individual_classifiers(self,run,w,**kwargs):
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
        ax1 = plt.subplot(gs[C],projection = '3d');
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
            ax1.scatter(self.x[ind,0],self.x[ind,1],self.y[ind],s = 50,color = self.colors[c],edgecolor = 'k',linewidth = 1.5)
            ax2.scatter(self.x[ind,0],self.x[ind,1],s = 50,color = self.colors[c],edgecolor = 'k', linewidth = 2)
      
        ax1.plot_surface(w1_vals,w2_vals,g_vals,alpha = 0.1,color = 'w',rstride=45, cstride=45,linewidth=0.25,edgecolor = 'k')

        for c in range(C):
            # plot separator curve in left plot z plane
            ax1.contour(w1_vals,w2_vals,g_vals - c,colors = 'k',levels = [0],linewidths = 3,zorder = 1)

            # color parts of plane with correct colors
            ax1.contourf(w1_vals,w2_vals,g_vals - c +0.5,colors = self.colors[c],alpha = 0.4,levels = [0,1])
             
        # plot separator in right plot
        ax2.contour(w1_vals,w2_vals,g_vals,colors = 'k',levels = range(0,C+1),linewidths = 3,zorder = 1)

        # adjust height of regressor to plot filled contours
        ax2.contourf(w1_vals,w2_vals,g_vals+0.5,colors = self.colors[:],alpha = 0.2,levels = range(0,C+1))

        ### clean up panels
        # set viewing limits on vertical dimension for 3d plot 
        minz = 0
        maxz = max(copy.deepcopy(self.y))
        gapz = (maxz - minz)*0.1
        minz -= gapz
        maxz += gapz
        ax1.set_zlim([minz,maxz])

        ax1.view_init(view[0],view[1]) 

        # clean up panel
        ax1.xaxis.pane.fill = False
        ax1.yaxis.pane.fill = False
        ax1.zaxis.pane.fill = False

        ax1.xaxis.pane.set_edgecolor('white')
        ax1.yaxis.pane.set_edgecolor('white')
        ax1.zaxis.pane.set_edgecolor('white')

        ax1.xaxis._axinfo["grid"]['color'] =  (1,1,1,0)
        ax1.yaxis._axinfo["grid"]['color'] =  (1,1,1,0)
        ax1.zaxis._axinfo["grid"]['color'] =  (1,1,1,0)

        self.move_axis_left(ax1)
        ax1.set_xlabel(r'$x_1$', fontsize = 16,labelpad = 5)
        ax1.set_ylabel(r'$x_2$', rotation = 0,fontsize = 16,labelpad = 5)
        ax1.set_zlabel(r'$y$', rotation = 0,fontsize = 16,labelpad = 5)

        ax2.set_xlabel(r'$x_1$', fontsize = 18,labelpad = 10)
        ax2.set_ylabel(r'$x_2$', rotation = 0,fontsize = 18,labelpad = 15)