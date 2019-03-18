# import standard plotting and animation
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from matplotlib import gridspec

# import autograd functionality
from autograd import grad as compute_grad   # The only autograd function you may ever need
import autograd.numpy as np
import math
import time
from IPython.display import clear_output

class visualizer: 
    '''
    Draw first through fourth Taylor series approximation to a function over points on 
    the interval [-3,3] and animate with a custom slider mechanism, allowing you to browse these
    approximations over various points of the input interval
    '''
    
    def __init__(self,**args):
        self.g = args['g']                                                # input function
        self.first_derivative = compute_grad(self.g)                      # first derivative of input function
        self.second_derivative = compute_grad(self.first_derivative)      # second derivative of input function
        self.third_derivative = compute_grad(self.second_derivative)      # third derivative of input function
        self.fourth_derivative = compute_grad(self.third_derivative)      # fourth derivative of input function

        self.colors = [[0,1,0.25],[0,0.75,1],[1,0.75,0],[1,0,0.75]]       # custom colors for plotting

    # draw taylor series approximations over a range of points and animate
    def draw_it(self,**args):
        # how many frames to plot
        num_frames = 5
        if 'num_frames' in args:
            num_frames = args['num_frames']
            
        # initialize figure
        fig = plt.figure(figsize = (10,3))
        artist = fig
        gs = gridspec.GridSpec(1, 4) 
        ax1 = fig.add_subplot(141)
        ax2 = fig.add_subplot(142)
        ax3 = fig.add_subplot(143)
        ax4 = fig.add_subplot(144)

        # generate function for plotting on each slide
        w_plot = np.linspace(-3,3,200)
        g_plot = self.g(w_plot)
        g_range = max(g_plot) - min(g_plot)
        ggap = g_range*0.5
        w_vals = np.linspace(-2.5,2.5,num_frames)
        print ('beginning animation rendering...')
        
        # animation sub-function
        def animate_it(k):
            # print rendering update
            if np.mod(k+1,25) == 0:
                print ('rendering animation frame ' + str(k+1) + ' of ' + str(num_frames))
            if k == num_frames - 1:
                print ('animation rendering complete!')
                time.sleep(1.5)
                clear_output()
                
            # clear out each panel for next slide
            ax1.cla()
            ax2.cla()
            ax3.cla()
            ax4.cla()

            # pick current value to plot approximations
            w_val = w_vals[k]
            g_val = self.g(w_val)

            # plot the original function in each panel
            ax1.plot(w_plot,g_plot,color = 'k',zorder = 0)                           
            ax2.plot(w_plot,g_plot,color = 'k',zorder = 0)                         
            ax3.plot(w_plot,g_plot,color = 'k',zorder = 0)                       
            ax4.plot(w_plot,g_plot,color = 'k',zorder = 0)                       
            
            # plot tangency point in each panel
            ax1.scatter(w_val,g_val,s = 90,c = 'r',edgecolor = 'k',linewidth = 0.7,zorder = 3)            
            ax2.scatter(w_val,g_val,s = 90,c = 'r',edgecolor = 'k',linewidth = 0.7,zorder = 3)            
            ax3.scatter(w_val,g_val,s = 90,c = 'r',edgecolor = 'k',linewidth = 0.7,zorder = 3)            
            ax4.scatter(w_val,g_val,s = 90,c = 'r',edgecolor = 'k',linewidth = 0.7,zorder = 3)            

            #### plot first order approximation? ####
            # plug in value into func and derivative
            g_first_val = self.first_derivative(w_val)

            # compute first order approximation
            wrange = np.linspace(-3,3,100)
            h = g_val + g_first_val*(wrange - w_val)

            # plot first order approximation in first panel
            ax1.plot(wrange,h,color = self.colors[0],linewidth = 2,zorder = 2)      # plot approx

            #### plot second order approximation? ####
            # plug in value into func and derivative
            g_second_val = self.second_derivative(w_val)

            # compute second order approximation
            h += 1/float(2)*g_second_val*(wrange - w_val)**2 

            # plot second order approximation in second panel
            ax2.plot(wrange,h,color = self.colors[1],linewidth = 3,zorder = 1)      # plot approx

            
            #### plot third order approximation? ####
            # plug in value into func and derivative
            g_third_val = self.third_derivative(w_val)

            # compute second order approximation
            h += 1/float(2*3)*g_third_val*(wrange - w_val)**3

            # plot second order approximation in second panel
            ax3.plot(wrange,h,color = self.colors[2],linewidth = 3,zorder = 1)      # plot approx

            
            #### plot fourth order approximation? ####
            # plug in value into func and derivative
            g_fourth_val = self.fourth_derivative(w_val)

            # compute second order approximation
            h += 1/float(2*3*4)*g_fourth_val*(wrange - w_val)**4

            # plot second order approximation in second panel
            ax4.plot(wrange,h,color = self.colors[3],linewidth = 3,zorder = 1)      # plot approx

            
            #### fix viewing limits ####
            ax1.set_xlim([-3,3])
            ax1.set_ylim([min(g_plot) - ggap,max(g_plot) + ggap])
            ax1.set_yticks([],[])
            ax1.set_title('first order approximation',fontsize = 13)
            
            ax2.set_xlim([-3,3])
            ax2.set_ylim([min(g_plot) - ggap,max(g_plot) + ggap])
            ax2.set_yticks([],[])
            ax2.set_title('second order approximation',fontsize = 13)
    
            ax3.set_xlim([-3,3])
            ax3.set_ylim([min(g_plot) - ggap,max(g_plot) + ggap])
            ax3.set_yticks([],[])
            ax3.set_title('third order approximation',fontsize = 13)
        
            ax4.set_xlim([-3,3])
            ax4.set_ylim([min(g_plot) - ggap,max(g_plot) + ggap])
            ax4.set_yticks([],[])
            ax4.set_title('fourth order approximation',fontsize = 13)
              
            return artist,

        anim = animation.FuncAnimation(fig, animate_it,frames=len(w_vals), interval=len(w_vals), blit=True)

        return(anim)