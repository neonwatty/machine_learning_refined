# import standard plotting and animation
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from mpl_toolkits.mplot3d import Axes3D

# import autograd functionality
from autograd import grad as compute_grad   # The only autograd function you may ever need
import autograd.numpy as np
import math
from matplotlib import gridspec
from IPython.display import clear_output
import time

class visualizer:
    '''
    This file illlustrates the sum of two functions in 3d.  Both functions are defined by the user.
    ''' 

    # animate the method
    def draw_it(self,h1,h2,savepath,**kwargs):
        # user input functions to add
        self.h1 = h1                            # input function
        self.h2 = h2                            # input function
        num_frames = 100
        if 'num_frames' in kwargs:
            num_frames = kwargs['num_frames']
            
        # turn axis on or off
        set_axis = 'on'
        if 'set_axis' in kwargs:
            set_axis = kwargs['set_axis']
  
        # set viewing angle on plot
        view = [20,50]
        if 'view' in kwargs:
            view = kwargs['view']
            
        epsmax = 2
        if 'epsmax' in kwargs:
            epsmax = kwargs['epsmax']
            
        # initialize figure
        fig = plt.figure(figsize = (15,5))
        artist = fig
        
        gs = gridspec.GridSpec(1, 3, width_ratios=[1,1, 1]) 
        ax1 = plt.subplot(gs[0],projection='3d');
        ax2 = plt.subplot(gs[1],projection='3d');
        ax3 = plt.subplot(gs[2],projection='3d');
        
        # generate input range for functions
        r = np.linspace(-3,3,100)
        w1_vals,w2_vals = np.meshgrid(r,r)
        w1_vals.shape = (len(r)**2,1)
        w2_vals.shape = (len(r)**2,1)
        h1_vals = self.h1([w1_vals,w2_vals])
        h2_vals = self.h2([w1_vals,w2_vals])
        
        # vals for cost surface
        w1_vals.shape = (len(r),len(r))
        w2_vals.shape = (len(r),len(r))
        h1_vals.shape = (len(r),len(r))
        h2_vals.shape = (len(r),len(r))

        # decide on number of slides
        epsilon_vals = np.linspace(0,epsmax,num_frames)

        # animation sub-function
        print ('starting animation rendering...')
        def animate(t):
            # clear panels for next slide
            ax1.cla()
            ax2.cla()
            ax3.cla()
            
            # print rendering update
            if np.mod(t+1,25) == 0:
                print ('rendering animation frame ' + str(t+1) + ' of ' + str(num_frames))
            if t == num_frames - 1:
                print ('animation rendering complete!')
                time.sleep(1.5)
                clear_output()
              
            # plot function 1
            ax1.plot_surface(w1_vals,w2_vals,h1_vals,alpha = 0.3,color = 'w',rstride=10, cstride=10,linewidth=2,edgecolor = 'k') 
            ax1.set_title("$h_1$",fontsize = 15)
            ax1.view_init(view[0],view[1])
            ax1.grid(False)


            # plot function 2
            ax2.plot_surface(w1_vals,w2_vals,h2_vals,alpha = 0.3,color = 'w',rstride=10, cstride=10,linewidth=2,edgecolor = 'k') 
            ax2.set_title("$h_2$",fontsize = 15)
            ax2.view_init(view[0],view[1])
            ax2.grid(False)
                
            # plot combination of both
            epsilon = epsilon_vals[t]
            h3_vals = h1_vals + epsilon*h2_vals
            ax3.plot_surface(w1_vals,w2_vals,h3_vals,alpha = 0.3,color = 'w',rstride=10, cstride=10,linewidth=2,edgecolor = 'k')
            ax3.grid(False)

            title = r'$h_1 + ' +  r'{:.2f}'.format(epsilon)  + 'h_2$'
            ax3.set_title(title,fontsize = 15)
            ax3.view_init(view[0],view[1])
            
            return artist,

        anim = animation.FuncAnimation(fig, animate ,frames=num_frames, interval=num_frames, blit=True)

        # produce animation and save
        fps = 50
        if 'fps' in kwargs:
            fps = kwargs['fps']
        anim.save(savepath, fps=fps, extra_args=['-vcodec', 'libx264'])
        clear_output()