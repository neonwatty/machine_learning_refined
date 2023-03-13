import numpy as np
import matplotlib.pyplot as plt

class Visualizer:
    '''
    Draw_Bases contains several functions for plotting 1-d examples of 
    elements or instances from 
    - a polynomial basis - 4 elements are shown
    - a neural net basis with 1 hidden layer - 4 random instances of a single element are shown
    - a neural net basis with 2 hidden layers - 4 random instances of a single element are shown
    - a decision tree basis with maxmimun depth defined by the user - 4 random instances of a single element are shown
    '''
    
    #### Polynomial Basis ####
    ## build 4 polynomial basis elements 
    # make 1d polynomial basis element
    def make_1d_poly(self,x,degree):
        f = x**degree
        return f
    
    # plot 4 elements of a poly basis
    def show_1d_poly(self):
        # build the first 4 non-constant polynomial basis elements
        x = np.linspace(-10,10,100)
        fig = plt.figure(figsize = (9,3))

        for m in range(1,5):
            # make basis element
            f_m = x**m

            # plot the current element
            ax = fig.add_subplot(1,4,m)
            ax.plot(x,f_m,color = [0,1/float(m),m/float(m+1)],linewidth = 3,zorder = 3)

            # clean up plot and show legend 
            if m == 1:
                ax.set_title('$f_1$',fontsize = 18)
            if m == 2:
                ax.set_title('$f_2$',fontsize = 18)
            if m == 3:
                ax.set_title('$f_3$',fontsize = 18)
            if m == 4:
                ax.set_title('$f_4$',fontsize = 18)
        
            # clean up plot
            ax.grid(True, which='both')
            ax.axhline(y=0, color='k')
            ax.axvline(x=0, color='k')
            
    # show N = 2 input polys        
    def show_2d_poly(self):
        # generate input values
        s = np.linspace(-2,2,100)
        x_1,x_2 = np.meshgrid(s,s)
        degree_dict = {}

        # build 4 polynomial basis elements
        fig = plt.figure(num=None, figsize = (12,4), dpi=80, facecolor='w', edgecolor='k')

        ### plot regression surface ###
        p =  [0,1,1,2]
        q = [1,2,1,3]
        for m in range(4):
            ax1 = plt.subplot(1,4,m+1,projection = '3d')
            f_m = (x_1**p[m])*(x_2**q[m])
            ax1.plot_surface(x_1,x_2,f_m,alpha = 0.5,color = 'w',zorder = 3,edgecolor = 'k',linewidth=1,cstride = 10, rstride = 10)
            ax1.view_init(10,20)  
            deg1 = ''
            if p[m] == 1:
                deg1 = 'x_1^{\,}'
            if p[m] >=2:
                deg1 = 'x_1^' + str(p[m])
            deg2 = ''
            if q[m] == 1:
                deg2 = 'x_2^{\,}'
            if q[m] >=2:
                deg2 = 'x_2^' + str(q[m])
            ax1.set_title('$f_'+str(m+1) + ' = ' + deg1 + deg2 + '$',fontsize = 18)
            ax1.axis('off')
        fig.subplots_adjust(left=0,right=1,bottom=0,top=1)   # remove whitespace around 3d figure

        plt.show()

    #### Neural Network Basis ####
    ## build 4 instances of a neural net basis with max activations and user-defined number of layers
    # make the parameter structure for a net basis element with given num_layers
    def make_1d_net(self,num_layers):
        # contariner for parameters
        parameters = []

        # number of hidden units in previous layer - initialize at 1 for first layer
        prev_num_hidden_units = 1

        # loop over layers
        for n in range(0,num_layers):

            # pick random integer for number of hidden units in layer
            layer_params = []

            # pick number of hidden units per layer
            num_hidden_units = 1
            if n < num_layers-1:
                num_hidden_units = 10

            for l in range(num_hidden_units):
                # choose first layer random weights to make nice single layer visualization
                c = 5*np.random.rand(1)[0] - 2.5
                v = -c - 2*np.sign(c)*np.random.rand(prev_num_hidden_units,1)
                
                c = np.random.randn(1)[0]
                v = np.random.randn(prev_num_hidden_units,1)
                
                # likewise, choose subsquent layer weights randomly as so for nice looking instances
                if n > 0:
                    c = np.random.randn(1)[0]
                    v = np.random.randn(prev_num_hidden_units,1)

                v = [s[0] for s in v]
                weights = [c,v]
                layer_params.append(weights)

            # store all weights 
            parameters.append(layer_params)

            prev_num_hidden_units = num_hidden_units

        return parameters

    # build 4 instances of a single hidden layer basis element
    def show_1d_net(self,num_layers,activation):
        fig = plt.figure(figsize = (9,3))
        m = 1
        while m < 5:
            # create instance of neural net basis element
            params = self.make_1d_net(num_layers)

            # create input
            x = np.linspace(-5,5,1000)
            x.shape = (1,len(x))
            f_prev = x

            # loop over each layer, pushing function composition, produce subsequent layer operations
            for n in range(0,num_layers):
                num_units = len(params[n])
                f_new = 0
                for u in range(0,num_units):
                    # grab parameters        
                    c = params[n][u][0]
                    v = params[n][u][1]

                    # loop over dimension of v, sum up components
                    f = 0
                    for i in range(0,len(v)):
                        f += f_prev[i,:]*v[i]

                    # evaluate through activation
                    temp = c + f
                    f_temp = 0
                    if activation == 'relu':
                        f_temp = a = np.maximum(np.zeros((np.shape(temp))),temp)
                    if activation == 'tanh':
                        f_temp = a = np.tanh(temp)
                    if type(f_new) == int:
                        f_new = f_temp
                    else:
                        f_new = np.vstack([f_new,f_temp])

                # update previous layer evaluations f_prev
                f_prev = f_new
                if f_prev.ndim == 1:
                    f_prev.shape = (1,len(f_prev))

            # plot the current instance
            f_m = f_prev

            # choose to plot interesting (non-zero) instance - if zero remake weights
            if np.std(f_m) > 0.01:
                ax = fig.add_subplot(1,4,m)
                ax.plot(x.ravel(),f_m.ravel()/max(f_m.ravel()),color = 'r',linewidth = 3)

                # clean up plot and show legend
                ax.set_title('$f^{\,(' + str(num_layers) + ')}(x)$', fontsize = 12)

                # clean up plot
                ax.grid(True, which='both')
                ax.axhline(y=0, color='k')
                ax.axvline(x=0, color='k')
                
                m+=1

    #### Decision Tree Basis ####
    ## build 4 instances of a decision tree basis with user defined depth
    # a recursive function for making a tree of defined depth over the interval [0,1], called by show_1d_tree
    def make_1d_tree(self,depth,intervals):
        new_intervals = []
        splits = []
        vals = []

        # loop over intervals, split, assign values 
        for length in intervals:
            new_split = float((length[1] - length[0])/float(2)*np.random.rand(1) + (length[1] + length[0])/float(2))
            splits.append(length[0])
            splits.append(new_split)

            h_1 = 10*np.random.rand(1) - 5
            h_2 = 10*np.random.rand(1) - 5

            vals.append(float(h_1))
            vals.append(float(h_2))

            new_intervals.append([length[0],new_split])
            new_intervals.append([new_split,length[1]])
        splits.append(length[1])

        # run again on new set of sub-intervals
        intervals = new_intervals
        depth -= 1

        # if we have reached desired depth then stop, else continue on each interval
        if depth == 0:
            return vals,splits
        else:
            return self.make_1d_tree(depth,intervals)
    
    # show 1-dimensional tree of user chosen depth
    def show_1d_tree(self,depth):
        # build 4 instances of a tree
        x = np.linspace(0,1,100)
        fig = plt.figure(figsize = (9,4))

        for m in range(1,5):
            # combine above into element
            f_m = np.zeros((len(x),1))

            # create instance tree basis element
            intervals = [[0,1]]
            vals,splits = self.make_1d_tree(depth,intervals)

            # build f_m
            f_m = []
            for i in range(len(splits)-1):
                level_in = [s for s in x if s >= splits[i] and s <= splits[i+1]]
                level_out = vals[i]*np.ones((len(level_in),1))
                f_m.append(level_out)
            f_m = [float(item) for sublist in f_m for item in sublist]

            # plot the current instance
            ax = fig.add_subplot(1,4,m)
            ax.plot(x,f_m,color = 'k',linewidth = 5)

            # clean up plot and show legend 
            ax.set_title('instance ' + str(m),fontsize = 18)

            # clean up plot
            ax.grid(True, which='both')
            ax.axhline(y=0, color='k')
            ax.axvline(x=0, color='k')
                

