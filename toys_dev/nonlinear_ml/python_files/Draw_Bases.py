import numpy as np
import matplotlib.pyplot as plt

class Draw_Bases:
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
        fig = plt.figure(figsize = (16,5))

        for m in range(1,5):
            # make basis element
            f_m = self.make_1d_poly(x,m)

            # plot the current element
            ax = fig.add_subplot(1,4,m)
            ax.plot(x,f_m,color = [0,1/float(m),m/float(m+1)],linewidth = 5)

            # clean up plot and show legend 
            ax.set_title('element ' + str(m),fontsize = 18)
            ax.set_yticks([], [])
            ax.axis('off')

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
                c = 10*np.random.rand(1)[0] - 5
                v = -c - 2*np.sign(c)*np.random.rand(prev_num_hidden_units,1)

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
    def show_1d_net(self,num_layers):
        x = np.linspace(0,1,100)
        fig = plt.figure(figsize = (16,5))
        m = 1
        while m < 5:
            # create instance of neural net basis element
            params = self.make_1d_net(num_layers)

            # create input
            x = np.linspace(0,1,100)
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
                    f_temp = a = np.maximum(np.zeros((np.shape(temp))),temp)
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
                ax.plot(x.ravel(),f_m.ravel(),color = 'k',linewidth = 3)

                # clean up plot and show legend 
                ax.set_title('instance ' + str(m),fontsize = 18)
                ax.set_yticks([], [])
                ax.axis('off')

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
        fig = plt.figure(figsize = (16,5))

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
            ax.set_ylim([-6,6])
            ax.set_yticks([], [])
            ax.axis('off')

