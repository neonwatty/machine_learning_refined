import autograd.numpy as np
import matplotlib.pyplot as plt
from matplotlib import gridspec

class Setup:
    def __init__(self,**kwargs):
        ### basic parameters of squiggle ###
        # An example 4 hidden layer network, with 10 units in each layer
        U_1 = 10
        U_2 = 10
        U_3 = 10
        U_4 = 10
        U_5 = 10

        # the list defines our network architecture
        self.encoder_layer_sizes = [2,U_1,1]
        self.decoder_layer_sizes = [1, U_1,U_2,U_3,U_4,U_5,2]
        self.activation = np.sinc
        self.scale = 0.5
        
        if 'activation' in kwargs:
            self.activation = kwargs['activation']
        if 'encoder_layer_sizes' in kwargs:
            self.encoder_layer_sizes = kwargs['encoder_layer_sizes']
        if 'decoder_layer_sizes' in kwargs:
            self.decoder_layer_sizes = kwargs['decoder_layer_sizes']
        if 'scale' in kwargs:
            self.scale = kwargs['scale']

    ####### squiggle maker functions ######
    def make_squiggle(self):
        # generate weights
        w1 = self.initialize_network_weights(self.encoder_layer_sizes, self.scale)
        w2 = self.initialize_network_weights(self.decoder_layer_sizes, self.scale)
        self.w = [w1,w2]
        
        # evaluate autoencoder over fine range of input
        a = np.linspace(-1,1,200)
        b = np.linspace(-1,1,200)
        s,t = np.meshgrid(a,b)
        s.shape = (1,len(a)**2)
        t.shape = (1,len(b)**2)
        z = np.vstack((s,t))

        # create encoded vectors
        v = self.encoder(z,self.w[0])

        # decode onto basis
        self.squiggle = self.decoder(v,self.w[1])
        
        # plot squiggle
        fig = plt.figure(figsize = (9,4))
        gs = gridspec.GridSpec(1,1) 
        ax = plt.subplot(gs[0]);ax.axis('off'); 
        #ax.set_xlabel(r'$x_1$',fontsize = 15);ax.set_ylabel(r'$x_2$',fontsize = 15,rotation = 0);
        ax.scatter(self.squiggle[0,:],self.squiggle[1,:],c = 'k',s = 5.5,edgecolor = 'k',linewidth = 0.5,zorder = 0)
        ax.scatter(self.squiggle[0,:],self.squiggle[1,:],c = 'r',s = 1.5,edgecolor = 'r',linewidth = 0.5,zorder = 0)
        plt.show()
        
    def make_so_many_squiggles(self):
        # evaluate autoencoder over fine range of input
        a = np.linspace(-1,1,200)
        b = np.linspace(-1,1,200)
        s,t = np.meshgrid(a,b)
        s.shape = (1,len(a)**2)
        t.shape = (1,len(b)**2)
        z = np.vstack((s,t))
        fig = plt.figure(figsize = (9,6))
        gs = gridspec.GridSpec(3,3) 
        for i in range(9):
            # generate weights
            w1 = self.initialize_network_weights(self.encoder_layer_sizes, self.scale)
            w2 = self.initialize_network_weights(self.decoder_layer_sizes, self.scale)
            self.w = [w1,w2]

            # create encoded vectors
            v = self.encoder(z,self.w[0])

            # decode onto basis
            self.squiggle = self.decoder(v,self.w[1])

            # plot squiggle
            ax = plt.subplot(gs[i]);ax.axis('off'); 
            #ax.set_xlabel(r'$x_1$',fontsize = 15);ax.set_ylabel(r'$x_2$',fontsize = 15,rotation = 0);
            ax.scatter(self.squiggle[0,:],self.squiggle[1,:],c = 'k',s = 5.5,edgecolor = 'k',linewidth = 0.5,zorder = 0)
            ax.scatter(self.squiggle[0,:],self.squiggle[1,:],c = 'r',s = 1.5,edgecolor = 'r',linewidth = 0.5,zorder = 0)
        plt.show()
        
    ####### network functions ######
    # create initial weights for arbitrary feedforward network
    def initialize_network_weights(self,layer_sizes, scale):
        # container for entire weight tensor
        weights = []

        # loop over desired layer sizes and create appropriately sized initial 
        # weight matrix for each layer
        for k in range(len(layer_sizes)-1):
            # get layer sizes for current weight matrix
            U_k = layer_sizes[k]
            U_k_plus_1 = layer_sizes[k+1]

            # make weight matrix
            weight = scale*np.random.randn(U_k+1,U_k_plus_1)
            weights.append(weight)

        # re-express weights so that w_init[0] = omega_inner contains all 
        # internal weight matrices, and w_init = w contains weights of 
        # final linear combination in predict function
        w_init = [weights[:-1],weights[-1]]

        return w_init

    # fully evaluate our network features using the tensor of weights in w
    def feature_transforms(self,a, w):    
        # loop through each layer matrix
        for W in w:
            #  pad with ones (to compactly take care of bias) for next layer computation        
            o = np.ones((1,np.shape(a)[1]))
            a = np.vstack((o,a))

            # compute inner product with current layer weights
            a = np.dot(a.T, W).T

            # output of layer activation
            a = self.activation(a)
        return a
    
    def encoder(self,x,w):    
        # feature transformation 
        f = self.feature_transforms(x,w[0])

        # tack a 1 onto the top of each input point all at once
        o = np.ones((1,np.shape(f)[1]))
        f = np.vstack((o,f))

        # compute linear combination and return
        a = np.dot(f.T,w[1])
        return a.T
    
    def decoder(self,v,w):
        # feature transformation 
        f = self.feature_transforms(v,w[0])

        # tack a 1 onto the top of each input point all at once
        o = np.ones((1,np.shape(f)[1]))
        f = np.vstack((o,f))

        # compute linear combination and return
        a = np.dot(f.T,w[1])
        return a.T