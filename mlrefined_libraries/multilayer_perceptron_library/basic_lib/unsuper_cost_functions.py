import autograd.numpy as np

class Setup:
    def __init__(self,name,**kwargs):                    
        # for autoencoder
        if name == 'autoencoder':
            self.cost = self.autoencoder
            
    ### insert feature transformations to use ###
    def define_encoder_decoder(self,feats1,feats2):
        self.feature_transforms = feats1
        self.feature_transforms_2 = feats2
   
    ### for autoencoder ###
    def encoder(self,x,w):    
        # feature transformation 
        f = self.feature_transforms(x,w[0])

        # compute linear combination and return
        a = w[1][0] + np.dot(f.T,w[1][1:])
        return a.T

    def decoder(self,v,w):
        # feature transformation 
        f = self.feature_transforms_2(v,w[0])

        # compute linear combination and return
        a = w[1][0] + np.dot(f.T,w[1][1:])
        return a.T
    
    def autoencoder(self,w,x,iter):
        x_p = x[:,iter]
        
        # encode input
        a = self.encoder(x_p,w[0])
        
        # decode result
        b = self.decoder(a,w[1])
        
        # compute Least Squares error
        cost = np.sum((b - x_p)**2)
        return cost/float(x_p.shape[1])