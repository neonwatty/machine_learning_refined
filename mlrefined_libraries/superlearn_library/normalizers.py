import autograd.numpy as np

###### standard normalization function ######
def standard_normalizer(x):
    # compute the mean and standard deviation of the input
    x_means = np.mean(x,axis = 1)[:,np.newaxis]
    x_stds = np.std(x,axis = 1)[:,np.newaxis]   

    # create standard normalizer based on mean / std
    normalizer = lambda data: (data - x_means)/x_stds

    # return normalizer and inverse_normalizer
    return normalizer