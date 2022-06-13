# This file is associated with the book
# "Machine Learning Refined", Cambridge University Press, 2016.
# by Jeremy Watt, Reza Borhani, and Aggelos Katsaggelos.

import numpy as np
import numpy.matlib
import matplotlib.pyplot as plt
from __future__ import division

def ploy_classification_hw():

    # parameters to play with
    poly_degs = np.arange(1,9)    # range of poly models to compare

    # load data
    X, y = load_data()

    # perform feature transformation + classification
    classify(X,y,poly_degs)
    plt.show()


### builds (poly) features based on input data ###
def poly_features(data, deg):

    # ---> YOUR CODE GOES HERE.

    return F


### sigmoid function for use with log_loss_newton ###
def sigmoid(z):
    return 1/(1+np.exp(-z))


def load_data():
    data = np.array(np.genfromtxt('2eggs_data.csv', delimiter=','))
    X = data[:,0:-1]
    y = data[:,-1]
    return X,y


### plots learned model ###
def plot_poly(w,deg,color):
    # Generate poly seperator
    o = np.arange(0,1,.01)
    s, t = np.meshgrid(o,o)
    s = np.reshape(s,(np.size(s),1))
    t = np.reshape(t,(np.size(t),1))
    f = np.zeros((np.size(s),1))
    count = 0

    for n in np.arange(0,deg+1):
        for m in np.arange(0,deg+1):
            if (n + m <= deg):
                f = f + w[count]*((s**n)*(t**m))
                count = count + 1

    s = np.reshape(s,(np.size(o),np.size(o)))
    t = np.reshape(t,(np.size(o),np.size(o)))
    f = np.reshape(f,(np.size(o),np.size(o)))

    # plot contour in original space
    plt.contour(s,t,f,1, color ='k')
    plt.gca().xaxis.set_major_locator(plt.NullLocator())
    plt.gca().yaxis.set_major_locator(plt.NullLocator())
    plt.hold(True)

### plots points for each fold ###
def plot_pts(X,y):

    # plot training set
    ind = np.nonzero(y==1)[0]
    plt.plot(X[ind,0],X[ind,1],'ro')
    ind = np.nonzero(y==-1)[0]
    plt.plot(X[ind,0],X[ind,1],'bo')
    plt.hold(True)


### plots training errors ###
def plot_errors(poly_degs, errors):

    plt.plot(np.arange(1,np.size(poly_degs)+1), errors,'m--')
    plt.plot(np.arange(1,np.size(poly_degs)+1), errors,'mo')

    #ax2.set_aspect('equal')
    plt.xlabel('D')
    plt.ylabel('error')


### newton's method for log-loss classifier ###
def log_loss_newton(D,b):
    # initialize
    w = np.random.randn(np.shape(D)[0],1)

    # precomputations
    H = np.dot(np.diag(b),D.T)
    grad = 1
    n = 1

    ### main ###
    while (n <= 30) & (np.linalg.norm(grad) > 1e-5):

        # prep gradient for logistic objective
        r = sigmoid(-np.dot(H,w))
        g = r*(1 - r)
        grad = -np.dot(H.T,r)



        hess = np.dot(D,np.dot(np.diag(np.ravel(g)),D.T))

        # take Newton step
        s = np.dot(hess,w) - grad
        w = np.dot(np.linalg.pinv(hess),s)
        n = n + 1

    return w


def classify(X,y,poly_degs):

    errors = []
    fig1 = plt.figure(facecolor = 'white')
    # solve for weights and collect errors
    for i in np.arange(1,np.shape(poly_degs)[0]+1):
        # generate features
        poly_deg = poly_degs[i-1]
        F = poly_features(X,poly_deg)

        # run logistic regression
        w = log_loss_newton(F.T,y)

        # output model
        ax1 = fig1.add_subplot(2,4,i)
        plot_poly(w, poly_deg,'b')
        title = 'D = ' + str(i)
        plt.title(title, fontsize = 12)
        plot_pts(X,y)

        # calculate training errors
        resid = evaluate(F,y,w)
        errors.append(resid)

    # plot training errors for visualization
    fig2 = plt.figure(facecolor = 'white')
    plot_errors(poly_degs, errors)


### evaluates error of a learned model ###
def evaluate(A,b,w):
    s = np.dot(A,w)
    s[np.nonzero(s>0)] = 1
    s[np.nonzero(s<=0)] = -1
    t = s*np.reshape(b,(np.size(b),1))
    t[np.nonzero(t<0)] = 0
    score = 1 - (t.sum()/np.size(t))
    return score


ploy_classification_hw()
