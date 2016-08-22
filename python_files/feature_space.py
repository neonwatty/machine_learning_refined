import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches

def plotting(x_train, y_train, f1, f2, title): # f1 and f2 are the index bounds of the two features

    color = ['ro', 'bo']
    for ind, p in enumerate(x_train):
        plt.plot(p[f1[0]:f1[1], f1[2]:f1[3]].sum(), p[f2[0]:f2[1], f2[2]:f2[3]].sum(), color[y_train[ind]])
    
    plt.xlabel('Feature #1')
    plt.ylabel('Feature #2')
    red_patch = mpatches.Patch(color='red', label='0')
    blue_patch = mpatches.Patch(color='blue', label='1')
    plt.legend(handles=[red_patch, blue_patch])
    plt.legend(handles=[red_patch, blue_patch], loc = 2)
    plt.title(title)
    return




