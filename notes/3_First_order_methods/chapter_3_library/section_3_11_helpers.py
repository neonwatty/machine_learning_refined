import matplotlib.pyplot as plt
import autograd.numpy as np

'''
Plot all three runs
- on the left, all three plotted according to the minibatch run
- on the right, all three plotted according to full batch run
'''
def compare_runs(batch_cost_hist,minibatch_cost_hist,minibatch_cost_hist_2):
    # create figure and color pattern
    fig = plt.figure(figsize = (10,3))
    colors = ['k','magenta','aqua','blueviolet','chocolate']

    ### plot all with respect to smallest mini-batch size ###
    ax = plt.subplot(1,2,1)

    # count number of steps in different runs
    num_stoch = len(minibatch_cost_hist)
    num_mini = len(minibatch_cost_hist_2)
    num_full = len(batch_cost_hist)

    # plot stochastic
    ax.plot(minibatch_cost_hist, label='batch size = 1',c = colors[0],linewidth=1.5)

    # plot mini and full batch with respect to stochastic
    range_mini = np.linspace(0,num_stoch,num_mini)
    ax.plot(range_mini,minibatch_cost_hist_2, label='batch size = 10',c = colors[1],linewidth=1)
    ax.scatter(range_mini,minibatch_cost_hist_2,c=colors[1],s=90,edgecolor = 'w',linewidth=0.5)

    range_full = np.linspace(0,num_stoch,num_full)
    ax.plot(range_full,batch_cost_hist, label='full batch',c = colors[2],linewidth=1)
    ax.scatter(range_full,batch_cost_hist,c=colors[2],s=90,edgecolor = 'w',linewidth=0.5)

    # label panel
    ax.set_title('progress with respect to batch size = 1 method')
    ax.set_xlabel('single summand')
    #plt.legend(loc = 1)

    ### plot with respect to epoch number ###
    ax = plt.subplot(1,2,2)

    # plot stochastic batch
    inds_stoch = np.linspace(0,num_stoch,num_full)
    inds_stoch = [int(v) for v in inds_stoch]
    inds_stoch[-1]-=1
    plot_stoch = [minibatch_cost_hist[v] for v in inds_stoch]
    ax.plot(plot_stoch, label='batch size = 1',c = colors[0],linewidth=1.5)
    ax.scatter(np.arange(num_full),plot_stoch,c=colors[0],s=90,edgecolor = 'w',linewidth=0.5)

    # plot mini and full batch with respect to stochastic
    inds_mini = np.linspace(0,num_mini,num_full)
    inds_mini = [int(v) for v in inds_mini]
    inds_mini[-1]-=1
    plot_mini = [minibatch_cost_hist_2[v] for v in inds_mini]
    ax.plot(plot_mini, label='batch size = 10',c = colors[1],linewidth=1)
    ax.scatter(np.arange(num_full),plot_mini,c=colors[1],s=90,edgecolor = 'w',linewidth=0.5)

    # plot full cost
    ax.plot(batch_cost_hist, label='full batch',c = colors[2],linewidth=1)
    ax.scatter(np.arange(num_full),batch_cost_hist,c=colors[2],s=90,edgecolor = 'w',linewidth=0.5)
    ax.set_title('progress with respect to full batch method')
    ax.set_xticks(np.arange(num_full))
    ax.set_xlabel('full epochs')

    # plot all with respect to epoch number
    plt.show()
