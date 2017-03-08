import matplotlib.pyplot as plt
import numpy as np
import matplotlib.animation as animation
from JSAnimation import IPython_display

class animator():
        
    ################## animation functions ##################
    ### animate validation runs ###
    def animate_validation_runs(self,gridworld,learner,starting_locations):
        # make local copies of input
        grid = gridworld
        Q = learner.Q
        starting_locs = starting_locations
        
        # initialize figure
        fig = plt.figure(figsize = (10,3))
        axs = []
        for i in range(len(starting_locs)):
            ax = fig.add_subplot(1,len(starting_locs),i+1,aspect = 'equal')
            axs.append(ax)
        
        # only one added subplot, axs must be in array format
        if len(starting_locs) == 1:
            axs = np.array(axs)
        
        ### produce validation runs ###
        validation_run_history = []
        for i in range(len(starting_locs)):
            # take random starting point - for short just from validation schedule
            grid.agent = starting_locs[i]
            
            # loop over max number of steps and try reach goal
            episode_path = []
            for j in range(grid.max_steps):
                # store current location
                episode_path.append(grid.agent)
                
                ### if you reach the goal end current episode immediately
                if grid.agent == grid.goal:
                    break
                
                # translate current agent location tuple into index
                s_k_1 = grid.state_tuple_to_index(grid.agent)
                    
                # get action
                a_k = grid.get_action(method = 'optimal',Q = Q)
                
                # move based on this action - if move takes you out of gridworld don't move and instead move randomly 
                s_k = grid.get_movin(action = a_k, illegal_move_response = 'random')
  
                # record next step in path
                grid.agent = grid.state_index_to_tuple(state_index = s_k)
                
            # record this episode's path
            validation_run_history.append(episode_path)
        
        ### compute maximum length of episodes animated ###
        max_len = 0
        for i in range(len(starting_locs)):
            l = len(validation_run_history[i])
            if l > max_len:
                max_len = l
    
        ### loop over the episode histories and plot the results ###
        def show_episode(step):
            # loop over subplots and plot current step of each episode history
            artist = fig

            for k in range(len(axs)):
                ax = axs[k]

                # take correct episode
                current_episode = validation_run_history[k]

                # define new location of agent
                loc = current_episode[min(step,len(current_episode)-1)]
                grid.agent = loc

                # color gridworld for this episode and step
                grid.color_gridworld(ax = ax)
                ax.set_title('fully trained run ' + str(k + 1))
                # fig.subplots_adjust(left=0,right=1,bottom=0,top=1)  ## gets rid of the white space around image

            return artist,

        # create animation object
        anim = animation.FuncAnimation(fig, show_episode,frames=min(100,max_len), interval=min(100,max_len), blit=True)

        # set frames per second in animation
        IPython_display.anim_to_html(anim,fps = min(100,max_len)/float(5))

        return(anim)
    
    ### compare training episodes from two q-learning settings ###    
    def animate_training_comparison(self,gridworld,learner_1,learner_2,episode):
        # make local copies of input
        grid = gridworld
        training_episodes_history_v1 = learner_1.training_episodes_history
        training_episodes_history_v2 = learner_2.training_episodes_history
        
        # initialize figure
        fig = plt.figure(figsize = (10,3))
        axs = []
        for i in range(2):
            ax = fig.add_subplot(1,2,i+1,aspect = 'equal')
            axs.append(ax)

        # compute maximum length of episodes animated
        max_len = 0
        key = episode
        L1 = len(training_episodes_history_v1[str(key)])
        L2 = len(training_episodes_history_v2[str(key)])
        max_len = max(L1,L2)
        
        # loop over the episode histories and plot the results
        rewards =  np.zeros((2,1))
        def show_episode(step):
            # loop over subplots and plot current step of each episode history
            artist = fig
            for k in range(len(axs)):
                ax = axs[k]
                
                # take correct episode
                current_episode = 0
                if k == 0:
                    current_episode = training_episodes_history_v1[str(key)]
                else:
                    current_episode = training_episodes_history_v2[str(key)]
                        
                # define new location of agent
                grid.agent = current_episode[min(step,len(current_episode)-1)]
                
                # color gridworld for this episode and step
                grid.color_gridworld(ax = ax)
            return artist,
           
        # create animation object
        anim = animation.FuncAnimation(fig, show_episode,frames=min(100,max_len), interval=min(100,max_len), blit=True)
        
        # set frames per second in animation
        IPython_display.anim_to_html(anim,fps = min(100,max_len)/float(5))
    
        return(anim)
    
    ### animate training episode from one version of q-learning ###
    def animate_training_runs(self,gridworld,learner,episodes):  
        # make local copies of input
        grid = gridworld
        training_episodes_history = learner.training_episodes_history

        # initialize figure
        fig = plt.figure(figsize = (10,3))
        axs = []
        for i in range(len(episodes)):
            ax = fig.add_subplot(1,len(episodes),i+1,aspect = 'equal')
            axs.append(ax)
            
        if len(episodes) == 1:
            axs = np.array(axs)
                
        # compute maximum length of episodes animated
        max_len = 0
        for key in episodes:
            l = len(training_episodes_history[str(key)])
            if l > max_len:
                max_len = l

        # loop over the episode histories and plot the results
        def show_episode(step):
            # loop over subplots and plot current step of each episode history
            artist = fig

            for k in range(len(axs)):
                ax = axs[k]
                
                # take correct episode
                episode_num = episodes[k]
                current_episode = training_episodes_history[str(episode_num)]
                                
                # define new location of agent
                grid.agent = current_episode[min(step,len(current_episode)-1)]
                
                # color gridworld for this episode and step
                grid.color_gridworld(ax = ax)
                ax.set_title('episode = ' + str(episode_num + 1))
                
            return artist,
           
        # create animation object
        anim = animation.FuncAnimation(fig, show_episode,frames=min(100,max_len), interval=min(100,max_len), blit=True)
        
        # set frames per second in animation
        IPython_display.anim_to_html(anim,fps = min(100,max_len)/float(5))
    
        return(anim)