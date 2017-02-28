import numpy as np
import time
from IPython import display
import matplotlib.pyplot as plt
import pandas as pd
import matplotlib.colors as mcolors
from matplotlib.colors import LinearSegmentedColormap
import matplotlib.animation as animation
from JSAnimation import IPython_display
import copy

class gridworld():
    def __init__(self,**args):
                
        ### initialize global containers and variables
        # initialize containers for grid, hazard locations, agent and goal locations, etc.,
        self.grid = []
        self.hazards = []
        self.agent = []
        self.goal = []
        self.training_episodes_history_v1 = []
        self.training_episodes_history_v2 = []
        self.training_start_schedule = []  # container for holding starting positions for traininig
        self.validation_start_schedule = []   # container for holding starting psoitions for validation
        self.training_reward_v1 = []
        self.training_reward_v2 = []
        self.validation_reward_v1 = []
        self.validation_reward_v2 = []
        
        # initialize global variables e.g., height and width of gridworld, hazard penalty value
        self.width = 0
        self.height = 0
        self.hazard_reward = 0
        self.goal_reward = 0
        self.standard_reward = -1
        self.num_episodes = 0
        self.training_episodes = 0
        self.validation_epislodes = 0
        self.world_size = ''
        self.world_type = ''
        
        # setup world
        world_name = ''
        if "world_size" not in args:
            print 'world_size parameter required, choose either small or large'
            return
        
        if "world_type" not in args:
            print 'world_type parameter required, choose maze, random, or moat'

        ### set world size ###    
        if args["world_size"] == 'small':
            self.world_size = 'small'
            self.width = 13
            self.height = 11

        if args["world_size"] == 'large':
            self.world_size = 'large'
            self.width = 41
            self.height = 15

        ### initialize grid based on world_size ###
        self.grid = np.zeros((self.height,self.width))

        # index states for Q matrix
        self.states = []
        for i in range(self.height):
            for j in range(self.width):
                block = [i,j]
                self.states.append(str(i) + str(j))
                
        ### with world type load in hazards ###
        if args["world_type"] == 'maze':
            self.world_type = 'maze'
            self.agent = [self.height-2, 0]   # initial location agent
            self.goal = [self.height-2, self.width-1]     # goal block                

        if args["world_type"] == 'random':
            self.world_type = 'random'
            self.agent = [0,0]   # initial location agent
            self.goal = [0,self.width-1]     # goal block

        if args["world_type"] == 'moat':
            self.world_type = 'moat'
            self.agent = [0,0]   # initial location agent
            self.goal = [0,self.width-1]     # goal block

        ### load in hazards for given world size and type ###    
        hazard_csvname = 'demo_datasets/RL_datasets/' + args["world_size"] + '_' + args["world_type"] + '_hazards.csv'

        # load in preset hazard locations from csv
        self.hazards = pd.read_csv(hazard_csvname,header = None)

        # setup hazard reward value
        self.hazard_reward = -200 
        if 'hazard_reward' in args:
            self.hazard_reward = args['hazard_reward'] 
            
        # initialize hazards locations
        temp = []
        for i in range(len(self.hazards)):
            block = list(self.hazards.iloc[i])
            self.grid[block[0]][block[1]] = 1   
            temp.append(block)

        # initialize goal location
        self.hazards = temp
                
        ### initialize state index, Q matrix, and action choices ###
        # initialize action choices
        self.action_choices = [[-1,0],[1,0],[0,-1],[0,1]]
        
        # initialize Q^* matrix
        self.Q_star_v1 = np.zeros((self.width*self.height,len(self.action_choices)))
        self.Q_star_v2 = np.zeros((self.width*self.height,len(self.action_choices)))

        ### create custom colormap for gridworld plotting ###
        vmax = 3.0
        self.my_cmap = LinearSegmentedColormap.from_list('mycmap', [(0 / vmax, [0.9,0.9,0.9]),
                                                        (1 / vmax, [1,0.5,0]),
                                                        (2 / vmax, 'lime'),
                                                        (3 / vmax, 'blue')]
                                                        )
        
        ### load preset random training start schedule or create new one ###
        train_csvname = 'demo_datasets/RL_datasets/' +  args['world_size'] + '_' + args['world_type'] + '_training_start_schedule.csv'
        if 'training_episodes' in args:
            # define num of training episodes
            self.training_episodes = args['training_episodes']
            
            # make new training start schedule
            self.training_start_schedule = self.make_start_schedule(episodes = self.training_episodes)
        
            # save new training start schedule for future use
            b = pd.DataFrame(self.training_start_schedule)
            b.to_csv(train_csvname,header = None,index = False)

        else:    # load pre-set random starting positions
            # preset number of training episodes value
            self.training_episodes = 200
            
            # load in preset random training starting positions
            temp = pd.read_csv(train_csvname,header = None)
            
            self.training_start_schedule = []
            for i in range(len(temp)):
                block = list(temp.iloc[i])
                self.training_start_schedule.append(block)
                
        ### load preset random validation start schedule or create new one ###
        validation_csvname = 'demo_datasets/RL_datasets/' + args['world_size'] + '_' + args['world_type'] + '_validation_start_schedule.csv'
        if 'validation_episodes' in args:
            # define num of testing episodes
            self.validation_episodes = args['validation_episodes']
            
            # make new testing start schedule
            self.validation_start_schedule = self.make_start_schedule(episodes = self.validation_episodes)
        
            # save new testing start schedule for future use
            b = pd.DataFrame(self.validation_start_schedule)
            b.to_csv(validation_csvname,header = None,index = False)

        else:    # load pre-set random starting positions
            # preset number of training episodes value
            self.validation_episodes = 200
            
            # load in preset random training starting positions
            temp = pd.read_csv(validation_csvname,header = None)
            
            self.validation_start_schedule = []
            for i in range(len(temp)):
                block = list(temp.iloc[i])
                self.validation_start_schedule.append(block)
                
        # default parameters for the qlearning 
        self.gamma = 0.8                           # short term / long term learning tradeoff param
        self.explore_exploit_param = 0.8           # exploration exploitation param tradeoff (with probability less than this param action chosen by exploitation)
        self.max_steps = 5*self.width*self.height  # maximum number of steps per episode

    ### world coloring function ###
    def color_gridworld(self,**args):
        # copy grid for plotting, add agent and goal location
        p_grid = copy.deepcopy(self.grid)
        p_grid[self.goal[0]][self.goal[1]] = 2   
        p_grid[self.agent[0]][self.agent[1]] = 3   
        
        # plot gridworld
        ax = 0
        if 'ax' in args:
            ax = args['ax']
        else: 
            fsize = 4
            if self.width > 20:
                fsize = 8
            fig = plt.figure(figsize = (fsize,fsize),frameon=False)
            ax = fig.add_subplot(111, aspect='equal')

        ax.pcolormesh(p_grid,edgecolors = 'k',linewidth = 0.01,cmap = self.my_cmap)

        # clean up plot
        ax.axis('off')
        ax.set_xlim(-0.1,self.width + 1.1);
        ax.set_ylim(-0.1,self.height + 1.1);  

 
    ### create starting schedule - starting position of each episode of training or testing ###
    def make_start_schedule(self,**args):
        num_episodes = args['episodes']
        start_schedule = []
        
        # create schedule of random starting positions for each episode
        if 'start_schedule' not in args or ('start_schedule' in args and args['start_schedule'] == 'random'):
            for i in range(num_episodes):
                loc = [np.random.randint(self.height),np.random.randint(self.width)]
                start_schedule.append(loc)
                
        # create exhaustive starting schedule - cycle through states sequentially
        if 'start_schedule' in args and args['start_schedule'] == 'exhaustive':
            i = 0
            while i <= num_episodes:
                for j in range(self.width):
                    for k in range(self.height):
                        loc = [j,k]
                        start_schedule.append(loc)
                        i+=1
        
        return start_schedule
                        
    ################## q-learning functions ##################
    def get_reward(self,location):
        r_k = 0

        # if new state is goal set reward of 0
        if location == self.goal:
            r_k = self.goal_reward
        elif location in self.hazards:
            r_k = self.hazard_reward
        else:  # standard non-hazard square
            r_k = self.standard_reward
        return r_k          
    
    ### save Q matrix ###
    def save_qmat(self,Q,csvname):
        states_print = []
        for i in range(len(self.states)):
            s = str(self.states[i])
            t = str('(') + s[0] + ',' + s[1] + str(')')
            states_print.append(t)
            
        df = pd.DataFrame(Q,columns=['up','down','left','right'], index=states_print)
        df.to_csv('demo_datasets/RL_datasets/' + self.world_size + '_' + self.world_type + '_' +  csvname + '.csv')
        
    ### Q-learning function - version 1 - random actions ###
    def qlearn_v1(self,**args):
        ### set basic parameters controlling training regiment ###
        # change these default parameters if user requests
        num_episodes = self.training_episodes
        if "gamma" in args:
            self.gamma = args['gamma']
        if 'max_steps' in args:
            self.max_steps = args['max_steps']    
        
        ### start main Q-learning loop ###
        self.training_episodes_history_v1 = {}
        self.training_reward_v1 = []
        self.validation_reward_v1 = []
        self.Q_star_v1 = np.zeros((self.width*self.height,len(self.action_choices)))
        for n in range(num_episodes): 
            # pick this episode's starting position
            loc = self.training_start_schedule[n]

            # update Q matrix while loc != goal
            episode_history = []      # container for storing this episode's journey
            episode_history.append(loc)
            total_episode_reward = 0
            for step in range(self.max_steps):    
                ### if you reach the goal end current episode immediately
                if loc == self.goal:
                    break
                    
                ### choose next action - left = 0, right = 1, up = 2, down = 3 --> if this leads you outside the gridworld you don't move ###
                # pick random actions
                k = np.random.randint(len(self.action_choices))  
                
                # update old location
                loc2 = [sum(x) for x in zip(loc, self.action_choices[k])] 
                ind_old = self.states.index(str(loc[0]) + str(loc[1]))

                # if new state is outside of boundaries of grid world do not move
                if loc2[0] > self.height-1 or loc2[0] < 0 or loc2[1] > self.width-1 or loc2[1] < 0:  
                    loc2 = loc
                    
                ### update episode history container
                episode_history.append(loc2)
                    
                ### recieve reward     
                r_k = self.get_reward(loc2)
                total_episode_reward += r_k
                
                ### Update Q function
                ind_new = self.states.index(str(loc2[0]) + str(loc2[1]))
                self.Q_star_v1[ind_old,k] = r_k + self.gamma*max(self.Q_star_v1[ind_new,:])
                    
                ### update current location of agent to one we just moved too (or stay still if grid world boundary met)
                self.agent = loc2
                loc = loc2
                
            # print out update
            if np.mod(n+1,50) == 0:
                print 'training episode ' + str(n+1) +  ' of ' + str(num_episodes) + ' complete'
            
            ### store this episode's training reward history
            self.training_episodes_history_v1[str(n)] = episode_history
            self.training_reward_v1.append(total_episode_reward)
            
            ### store this episode's validation reward history
            if 'validate' in args:
                if args['validate'] == True:
                    reward = self.run_validation_episode(self.Q_star_v1)
                    self.validation_reward_v1.append(reward)

        # save Q function
        csvname = 'algo1_Qmat'
        self.save_qmat(self.Q_star_v1,csvname)
            
        print 'q-learning version 1 algorithm complete'

    ### Q-learning function - version 2 - with exploration vs exploitation chosen actions ###
    def qlearn_v2(self,**args):
        ### set basic parameters controlling training regiment ###

        # change these default parameters if user requests
        num_episodes = self.training_episodes
        if "gamma" in args:
            self.gamma = args['gamma']
        if 'max_steps' in args:
            self.max_steps = args['max_steps']  
        if 'explore_exploit_param' in args:
            self.explore_exploit_param = args['explore_exploit_param']

        ### start main Q-learning loop ###
        self.training_episodes_history_v2 = {}
        self.training_reward_v2 = []
        self.validation_reward_v2 = []
        self.Q_star_v2 = np.zeros((self.width*self.height,len(self.action_choices)))
        for n in range(num_episodes):    
            # pick this episode's starting position
            loc = self.training_start_schedule[n]
            
            # update Q matrix while loc != goal
            episode_history = []      # container for storing this episode's journey
            episode_history.append(loc)
            total_episode_reward = 0
            for step in range(self.max_steps):    
                ### if you reach the goal end current episode immediately
                if loc == self.goal:
                    break
                    
                ### choose next action - left = 0, right = 1, up = 2, down = 3 --> if this leads you outside the gridworld you don't move ###
                k = 0
                # choose action based on max
                r = np.random.rand(1)
                if r < self.explore_exploit_param:
                    ind_old = self.states.index(str(loc[0]) + str(loc[1]))
                    k = np.argmax(self.Q_star_v2[ind_old,:])
                else:
                    # pick random actions
                    k = np.random.randint(len(self.action_choices))
                
                # update old location
                loc2 = [sum(x) for x in zip(loc, self.action_choices[k])] 
                ind_old = self.states.index(str(loc[0]) + str(loc[1]))

                # if new state is outside of boundaries of grid world do not move
                if loc2[0] > self.height-1 or loc2[0] < 0 or loc2[1] > self.width-1 or loc2[1] < 0:  
                    loc2 = loc
                    
                ### update episode history container
                episode_history.append(loc2)
                    
                ### recieve reward     
                r_k = self.get_reward(loc2)
                total_episode_reward += r_k
                
                ### Update Q function
                ind_new = self.states.index(str(loc2[0]) + str(loc2[1]))
                self.Q_star_v2[ind_old,k] = r_k + self.gamma*max(self.Q_star_v2[ind_new,:])
                    
                ### update current location of agent to one we just moved too (or stay still if grid world boundary met)
                self.agent = loc2
                loc = loc2
             
            # print out update
            if np.mod(n+1,50) == 0:
                print 'training episode ' + str(n+1) +  ' of ' + str(num_episodes) + ' complete'
            
            ### store this episode's history
            self.training_episodes_history_v2[str(n)] = episode_history
            self.training_reward_v2.append(total_episode_reward)
            
            ### store this episode's validation reward history
            if 'validate' in args:
                if args['validate'] == True:
                    reward = self.run_validation_episode(self.Q_star_v2)
                    self.validation_reward_v2.append(reward)
            
        print 'q-learning version 2 algorithm complete'
        
        # save Q function
        csvname = 'qlearn_algo2_Qmat'
        self.save_qmat(self.Q_star_v2,csvname)
            
    ################## animation functions ##################
    ### animate training episode from one version of q-learning ###
    def animate_single_version_training_episodes(self,episodes):
        # initialize figure
        fig = plt.figure(figsize = (10,3))
        axs = []
        for i in range(len(episodes)):
            ax = fig.add_subplot(1,len(episodes),i+1)
            axs.append(ax)
            
        if len(episodes) == 1:
            axs = np.array(axs)

        # make a copy of the original gridworld - set at initialization
        gridworld_orig = self.grid.copy()
        
        # compute maximum length of episodes animated
        max_len = 0
        for key in episodes:
            l = len(self.training_episodes_history_v1[str(key)])
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
                current_episode = self.training_episodes_history_v1[str(episode_num)]
                                
                # define new location of agent
                loc = current_episode[min(step,len(current_episode)-1)]
                self.agent = loc
                
                # color gridworld for this episode and step
                self.color_gridworld(ax = ax)
                ax.set_title('episode = ' + str(episode_num + 1))
            return artist,
           
        # create animation object
        anim = animation.FuncAnimation(fig, show_episode,frames=min(100,max_len), interval=min(100,max_len), blit=True)
        
        # set frames per second in animation
        IPython_display.anim_to_html(anim,fps = min(100,max_len)/float(5))
    
        return(anim)
    
    ### animate training episode from one version of q-learning ###
    def compare_v1v2_training_episode(self,episode_to_compare):
        # initialize figure
        episode_to_compare = episode_to_compare - 1
        fig = plt.figure(figsize = (10,3))
        axs = []
        for i in range(2):
            ax = fig.add_subplot(1,2,i+1)
            axs.append(ax)

        # make a copy of the original gridworld - set at initialization
        gridworld_orig = self.grid.copy()
        
        # compute maximum length of episodes animated
        max_len = 0
        key = episode_to_compare
        L1 = len(self.training_episodes_history_v1[str(key)])
        L2 = len(self.training_episodes_history_v2[str(key)])
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
                    current_episode = self.training_episodes_history_v1[str(key)]
                else:
                    current_episode = self.training_episodes_history_v2[str(key)]
                        
                # define new location of agent
                loc = current_episode[min(step,len(current_episode)-1)]
                self.agent = loc
                
                # recieve reward     
                r_k = self.get_reward(loc)
                rewards[k] += r_k
                
                # color gridworld for this episode and step
                self.color_gridworld(ax = ax)
#                 ax.set_title('total reward = ' + str(rewards[k][0]),fontsize = 18)                
            return artist,
           
        # create animation object
        anim = animation.FuncAnimation(fig, show_episode,frames=min(100,max_len), interval=min(100,max_len), blit=True)
        
        # set frames per second in animation
        IPython_display.anim_to_html(anim,fps = min(100,max_len)/float(5))
    
        return(anim)
    
    ### a simple plot comparing qlearning v1 and v2 training costs ###
    def compare_v1v2_training_rewards(self):
        # initialize figure
        fig = plt.figure(figsize = (12,4))
        
        # plot each reward history
        plt.plot(self.training_reward_v1,color = 'b',linewidth = 2)
        plt.plot(self.training_reward_v2,color = 'r',linewidth = 2)
        
        # clean up panel
        ymin = min(min(self.training_reward_v1),min(self.training_reward_v2))
        ymax = max(max(self.training_reward_v1),max(self.training_reward_v2))
        ygap = abs((ymax - ymin)/float(10))
        plt.ylim([ymin - ygap,ygap])
        plt.xlabel('episode')
        plt.ylabel('total reward')
        plt.title('qlearn v1 vs v2 training rewards per episode')
        plt.legend(['qlearn v1','qlearn v2'],loc='center left', bbox_to_anchor=(1, 0.5))

    ################## test trained agent ##################
    # animate the testing phase based on completed Q-learning cycle
    def animate_trained_run(self,loc,algorithm):
        # show movement based on an initial 
        self.agent = loc # initial agent location

        # if you chose an invalid starting position, break out and try again
        if  loc == self.goal or loc[0] > self.height-1 or loc[0] < 0 or loc[1] > self.width-1 or loc[1] < 0:
            print 'initialization is outside of gridworld or is goal, try again'
            return
            
        ### produce sequence run starting at user-defined loc ###
        episode_history = []
        loc2 = [-1,-1]
        for i in range(self.max_steps):
            # record location
            episode_history.append(loc)

            ### if you reach the goal end current episode immediately
            if loc == self.goal:
                break
                    
            # pick action
            k = 0
            ind_old = self.states.index(str(loc[0]) + str(loc[1]))
            if algorithm == 1:
                k = np.argmax(self.Q_star_v1[ind_old,:])
            else:
                k = np.argmax(self.Q_star_v2[ind_old,:]) 
                
            # convert action to new sstate 
            loc2 = [sum(x) for x in zip(loc, self.action_choices[k])]
            
            # if loc2 is outside of gridworld, pick action randomly until move in a valid direction
            while loc2[0] > self.height-1 or loc2[0] < 0 or loc2[1] > self.width-1 or loc2[1] < 0:
                k = np.random.randint(len(self.action_choices))
                loc2 = [sum(x) for x in zip(loc, self.action_choices[k])]

            # append location
            loc = copy.deepcopy(loc2)
        
        ### animate run ###
        # initialize figure
        fig = plt.figure(figsize = (4,4))
        ax = fig.add_subplot(111, aspect='equal',frameon=False)
            
        # animation function
        def show_episode(step):
            # loop over subplots and plot current step of each episode history
            artist = fig
            
            # define new location of agent
            loc = episode_history[step]
            self.agent = loc

            # color gridworld for this episode and step
            self.color_gridworld(ax = ax)
            fig.subplots_adjust(left=0,right=1,bottom=0,top=1)  ## gets rid of the white space around image

            return artist,
           
        # create animation object
        anim = animation.FuncAnimation(fig, show_episode,frames=min(100,len(episode_history)), interval=min(100,len(episode_history)), blit=True)
        
        # set frames per second in animation
        IPython_display.anim_to_html(anim,fps = min(100,len(episode_history))/float(5))
    
        return(anim)
    
    ################## validate trained agent ##################
    # run validation experiments
    def run_validation_episode(self,Q):
        # run validation episodes
        total_reward = []
        
        # run over validation episodes
        for i in range(self.validation_episodes):  
            
            # get this episode's starting position
            loc = self.validation_start_schedule[i]
            
            # reward container for this episode
            episode_reward = 0
            
            # run over steps in single episode
            for j in range(self.max_steps):
                ### if you reach the goal end current episode immediately
                if loc == self.goal:
                    break

                # choose new action based on policy
                ind_old = self.states.index(str(loc[0]) + str(loc[1]))
                k = np.argmax(Q[ind_old,:])
      
                # convert action to new state 
                loc2 = [sum(x) for x in zip(loc, self.action_choices[k])]

                # if loc2 is outside of gridworld, pick action randomly until move in a valid direction
                while loc2[0] > self.height-1 or loc2[0] < 0 or loc2[1] > self.width-1 or loc2[1] < 0:
                    k = np.random.randint(len(self.action_choices))
                    loc2 = [sum(x) for x in zip(loc, self.action_choices[k])]
                
                # update episode reward
                r_k = self.get_reward(loc2)
                episode_reward += r_k
                
                # update location
                loc = copy.deepcopy(loc2)
                
            # after each episode append to total reward
            total_reward.append(episode_reward)
        
        # return total reward
        return np.median(total_reward)
    
    ### a simple plot comparing qlearning v1 and v2 training costs ###
    def show_single_validation_history(self,**args):
        # grab series
        series = args['validation_history']
        
        # initialize figure
        fig = plt.figure(figsize = (12,4))
        
        # plot each reward history
        plt.plot(series,color = 'b',linewidth = 2)
        
        # clean up panel
        ymin = min(series)
        ymax = max(series)
        ygap = abs((ymax - ymin)/float(10))
        plt.ylim([ymin - ygap,ygap])
        plt.xlabel('episode')
        plt.ylabel('total reward')
        plt.title('validation history')
        
    ### a simple plot comparing qlearning v1 and v2 training costs ###
    def compare_v1v2_validation_rewards(self):
        # initialize figure
        fig = plt.figure(figsize = (12,4))
        
        # plot each reward history
        plt.plot(self.validation_reward_v1,color = 'b',linewidth = 2)
        plt.plot(self.validation_reward_v2,color = 'r',linewidth = 2)
        
        # clean up panel
        ymin = min(min(self.validation_reward_v1),min(self.validation_reward_v2))
        ymax = max(max(self.validation_reward_v1),max(self.validation_reward_v2))
        ygap = abs((ymax - ymin)/float(10))
        plt.ylim([ymin - ygap,ygap])
        plt.xlabel('episode')
        plt.ylabel('total reward')
        plt.title('qlearn v1 vs v2 validation rewards per episode')
        plt.legend(['qlearn v1','qlearn v2'],loc='center left', bbox_to_anchor=(1, 0.5))
        
    # animate validation runs
    def show_validation_runs(self,Q,starting_locs):
        # initialize figure
        fsize = 3
        if len(starting_locs) > 1:
            fsize = 13
        fig = plt.figure(figsize = (fsize,3))
        axs = []
        for i in range(len(starting_locs)):
            ax = fig.add_subplot(1,len(starting_locs),i+1)
            axs.append(ax)

        if len(starting_locs) == 1:
            axs = np.array(axs)

        # make a copy of the original gridworld - set at initialization
        gridworld_orig = self.grid.copy()
        
        ### produce validation runs ###
        validation_run_history = []
        for i in range(len(starting_locs)):
            # take random starting point - for short just from validation schedule
            loc = starting_locs[i]
            
            # loop over max number of steps and try reach goal
            episode_path = [loc]
            for j in range(self.max_steps):
                # if you reach the goal end current episode immediately
                if loc == self.goal:
                    break

                # choose new action based on policy
                ind_old = self.states.index(str(loc[0]) + str(loc[1]))
                k = np.argmax(Q[ind_old,:])
      
                # convert action to new state 
                loc2 = [sum(x) for x in zip(loc, self.action_choices[k])]

                # if loc2 is outside of gridworld, pick action randomly until move in a valid direction
                while loc2[0] > self.height-1 or loc2[0] < 0 or loc2[1] > self.width-1 or loc2[1] < 0:
                    k = np.random.randint(len(self.action_choices))
                    loc2 = [sum(x) for x in zip(loc, self.action_choices[k])]
                
                # record next step in path
                loc = loc2
                episode_path.append(loc)
                
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
                self.agent = loc

                # color gridworld for this episode and step
                self.color_gridworld(ax = ax)
                ax.set_title('fully trained run ' + str(k + 1))
            return artist,

        # create animation object
        anim = animation.FuncAnimation(fig, show_episode,frames=min(50,max_len), interval=min(50,max_len), blit=True)

        # set frames per second in animation
        IPython_display.anim_to_html(anim,fps = min(50,max_len)/float(5))

        return(anim)