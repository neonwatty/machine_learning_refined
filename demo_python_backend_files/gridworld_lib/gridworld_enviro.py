import numpy as np
import time
from IPython import display
import matplotlib.pyplot as plt
import pandas as pd
import matplotlib.colors as mcolors
from matplotlib.colors import LinearSegmentedColormap
import copy

class environment():
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
                self.states.append(str(i) + ',' + str(j))
                
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
       
        ### create custom colormap for gridworld plotting ###
        vmax = 3.0
        self.my_cmap = LinearSegmentedColormap.from_list('mycmap', [(0 / vmax, [0.9,0.9,0.9]),
                                                        (1 / vmax, [1,0.5,0]),
                                                        (2 / vmax, 'lime'),
                                                        (3 / vmax, 'blue')]
                                                        )
        
        # create training episodes
        self.training_episodes = 10000
        if 'training_episodes' in args:
            # define num of training episodes
            self.training_episodes = args['training_episodes']
            
        # make new training start schedule
        self.training_start_schedule = self.make_start_schedule(episodes = self.training_episodes)

        # preset number of training episodes value
        self.validation_episodes = 1000
        if 'validation_episodes' in args:
            # define num of testing episodes
            self.validation_episodes = args['validation_episodes']
            
        # make new testing start schedule
        self.validation_start_schedule = self.make_start_schedule(episodes = self.validation_episodes)
        
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
                        
    ################## environment rules ##################
    # convert tuple location in gridworld to index for Q matrix updating
    def state_tuple_to_index(self,state_tuple):
        index = self.states.index(str(state_tuple[0]) + ',' + str(state_tuple[1]))
        return index
        
    # convert index location to tuple for movement updates
    def state_index_to_tuple(self,state_index):
        tup = self.states[state_index].split(',')
        tup1 = int(tup[0])
        tup2 = int(tup[1])
        tup = [tup1,tup2]
        return tup
        
    # convert action index to action tuple
    def action_index_to_tuple(self,action_index):
        action_tuple = self.action_choices[action_index]
        return action_tuple
    
    ### reward rule ###
    def get_reward(self,state_index):
        r_k = 0
        
        # convert state index to tuple
        state_tuple = self.state_index_to_tuple(state_index)
        
        # if new state is goal set reward of 0
        if state_tuple == self.goal:
            r_k = self.goal_reward
        elif state_tuple in self.hazards:
            r_k = self.hazard_reward
        else:  # standard non-hazard square
            r_k = self.standard_reward
        return r_k          
      
    ### choose next action - left = 0, right = 1, up = 2, down = 3 based on random or exploit method ###
    def get_action(self,**args):
        a_k = 0
        # choose action index randomly
        if args['method'] == 'random':
            a_k = np.random.randint(len(self.action_choices))
            
        # choose action index based on explore/exploit
        elif args['method'] == 'exploit':
            exploit_param = args['exploit_param']
            r = np.random.rand(1)
            if r < exploit_param:
                Q = args['Q']
                s_k_1 = self.state_tuple_to_index(self.agent)
                a_k = np.argmax(Q[s_k_1,:])
            else:
                # pick random actions
                a_k = np.random.randint(len(self.action_choices))
                
        # choose action based on optimal policy
        elif args['method'] == 'optimal':
            Q = args['Q']
            s_k_1 = self.state_tuple_to_index(state_tuple = self.agent)
            a_k = np.argmax(Q[s_k_1,:])
            
        return a_k  # return action index

    ### move according to action ###
    def get_movin(self,**args):
        # get action
        a = args['action']
        
        # update old location
        loc2 = [sum(x) for x in zip(self.agent, self.action_choices[a])] 

        # switch for how to deal with the possibility of new state being outside of gridworld
        if 'illegal_move_response' not in args or args['illegal_move_response'] == 'none':
            # if new state is outside of boundaries of grid world either a) do not move or b) move in random location depending on application
            if loc2[0] > self.height-1 or loc2[0] < 0 or loc2[1] > self.width-1 or loc2[1] < 0:  
                loc2 = self.agent
                
        elif args['illegal_move_response'] == 'random':
            # if loc2 is outside of gridworld, pick action randomly until move in a valid direction
            while loc2[0] > self.height-1 or loc2[0] < 0 or loc2[1] > self.width-1 or loc2[1] < 0:
                a_k = self.get_action(method = 'random')
                loc2 = [sum(x) for x in zip(self.agent, self.action_choices[a_k])]
            
        # convert tuple location to index and return
        s = self.state_tuple_to_index(loc2)
        return s         
    

    
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
                s_k_1 = self.states.index(str(loc[0]) + str(loc[1]))
                a_k = np.argmax(Q[s_k_1,:])
      
                # convert action to new state 
                loc2 = [sum(x) for x in zip(loc, self.action_choices[a_k])]

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
        
    ################## validation functions ##################
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
                s_k_1 = self.states.index(str(loc[0]) + str(loc[1]))
                a_k = np.argmax(Q[s_k_1,:])
      
                # convert action to new state 
                loc2 = [sum(x) for x in zip(loc, self.action_choices[a_k])]

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
       