from large_gridworld_ipythonblocks import BlockGrid
import numpy as np
import time
from IPython import display
import matplotlib.pyplot as plt
import pandas as pd

class my_gridworld():
    
    def __init__(self,grid_size):
        # use either a small pre-staged grid world or big one with random hazards
        
        ### initialize grid, agent, obstacles, etc.,
        self.width = 5
        self.height = 4
        self.grid = BlockGrid(self.width,self.height, fill=(200, 200, 200))
        
        # decide on hazards and goal locations
        self.hazards = [[0,3],[1,3],[2,3]]  # impenetrable obstacle locations          
        self.goal = [0,4]     # goal block
        self.player = [0,0]   # initial location player
        
        # index states for Q matrix
        self.states = []
        for i in range(self.grid.height):
            for j in range(self.grid.width):
                block = [i,j]
                self.states.append(str(i) + str(j))
        
        if grid_size == 'large':
            ### initialize grid, agent, obstacles, etc.,
            self.width = 20
            self.height = 10
            self.grid = BlockGrid(self.width,self.height, fill=(200, 200, 200))

            # decide on player and goal locations
            self.goal = [0,self.width-1]     # goal block
            self.player = [0,0]   # initial location player
            
            # index states for Q matrix
            self.states = []
            for i in range(self.grid.height):
                for j in range(self.grid.width):
                    block = [i,j]
                    self.states.append(str(i) + str(j))

            # decide on hazard locations
            num_hazards = 50
            self.hazards = []
            inds = np.random.permutation(self.grid.width*self.grid.height)
            inds = inds[:num_hazards]
            k = 0
            for i in range(self.grid.height):
                for j in range(self.grid.width):
                    if k in inds: 
                        block = [i,j]
                        if block != self.goal:
                            self.hazards.append(block)
                    k+=1
                                  
        if grid_size[0:4] == 'maze':
            
            df = pd.read_csv('maze_small.csv',header = None)
            ### initialize grid, agent, obstacles, etc.,            
            self.width = 13
            self.height = 11
       
            if grid_size == 'maze_large':
                df = pd.read_csv('maze_large.csv',header = None)
                ### initialize grid, agent, obstacles, etc.,            
                self.width = 41
                self.height = 15

            self.grid = BlockGrid(self.width,self.height, fill=(200, 200, 200))

            # decide on player and goal locations
            self.goal = [self.height-2, self.width-1]     # goal block
            self.player = [self.height-2, 0]   # initial location player
            
            # index states for Q matrix
            self.states = []
            for i in range(self.grid.height):
                for j in range(self.grid.width):
                    block = [i,j]
                    self.states.append(str(i) + str(j))

            # decide on hazard locations
            self.hazards = []
            inds = df.values[0]
            k = 0
            for i in range(self.grid.height):
                for j in range(self.grid.width):
                    if k in inds: 
                        block = [i,j]
                        if block != self.goal:
                            self.hazards.append(block)
                    k+=1   
                        
        # initialize action choices
        self.action_choices = [[-1,0],[1,0],[0,-1],[0,1]]
        
        # initialize Q^* matrix
        self.Q_star = np.zeros((self.grid.width*self.grid.height,len(self.action_choices)))
        
    def color_grid(self):                            
        # remake + recolor grid
        self.grid = BlockGrid(self.width,self.height, fill=(200, 200, 200))

        # color obstacles
        for i in range(len(self.hazards)):
            self.grid[self.hazards[i][0],self.hazards[i][1]].green = 100
            self.grid[self.hazards[i][0],self.hazards[i][1]].red = 250
            self.grid[self.hazards[i][0],self.hazards[i][1]].blue = 0

        # make and color goal
        self.grid[self.goal[0],self.goal[1]].green = 255
        self.grid[self.goal[0],self.goal[1]].red = 0
        self.grid[self.goal[0],self.goal[1]].blue = 0
        
        # color player location
        self.grid[self.player[0],self.player[1]].green = 0
        self.grid[self.player[0],self.player[1]].red = 0
        self.grid[self.player[0],self.player[1]].blue = 200
        
        self.grid.show()
        
    ## Q-learning function
    def qlearn(self,gamma,hazard_penalty,num_train_animate):
        num_episodes = 300
        num_complete = 0
        
        # loop over episodes, for each run simulation and update Q
        for n in range(num_episodes):
            # pick random initialization 
            obstical_free = 0
            loc = [np.random.randint(self.grid.height),np.random.randint(self.grid.width)]
           
            # update Q matrix while loc != goal
            steps = 0  # step counter
            max_steps = 10*self.grid.width*self.grid.height  # maximum number of steps per episode
            while steps < max_steps:    
                # when you reach the goal end current episode
                if loc == self.goal:
                    break
                    
                ### choose action - left = 0, right = 1, up = 2, down = 3
                k = np.random.randint(len(self.action_choices))  
                loc2 = [sum(x) for x in zip(loc, self.action_choices[k])] 
                ind_old = self.states.index(str(loc[0]) + str(loc[1]))

                ### set reward    
                # is the new location just a regular square?  Than small negative reward
                r_k = int(-1)

                # if new state is hazard penalize with medium negative value - this needs to be set properly if you're trying to prove a point i.e., that a trained agent won't walk over one of these unless going around costs more
                if loc2 in self.hazards:
                    r_k = int(hazard_penalty)

                # if new state is goal set reward of 0
                if loc2 == self.goal:
                    r_k = int(0)
                    
                # if new state is outside of boundaries of grid world penalize set reward to small negative value (like -1) and do not move
                if loc2[0] > self.grid.height-1 or loc2[0] < 0 or loc2[1] > self.grid.width-1 or loc2[1] < 0:  
                    r_k = int(-1)
                    loc2 = loc
                
                ### Update Q
                ind_new = self.states.index(str(loc2[0]) + str(loc2[1]))
                self.Q_star[ind_old,k] = r_k + gamma*max(self.Q_star[ind_new,:])
                    
                # update current location of player to one we just moved too (or stay still if grid world boundary met)
                self.player = loc2
                loc = loc2
                
                # the next few lines just animate the first few steps during the first few episodes
                if n < num_train_animate and steps < 200:
                    self.color_grid()
                    time.sleep(0.1)
                    display.clear_output(wait=True)
                    
                # update counter
                steps+=1
        
            # pause briefly between first few episodes to show user cutoff
            if n < num_train_animate:
                time.sleep(1)
                print 'finished episode, animating next episode'
                time.sleep(1)
            if n == num_train_animate:
                print 'continuing with remaining episodes without animation...'
                
        print 'q-learning process complete'
                
    # print out
    def show_qmat(self):        
        states_print = []
        for i in range(len(self.states)):
            s = str(self.states[i])
            t = str('(') + s[0] + ',' + s[1] + str(')')
            states_print.append(t)
            
        df = pd.DataFrame(self.Q_star,columns=['up','down','left','right'], index=states_print)
        print df.round(3)             
            
    # animate the testing phase based on completed Q-learning cycle
    def animate_testing_phase(self,loc):
        # show movement based on an initial 
        self.player = loc # initial agent location

        # if you chose an invalid starting position, break out and try again
        if  loc == self.goal or loc[0] > self.grid.height-1 or loc[0] < 0 or loc[1] > self.grid.width-1 or loc[1] < 0:
            print 'initialization is outside of gridworld or is goal, try again'
        else:  
            # now use the learned Q* matrix to run from any (valid) initial point to the goal
            self.color_grid()
            time.sleep(0.3)
            display.clear_output(wait=True)
            count = 0
            max_count = self.grid.width*self.grid.height
            while count < max_count:
                # find next state using max Q* value
                ind_old = self.states.index(str(self.player[0]) + str(self.player[1]))
                
                # find biggest value in Q* and determine block location
                action_ind = np.argmax(self.Q_star[ind_old,:])
                action = self.action_choices[action_ind]
                
                # move player to new location and recolor - if you move out of gridworld halt and report this to user
                new_loc = [sum(x) for x in zip(self.player,action)] 
                
                if new_loc[0] > self.grid.height-1 or new_loc[0] < 0 or new_loc[1] > self.grid.width-1 or new_loc[1] < 0:
                    print 'something went wrong - the player has left the gridworld arena'
                    print "your episodes did not the states reached by your initialization enough to train Q properly here"
                    print "this is likely because you didn't traiin long enough - up the number of steps / episodes and try again"
                    
                self.player = new_loc
                
                # clear current screen for next step
                self.color_grid()
                time.sleep(0.2)
                if self.player == self.goal:
                    break
                display.clear_output(wait=True)
                count+=1