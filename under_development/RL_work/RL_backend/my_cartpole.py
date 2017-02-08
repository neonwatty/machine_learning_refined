import gym
import numpy as np
import math
import time
import matplotlib.pyplot as plt

class my_cartpole():

    def __init__(self):
        # initialize enviroment
        self.env = gym.make('CartPole-v0')                           
        
        # initialize Q matrix parameters
        self.thetas = np.zeros((np.shape(self.env.observation_space)[0]+1,self.env.action_space.n))

    ## Q-learning function
    def qlearn(self,gamma,num_episodes):
        # run num_rounds of qlearning with linear function approximation of num_episodes number of episodes, and keep the parameters that provide the best results of many test runs
        num_rounds = 100
        ave_test = 0
        for r in range(num_rounds):
            thetas = np.zeros((np.shape(self.env.observation_space)[0]+1,self.env.action_space.n))
            
            # loop over episodes, for each run simulation and update Q
            for n in range(num_episodes):
                # we use the same initialization every time
                observation = self.env.reset()  # note here 'observation' = state

                '''
                 Note here that there are four features to the given state:
                - the cart position -> the 1st entry of the state
                - the cart velocity -> the 2nd entry of the state
                - the angle of the pole measured as its deviation from the vertical position - 3rd entry of state (in radians)
                - the angular velocity of the pole - 4th entry of state
                '''

                # start episode of simulation training
                steps = 0
                max_steps = 500
                while steps < max_steps:                  
                    # animate first trials
                    if n < 20 and r == 0:
                        self.env.render()
                    
                    action = self.env.action_space.sample()
                    observation, reward, done, info = self.env.step(action)  # reward = +1 for every time unit the pole is above a threshold angle, 0 else

                    # if done == true alter reward to communicate failure to system reward structure
                    if done:
                        reward = -1

                    # discretize, i.e., round observation features
                    observation = np.around(observation,3)

                    # compute long term reward based on proper linear model from Q
                    state = np.insert(observation,0,1)
                    state.shape = (len(state),1)            
                    old_theta = thetas[:,action]
                    old_theta.shape = (len(old_theta),1)
                    h_i = np.dot(old_theta.T,state)

                    # compute Q cost value
                    q_k = reward + gamma*h_i

                    # choose step length
                    alpha = 1/float(n + 1)

                    # update theta_i
                    grad = (h_i - q_k)*state
                    grad.shape = (len(grad),1)

                    theta_i = old_theta - alpha*grad      
                    thetas[:,action] = theta_i.ravel()

                    # if pole goes below threshold angle restart - new episode
                    if done:
                        break

                    # update counter
                    steps+=1
            
            # close animation of first trials
            self.env.render(close=True)
                
            # all episodes have run, compute testing error, keep best parameters
            ave_test2 = self.testing(thetas)
            if ave_test2 > ave_test:
                self.thetas = thetas.copy()
                ave_test = ave_test2
                               
        print 'q-learning process complete, best number of average steps in test runs = ' + str(ave_test)     

        # a function to run several test rounds using a candidate set of parameters
    def testing(self,thetas):
        num_testing_rounds = 20
        ave_steps = 0
        for i in range(num_testing_rounds):
            # start up simulation episode
            observation = self.env.reset()  # note here 'observation' = state

            # start testing phase
            steps = 0
            max_steps = 500
            while steps < max_steps:
                state = np.insert(observation,0,1)
                state.shape = (len(state),1)

                # pick best action based on input state
                h = np.dot(thetas.T,state)
                action = int(np.argmax(h))

                # take action 
                observation, reward, done, info = self.env.step(action)

                # if pole goes below threshold then end simulation
                if done:
                    ave_steps+=steps
                    break
                else:
                    steps+=1
                    
        # return average number of steps taken         
        ave_steps = ave_steps/float(num_testing_rounds)
        return ave_steps
    
    # animate a single test run
    def animate_test_run(self):
        # start up simulation episode
        observation = self.env.reset()  # note here 'observation' = state

        # start testing phase
        steps = 0
        max_steps = 500
        while steps < max_steps:
            # render action in animation 
            self.env.render()
            
            state = np.insert(observation,0,1)
            state.shape = (len(state),1)

            # pick best action based on input state
            h = np.dot(self.thetas.T,state)
            action = int(np.argmax(h))

            # take action 
            observation, reward, done, info = self.env.step(action)

            # if pole goes below threshold then end simulation
            if done:
                print("lasted {} timesteps".format(steps))
                break

            steps+=1
        self.env.render(close=True)

