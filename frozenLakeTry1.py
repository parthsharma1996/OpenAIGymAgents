import gym
import numpy as np
import random


env = gym.make('FrozenLake-v0')


def selectAction(s):
    #following an epsilon-greedy policy
    if epsilon >= random.random():
        a = random.randint(0,3)
        
    else:
        a = np.argmax(q(s,:))


def updateQValues(prev_s,cur_s):

    q(prev_s,action) += q(prev_s,action) + learning_rate*(r + np.max(q(cur_s,:))-q(prev_s,action))

for num_episode in range(0,max_episodes):


    for num_steps in range(0,max_steps):

        obs, r, done, info  = env.step(action)


        prev_obs = obs
        updateQValues
