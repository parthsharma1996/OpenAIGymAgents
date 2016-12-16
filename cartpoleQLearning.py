import gym
import numpy as np
import math

obs_dim = 4
action_dim = 2 
max_trials = 1000000
env = gym.make('CartPole-v0')
max_steps = 1000
r_list= []
w = np.zeros(obs_dim,action_dim)

def updateWeights(w,q_target,q_present):

    w = w + learning_rate * (q_target - q_present)*prev_obs
    return w

def selectAction(w,obs):

    if random.rand() < epsilon:
        action = random.randint(0,1)
    else:
        action = np.argmax(w*next_obs)

    return action


for trial_num in range(0,max_trials):


    for step_num in range(0,max_steps):

        obs,r,done,info = env.step(action)

        
        

        prev_obs = obs
        w = updateWeights(q_target,q_present)
        a = selectAction(obs)
