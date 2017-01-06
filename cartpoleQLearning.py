import gym
import numpy as np
import math
import random

obs_dim = 4
action_dim = 2 
learning_rate = 0.001
max_trials = 1000000
env = gym.make('CartPole-v0')
max_steps = 1000
r_list= []
epr_list = []
w = np.zeros((action_dim,obs_dim))
ep_start = 1
ep_end = 0.1
gamma = 0.97
print_freq = 100
solv_req = 195

def updateWeights(prev_obs,next_obs,action):
    global w #asking python to use the global variable w
    q_target = r + gamma * np.argmax(w * next_obs)
    q_present = w*next_obs[action]
    w = w + learning_rate * (q_target - q_present)*prev_obs

def selectAction(epsilon,obs):
    global w

    print(epsilon)
    if np.random.rand() < epsilon:
        action = random.randint(0,1)
    else:
        action = np.argmax(w*obs)
        print("size of w is", w)

    return action


for trial_num in range(0,max_trials):
    
    obs = env.reset()
    prev_obs = obs
    
    for num_steps in range(0,max_steps):

        action = selectAction(ep_end+1/(trial_num+1),obs)
        print("action was",action)
        obs,r,done,info = env.step(action)
        updateWeights(prev_obs,obs,action)
        r_list.append(r)
        prev_obs = obs
        
        if done:
            epr_list.append(sum(r_list))
            r_list = []
            break
    if trial_num%print_freq == 0:
        print("Average reward in the last 100 epsiodes was",sum(epr_list[-100:])/100)

    if sum(r_list[-100:])/100 >=solv_req:
        print(trial_num)
        break

