import gym
import numpy as np
import math
import random

obs_dim = 4
action_dim = 2 
learning_rate = 0.0001
max_trials = 1000000
env = gym.make('CartPole-v0')
max_steps = 100000
r_list= []
epr_list = []
w = np.zeros((action_dim,obs_dim))
ep_start = 1
ep_end = 0.1
gamma = 0.95
print_freq = 10000
solv_req = 195
record =True

if record:
    env.monitor.start('/tmp/cartPole-experiment-1',force=True)
    
def updateWeights(prev_obs,next_obs,action,r):
    global w #asking python to use the global variable w
    prev_obs = np.asarray(prev_obs)
    next_obs = np.asarray(next_obs)
    q_target = r + gamma * np.amax(np.dot(w , next_obs))
    q_present = np.dot(w,prev_obs)[action]
    w[action,:] = w[action,:] + learning_rate * (q_target - q_present)*prev_obs

def selectAction(epsilon,obs):
    global w
    obs = np.asarray(obs)
    if np.random.rand() < epsilon:
        action = random.randint(0,1)
    else:
        action = np.argmax(np.dot(w,obs))

    return action


for trial_num in range(0,max_trials):
    
    obs = env.reset()
    prev_obs = obs
    
    for num_steps in range(0,max_steps):

        action = selectAction(ep_end+1/(trial_num+1),obs)
        obs,r,done,info = env.step(action)
        updateWeights(prev_obs,obs,action,r)
        r_list.append(r)
        prev_obs = obs
        
        if done:
            epr_list.append(sum(r_list))
            r_list = []
            break
    if trial_num%print_freq == 0:
        print("Average reward in the last 100 epsiodes was",sum(epr_list[-100:])/100)

    if sum(epr_list[-100:])/100 >=solv_req:
        print(trial_num)
        break

if record:
    env.monitor.close()
    gym.upload('/tmp/cartPole-experiment-1')
