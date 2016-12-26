import gym
import numpy as np
import random

obs_dim = 8*8 
action_dim = 4
max_episodes = 30000000
max_steps = 1000000
learning_rate = 0.001
eps_start  = 1 
eps_end = 0.001
eps_step = (eps_start -eps_end)/max_episodes
eps_threshold = 0.5
gamma = 0.97 
solv_req = 0.99
record = True

env = gym.make('FrozenLake8x8-v0')
if record:
    env.monitor.start('/tmp/frozenLake-experiment-1')

q = np.zeros((obs_dim,action_dim))
r_list = []

def selectAction(s,epsilon):
    #print("e",epsilon)
    #following an epsilon-greedy policy
    if epsilon >= random.random():
        a = random.randint(0,3)
    else:
        a = np.argmax(q[s,:])
   # print("action chosen was",a)
    return a;

def updateQValues(prev_s,cur_s,r):

    q[prev_s,action] = q[prev_s,action]+ learning_rate*(r + gamma*np.max(q[cur_s,:])-q[prev_s,action])

counter  =0 
epsilon = eps_start
for num_episode in range(0,max_episodes):

    obs = env.reset()
    prev_obs = obs
    #print(epsilon)
    for num_steps in range(0,max_steps):
        action  = selectAction(prev_obs,epsilon)
        obs, r, done, info  = env.step(action)
        updateQValues(prev_obs, obs,r)
        if r==1 :
            counter +=1
            #print("kuch to mila", num_episode,num_steps,counter)
        if done:
            #print("Reward was " ,r)
            #print(q)
            r_list.append(r)
            break;
        prev_obs = obs
    if num_episode%100==0:
        print("Average reward in the last 100 episodes was ",sum(r_list[-100:])/100,"\n",epsilon )
    
    if sum(r_list[-100:])/100 >=eps_threshold:
        epsilon = 0.01
    else:
        epsilon = epsilon - eps_step
    if sum(r_list[-100:])/100 >=solv_req:
        print(num_episode)
        break

print("Average reward in the last 100 episodes was ",sum(r_list[-100:])/100)
print(q)
if record:
    env.monitor.close()
    gym.upload('/tmp/frozenLake-experiment-1', api_key='sk_FjvBIfXASuuFdqgLn83bHw')

