import gym
import numpy as np
import math


num_trials = 1000000
env = gym.make('CartPole-v0')
running_sum = 0
max_steps = 1000
r_list= []
r_best = -math.inf
w_best = [[ 0.31363913, 0.97461077,  0.81859487,  0.0653391 ]]
r_best = 171.0
for trial_num in range(0,num_trials):

    w = np.random.rand(1,4)
    r_episode = 0

    obs = env.reset()
    for step in range(max_steps):
        if np.dot(w,obs)>0:
            action = 0
        else:
            action = 1 
        obs,r,done,info = env.step(action)
        r_episode += r
        if done == True:
            break

    r_list.append(r_episode)

    if trial_num == 100:
        running_sum = sum(r_list)
    if trial_num >100:
        running_sum = running_sum -r_list[-101]+r_list[-1]
    if trial_num %10000==0:
        print(trial_num,r_best)
    if r_episode > r_best:

        r_best = r_episode
        w_best = w
        print(w_best)

print("r_best is", r_best," Running mean is ", running_sum/100)
