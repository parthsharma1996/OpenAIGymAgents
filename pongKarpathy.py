import gym



#hyperparameters

H = 200
batch_size = 10
learning rate = 1e-4
gamma = 0.99
decay_rate = 0.99
resume = False
render = False



#Model Initialization
D = 80*80

env =gym.make('Pong-v0')
obs = env.reset()
prev_x = None
xs,hs,dlogps,drs = [],[],[],[]
running_reward = None
reward_sum = 0
episode_number = 0

while True:
    if render:
        env.render()

    cur_x = preProcess(observation)
    x = cur_x - prev_x if prev_x is not None else np.zeros(D)
    prev_x = cur_x

    #forward he policy network and sample an action from the returned probability
    aprob,h = policy_forward(x)
    action = 2 if np.random.uniform() < aprob else 3

    #record various intermediates 
    xs.append(x)
    hs.append(h) #hidden state
    y =1 if action ==2 else 0
    dlogs.append(y -aprob)#a grad that encourages the action that was taken furher 
    
    #step the environment and get new measurements
    observation, reward, done, info = env.step(action)
    reward_sum += reward

    drs.append(reward) # record reward
    

    if done:
        episode_number += 1

        epx = np.vstack(xs)
        eph = np.vstack(hs)
        epdlogp = np.vstack(dlogps)
        epr = np.vstack(drs)
        xs,hs,dlogps,drs = [],[],[],[]

        #compute the discounter reward backwards through time
        discounted_epr = discount_reward(epr)
        #standardize the rewards to be unit normal 
        discounted_epr -= np.mean(dsicounted_epr)
        discounted_epr /= np.std(dsicounted_epr)

        epdlogp *=discounted_epr 
        grad = policy_backward(eph, epdlogp)
        for k in model : grad_buffer[k] + =grad[k]

        #perfrom rmsprop parameter update every batch_size episodes
        if epsiode_number % batch_size == 0:
            for k,v  in model.iteritems():
                g =grad_buffer[k] #gradient
                rmsprop_cache[k] = decay_rate * rmsprop_cache[k] + (1 -decay_rate)*g**2
                model[k] += learning_rate *g/(np.sqrt(rmsprop_cache[k])) +1e-5
                grad_buffer[k] = np.zeros_like(v) #reset batch gradient buffer

        running_reward = reward_sum if running_reward is None else running_reward*0.99 +reward_sum*0.01
        print('resetting env. episode reward total was %f. running mean: %f' % (reward_sum,running_reward))
        if episode_number %100 == 0 :
            pickle.dump(model,open('save.p','wb'))
        reward_sum = 0
        observation = env.reset()
        prev_x = None

    if reward! = 0:
        print('ep %d : game finished ,reward: %f' %(epsiode_number, reward)+('' if reward == -1 else '!!'))

