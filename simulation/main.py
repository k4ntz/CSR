import numpy as np
import tensorflow as tf

from env import env_pomdp
from model import VAE
from agent import DQNAgent

from rollout import ReplayBuffer
from utils import plot_reward

obs_dim = 256
state_dim = 4
cont_state_dim = 4
action_dim = 33
theta_dim = 1
MAX_TRAJ = 256
MAX_EPISODE = 100


# train  world model
def train_wm_ours(adapt=False):
    data_replay = ReplayBuffer()
    for episode in range(MAX_EPISODE):
        nn_env.reset()
        for idx in range(MAX_TRAJ):
            if not idx:
                obs = nn_env.o_1
            else:
                obs = next_obs            
            action = [(np.random.choice(action_dim) - (action_dim - 1)/2) / ((action_dim - 1)/2)]
            next_obs, reward = nn_env(action)
            if episode < (MAX_EPISODE * 0.8):
                data_replay.add_sample(obs, action, reward, next_obs)
            else:
                data_replay.add_sample(obs, action, reward, next_obs, is_test=True)
    train_data, test_data = data_replay.get_all_samples()
    if not adapt:
        print('Training our world model.')
        wm_ours.fit(train_data, batch_size=4, epochs=32, verbose=0)
    else:
        print('Adapting our world model.')
        wm_ours.fit(train_data, batch_size=4, epochs=8, verbose=0)
    threshold = metric(wm_ours(test_data), test_data['next_observation'])
    return threshold, train_data


# expand our world model
# without rl search, directly assign
def expand_wm_ours(expand_actions):
    data_replay = ReplayBuffer()
    for episode in range(MAX_EPISODE):
        nn_env.reset()
        for idx in range(MAX_TRAJ):
            if not idx:
                obs = nn_env.o_1
            else:
                obs = next_obs            
            action = [(np.random.choice(action_dim) - (action_dim - 1)/2) / ((action_dim - 1)/2)]
            next_obs, reward = nn_env(action)        
            if episode < (MAX_EPISODE * 0.8):
                data_replay.add_sample(obs, action, reward, next_obs)
            else:
                data_replay.add_sample(obs, action, reward, next_obs, is_test=True)                
    train_data, test_data = data_replay.get_all_samples()
   
    wm_ours.record()
    wm_ours.search(expand_actions)
    wm_ours.update(expand_actions)
    wm_ours.compile(tf.keras.optimizers.Adam(), tf.keras.losses.MeanSquaredError())
    wm_ours.fit(train_data, batch_size=4, epochs=32, verbose=0)
    threshold = metric(wm_ours(test_data), test_data['next_observation'])
    return expand_actions, threshold


# policy learning
def train_policy(wm, agent, model_based=True):
    agent.buffer.clear()
    returns = []
    for episode in range(MAX_EPISODE):
        sum_reward = 0
        nn_env.reset()
        if model_based: 
            state = wm.encode(nn_env.o_1.numpy())
        else:
            state = nn_env.o_1
        for idx in range(MAX_TRAJ):           
            agent_action = agent.choose_action(state)
            action = [(agent_action - (action_dim - 1)/2) / ((action_dim - 1)/2)]
            next_obs, reward = nn_env(action)
            if model_based:
                next_state = wm.encode(next_obs.numpy())
            else:
                next_state = next_obs
            sum_reward += reward
            agent.buffer.push(state, agent_action, reward, next_state)
        agent.train()
        returns.append(sum_reward / MAX_TRAJ)
    return returns


if __name__ == '__main__':
    returns_all_eps = []
    # initialize
    nn_env = env_pomdp(obs_dim, state_dim, theta_dim)
    wm_ours = VAE(obs_dim, state_dim, theta_dim)
    wm_list = [wm_ours]
    agent_ours = DQNAgent(state_dim, action_dim)
    agent_list = [agent_ours]
    metric = tf.keras.losses.MeanSquaredError()
    print("##################################################################################################################")
    print('Task 0 with state space:\t', nn_env.get_Space())

    # train world models then agents
    threshold, train_data = train_wm_ours()

    returns_each_ep = []
    for wm, agent in zip(wm_list, agent_list):
        if wm is not None:
            returns = train_policy(wm, agent)
        else:
            returns = train_policy(wm, agent, False)
        returns_each_ep.append(returns)
    returns_all_eps.append(returns_each_ep)
    print('Policy learning done.')

    # generalization tasks
    for idx in range(3):
        print("##################################################################################################################")
        # environemt settings
        if idx == 0:
            nn_env.change_dist()
        elif idx == 1:
            grow_matrix = nn_env.grow_latent_space()
        else:
            nn_env.change_dist()
            grow_matrix = nn_env.grow_latent_space()                
        print('Task {} with state space:\t'.format(idx + 1), nn_env.get_Space())

        # expand world models and agents 
        test_loss, train_data = train_wm_ours(adapt=True)        
        if test_loss > threshold:
            print('Expanding our world model.')
            expand_dim, threshold = expand_wm_ours(grow_matrix)
            print('Add {} causal variables.'.format(expand_dim))
            state_dim += expand_dim
            agent_ours.update(expand_dim)
        else:
            print('Achieve threshold with:', test_loss)     
        
        # update agents
        returns_each_ep = []
        for wm, agent in zip(wm_list, agent_list):
            if wm is not None:
                returns = train_policy(wm, agent)
            else:
                returns = train_policy(wm, agent, False)
            returns_each_ep.append(returns)
        returns_all_eps.append(returns_each_ep)
    print("##################################################################################################################")
    
    print('All tasks ended.')
    plot_reward(returns_all_eps)
