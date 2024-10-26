import random
import numpy as np
from datetime import datetime
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors

plt.rcParams['axes.linewidth'] = 1
plt.rcParams['figure.dpi'] = 600
plt.rc('font', family='times new roman')
font = {'style': 'times new roman', 'weight': 'black'}

date_format = '%A_%d_%B_%Y_%Hh_%Mm_%Ss'


def moving_average(a, window_size):
    cumulative_sum = np.cumsum(np.insert(a, 0, 0))
    middle = (cumulative_sum[window_size:] - cumulative_sum[:-window_size]) / window_size
    r = np.arange(1, window_size - 1, 2)
    begin = np.cumsum(a[:window_size - 1])[::2] / r
    end = (np.cumsum(a[:-window_size:-1])[::2] / r)[::-1]
    return np.concatenate((begin, middle, end))


def plot_reward(returns_all_eps):
    time = datetime.now().strftime(date_format)
    np.save('./data/reward_{}.npy'.format(time), np.array(returns_all_eps))

    figure_num = len(returns_all_eps)
    fig, axs = plt.subplots(1, figure_num, figsize=(4 * figure_num, 4))

    for i in range(figure_num):
        reward = np.array(returns_all_eps[i][0])[:50]
        reward = moving_average(reward, 9)
        t = np.arange(reward.shape[0])
        axs[i].plot(t, reward, color="#c44e52")
        axs[i].set_title('Task {}'.format(i), fontsize=15)
        axs[i].tick_params(axis='x', labelsize=15)
        axs[i].tick_params(axis='y', labelsize=15)
        axs[i].set_xticks([0, 10, 20, 30, 40, 50])
        axs[i].set_xlabel("Episode", fontsize=15)
        axs[i].set_ylabel("Mean accumulared Reward", fontsize=15)
        axs[i].grid(alpha=0.8, linestyle="--")
        axs[i].legend(loc="lower right", ncol=1, fontsize=10)

    plt.tight_layout()
    plt.savefig('./figures/reward_{}.png'.format(time))


def init_sparse_mask(input_dim, output_dim, seed=None):
    """
    Initialize a sparse environment mask.

    Parameters:
        - input_dim (int): The number of input dimensions.
        - output_dim (int): The number of output dimensions.
        - seed (int): Seed for random number generation (optional).

    Returns:
        - np.ndarray: Sparse environment mask.
    """
    if seed is not None:
        np.random.seed(seed)

    # Glorot uniform initialization
    limit = np.sqrt(6 / (input_dim + output_dim))
    g_random = np.random.uniform(low=-limit, high=limit, size=(input_dim, output_dim))

    return g_random


def init_grow_state_mask(old_mask, new_mask, theta_dim, state_grow, seed=None):
    # old_mask: (theta + s + a) * (s)
    # state_grow: s' ;   action_grow: a'
    # [theta, s, a, s', a'] to [theta, s, s', a, a']
    base_mask = np.where(old_mask != 0, 1, 0)

    new_mask[:theta_dim + old_mask.shape[1], :old_mask.shape[1]] *= base_mask[:theta_dim + old_mask.shape[1], :]
    new_mask[theta_dim + old_mask.shape[1] + state_grow:old_mask.shape[0] + state_grow, :old_mask.shape[1]] *= base_mask[theta_dim + old_mask.shape[1]:old_mask.shape[0], :]
           
    return new_mask


def init_grow_reward_mask(old_mask, new_mask, state_grow, seed=None):
    base_mask = np.where(old_mask != 0, 1, 0)

    new_mask[:old_mask.shape[0] - 1, :] *= base_mask[:old_mask.shape[0] - 1, :]
    new_mask[old_mask.shape[0] - 1 + state_grow:, :] *= base_mask[old_mask.shape[0] - 1:, :]

    return new_mask


def init_grow_obs_mask(old_mask, new_mask, seed=None):
    base_mask = np.where(old_mask != 0, 1, 0)

    new_mask[:old_mask.shape[0], :] *= base_mask

    return new_mask


def init_grow_encoder_mask(old_mask, state_grow, obs_dim, seed=None):
    col_grow_mask = init_sparse_mask(obs_dim, state_grow, seed=seed)
    new_mask = np.hstack((old_mask, col_grow_mask))
    return new_mask

def init_grow_tran_reward_l1(old_mask, state_grow, seed=None):
    old_state_dim = old_mask.shape[0] - 1
    row_grow_mask = init_sparse_mask(state_grow, old_mask.shape[1], seed=seed)
    new_mask = np.vstack((old_mask, row_grow_mask))

    copy_mask = new_mask.copy()
    # s: s + s'  \leftarrow s + a : s + a + s'
    copy_mask[old_state_dim:old_state_dim + state_grow, :] = new_mask[old_mask.shape[0]:old_mask.shape[0] + state_grow, :]
    # s + s' : s + s' + a \leftarrow s : s + a
    copy_mask[old_state_dim + state_grow:old_mask.shape[0] + state_grow, :] = new_mask[old_state_dim:old_mask.shape[0], :]      
    new_mask = copy_mask

    return new_mask

def init_grow_decoder_mask(old_mask, state_grow, obs_dim, seed=None):
    row_grow_mask = init_sparse_mask(state_grow, obs_dim, seed=seed)
    new_mask = np.vstack((old_mask, row_grow_mask))

    return new_mask

def update_vae_model(late_z_mean_w, late_z_log_var_w, late_tran_w, late_outputs_w, late_reward_ws, theta_dim, state_grow, seed=None):
    z_mean_w = late_z_mean_w.copy()
    z_log_var_w = late_z_log_var_w.copy()
    tran_w = late_tran_w.copy()
    outputs_w = late_outputs_w.copy()
    reward_ws = late_reward_ws.copy()

    obs_dim = late_z_mean_w.shape[0]
    
    z_mean_w = init_grow_encoder_mask(z_mean_w, state_grow, obs_dim, seed=seed)
    z_log_var_w = init_grow_encoder_mask(z_log_var_w, state_grow, obs_dim, seed=seed)

    tran_w = init_grow_tran_reward_l1(tran_w, state_grow, seed=seed)
    outputs_w = init_grow_decoder_mask(outputs_w, state_grow, obs_dim, seed=seed)
    reward_ws[0] = init_grow_tran_reward_l1(reward_ws[0], state_grow, seed=seed)
    
    return z_mean_w, z_log_var_w, tran_w, outputs_w, reward_ws


def update_dqn_agent(old_mask, new_mask):
    new_mask[:old_mask.shape[0], :] = old_mask
    return new_mask


def cons_state(state_dim, hdr, loss):
    hd = [state_dim] + hdr
    model_state = np.zeros((len(hd), 3))
    for idx in range(len(hd)):
        model_state[idx] = np.array([idx + 1, hd[idx], loss])

    return model_state


class BufferReplay:
    def __init__(self, capacity=100000, batch_size=64):
        self.capacity = capacity
        self.buffer = []
        self.position = 0
        self.batch_size = batch_size

    def push(self, state, action, reward, next_state):
        if len(self.buffer) < self.capacity:
            self.buffer.append(None)
        self.buffer[self.position] = (state, action, reward, next_state)
        self.position = int((self.position + 1) % self.capacity)

    def sample(self):
        batch = random.sample(self.buffer, self.batch_size)
        state, action, reward, next_state = map(np.stack, zip(*batch))

        return state, action, reward, next_state

    def clear(self):
        self.buffer = []
        self.position = 0


def init_SSLs(state_grow, SSL_A, SSL_B, SSL_D, SSL_E, SSL_F, SSL_P, seed=None):
    state_dim = SSL_D.shape[0]
    action_dim = SSL_E.shape[0]
    theta_dim = SSL_F.shape[0]

    add_SSL_A = np.random.randint(2, size=(state_grow,)).astype(SSL_A.dtype)
    add_SSL_B = np.random.randint(2, size=(state_grow,)).astype(SSL_B.dtype)
    add_SSL_D1 = np.random.randint(2, size=(state_dim, state_grow)).astype(SSL_D.dtype)
    add_SSL_D2 = np.random.randint(2, size=(state_grow, state_dim + state_grow)).astype(SSL_D.dtype)
    add_SSL_E = np.random.randint(2, size=(action_dim, state_grow)).astype(SSL_E.dtype)
    add_SSL_F = np.random.randint(2, size=(theta_dim, state_grow)).astype(SSL_F.dtype)
    add_SSL_P = np.ones(state_grow, dtype=np.float32)

    add_SSL_A = np.concatenate((SSL_A, add_SSL_A))
    add_SSL_B = np.concatenate((SSL_B, add_SSL_B))

    add_SSL_D = np.hstack((SSL_D, add_SSL_D1))
    add_SSL_D = np.vstack((add_SSL_D, add_SSL_D2))

    add_SSL_E = np.hstack((SSL_E, add_SSL_E))
    add_SSL_F = np.hstack((SSL_F, add_SSL_F))
    add_SSL_P = np.concatenate((SSL_P, add_SSL_P))

    return add_SSL_A, add_SSL_B, add_SSL_D, add_SSL_E, add_SSL_F, add_SSL_P
