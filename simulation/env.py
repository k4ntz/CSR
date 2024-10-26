import tensorflow as tf
import numpy as np
from tensorflow.keras import layers, Sequential
from tensorflow.keras.initializers import glorot_uniform
from utils import init_grow_state_mask, init_grow_obs_mask


def safety_reward(s, a):
    distance_to_obstacle = s[0]
    reward = tf.where(distance_to_obstacle > 0, 1.0, -1.0)
    return reward

def efficiency_reward(s):
    distance_to_destination = s[1]
    reward = -distance_to_destination
    return reward

def value_reward(s):
    reward = 0
    for item in s:
        if item > 0:
            reward += item**2
    return reward

def comfort_reward(a):
    acceleration = a[0]
    reward = -tf.abs(acceleration)
    return reward

def total_reward(s, a, w1=0.5, w2=0.3, w3=1, w4=0.2):
    return w1 * safety_reward(s, a) + w2 * efficiency_reward(s) + w3 * value_reward(s) + w4 * comfort_reward(a)


class SparseGlorotUniform(tf.keras.initializers.Initializer):
    def __init__(self, sparsity=0.1):
        self.sparsity = sparsity

    def __call__(self, shape, dtype=None):
        initializer = glorot_uniform()
        weights = initializer(shape, dtype=dtype)
        mask = np.random.choice([0, 1], size=shape, p=[self.sparsity, 1 - self.sparsity])
        sparse_weights = weights * mask
        return sparse_weights


class env_pomdp(tf.keras.Model):
    def __init__(self, obs_dim, state_dim, theta_dim, is_pomdp=True):
        super(env_pomdp, self).__init__()
        self.obs_dim = obs_dim
        self.state_dim = state_dim
        self.theta_dim = theta_dim
        self.is_pomdp = is_pomdp
        self.theta = tf.random.uniform(shape=(theta_dim,), minval=-1, maxval=1)

        self.state_tran_net = self._create_state_tran_net()
        if self.is_pomdp:
            self.obs_tran_net = self._create_obs_tran_net()
        
        self._build()

    def _create_state_tran_net(self):
        return layers.Dense(self.state_dim, activation="tanh", kernel_initializer=SparseGlorotUniform(sparsity=0.1))

    def _create_obs_tran_net(self):
        return layers.Dense(self.obs_dim, kernel_initializer=SparseGlorotUniform(sparsity=0.1))

    def _build(self):
        self.state_tran_net.build(input_shape=(None, self.theta_dim + self.state_dim + 1))
        if self.is_pomdp:
            self.obs_tran_net.build(input_shape=(None, self.state_dim))

    def change_dist(self):
        self.theta = tf.random.uniform(shape=(self.theta_dim,), minval=-1, maxval=1)

    def get_Space(self):
        state_space = ' * '.join([f'S{i}' for i in range(1, self.state_dim + 1)])
        return state_space

    def reset(self):
        self.s_1 = tf.random.normal(shape=(self.state_dim,), mean=0.0, stddev=0.02)
        self.last_state = self.s_1
        self.o_1 = self.obs_tran_net(tf.expand_dims(self.s_1, axis=0))[0] if self.is_pomdp else self.s_1

    def _grow_state_tran_net(self):
        new_state_tran_net = layers.Dense(self.state_dim, activation="tanh", kernel_initializer=SparseGlorotUniform(sparsity=0.1))
        new_state_tran_net.build(input_shape=(None, self.theta_dim + self.state_dim + 1))
        state_tran_w = init_grow_state_mask(self.state_tran_net.get_weights()[0], new_state_tran_net.get_weights()[0], self.theta_dim, self.state_grow)
        return layers.Dense(self.state_dim, activation="tanh", kernel_initializer=tf.constant_initializer(state_tran_w))

    def _grow_obs_tran_net(self):
        new_obs_tran_net = layers.Dense(self.obs_dim, kernel_initializer=SparseGlorotUniform(sparsity=0.1))
        new_obs_tran_net.build(input_shape=(None, self.state_dim))
        obs_tran_w = init_grow_obs_mask(self.obs_tran_net.get_weights()[0], new_obs_tran_net.get_weights()[0])
        return layers.Dense(self.obs_dim, kernel_initializer=tf.constant_initializer(obs_tran_w))

    def grow_latent_space(self, expand_state_dim=None):
        self.state_grow = np.random.randint(3, 7) if expand_state_dim is None else expand_state_dim
        self.state_dim += self.state_grow

        self.state_tran_net = self._grow_state_tran_net()
        if self.is_pomdp:
            self.obs_tran_net = self._grow_obs_tran_net()

        self._build()
        return self.state_grow

    def call(self, inputs):
        theta_state_action = tf.expand_dims(tf.concat([self.theta, self.last_state, inputs], axis=-1), axis=0)        
        s_t = self.state_tran_net(theta_state_action)[0]
        self.last_state = s_t
        
        r_t = total_reward(self.last_state, inputs)
        o_t = self.obs_tran_net(tf.expand_dims(s_t, axis=0))[0] if self.is_pomdp else s_t
        return o_t, r_t
