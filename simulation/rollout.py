import tensorflow as tf
import numpy as np
from collections import deque


class ReplayBuffer:
    def __init__(self,
                buffer_size=100000):

        self.buffer_size = buffer_size
        self.buffer = deque(maxlen=buffer_size)
        self.test_buffer = deque(maxlen=buffer_size)

    def add_sample(self, obs, action, reward, next_obs, is_test=False):
        obs = tf.convert_to_tensor(obs, dtype=tf.float32)
        action = tf.convert_to_tensor(action, dtype=tf.float32)
        reward = tf.convert_to_tensor(reward, dtype=tf.float32)
        next_obs = tf.convert_to_tensor(next_obs, dtype=tf.float32)

        sample = {'observation': obs, 'action': action, 'reward': reward, 'next_observation': next_obs}
        if not is_test:
            self.buffer.append(sample)
        else:
            self.test_buffer.append(sample)
        
    def reset_buffer(self):
        self.buffer.clear()
        self.test_buffer.clear()

    def sample_batch(self, batch_size):
        indices = np.random.choice(len(self.buffer), size=batch_size)
        batch = {key: tf.stack([self.buffer[i][key] for i in indices]) for key in self.buffer[0]}
        return batch

    def get_all_samples(self):
        return {key: tf.stack([sample[key] for sample in self.buffer]) for key in self.buffer[0]}, {key: tf.stack([sample[key] for sample in self.test_buffer]) for key in self.test_buffer[0]}
