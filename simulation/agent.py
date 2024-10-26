import os
import numpy as np
import tensorflow as tf
from tensorflow.keras import layers, models
from utils import update_dqn_agent, BufferReplay


class DQNAgent:
    def __init__(self,
                state_dim,
                action_dim,
                seed=None):
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.batch_size = 64

        self.model = self.create_model(self.state_dim)
        self.target_model = self.create_model(self.state_dim)
        self.model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=0.01), loss='mse')
        self.target_model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=0.01), loss='mse')
        self.target_model.set_weights(self.model.get_weights())

        self.epsilon = 0.1
        self.epsilon_decrement = 0.005
        self.epsilon_min = 0.01
        self.gamma = 0.95

        self.buffer = BufferReplay()    
    
    def create_model(self, input_state_shape):
        input_layer = layers.Input(shape=input_state_shape)
        layer_1 = layers.Dense(32, activation='relu')(input_layer)
        layer_2 = layers.Dense(16, activation='relu')(layer_1)
        output_layer = layers.Dense(self.action_dim)(layer_2)
    
        model = models.Model(inputs=input_layer, outputs=output_layer)
        return model

    def target_update(self):
        self.target_model.set_weights(self.model.get_weights())

    def record(self):
        last_weights = []
        for weight in self.model.trainable_weights:
            last_weights.append(weight.numpy())

        return last_weights
        
    def update(self, expand_dim):
        last_weights = self.record()
        self.state_dim += expand_dim
        self.model = self.create_model(self.state_dim)
        self.model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=0.005), loss='mse')     
        tmp_weights = self.record()
        tmp_weights[0] = update_dqn_agent(last_weights[0], tmp_weights[0])        

        for idx, weights in enumerate(self.model.trainable_weights):
            weights.assign(tmp_weights[idx])

        self.target_model = self.create_model(self.state_dim)        
        self.target_model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=0.005), loss='mse')
        self.target_model.set_weights(self.model.get_weights())

    def choose_action(self, state):
        if np.random.uniform() < self.epsilon:
            return np.random.choice(self.action_dim)
        else:
            q_value = self.model(state[np.newaxis, :])[0]
            return np.argmax(q_value)

    def replay(self):
        if len(self.buffer.buffer) < self.batch_size:
            return

        states, actions, rewards, next_states = self.buffer.sample()

        target = self.target_model(states).numpy()
        next_target = self.target_model(next_states)
        next_q_value = tf.reduce_max(next_target, axis=1)

        target[range(self.batch_size), actions] = rewards + self.gamma * next_q_value

        with tf.GradientTape() as tape:
            q_pred = self.model(states)
            loss = tf.losses.mean_squared_error(target, q_pred)

        grads = tape.gradient(loss, self.model.trainable_weights)
        self.model.optimizer.apply_gradients(zip(grads, self.model.trainable_weights))

    def train(self):
        self.replay()
        self.target_update()
        self.epsilon = max(self.epsilon - self.epsilon_decrement, self.epsilon_min)
