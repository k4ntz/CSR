import numpy as np
import tensorflow as tf
from tensorflow.keras import layers, Sequential
from utils import update_vae_model, init_SSLs


class Encoder(tf.keras.Model):
    def __init__(
        self,
        obs_dim,
        state_dim,
        z_mean_w=None,
        z_log_var_w=None):
        super(Encoder, self).__init__()

        self.state_dim = state_dim

        if z_mean_w is not None:
            self.z_mean = layers.Dense(state_dim, input_shape=(obs_dim,), kernel_initializer=tf.constant_initializer(z_mean_w), name='z_mean')
        else:
            self.z_mean = layers.Dense(state_dim, input_shape=(obs_dim,), kernel_initializer='glorot_uniform', name='z_mean')

        if z_log_var_w is not None:
            self.z_log_var = layers.Dense(state_dim, input_shape=(obs_dim,), kernel_initializer=tf.constant_initializer(z_log_var_w), name='z_log_var')
        else:
            self.z_log_var = layers.Dense(state_dim, input_shape=(obs_dim,), kernel_initializer='glorot_uniform', name='z_log_var')

    def sampling(self, z_mean, z_log_var):
        batch = tf.shape(z_mean)[0]
        dim = tf.shape(z_mean)[1]
        epsilon = tf.random.normal(shape=(batch, dim))
        return z_mean + tf.exp(0.5 * z_log_var) * epsilon        

    def call(self, inputs):
        z_mean = self.z_mean(inputs)
        z_log_var = self.z_log_var(inputs)
        z = self.sampling(z_mean, z_log_var)
        return z_mean, z_log_var, z


class Reward_func(tf.keras.Model):
    def __init__(
        self,
        input_dim,
        reward_ws=None):
        super(Reward_func, self).__init__()

        self.reward_net = Sequential([
            layers.Dense(128, input_shape=(input_dim,), activation='relu', kernel_initializer='he_uniform'),
            layers.Dense(64, activation='relu', kernel_initializer='he_uniform'),
            layers.Dense(1, kernel_initializer='glorot_uniform')
        ])

        if reward_ws is not None:
            for idx, weights in enumerate(self.reward_net.trainable_weights):
                weights.assign(reward_ws[idx])

    def call(self, inputs):
        return self.reward_net(inputs)


class Decoder(tf.keras.Model):
    def __init__(
        self,
        state_dim,
        obs_dim,
        outputs_w=None):
        super(Decoder, self).__init__()

        self.obs_dim = obs_dim
        if outputs_w is not None:
            self.outputs = layers.Dense(self.obs_dim, input_shape=(state_dim,), kernel_initializer=tf.constant_initializer(outputs_w), name='outputs')
        else:
            self.outputs = layers.Dense(self.obs_dim, input_shape=(state_dim,), kernel_initializer='glorot_uniform', name='outputs')

    def call(self, inputs):
        return self.outputs(inputs)


class Transition(tf.keras.Model):
    def __init__(
        self,
        input_dim,
        tran_w=None):
        super(Transition, self).__init__()

        if tran_w is not None:
            self.input_layer = layers.Dense(128, input_shape=(input_dim,), activation='relu', kernel_initializer=tf.constant_initializer(tran_w))
        else:
            self.input_layer = layers.Dense(128, input_shape=(input_dim,), activation='relu', kernel_initializer='he_uniform')

        self.trans = Sequential([
            self.input_layer,
            layers.Dropout(0.2),
            layers.Dense(64, activation='relu', kernel_initializer='he_uniform'),
            layers.Dropout(0.2),
            layers.Dense(1, activation="tanh", kernel_initializer='glorot_uniform')
        ])

    def call(self, inputs):
        return self.trans(inputs)


class VAE(tf.keras.Model):
    def __init__(self,
                obs_dim,
                state_dim,
                theta_dim,
                is_pomdp=True,
                seed=None):
        super(VAE, self).__init__()

        self.obs_dim = obs_dim
        self.state_dim = state_dim
        self.theta_dim = theta_dim
        self.action_dim = 1

        self.is_pomdp = is_pomdp
        self.seed = seed

        self.encoder = Encoder(obs_dim, state_dim)
        self.decoder = Decoder(state_dim, obs_dim)
        self.transition = Transition(theta_dim + state_dim + self.action_dim)
        self.reward_func = Reward_func(state_dim + self.action_dim)

        self.SSL_A = self.add_weight(shape=(self.state_dim,), initializer='glorot_uniform', trainable=True, name='SSL_A')  # parameter D_{s->o} for o_t = f(A * s_t, e_t)
        self.SSL_B = self.add_weight(shape=(self.state_dim,), initializer='glorot_uniform', trainable=True, name='SSL_B')  # parameter D_{s->r} for r_t = g(B * s_t, C * a_t, epsilon_t)
        self.SSL_C = self.add_weight(shape=(self.action_dim,), initializer='glorot_uniform', trainable=True, name='SSL_C')  # parameter D_{a->r} for r_t = g(B * s_t, C * a_t, epsilon_t)
        self.SSL_D = self.add_weight(shape=(self.state_dim, self.state_dim), initializer='glorot_uniform', trainable=True, name='SSL_D')  # parameter D_{s->s} for s_{t+1} = h(D * s_t, E * a_t, eta_t)
        self.SSL_E = self.add_weight(shape=(self.action_dim, self.state_dim), initializer='glorot_uniform', trainable=True, name='SSL_E')  # parameter D_{a->s} for s_{t+1} = h(D * s_t, E * a_t, eta_t)
        self.SSL_F = self.add_weight(shape=(self.theta_dim, self.state_dim), initializer='glorot_uniform', trainable=True, name='SSL_F')  # the parameter before theta_s
        self.theta = self.add_weight(shape=(self.theta_dim,), initializer='random_normal', trainable=True, name='theta')
        self.SSL_P = tf.Variable(np.ones(self.state_dim, dtype=np.float32), trainable=False)

        self.compile(tf.keras.optimizers.Adam(), tf.keras.losses.MeanSquaredError())

    def compile(self, optimizer, loss_fn):
        super(VAE, self).compile()
        self.optimizer = optimizer
        self.loss_fn = loss_fn

    def record(self):
        self.late_z_mean_w = self.encoder.z_mean.get_weights()[0]
        self.late_z_log_var_w = self.encoder.z_log_var.get_weights()[0]
        self.late_tran_w = self.transition.input_layer.get_weights()[0]
        self.late_outputs_w = self.decoder.outputs.get_weights()[0]
        self.late_reward_ws = []
        self.reward_func.reward_net.trainable = True
        for weight in self.reward_func.reward_net.trainable_weights:
            self.late_reward_ws.append(weight.numpy())

        self.late_SSL_A = self.SSL_A.numpy().copy()
        self.late_SSL_B = self.SSL_B.numpy().copy()
        self.late_SSL_C = self.SSL_C.numpy().copy()
        self.late_SSL_D = self.SSL_D.numpy().copy()
        self.late_SSL_E = self.SSL_E.numpy().copy()
        self.late_SSL_F = self.SSL_F.numpy().copy()
        self.late_SSL_P = self.SSL_P.numpy().copy()
        self.late_state_dim = self.state_dim

    def search(self, expand_state_dim):
        z_mean_w, z_log_var_w, tran_w, outputs_w, reward_ws = update_vae_model(self.late_z_mean_w,
                                                                            self.late_z_log_var_w,
                                                                            self.late_tran_w,
                                                                            self.late_outputs_w,
                                                                            self.late_reward_ws,
                                                                            self.theta_dim,
                                                                            expand_state_dim)
        self.state_dim = self.late_state_dim + expand_state_dim

        self.encoder = Encoder(self.obs_dim, self.state_dim, z_mean_w, z_log_var_w)
        self.decoder = Decoder(self.state_dim, self.obs_dim, outputs_w)
        self.transition = Transition(self.theta_dim + self.state_dim + self.action_dim, tran_w)
        self.reward_func = Reward_func(self.state_dim + self.action_dim, reward_ws)

        SSL_A, SSL_B, SSL_D, SSL_E, SSL_F, SSL_P = init_SSLs(expand_state_dim, self.late_SSL_A, self.late_SSL_B, self.late_SSL_D, self.late_SSL_E, self.late_SSL_F, self.late_SSL_P)
        self.SSL_A = self.add_weight(shape=(self.state_dim,), initializer=tf.constant_initializer(SSL_A), trainable=True, name='SSL_A')
        self.SSL_B = self.add_weight(shape=(self.state_dim,), initializer=tf.constant_initializer(SSL_B), trainable=True, name='SSL_B')
        self.SSL_D = self.add_weight(shape=(self.state_dim, self.state_dim), initializer=tf.constant_initializer(SSL_D), trainable=True, name='SSL_D')
        self.SSL_E = self.add_weight(shape=(self.action_dim, self.state_dim), initializer=tf.constant_initializer(SSL_E), trainable=True, name='SSL_E')
        self.SSL_F = self.add_weight(shape=(self.theta_dim, self.state_dim), initializer=tf.constant_initializer(SSL_F), trainable=True, name='SSL_F')
        self.SSL_P = tf.Variable(SSL_P, trainable=False)
                                    
    def update(self, expand_state_dim):
        self.state_dim = self.late_state_dim + expand_state_dim
    
    def train_step(self, data):
        obs = data[0]['observation']
        action = data[0]['action']
        reward = data[0]['reward']
        next_obs = data[0]['next_observation']

        ssl_ss, ssl_as, ssl_ds = [], [], []

        with tf.GradientTape() as tape:
            z_mean, z_log_var, z = self.encoder(obs)
            z = tf.multiply(z, self.SSL_P)

            reconstructed = self.decoder(tf.multiply(z, self.SSL_A))
            reconstruction_loss = self.loss_fn(obs, reconstructed)
            kl_loss = -0.5 * tf.reduce_sum(1 + z_log_var - tf.square(z_mean) - tf.exp(z_log_var), axis=-1)

            theta = tf.tile(tf.expand_dims(self.theta, axis=0), [tf.shape(obs)[0], 1])
            pred_reward = self.reward_func(tf.concat([tf.multiply(z, self.SSL_B), tf.multiply(action, self.SSL_C)], axis=-1))
            reward_loss = self.loss_fn(reward, pred_reward)

            for i in range(self.state_dim):
                ssl_ss.append(tf.multiply(z, self.SSL_D[:, i]))
                ssl_as.append(tf.multiply(action, self.SSL_E[:, i]))
                ssl_ds.append(tf.multiply(theta, self.SSL_F[:, i]))

            ssl_ss = tf.convert_to_tensor(ssl_ss)  # ssl_ss: state_dim x batch_size x state_dim
            ssl_ss = tf.reshape(ssl_ss, [-1, self.state_dim])  # ssl_ss: (state_dim x batch_size) x state_dim

            ssl_as = tf.convert_to_tensor(ssl_as)  # ssl_zz: state_dim x batch_size x action_dim
            ssl_as = tf.reshape(ssl_as, [-1, self.action_dim])  # ssl_zz: (state_dim x batch_size) x action_dim

            ssl_ds = tf.convert_to_tensor(ssl_ds)
            ssl_ds = tf.reshape(ssl_ds, [-1, self.theta_dim])
            next_z_input = tf.reshape(tf.concat([ssl_ds, ssl_ss, ssl_as], 1),
                                [-1,  self.theta_dim + self.state_dim + self.action_dim])

            next_z = self.transition(next_z_input)
            next_z = tf.transpose(tf.reshape(next_z, [self.state_dim, -1]))
            next_z = tf.multiply(next_z, self.SSL_P)

            prediction = self.decoder(tf.multiply(next_z, self.SSL_A))
            prediction_loss = self.loss_fn(next_obs, prediction)

            total_loss = reconstruction_loss + tf.reduce_mean(kl_loss) + prediction_loss + reward_loss

        gradients = tape.gradient(total_loss, self.trainable_weights)
        self.optimizer.apply_gradients(zip(gradients, self.trainable_weights))

        return {
            "loss": total_loss,
            "reconstruction_loss": reconstruction_loss,
            "kl_loss": kl_loss,
            "prediction_loss": prediction_loss,
            "reward_loss": reward_loss
        }

    def encode(self, input_obs):
        _, _, z = self.encoder(input_obs.reshape(-1, self.obs_dim))
        z = tf.multiply(z, self.SSL_P)

        return z[0]
    
    def pruning(self):
        tmp_SSL_P = np.ones(self.state_dim)
        for i in range(self.state_dim):
            tmp_SSL_P[i] = np.abs(self.SSL_A.numpy()[i]) + np.abs(self.SSL_B.numpy()[i]) + (np.sum(np.abs(self.SSL_D.numpy()[:, i])) - np.abs(self.SSL_D.numpy()[i, i]))

        tmp_SSL_P = np.where(tmp_SSL_P, 1, 0).astype(np.float32)
        self.SSL_P = tf.Variable(tmp_SSL_P, trainable=False)

    def call(self, inputs):
        obs = inputs['observation']
        action = inputs['action']

        _, _, z = self.encoder(obs)
        z = tf.multiply(z, self.SSL_P)

        theta = tf.tile(tf.expand_dims(self.theta, axis=0), [obs.shape[0], 1])

        ssl_ss, ssl_as, ssl_ds = [], [], []

        for i in range(self.state_dim):
            ssl_ss.append(tf.multiply(z, self.SSL_D[:, i]))
            ssl_as.append(tf.multiply(action, self.SSL_E[:, i]))
            ssl_ds.append(tf.multiply(theta, self.SSL_F[:, i]))

        ssl_ss = tf.convert_to_tensor(ssl_ss)  # ssl_ss: state_dim x batch_size x state_dim
        ssl_ss = tf.reshape(ssl_ss, [-1, self.state_dim])  # ssl_ss: (state_dim x batch_size) x state_dim

        ssl_as = tf.convert_to_tensor(ssl_as)  # ssl_zz: state_dim x batch_size x action_dim
        ssl_as = tf.reshape(ssl_as, [-1, self.action_dim])  # ssl_zz: (state_dim x batch_size) x action_dim

        ssl_ds = tf.convert_to_tensor(ssl_ds)
        ssl_ds = tf.reshape(ssl_ds, [-1, self.theta_dim])
        next_z_input = tf.reshape(tf.concat([ssl_ds, ssl_ss, ssl_as], 1),
                            [-1,  self.theta_dim + self.state_dim + self.action_dim])

        next_z = self.transition(next_z_input)
        next_z = tf.transpose(tf.reshape(next_z, [self.state_dim, -1]))
        next_z = tf.multiply(next_z, self.SSL_P)

        prediction = self.decoder(tf.multiply(next_z, self.SSL_A))

        return prediction    
