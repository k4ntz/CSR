import numpy as np
import torch
import torch.optim as optim
import os, re
import torch.nn.functional as F
from mymodel.utils.module import get_parameters, FreezeParameters
from mymodel.utils.algorithm import compute_return

from mymodel.models.actor import DiscreteActionModel
from mymodel.models.dense import DenseModel
from mymodel.models.drssm_cartpole import DRSSM
from mymodel.models.pixel_cartpole import ObsDecoder, ObsEncoder
from mymodel.utils.buffer import TransitionBuffer
from mymodel.utils.log import Log
from functools import reduce
import pdb


class CPOTrainer(object):
    def __init__(
            self,
            config,
            device,
            action_expansion=False,
    ):
        self.device = device
        self.config = config
        if not action_expansion:
            self.action_size = config.action_size
        else:
            self.action_size = config.action_size - 4
        self.pixel = config.pixel
        self.kl_info = config.kl
        self.seq_len = config.seq_len
        self.batch_size = config.batch_size
        self.collect_intervals = config.collect_intervals
        self.seed_steps = config.seed_steps
        self.discount = config.discount_
        self.lambda_ = config.lambda_
        self.horizon = config.horizon
        self.loss_scale = config.loss_scale
        self.grad_clip_norm = config.grad_clip
        self.logger = Log(os.path.join(self.config.model_dir, 'train.log')).logger
        self.train_steps = 0

        self._model_initialize(config)
        self._optim_initialize(config)
        # self._print_summary()

    def collect_seed_episodes(self, env):
        s, done = env.reset(), False
        for i in range(self.seed_steps):
            a = env.action_space.sample()
            ns, r, done, _ = env.step(a)
            if done:
                self.buffer.add(s, a, r, done)
                s, done = env.reset(), False
            else:
                self.buffer.add(s, a, r, done)
                s = ns

    def resume_training(self, model_dir, resume_step=200000):
        last_model_name = "models_best.pth"
        save_dict = torch.load(os.path.join(model_dir, last_model_name))
        self.load_save_dict(save_dict)
        return resume_step

    def train_batch(self, train_metrics, expand_deter=False):
        """
        trains the world model and imagination actor and critic for collect_interval times using sequence-batch data from buffer
        """
        obs_l = []
        obs_mse_l = []
        model_l = []
        reward_l = []
        reward_mse_l = []

        prior_ent_l = []
        post_ent_l = []
        kl_l = []
        pcont_l = []
        for i in range(self.collect_intervals):
            obs, actions, rewards, terms = self.buffer.sample()
            uint8_flag = True if obs.dtype == np.uint8 else False
            obs = torch.tensor(obs, dtype=torch.float32).to(self.device)  # t, t+seq_len
            if uint8_flag:
                obs = obs.div(255).sub_(0.5)
            actions = torch.tensor(actions, dtype=torch.float32).to(self.device)  # t-1, t+seq_len-1
            rewards = torch.tensor(rewards, dtype=torch.float32).to(self.device).unsqueeze(-1)  # t-1 to t+seq_len-1
            nonterms = torch.tensor(1 - terms, dtype=torch.float32).to(self.device).unsqueeze(-1)  # t-1 to t+seq_len-1
            prior, posterior = self._get_prior_posterior(obs, actions, nonterms, expand_deter)

            model_loss, kl_s, obs_loss, obs_mse_loss, reward_loss, reward_mse_loss, pcont_loss, prior_dist, post_dist, posterior = self.representation_loss(
                prior, posterior, obs, actions, rewards, nonterms, expand_deter)
            if expand_deter:
                self.new_model_optimizer.zero_grad()
                # with torch.autograd.detect_anomaly():
                torch.autograd.backward(model_loss, inputs=get_parameters(self.new_world_list))
                has_nan = False
                for model in self.new_world_list:
                    for name, param in model.named_parameters():
                        if param.grad is not None and torch.isnan(param.grad).any():
                            print("nan gradient found")
                            print("name:",name)
                            print("param:",param.grad)
                            has_nan = True
                            # raise SystemExit
                if has_nan:
                    pdb.set_trace()
                torch.nn.utils.clip_grad_norm_(get_parameters(self.new_world_list), self.grad_clip_norm)
                self.new_model_optimizer.step()
                if torch.stack([torch.isnan(p).any() for p in self.new_RSSM.rnn.parameters()]).any():
                    is_nan_reward = torch.stack([torch.isnan(p).any() for p in self.new_RewardDecoder.parameters()]).any()
                    is_nan_encoder = torch.stack([torch.isnan(p).any() for p in self.ObsEncoder.parameters()]).any()
                    is_nan_decoder = torch.stack([torch.isnan(p).any() for p in self.new_ObsDecoder.parameters()]).any()
                    print(f'Have nan: reward: {is_nan_reward}, obsencoder: {is_nan_encoder}, obsdecoder: {is_nan_decoder}, action: {is_nan_action}, value: {is_nan_value}')
            else:
                self.model_optimizer.zero_grad()
                # with torch.autograd.detect_anomaly():
                torch.autograd.backward(model_loss, inputs=get_parameters(self.world_list))
                has_nan = False
                for model in self.world_list:
                    for name, param in model.named_parameters():
                        if param.grad is not None and torch.isnan(param.grad).any():
                            print("nan gradient found")
                            print("name:",name)
                            print("param:",param.grad)
                            has_nan = True
                            # raise SystemExit
                if has_nan:
                    pdb.set_trace()
                torch.nn.utils.clip_grad_norm_(get_parameters(self.world_list), self.grad_clip_norm)
                self.model_optimizer.step()
                if torch.stack([torch.isnan(p).any() for p in self.RSSM.rnn.parameters()]).any():
                    is_nan_reward = torch.stack([torch.isnan(p).any() for p in self.RewardDecoder.parameters()]).any()
                    is_nan_encoder = torch.stack([torch.isnan(p).any() for p in self.ObsEncoder.parameters()]).any()
                    is_nan_decoder = torch.stack([torch.isnan(p).any() for p in self.ObsDecoder.parameters()]).any()
                    print(f'Have nan: reward: {is_nan_reward}, obsencoder: {is_nan_encoder}, obsdecoder: {is_nan_decoder}, action: {is_nan_action}, value: {is_nan_value}')

            obs_l.append(obs_loss.item())
            obs_mse_l.append(obs_mse_loss.item())
            model_l.append(model_loss.item())
            reward_l.append(reward_loss.item())
            kl_l.append(kl_s.item())
            pcont_l.append(pcont_loss.item())
            reward_mse_l.append(reward_mse_loss.item())

        train_metrics['model_loss'] = np.mean(model_l)
        train_metrics['kl_loss'] = np.mean(kl_l)
        train_metrics['reward_mse_loss'] = np.mean(reward_mse_l)
        train_metrics['obs_loss'] = np.mean(obs_l)
        train_metrics['obs_mse_loss'] = np.mean(obs_mse_l)
        train_metrics['pcont_loss'] = np.mean(pcont_l)
        self.train_steps += 1
        return train_metrics

    def adapt_batch(self, train_metrics):
        """
        trains the world model and imagination actor and critic for collect_interval times using sequence-batch data from buffer
        """
        obs_l = []
        obs_mse_l = []
        model_l = []
        reward_l = []
        reward_mse_l = []

        prior_ent_l = []
        post_ent_l = []
        kl_l = []
        pcont_l = []
        for i in range(self.collect_intervals):
            obs, actions, rewards, terms = self.buffer.sample()
            uint8_flag = True if obs.dtype == np.uint8 else False
            obs = torch.tensor(obs, dtype=torch.float32).to(self.device)  # t, t+seq_len
            if uint8_flag:
                obs = obs.div(255).sub_(0.5)
            actions = torch.tensor(actions, dtype=torch.float32).to(self.device)  # t-1, t+seq_len-1
            rewards = torch.tensor(rewards, dtype=torch.float32).to(self.device).unsqueeze(-1)  # t-1 to t+seq_len-1
            nonterms = torch.tensor(1 - terms, dtype=torch.float32).to(self.device).unsqueeze(-1)  # t-1 to t+seq_len-1
            prior, posterior = self._get_prior_posterior(obs, actions, nonterms)

            with FreezeParameters(self.world_list, adapt_theta=True):
                model_loss, kl_s, obs_loss, obs_mse_loss, reward_loss, reward_mse_loss, pcont_loss, prior_dist, post_dist, posterior = self.representation_loss(
                    prior, posterior, obs, actions, rewards, nonterms)
                self.theta_optimizer.zero_grad()
                torch.autograd.backward(model_loss, inputs=self.theta_list)
                torch.nn.utils.clip_grad_norm_(self.theta_list, self.grad_clip_norm)
                self.theta_optimizer.step()

            obs_l.append(obs_loss.item())
            obs_mse_l.append(obs_mse_loss.item())
            model_l.append(model_loss.item())
            reward_l.append(reward_loss.item())
            kl_l.append(kl_s.item())
            pcont_l.append(pcont_loss.item())
            reward_mse_l.append(reward_mse_loss.item())

        train_metrics['model_loss'] = np.mean(model_l)
        train_metrics['kl_loss'] = np.mean(kl_l)
        train_metrics['reward_mse_loss'] = np.mean(reward_mse_l)
        train_metrics['obs_loss'] = np.mean(obs_l)
        train_metrics['obs_mse_loss'] = np.mean(obs_mse_l)
        train_metrics['pcont_loss'] = np.mean(pcont_l)
        self.train_steps += 1
        return train_metrics

    def _get_prior_posterior(self, obs, actions, nonterms, expand_deter=False):
        embed = self.ObsEncoder(obs)  # t to t+seq_len
        if expand_deter:
            prev_rssm_state = self.new_RSSM._init_rssm_state(self.batch_size)
            prior, posterior = self.new_RSSM.rollout_observation(self.seq_len, embed, actions, nonterms, prev_rssm_state)
        else:
            prev_rssm_state = self.RSSM._init_rssm_state(self.batch_size)
            prior, posterior = self.RSSM.rollout_observation(self.seq_len, embed, actions, nonterms, prev_rssm_state)
        return prior, posterior

    def representation_loss(self, prior, posterior, obs, actions, rewards, nonterms, expand_deter=False):
        if expand_deter:
            post_modelstate = self.new_RSSM.get_model_state(posterior)  # t to t+seq_len
            obs_dist = self.new_ObsDecoder(post_modelstate[:-1])  # t to t+seq_len-1
            reward_dist = self.new_RewardDecoder(post_modelstate[:-1])  # t to t+seq_len-1
            pcont_dist = self.new_DiscountModel(post_modelstate[:-1])  # t to t+seq_len-1
        else:
            post_modelstate = self.RSSM.get_model_state(posterior)  # t to t+seq_len
            obs_dist = self.ObsDecoder(post_modelstate[:-1])  # t to t+seq_len-1
            reward_dist = self.RewardDecoder(post_modelstate[:-1])  # t to t+seq_len-1
            pcont_dist = self.DiscountModel(post_modelstate[:-1])  # t to t+seq_len-1

        obs_loss, obs_mse_loss = self._obs_loss(obs_dist, obs[:-1])
        reward_loss, reward_mse_loss = self._reward_loss(reward_dist, rewards[1:])

        pcont_loss = self._pcont_loss(pcont_dist, nonterms[1:])
        prior_dist, post_dist, kl_s = self._kl_loss(prior, posterior, expand_deter)

        model_loss = self.loss_scale['kl'] * kl_s + self.loss_scale['reward'] * reward_loss + obs_loss + \
                     self.loss_scale['discount'] * pcont_loss
        return model_loss, kl_s, obs_loss, obs_mse_loss, reward_loss, reward_mse_loss, pcont_loss, prior_dist, post_dist, posterior

    def _obs_loss(self, obs_dist, obs):
        obs_loss = -torch.mean(obs_dist.log_prob(obs))
        obs_mse_loss = F.mse_loss(obs_dist.mean.detach(), obs)
        return obs_loss, obs_mse_loss

    def _kl_loss(self, prior, posterior, expand_deter=False):
        if expand_deter:
            prior_dist = self.new_RSSM.get_dist(prior)
            post_dist = self.new_RSSM.get_dist(posterior)
        else:
            prior_dist = self.RSSM.get_dist(prior)
            post_dist = self.RSSM.get_dist(posterior)
        if self.kl_info['use_kl_balance']:
            alpha = self.kl_info['kl_balance_scale']
            if expand_deter:
                kl_lhs = self._kl(self.new_RSSM.get_dist(self.new_RSSM.rssm_detach(posterior)),
                                                    prior_dist)
                kl_rhs = self._kl(post_dist,
                                    self.new_RSSM.get_dist(self.new_RSSM.rssm_detach(prior)))
            else:
                kl_lhs = self._kl(self.RSSM.get_dist(self.RSSM.rssm_detach(posterior)),
                                                    prior_dist)
                kl_rhs = self._kl(post_dist,
                                    self.RSSM.get_dist(self.RSSM.rssm_detach(prior)))
            if self.kl_info['use_free_nats']:
                free_nats = self.kl_info['free_nats']
                kl_lhs = torch.max(kl_lhs, kl_lhs.new_full(kl_lhs.size(), free_nats))
                kl_rhs = torch.max(kl_rhs, kl_rhs.new_full(kl_rhs.size(), free_nats))
            kl_loss = alpha * kl_lhs + (1 - alpha) * kl_rhs
        else:
            kl_loss = self._kl(post_dist, prior_dist)
            if self.kl_info['use_free_nats']:
                free_nats = self.kl_info['free_nats']
                kl_loss = torch.max(kl_loss, kl_loss.new_full(kl_loss.size(), free_nats))
        return prior_dist, post_dist, kl_loss

    def _kl(self, posterior_dist, prior_dist):
        poster_mean, poster_stddev = posterior_dist.mean, posterior_dist.stddev
        prior_mean, prior_stddev = prior_dist.mean, prior_dist.stddev
        po_ = torch.distributions.Independent(torch.distributions.Normal(poster_mean, poster_stddev), 1)
        pr_ = torch.distributions.Independent(torch.distributions.Normal(prior_mean, prior_stddev), 1)
        kl_s = torch.mean(torch.distributions.kl.kl_divergence(po_, pr_))
        return kl_s

    def _reward_loss(self, reward_dist, rewards):
        reward_loss = -torch.mean(reward_dist.log_prob(rewards))
        reward_mse_loss = F.mse_loss(reward_dist.mean.detach(), rewards)
        return reward_loss, reward_mse_loss

    def _pcont_loss(self, pcont_dist, nonterms):
        pcont_target = nonterms.float()
        pcont_loss = -torch.mean(pcont_dist.log_prob(pcont_target))
        return pcont_loss

    def save_model(self, iter):
        save_dict = self.get_save_dict()
        model_dir = os.path.join(self.config.model_dir, 'models_%d' % iter)
        os.makedirs(model_dir, exist_ok=True)
        save_path = os.path.join(model_dir, 'models_%d.pth' % iter)
        torch.save(save_dict, save_path)
        return model_dir

    def get_save_dict(self):
        return {
            "RSSM": self.RSSM.state_dict(),
            "ObsEncoder": self.ObsEncoder.state_dict(),
            "ObsDecoder": self.ObsDecoder.state_dict(),
            "RewardDecoder": self.RewardDecoder.state_dict(),
            "DiscountModel": self.DiscountModel.state_dict(),
        }

    def load_save_dict(self, saved_dict):
        self.RSSM.load_state_dict(saved_dict["RSSM"])
        self.ObsEncoder.load_state_dict(saved_dict["ObsEncoder"])
        self.ObsDecoder.load_state_dict(saved_dict["ObsDecoder"])
        self.RewardDecoder.load_state_dict(saved_dict["RewardDecoder"])
        self.DiscountModel.load_state_dict(saved_dict['DiscountModel'])

    def load_save_dict_new(self, saved_dict):
        self.new_RSSM.load_state_dict(saved_dict["RSSM"])
        self.ObsEncoder.load_state_dict(saved_dict["ObsEncoder"])
        self.new_ObsDecoder.load_state_dict(saved_dict["ObsDecoder"])
        self.new_RewardDecoder.load_state_dict(saved_dict["RewardDecoder"])
        self.new_DiscountModel.load_state_dict(saved_dict['DiscountModel'])

    def _model_initialize(self, config):
        self.obs_shape = config.obs_shape
        self.deter_size = config.rssm_info['deter_size']
        if config.rssm_type == 'continuous':
            self.stoch_size = config.rssm_info['stoch_size']
        elif config.rssm_type == 'discrete':
            category_size = config.rssm_info['category_size']
            class_size = config.rssm_info['class_size']
            self.stoch_size = category_size * class_size

        self.embedding_size = config.embedding_size
        self.rssm_node_size = config.rssm_node_size
        self.modelstate_size = self.stoch_size + self.deter_size
        self.asrstate_size = self.deter_size + self.stoch_size
        self.reward_size = self.deter_size + self.stoch_size
        self.theta_deter_size = config.theta_deter_size
        self.theta_stoch_size = config.theta_stoch_size
        self.theta_size = self.theta_deter_size + self.theta_stoch_size

        self.buffer = TransitionBuffer(config.capacity, self.obs_shape, config.action_size, config.seq_len, config.batch_size,
                                       config.obs_dtype, config.action_dtype)
        self.RSSM = DRSSM(self.action_size, self.rssm_node_size, self.embedding_size, self.theta_deter_size, self.theta_stoch_size,
                          self.device, config.rssm_type,
                          config.rssm_info).to(self.device)
        self.RewardDecoder = DenseModel((1,), self.reward_size, config.reward).to(self.device)

        self.ActionModel = DiscreteActionModel(config.action_size, self.deter_size, self.stoch_size,
                                               self.embedding_size, config.actor,
                                               config.expl, self.theta_size).to(self.device)

        if config.discount['use']:
            self.DiscountModel = DenseModel((1,), self.asrstate_size, config.discount).to(self.device)
        if config.pixel:
            self.ObsEncoder = ObsEncoder(self.obs_shape, self.embedding_size, config.obs_encoder).to(self.device)
            self.ObsDecoder = ObsDecoder(self.obs_shape, self.modelstate_size, config.obs_decoder).to(self.device)
        else:
            self.ObsEncoder = DenseModel((self.embedding_size,), int(np.prod(self.obs_shape)), config.obs_encoder).to(self.device)
            self.ObsDecoder = DenseModel(self.obs_shape, self.modelstate_size, config.obs_decoder).to(self.device)    

    def _optim_initialize(self, config):
        model_lr = config.lr['model']
        self.world_list = [self.ObsEncoder, self.RSSM, self.RewardDecoder, self.ObsDecoder, self.DiscountModel]
        self.model_optimizer = optim.Adam(get_parameters(self.world_list), model_lr)

        self.theta_list = [self.RSSM.theta_deter, self.RSSM.theta_stoch]
        self.theta_optimizer = optim.Adam(self.theta_list, model_lr)

    def _action_expansion_initialize(self, config, expand_action_size=4):
        self.action_size += expand_action_size
        self.RSSM.action_size += expand_action_size
        self.RSSM._expand_embed_state_action()

        self._optim_initialize(config)
        
    def _model_expand_initialize(self, config, expand_deter_size=10, expand_stoch_size=4):
        model_lr = config.lr['model']
        self.expand_deter_size = expand_deter_size
        self.expand_stoch_size = expand_stoch_size
        new_deter_size = self.deter_size + expand_deter_size
        new_stoch_size = self.stoch_size + expand_stoch_size
        new_modelstate_size = new_stoch_size + new_deter_size
        new_asrstate_size = new_stoch_size + new_deter_size
        new_reward_size = new_stoch_size + new_deter_size

        self.new_RSSM = DRSSM(self.action_size, self.rssm_node_size, self.embedding_size,
                              self.theta_deter_size, self.theta_stoch_size,
                              self.device, config.rssm_type, config.rssm_info,
                              new_deter_size, new_stoch_size).to(self.device)
        self.new_RewardDecoder = DenseModel((1,), new_reward_size, config.reward).to(self.device)

        self.new_ActionModel = DiscreteActionModel(self.action_size, new_deter_size, new_stoch_size,
                                               self.embedding_size, config.actor,
                                               config.expl, self.theta_size).to(self.device)

        if config.discount['use']:
            self.new_DiscountModel = DenseModel((1,), new_asrstate_size, config.discount).to(self.device)
        if config.pixel:
            self.new_ObsDecoder = ObsDecoder(self.obs_shape, new_modelstate_size, config.obs_decoder).to(self.device)
        else:
            self.new_ObsDecoder = DenseModel(self.obs_shape, new_modelstate_size, config.obs_decoder).to(self.device)

        self.new_world_list = [self.ObsEncoder, self.new_RSSM, self.new_RewardDecoder, self.new_ObsDecoder, self.new_DiscountModel]
        self.param_succeed(config)
        self.new_model_optimizer = optim.Adam(get_parameters(self.new_world_list), model_lr)

    def _model_update(self, config, expand_deter_size=10, expand_stoch_size=4):
        self.deter_size += expand_deter_size
        self.stoch_size += expand_stoch_size
        self.modelstate_size = self.stoch_size + self.deter_size
        self.asrstate_size = self.stoch_size + self.deter_size
        self.reward_size = self.stoch_size + self.deter_size

        self.RSSM = self.new_RSSM
        self.RewardDecoder = self.new_RewardDecoder
        self.ActionModel = self.new_ActionModel
        if config.discount['use']:
            self.DiscountModel = self.new_DiscountModel
        if config.pixel:
            self.ObsDecoder = self.new_ObsDecoder
        else:
            self.ObsDecoder = self.new_ObsDecoder

    def param_succeed(self, config):
        for old_module, new_module in zip(self.world_list, self.new_world_list):
            if isinstance(old_module, ObsEncoder):
                pass
            elif isinstance(old_module, DRSSM):
                param_copy(old_module, new_module, 'theta_deter')
                param_copy(old_module, new_module, 'theta_stoch')
                for i in range(3):
                    start_old = i * self.deter_size
                    start_new = i * (self.deter_size + self.expand_deter_size)
                    param_copy(old_module, new_module, 'rnn.weight_ih', target_row_start=start_new, target_row_end=start_new + self.deter_size,
                                                                        source_row_start=start_old, source_row_end=start_old + self.deter_size)
                    param_copy(old_module, new_module, 'rnn.weight_hh', target_row_start=start_new, target_row_end=start_new + self.deter_size,
                                                                        source_row_start=start_old, source_row_end=start_old + self.deter_size,
                                                                        target_col_end=self.deter_size)
                    param_copy(old_module, new_module, 'rnn.bias_ih', target_row_start=start_new, target_row_end=start_new + self.deter_size,
                                                                      source_row_start=start_old, source_row_end=start_old + self.deter_size)
                    param_copy(old_module, new_module, 'rnn.bias_hh', target_row_start=start_new, target_row_end=start_new + self.deter_size,
                                                                      source_row_start=start_old, source_row_end=start_old + self.deter_size)
                param_copy(old_module, new_module, 'fc_embed.0.weight', target_row_end=self.rssm_node_size,
                                                                        target_col_end=self.theta_deter_size + self.stoch_size)
                param_copy(old_module, new_module, 'fc_embed.0.weight', target_row_end=self.rssm_node_size,
                                                                        target_col_start=self.theta_deter_size + self.stoch_size + self.expand_stoch_size,
                                                                        target_col_end=self.theta_deter_size + self.stoch_size + self.expand_stoch_size + self.action_size,
                                                                        source_col_start=self.theta_deter_size + self.stoch_size,
                                                                        source_col_end=self.theta_deter_size + self.stoch_size + self.action_size)
                param_copy(old_module, new_module, 'fc_embed.0.bias')
                param_copy(old_module, new_module, 'fc_prior.prior.0.weight', target_row_end=self.rssm_node_size,
                                                                              target_col_end=self.theta_stoch_size + self.deter_size)
                param_copy(old_module, new_module, 'fc_prior.prior.0.bias')
                param_copy(old_module, new_module, 'fc_prior.prior.2.weight', target_row_end=self.stoch_size, target_col_end=self.rssm_node_size)
                param_copy(old_module, new_module, 'fc_prior.prior.2.weight', target_row_start=self.stoch_size + self.expand_stoch_size,
                                                                              target_row_end=2 * self.stoch_size + self.expand_stoch_size,
                                                                              target_col_end=self.rssm_node_size,
                                                                              source_row_start=self.stoch_size,
                                                                              source_row_end=2 * self.stoch_size)
                param_copy(old_module, new_module, 'fc_prior.prior.2.bias', target_row_end=self.stoch_size)
                param_copy(old_module, new_module, 'fc_prior.prior.2.bias', target_row_start=self.stoch_size + self.expand_stoch_size,
                                                                            target_row_end=2 * self.stoch_size + self.expand_stoch_size,
                                                                            source_row_start=self.stoch_size,
                                                                            source_row_end=2 * self.stoch_size)
                param_copy(old_module, new_module, 'fc_posterior.posterior.0.weight', target_row_end=self.rssm_node_size,
                                                                                      target_col_end=self.embedding_size + self.deter_size)
                param_copy(old_module, new_module, 'fc_posterior.posterior.0.bias')
                param_copy(old_module, new_module, 'fc_posterior.posterior.2.weight', target_row_end=self.stoch_size, target_col_end=self.rssm_node_size)
                param_copy(old_module, new_module, 'fc_posterior.posterior.2.weight', target_row_start=self.stoch_size + self.expand_stoch_size,
                                                                                      target_row_end=2 * self.stoch_size + self.expand_stoch_size,
                                                                                      target_col_end=self.rssm_node_size,
                                                                                      source_row_start=self.stoch_size,
                                                                                      source_row_end=2 * self.stoch_size)
                param_copy(old_module, new_module, 'fc_posterior.posterior.2.bias', target_row_end=self.stoch_size)
                param_copy(old_module, new_module, 'fc_posterior.posterior.2.bias', target_row_start=self.stoch_size + self.expand_stoch_size,
                                                                                    target_row_end=2 * self.stoch_size + self.expand_stoch_size,
                                                                                    source_row_start=self.stoch_size,
                                                                                    source_row_end=2 * self.stoch_size)
            elif isinstance(old_module, ObsDecoder):
                param_copy(old_module, new_module, 'linear.weight', target_col_end=self.stoch_size)
                param_copy(old_module, new_module, 'linear.weight', target_col_start=self.stoch_size + self.expand_stoch_size,
                                                                    target_col_end=self.modelstate_size + self.expand_stoch_size,
                                                                    source_col_start=self.stoch_size,
                                                                    source_col_end=self.modelstate_size)
                param_copy(old_module, new_module, 'linear.bias') 
                param_copy(old_module, new_module, 'decoder.0.weight') 
                param_copy(old_module, new_module, 'decoder.0.bias')
                param_copy(old_module, new_module, 'decoder.2.weight') 
                param_copy(old_module, new_module, 'decoder.2.bias')
                param_copy(old_module, new_module, 'decoder.4.weight') 
                param_copy(old_module, new_module, 'decoder.4.bias')
                param_copy(old_module, new_module, 'decoder.6.weight') 
                param_copy(old_module, new_module, 'decoder.6.bias')
            else:
                param_copy(old_module, new_module, 'model.0.weight', target_col_end=self.stoch_size)
                param_copy(old_module, new_module, 'model.0.weight', target_col_start=self.stoch_size + self.expand_stoch_size,
                                                                     target_col_end=self.stoch_size + self.expand_stoch_size + self.deter_size,
                                                                     source_col_start=self.stoch_size,
                                                                     source_col_end=self.stoch_size + self.deter_size)
                param_copy(old_module, new_module, 'model.0.bias')
                param_copy(old_module, new_module, 'model.2.weight') 
                param_copy(old_module, new_module, 'model.2.bias')
                param_copy(old_module, new_module, 'model.4.weight') 
                param_copy(old_module, new_module, 'model.4.bias')

    def _print_summary(self):
        print('\n Obs encoder: \n', self.ObsEncoder)
        print('\n RSSM model: \n', self.RSSM)
        print('\n Reward decoder: \n', self.RewardDecoder)
        print('\n Obs decoder: \n', self.ObsDecoder)
        if self.config.discount['use']:
            print('\n Discount decoder: \n', self.DiscountModel)


def param_copy(
    source_module, target_module, param_name,
    target_row_start=0, target_row_end=None, target_col_start=0, target_col_end=None,
    source_row_start=None, source_row_end=None, source_col_start=None, source_col_end=None
):
    with torch.no_grad():
        attrs = param_name.split('.')
        source_param = getattr(reduce(getattr, attrs[:-1], source_module), attrs[-1])
        target_param = getattr(reduce(getattr, attrs[:-1], target_module), attrs[-1])

        if target_row_end is not None or target_col_end is not None:
            if 'weight' in param_name:
                target_rows = slice(target_row_start, target_row_end) if target_row_end is not None else slice(target_row_start, target_param.data.size(0))
                target_cols = slice(target_col_start, target_col_end) if target_col_end is not None else slice(target_col_start, target_param.data.size(1))
                if source_row_start is not None:
                    source_rows = slice(source_row_start, source_row_end)
                else:
                    source_rows = target_rows
                if source_col_start is not None:
                    source_cols = slice(source_col_start, source_col_end)
                else:
                    source_cols = target_cols                
                target_param.data[target_rows, target_cols] = source_param.data[source_rows, source_cols].clone()
            elif 'bias' in param_name:
                target_rows = slice(target_row_start, target_row_end) if target_row_end is not None else slice(target_row_start, target_param.data.size(0))
                if source_row_start is not None:
                    source_rows = slice(source_row_start, source_row_end)
                else:
                    source_rows = target_rows
                target_param.data[target_rows] = source_param.data[source_rows].clone()
        else:
            target_param.data = source_param.data.clone()
