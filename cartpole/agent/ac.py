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


class ACAgent(object):
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
        self.seq_len = config.seq_len
        self.batch_size = config.batch_size
        self.collect_intervals = config.collect_intervals
        self.seed_steps = config.seed_steps
        self.discount = config.discount_
        self.lambda_ = config.lambda_
        self.horizon = config.horizon
        self.actor_entropy_scale = config.actor_entropy_scale
        self.grad_clip_norm = config.grad_clip
        self.train_steps = 0

        self._model_initialize(config)
        self._optim_initialize(config)

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

    def policy_learning(self, train_metrics):
        actor_l = []
        value_l = []
        mean_targ = []
        min_targ = []
        max_targ = []
        std_targ = []

        for i in range(self.collect_intervals):
            obs, actions, rewards, terms = self.buffer.sample()
            uint8_flag = True if obs.dtype == np.uint8 else False
            obs = torch.tensor(obs, dtype=torch.float32).to(self.device)  # t, t+seq_len
            if uint8_flag:
                obs = obs.div(255).sub_(0.5)
            actions = torch.tensor(actions, dtype=torch.float32).to(self.device)  # t-1, t+seq_len-1
            rewards = torch.tensor(rewards, dtype=torch.float32).to(self.device).unsqueeze(-1)  # t-1 to t+seq_len-1
            nonterms = torch.tensor(1 - terms, dtype=torch.float32).to(self.device).unsqueeze(-1)  # t-1 to t+seq_len-1
            _, posterior = self._get_prior_posterior(obs, actions, nonterms)

            actor_loss, value_loss, target_info = self.actorcritc_loss(posterior)
            assert torch.isnan(actor_loss).sum() == 0, print('actor loss nan', actor_loss)

            self.actor_optimizer.zero_grad()
            self.value_optimizer.zero_grad()

            actor_loss.backward()
            value_loss.backward()

            grad_norm_actor = torch.nn.utils.clip_grad_norm_(get_parameters(self.actor_list), self.grad_clip_norm)
            grad_norm_value = torch.nn.utils.clip_grad_norm_(get_parameters(self.value_list), self.grad_clip_norm)

            self.actor_optimizer.step()
            self.value_optimizer.step()

            actor_l.append(actor_loss.item())
            value_l.append(value_loss.item())
            mean_targ.append(target_info['mean_targ'])
            min_targ.append(target_info['min_targ'])
            max_targ.append(target_info['max_targ'])
            std_targ.append(target_info['std_targ'])

        train_metrics['mean_targ'] = np.mean(mean_targ)
        train_metrics['min_targ'] = np.mean(min_targ)
        train_metrics['max_targ'] = np.mean(max_targ)
        train_metrics['std_targ'] = np.mean(std_targ)
        train_metrics['value_loss'] = np.mean(value_l)
        train_metrics['actor_loss'] = np.mean(actor_l)
        self.train_steps += 1
        return train_metrics

    def actorcritc_loss(self, posterior):
        with torch.no_grad():
            batched_posterior = self.RSSM.rssm_detach(
                self.RSSM.rssm_seq_to_batch(posterior, self.batch_size, self.seq_len - 1))

        with FreezeParameters(self.world_list):
            imag_rssm_states, imag_log_prob, policy_entropy = self.RSSM.rollout_imagination(self.horizon,
                                                                                            self.ActionModel,
                                                                                            batched_posterior)

        imag_asrstates = self.RSSM.get_model_state(imag_rssm_states)
        imag_rewardstates = self.RSSM.get_model_state(imag_rssm_states)
        with FreezeParameters(self.world_list + self.value_list + [self.TargetValueModel] + [self.DiscountModel]):
            imag_reward_dist = self.RewardDecoder(imag_rewardstates)
            imag_reward = imag_reward_dist.mean
            imag_value_dist = self.TargetValueModel(imag_asrstates)
            imag_value = imag_value_dist.mean
            discount_dist = self.DiscountModel(imag_asrstates)
            discount_arr = self.discount * torch.round(discount_dist.base_dist.probs)  # mean = prob(disc==1)
        actor_loss, discount, lambda_returns = self._actor_loss(imag_reward, imag_value, discount_arr, imag_log_prob,
                                                                policy_entropy)
        value_loss = self._value_loss(imag_asrstates, discount, lambda_returns)

        mean_target = torch.mean(lambda_returns, dim=1)
        max_targ = torch.max(mean_target).item()
        min_targ = torch.min(mean_target).item()
        std_targ = torch.std(mean_target).item()
        mean_targ = torch.mean(mean_target).item()
        target_info = {
            'min_targ': min_targ,
            'max_targ': max_targ,
            'std_targ': std_targ,
            'mean_targ': mean_targ,
        }

        return actor_loss, value_loss, target_info

    def _get_prior_posterior(self, obs, actions, nonterms):
        embed = self.ObsEncoder(obs)  # t to t+seq_len
        prev_rssm_state = self.RSSM._init_rssm_state(self.batch_size)
        prior, posterior = self.RSSM.rollout_observation(self.seq_len, embed, actions, nonterms, prev_rssm_state)
        return prior, posterior

    def _actor_loss(self, imag_reward, imag_value, discount_arr, imag_log_prob, policy_entropy):

        lambda_returns = compute_return(imag_reward[:-1], imag_value[:-1], discount_arr[:-1], bootstrap=imag_value[-1],
                                        lambda_=self.lambda_)

        if self.config.actor_grad == 'reinforce':
            advantage = (lambda_returns - imag_value[:-1]).detach()
            objective = imag_log_prob[1:].unsqueeze(-1) * advantage

        elif self.config.actor_grad == 'dynamics':
            objective = lambda_returns
        else:
            raise NotImplementedError

        discount_arr = torch.cat([torch.ones_like(discount_arr[:1]), discount_arr[1:]])
        discount = torch.cumprod(discount_arr[:-1], 0)
        policy_entropy = policy_entropy[1:].unsqueeze(-1)
        actor_loss = -torch.sum(torch.mean(discount * (objective + self.actor_entropy_scale * policy_entropy), dim=1))
        return actor_loss, discount, lambda_returns

    def _value_loss(self, imag_asrstates, discount, lambda_returns):
        with torch.no_grad():
            value_modelstates = imag_asrstates[:-1].detach()
            value_discount = discount.detach()
            value_target = lambda_returns.detach()

        value_dist = self.ValueModel(value_modelstates)
        value_loss = -torch.mean(value_discount * value_dist.log_prob(value_target).unsqueeze(-1))
        return value_loss

    def update_target(self):
        mix = self.config.slow_target_fraction if self.config.use_slow_target else 1
        for param, target_param in zip(self.ValueModel.parameters(), self.TargetValueModel.parameters()):
            target_param.data.copy_(mix * param.data + (1 - mix) * target_param.data)

    def save_model(self, iter):
        save_dict = self.get_save_dict()
        model_dir = os.path.join(self.config.model_dir, 'agent_%d' % iter)
        os.makedirs(model_dir, exist_ok=True)
        save_path = os.path.join(model_dir, 'agent_%d.pth' % iter)
        torch.save(save_dict, save_path)
        return save_path

    def get_trainer_dict(self):
        return {
            "RSSM": self.RSSM.state_dict(),
            "ObsEncoder": self.ObsEncoder.state_dict(),
            "ObsDecoder": self.ObsDecoder.state_dict(),
            "RewardDecoder": self.RewardDecoder.state_dict(),
            "DiscountModel": self.DiscountModel.state_dict(),
        }

    def get_save_dict(self):
        return {
            "ActionModel": self.ActionModel.state_dict(),
            "ValueModel": self.ValueModel.state_dict(),
        }

    def load_trainer_dict(self, saved_dict):
        self.RSSM.load_state_dict(saved_dict["RSSM"])
        self.ObsEncoder.load_state_dict(saved_dict["ObsEncoder"])
        self.ObsDecoder.load_state_dict(saved_dict["ObsDecoder"])
        self.RewardDecoder.load_state_dict(saved_dict["RewardDecoder"])
        self.DiscountModel.load_state_dict(saved_dict['DiscountModel'])

    def load_save_dict(self, saved_dict):
        self.ActionModel.load_state_dict(saved_dict["ActionModel"])
        self.ValueModel.load_state_dict(saved_dict["ValueModel"])
        
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
        self.RSSM = DRSSM(config.action_size, self.rssm_node_size, self.embedding_size, self.theta_deter_size, self.theta_stoch_size,
                          self.device, config.rssm_type,
                          config.rssm_info).to(self.device)
        self.RewardDecoder = DenseModel((1,), self.reward_size, config.reward).to(self.device)

        self.ActionModel = DiscreteActionModel(self.action_size, self.deter_size, self.stoch_size,
                                               self.embedding_size, config.actor,
                                               config.expl, self.theta_size).to(self.device)

        self.ValueModel = DenseModel((1,), self.asrstate_size, config.critic).to(self.device)
        self.TargetValueModel = DenseModel((1,), self.asrstate_size, config.critic).to(self.device)
        self.TargetValueModel.load_state_dict(self.ValueModel.state_dict())

        if config.discount['use']:
            self.DiscountModel = DenseModel((1,), self.asrstate_size, config.discount).to(self.device)
        if config.pixel:
            self.ObsEncoder = ObsEncoder(self.obs_shape, self.embedding_size, config.obs_encoder).to(self.device)
            self.ObsDecoder = ObsDecoder(self.obs_shape, self.modelstate_size, config.obs_decoder).to(self.device)
        else:
            self.ObsEncoder = DenseModel((self.embedding_size,), int(np.prod(self.obs_shape)), config.obs_encoder).to(self.device)
            self.ObsDecoder = DenseModel(self.obs_shape, self.modelstate_size, config.obs_decoder).to(self.device)

    def _optim_initialize(self, config):
        actor_lr = config.lr['actor']
        value_lr = config.lr['critic']
        self.world_list = [self.ObsEncoder, self.RSSM, self.RewardDecoder, self.ObsDecoder, self.DiscountModel]
        self.actor_list = [self.ActionModel]
        self.value_list = [self.ValueModel]
        self.actor_optimizer = optim.Adam(get_parameters(self.actor_list), actor_lr)
        self.value_optimizer = optim.Adam(get_parameters(self.value_list), value_lr)

    def _model_expand_initialize(self, config, expand_deter_size=10, expand_stoch_size=4):
        self.expand_deter_size = expand_deter_size
        self.expand_stoch_size = expand_stoch_size
        new_deter_size = self.deter_size + expand_deter_size
        new_stoch_size = self.stoch_size + expand_stoch_size
        new_modelstate_size = new_stoch_size + new_deter_size
        new_asrstate_size = new_stoch_size + new_deter_size
        new_reward_size = new_stoch_size + new_deter_size

        self.RSSM = DRSSM(self.action_size, self.rssm_node_size, self.embedding_size,
                          self.theta_deter_size, self.theta_stoch_size,
                          self.device, config.rssm_type, config.rssm_info,
                          new_deter_size, new_stoch_size).to(self.device)
        
        self.RewardDecoder = DenseModel((1,), new_reward_size, config.reward).to(self.device)

        self.new_ActionModel = DiscreteActionModel(self.action_size, new_deter_size, new_stoch_size,
                                               self.embedding_size, config.actor,
                                               config.expl, self.theta_size).to(self.device)

        if config.discount['use']:
            self.DiscountModel = DenseModel((1,), new_asrstate_size, config.discount).to(self.device)
        if config.pixel:
            self.ObsDecoder = ObsDecoder(self.obs_shape, new_modelstate_size, config.obs_decoder).to(self.device)
        else:
            self.ObsDecoder = DenseModel(self.obs_shape, new_modelstate_size, config.obs_decoder).to(self.device)

        self.new_ValueModel = DenseModel((1,), new_asrstate_size, config.critic).to(self.device)
        self.param_succeed(config)

        self.ActionModel = self.new_ActionModel
        self.ValueModel = self.new_ValueModel
        self.TargetValueModel = DenseModel((1,), new_asrstate_size, config.critic).to(self.device)
        self.TargetValueModel.load_state_dict(self.ValueModel.state_dict())
        self.deter_size = new_deter_size

        self._optim_initialize(config)

    def _action_expansion_initialize(self, config, expand_action_size=4):
        self.expand_action_size = expand_action_size
        new_action_size = self.action_size + expand_action_size

        self.new_ActionModel = DiscreteActionModel(new_action_size, self.deter_size, self.stoch_size,
                                               self.embedding_size, config.actor,
                                               config.expl, self.theta_size).to(self.device)

        param_copy(self.ActionModel, self.new_ActionModel, 'model.0.weight')
        param_copy(self.ActionModel, self.new_ActionModel, 'model.0.bias')
        param_copy(self.ActionModel, self.new_ActionModel, 'model.2.weight') 
        param_copy(self.ActionModel, self.new_ActionModel, 'model.2.bias')
        param_copy(self.ActionModel, self.new_ActionModel, 'model.4.weight', target_row_end=self.action_size) 
        param_copy(self.ActionModel, self.new_ActionModel, 'model.4.bias', target_row_end=self.action_size)

        self.ActionModel = self.new_ActionModel
        
        self.action_size = new_action_size

    def param_succeed(self, config):        
        param_copy(self.ActionModel, self.new_ActionModel, 'model.0.weight', target_col_end=self.theta_size + self.stoch_size)
        param_copy(self.ActionModel, self.new_ActionModel, 'model.0.weight', target_col_start=self.theta_size + self.stoch_size + self.expand_stoch_size,
                                                                             target_col_end=self.theta_size + self.stoch_size + self.expand_stoch_size + self.deter_size,
                                                                             source_col_start=self.theta_size + self.stoch_size,
                                                                             source_col_end=self.theta_size + self.stoch_size + self.deter_size)
        param_copy(self.ActionModel, self.new_ActionModel, 'model.0.bias')
        param_copy(self.ActionModel, self.new_ActionModel, 'model.2.weight') 
        param_copy(self.ActionModel, self.new_ActionModel, 'model.2.bias')
        param_copy(self.ActionModel, self.new_ActionModel, 'model.4.weight') 
        param_copy(self.ActionModel, self.new_ActionModel, 'model.4.bias')
        param_copy(self.ValueModel, self.new_ValueModel, 'model.0.weight', target_col_end=self.stoch_size)
        param_copy(self.ValueModel, self.new_ValueModel, 'model.0.weight', target_col_start=self.stoch_size + self.expand_stoch_size,
                                                                           target_col_end=self.asrstate_size + self.expand_stoch_size,
                                                                           source_col_start=self.stoch_size,
                                                                           source_col_end=self.asrstate_size)
        param_copy(self.ValueModel, self.new_ValueModel, 'model.0.bias')
        param_copy(self.ValueModel, self.new_ValueModel, 'model.2.weight') 
        param_copy(self.ValueModel, self.new_ValueModel, 'model.2.bias')
        param_copy(self.ValueModel, self.new_ValueModel, 'model.4.weight') 
        param_copy(self.ValueModel, self.new_ValueModel, 'model.4.bias')


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
