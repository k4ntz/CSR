import torch
import torch.nn as nn
import torch.distributions as td
from mymodel.utils.drssm import DRSSMUtils, RSSMContState, RSSMDiscState, gumbel_sigmoid
from typing import *
import pdb


class TemporalPrior(nn.Module):
    def __init__(self, deter_size,
                 stoch_size, theta_stoch_size, node_size, rssm_type="continuous",
                 act_fn=nn.ELU) -> None:
        super().__init__()
        self.deter_size = deter_size
        self.stoch_size = stoch_size
        self.theta_stoch_size = theta_stoch_size
        self.node_size = node_size
        self.rssm_type = rssm_type
        self.act_fn = act_fn
        self.prior = None
        self._build_model()

    def _build_model(self):
        if self.deter_size > 0:
            temporal_prior = [nn.Linear(self.theta_stoch_size + self.deter_size, self.node_size), self.act_fn()]
        if self.rssm_type == 'discrete':
            raise NotImplementedError
        elif self.rssm_type == 'continuous':
            if self.deter_size > 0:
                temporal_prior += [nn.Linear(self.node_size, 2 * self.stoch_size)]
                self.prior = nn.Sequential(*temporal_prior)

    def forward(self, input_tensor):
        mean_result_list = []
        std_result_list = []
        if self.rssm_type == 'discrete':
            raise NotImplementedError
        if self.rssm_type == 'continuous':
            if self.prior is not None:
                output_stoch_mean, output_stoch_std = torch.chunk(self.prior(input_tensor), 2, dim=-1)
                mean_result_list.append(output_stoch_mean)
                std_result_list.append(output_stoch_std)
            return torch.cat(mean_result_list + std_result_list, dim=-1)


class TemporalPosterior(nn.Module):
    def __init__(self, deter_size,
                 stoch_size, embedding_size, node_size,
                 rssm_type="continuous", act_fn=nn.ELU) -> None:
        super().__init__()
        self.deter_size = deter_size
        self.stoch_size = stoch_size
        self.embedding_size = embedding_size
        self.node_size = node_size
        self.rssm_type = rssm_type
        self.act_fn = act_fn
        self.posterior = None
        self._build_model()

    def _build_model(self):
        if self.deter_size > 0:
            temporal_posterior = [
                nn.Linear(self.embedding_size + self.deter_size, self.node_size), self.act_fn()]
        if self.rssm_type == 'discrete':
            raise NotImplementedError
        elif self.rssm_type == 'continuous':
            if self.deter_size > 0:
                temporal_posterior += [nn.Linear(self.node_size, 2 * self.stoch_size)]
                self.posterior = nn.Sequential(*temporal_posterior)

    def forward(self, input_tensor):
        mean_result_list = []
        std_result_list = []
        if self.rssm_type == 'discrete':
            raise NotImplementedError
        if self.rssm_type == 'continuous':
            if self.posterior is not None:
                output_stoch_mean, output_stoch_std = torch.chunk(self.posterior(input_tensor), 2, dim=-1)
                mean_result_list.append(output_stoch_mean)
                std_result_list.append(output_stoch_std)
            return torch.cat(mean_result_list + std_result_list, dim=-1)


class DRSSM(nn.Module, DRSSMUtils):
    def __init__(
            self,
            action_size,
            rssm_node_size,
            embedding_size,
            theta_deter_size,
            theta_stoch_size,
            device,
            rssm_type,
            info,
            deter_size=None,
            stoch_size=None,
            act_fn=nn.ELU,
    ):
        nn.Module.__init__(self)
        DRSSMUtils.__init__(self, rssm_type=rssm_type, info=info, deter_size=deter_size, stoch_size=stoch_size)
        self.device = device
        self.action_size = action_size
        self.node_size = rssm_node_size
        self.embedding_size = embedding_size
        self.theta_deter_size = theta_deter_size
        self.theta_stoch_size = theta_stoch_size
        self.act_fn = act_fn
        self.rnn = nn.GRUCell(self.deter_size, self.deter_size)
        self._build_embed_state_action()
        self._build_theta()
        self.fc_prior = self._build_temporal_prior()
        self.fc_posterior = self._build_temporal_posterior()

    def _build_theta(self):
        self.theta_deter = None
        self.theta_stoch = None
        if self.theta_deter_size > 0:
            self.theta_deter = nn.Parameter(torch.empty(self.theta_deter_size))
            nn.init.xavier_uniform_(self.theta_deter.unsqueeze(0))
        if self.theta_stoch_size > 0:
            self.theta_stoch = nn.Parameter(torch.empty(self.theta_stoch_size))
            nn.init.xavier_uniform_(self.theta_stoch.unsqueeze(0))

    def _build_embed_state_action(self):
        """
        model is supposed to take in previous stochastic state and previous action
        and embed it to deter size for rnn input
        """
        self.fc_embed = None
        if self.deter_size > 0:
            fc_embed_sa = [nn.Linear(self.theta_deter_size + self.stoch_size + self.action_size, self.deter_size), self.act_fn()]
            self.fc_embed = nn.Sequential(*fc_embed_sa)

    def _expand_embed_state_action(self):
        self.new_fc_embed = None
        if self.deter_size > 0:
            # Define new sequential layer
            new_fc_embed_sa = [nn.Linear(self.theta_deter_size + self.stoch_size + self.action_size, self.deter_size), self.act_fn()]
            self.new_fc_embed = nn.Sequential(*new_fc_embed_sa).to(self.device)
        
            # Copy the parameters from the old layer to the new one
            with torch.no_grad():
                self.new_fc_embed[0].weight[:, :self.fc_embed[0].weight.size(1)] = self.fc_embed[0].weight.clone()
                self.new_fc_embed[0].bias[:] = self.fc_embed[0].bias.clone()
        
            self.fc_embed = self.new_fc_embed
            del self.new_fc_embed

    def _build_temporal_prior(self):
        """
        model is supposed to take in latest deterministic state
        and output prior over stochastic state
        """
        return TemporalPrior(self.deter_size,
                             self.stoch_size,
                             self.theta_stoch_size,
                             self.node_size, self.rssm_type, self.act_fn)

    def _build_temporal_posterior(self):
        """
        model is supposed to take in latest embedded observation and deterministic state
        and output posterior over stochastic states
        """
        return TemporalPosterior(self.deter_size,
                                 self.stoch_size,
                                 self.embedding_size, self.node_size, self.rssm_type, self.act_fn)

    def forward_embed_state(self, stoch_state, prev_action):
        prev_theta_deter = self.theta_deter.unsqueeze(0).expand(stoch_state.size(0), -1)
        return self.fc_embed(torch.cat([prev_theta_deter, stoch_state, prev_action], dim=-1))

    def forward_rnn(self, state_embed, prev_deter_state):
        deter_state = self.rnn(state_embed, prev_deter_state)
        return deter_state

    def rssm_imagine(self, prev_action, prev_rssm_state, nonterms=True):
        state_embed = self.forward_embed_state(prev_rssm_state.stoch * nonterms, prev_action)
        deter_state = self.forward_rnn(state_embed, prev_rssm_state.deter * nonterms)
        prev_theta_stoch = self.theta_stoch.unsqueeze(0).expand(deter_state.size(0), -1)
        if self.rssm_type == 'discrete':
            prior_logit = self.fc_prior(torch.cat([prev_theta_stoch, deter_state], dim=-1))
            stats = {'logit': prior_logit}
            prior_stoch_state = self.get_stoch_state(stats)
            prior_rssm_state = RSSMDiscState(prior_logit, prior_stoch_state, deter_state)
        elif self.rssm_type == 'continuous':
            prior_mean, prior_std = torch.chunk(self.fc_prior(torch.cat([prev_theta_stoch, deter_state], dim=-1)), 2, dim=-1)
            stats = {'mean': prior_mean, 'std': prior_std}
            prior_stoch_state, std = self.get_stoch_state(stats)
            prior_rssm_state = RSSMContState(prior_mean, std, prior_stoch_state, deter_state)
        return prior_rssm_state

    def rollout_imagination(self, horizon: int, actor: nn.Module, prev_rssm_state):
        rssm_state = prev_rssm_state
        next_rssm_states = []
        action_entropy = []
        imag_log_probs = []
        for t in range(horizon):
            try:
                action, action_dist = actor((self.get_actor_state(rssm_state, self.theta_deter, self.theta_stoch)).detach())
            except Exception:
                pdb.set_trace()
                
            rssm_state = self.rssm_imagine(action, rssm_state)
            next_rssm_states.append(rssm_state)
            action_entropy.append(action_dist.entropy())
            imag_log_probs.append(action_dist.log_prob(torch.round(action.detach())))

        next_rssm_states = self.rssm_stack_states(next_rssm_states, dim=0)
        action_entropy = torch.stack(action_entropy, dim=0)
        imag_log_probs = torch.stack(imag_log_probs, dim=0)
        return next_rssm_states, imag_log_probs, action_entropy

    def rssm_observe(self, obs_embed, prev_action, prev_nonterm, prev_rssm_state):
        prior_rssm_state = self.rssm_imagine(prev_action, prev_rssm_state, prev_nonterm)
        deter_state = prior_rssm_state.deter
        x = torch.cat([obs_embed, deter_state], dim=-1)
        if self.rssm_type == 'discrete':
            posterior_logit = self.fc_posterior(x)
            stats = {'logit': posterior_logit}
            posterior_stoch_state = self.get_stoch_state(stats)
            posterior_rssm_state = RSSMDiscState(posterior_logit, posterior_stoch_state, deter_state)

        elif self.rssm_type == 'continuous':
            posterior_mean, posterior_std = torch.chunk(self.fc_posterior(x), 2, dim=-1)
            stats = {'mean': posterior_mean, 'std': posterior_std}
            posterior_stoch_state, std = self.get_stoch_state(stats)
            posterior_rssm_state = RSSMContState(posterior_mean, std, posterior_stoch_state, deter_state)
        return prior_rssm_state, posterior_rssm_state

    def rollout_observation(self, seq_len: int, obs_embed: torch.Tensor, action: torch.Tensor, nonterms: torch.Tensor,
                            prev_rssm_state):
        priors = []
        posteriors = []
        for t in range(seq_len):
            prev_action = action[t] * nonterms[t]
            prior_rssm_state, posterior_rssm_state = self.rssm_observe(obs_embed[t], prev_action, nonterms[t],
                                                                       prev_rssm_state)
            priors.append(prior_rssm_state)
            posteriors.append(posterior_rssm_state)
            prev_rssm_state = posterior_rssm_state
        prior = self.rssm_stack_states(priors, dim=0)
        post = self.rssm_stack_states(posteriors, dim=0)
        return prior, post
