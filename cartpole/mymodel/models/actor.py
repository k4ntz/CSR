import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import torch.distributions as td
from torch.distributions.independent import Independent
from torch.distributions.transformed_distribution import TransformedDistribution
from torch.distributions.transforms import TanhTransform
from ..utils.dist import TruncNormalDist


class ClipGradTanhBijector(torch.distributions.Transform):
    domain = TanhTransform.domain
    codomain = TanhTransform.codomain

    def __init__(self):
        super().__init__()
        self.bijective = True

    @property
    def sign(self):
        return 1.0

    def _call(self, x):
        return torch.tanh(x)

    def _inverse(self, y: torch.Tensor):
        y = torch.where(torch.abs(y) <= 1.0, torch.clamp(y, -0.99999997, 0.99999997), y)
        y = torch.atanh(y)
        return y

    def log_abs_det_jacobian(self, x, y):
        return 2.0 * (np.log(2) - x - F.softplus(-2.0 * x))


class SampleDist:
    def __init__(self, dist: torch.distributions.Distribution, samples=100):
        self._dist = dist
        self._samples = samples

    @property
    def name(self):
        return "SampleDist"

    def __getattr__(self, name):
        return getattr(self._dist, name)

    def normal_std(self):
        base_dist = self._dist.base_dist.base_dist
        return base_dist.scale.mean()

    @property
    def mean(self):
        dist: torch.distributions.Distribution = self._dist.expand((self._samples, *self._dist.batch_shape))
        sample = dist.rsample()
        return torch.mean(sample, 0)

    @property
    def mode(self):
        dist: torch.distributions.Distribution = self._dist.expand((self._samples, *self._dist.batch_shape))
        sample: torch.Tensor = dist.rsample()
        logprob = dist.log_prob(sample)
        batch_size = sample.size(1)
        feature_size = sample.size(2)
        indices = torch.argmax(logprob, dim=0).reshape(1, batch_size, 1).expand(1, batch_size, feature_size)
        return torch.gather(sample, 0, indices).squeeze(0)

    def entropy(self):
        dist: torch.distributions.Distribution = self._dist.expand((self._samples, *self._dist.batch_shape))
        sample: torch.Tensor = dist.rsample()
        logprob = dist.log_prob(sample)
        return -torch.mean(logprob, 0)

    def sample(self):
        return self._dist.sample()

    def rsample(self):
        return self._dist.rsample()

    def log_prob(self, value):
        return self._dist.log_prob(value)


class DiscreteActionModel(nn.Module):
    def __init__(
            self,
            action_size,
            deter_size,
            stoch_size,
            embedding_size,
            actor_info,
            expl_info,
            theta_size=0,
            min_std=1e-4,
            init_std=2,
            mean_scale=5,
    ):
        super().__init__()
        self.action_size = action_size
        self.deter_size = deter_size
        self.stoch_size = stoch_size
        self.theta_size = theta_size
        self.embedding_size = embedding_size
        self.layers = actor_info['layers']
        self.node_size = actor_info['node_size']
        self.act_fn = actor_info['activation']
        self.dist = actor_info['dist']
        self.act_fn = actor_info['activation']
        self.train_noise = expl_info['train_noise']
        self.eval_noise = expl_info['eval_noise']
        self.expl_min = expl_info['expl_min']
        self.expl_decay = expl_info['expl_decay']
        self.expl_type = expl_info['expl_type']
        self.model = self._build_model()

        self._min_std = min_std
        self._init_std = init_std
        self._mean_scale = mean_scale
        self._raw_init_std = np.log(np.exp(self._init_std) - 1)

    def _build_model(self):
        model = [nn.Linear(self.deter_size + self.stoch_size + self.theta_size, self.node_size)]
        model += [self.act_fn()]
        for i in range(1, self.layers):
            model += [nn.Linear(self.node_size, self.node_size)]
            model += [self.act_fn()]

        if self.dist == 'one_hot':
            model += [nn.Linear(self.node_size, self.action_size)]
        elif self.dist == 'normal' or self.dist == 'truncated_normal':
            model += [nn.Linear(self.node_size, self.action_size * 2)]
        else:
            raise NotImplementedError
        return nn.Sequential(*model)

    def forward(self, model_state):
        action_dist = self.get_action_dist(model_state)
        if self.dist == 'one_hot':
            action = action_dist.sample()
            action = action + action_dist.probs - action_dist.probs.detach()
        elif self.dist == 'normal':
            action = action_dist.rsample()
            action = action.clamp(-1, 1)
        return action, action_dist

    def get_action_dist(self, modelstate):
        logits = self.model(modelstate)
        if self.dist == 'one_hot':
            return torch.distributions.OneHotCategorical(logits=logits)
        elif self.dist == 'normal':
            mean, std = torch.chunk(logits, 2, dim=-1)
            # mean = torch.tanh(mean)
            # lo, hi = self._min_std, self._max_std
            # std = (hi - lo) * torch.sigmoid(std + 2.0) + lo
            # return torch.distributions.Independent(td.Normal(mean, std), 1)

            mean = self._mean_scale * torch.tanh(mean / self._mean_scale)
            std = F.softplus(std + self._raw_init_std) + self._min_std

            dist = td.Normal(mean, std)
            dist = TransformedDistribution(dist, ClipGradTanhBijector())
            dist = torch.distributions.Independent(dist, 1)
            dist = SampleDist(dist)
            return dist
        elif self.dist == 'truncated_normal':
            mean, std = torch.chunk(logits, 2, dim=-1)
            std = 2 * torch.sigmoid((std + self._raw_init_std) / 2) + self._min_std
            dist = TruncNormalDist(torch.tanh(mean), std, -1, 1)
            # dist = torch.distributions.Independent(dist, 1)
            return dist
        else:
            raise NotImplementedError

    def add_exploration(self, action: torch.Tensor, itr: int, mode='train'):
        if mode == 'train':
            expl_amount = self.train_noise
            expl_amount = expl_amount - itr / self.expl_decay
            expl_amount = max(self.expl_min, expl_amount)
        elif mode == 'eval':
            expl_amount = self.eval_noise
        else:
            raise NotImplementedError

        if self.expl_type == 'epsilon_greedy':
            if np.random.uniform(0, 1) < expl_amount:
                if self.dist == "one_hot":
                    index = torch.randint(0, self.action_size, action.shape[:-1], device=action.device)
                    action = torch.zeros_like(action)
                    action[:, index] = 1
                elif self.dist == "normal":
                    # action = torch.rand_like(action) * 2 - 1
                    action = td.Normal(action, expl_amount).sample().clamp(-0.99999997, 0.99999997)
                else:
                    raise NotImplementedError
        elif self.expl_type == 'add_noise':
            assert self.dist == "normal"
            action = td.Normal(action, expl_amount).sample().clamp(-0.99999997, 0.99999997)
        elif self.expl_type == 'no':
            pass
        else:
            raise NotImplementedError

        return action

    def optimal_action(self, model_state, sample_num=100):
        if self.dist == "normal":
            # print('optimal action')
            logits = self.model(model_state)
            mean, std = torch.chunk(logits, 2, dim=-1)
            # mean = torch.tanh(mean)
            # lo, hi = self._min_std, self._max_std
            # std = (hi - lo) * torch.sigmoid(std + 2.0) + lo
            # return torch.distributions.Independent(td.Normal(mean, std), 1)
            mean = self._mean_scale * torch.tanh(mean / self._mean_scale)
            return torch.tanh(mean)
        action_dist = self.get_action_dist(model_state)
        if self.dist == "one_hot":
            return action_dist.mode
        elif self.dist == "normal" or self.dist == "truncated_normal":
            return action_dist.mode
        else:
            raise NotImplementedError
