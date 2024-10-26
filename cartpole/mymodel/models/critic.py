from IFactor.models.dense import DenseModel
import torch as th
from typing import Tuple

class QCritic(th.nn.Module):
    def __init__(
        self,
        state_size,
        action_size,
        config,
        n_critics: int = 2,
    ):
        super(QCritic, self).__init__()
        self.n_critics = n_critics
        self.q_networks = []
        print(config.critic)
        for idx in range(n_critics):
            q_net = DenseModel((1,), state_size+action_size, config.critic)
            self.add_module(f"qf{idx}", q_net)
            self.q_networks.append(q_net)

    def forward(self, states: th.Tensor, actions: th.Tensor) -> Tuple[th.Tensor, ...]:
        # Learn the features extractor using the policy loss only
        # when the features_extractor is shared with the actor
        qvalue_input = th.cat([states, actions], dim=-1)
        return tuple(q_net(qvalue_input) for q_net in self.q_networks)

    def q1_forward(self, states: th.Tensor, actions: th.Tensor) -> th.Tensor:
        """
        Only predict the Q-value using the first network.
        This allows to reduce computation when all the estimates are not needed
        (e.g. when updating the policy in TD3).
        """
        return self.q_networks[0](th.cat([states, actions], dim=1))