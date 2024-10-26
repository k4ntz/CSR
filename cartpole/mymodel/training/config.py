import numpy as np
import torch.nn as nn
from dataclasses import dataclass, field
from typing import Any, Tuple, Dict


@dataclass
class CartPoleFricConfig():
    '''default HPs that are known to work for Cartpole envs'''
    # env desc
    env: str = "cartpole"
    obs_shape: Tuple = (3, 64, 64)
    action_size: int = 8
    theta_deter_size: int = 1
    theta_stoch_size: int = 1
    pixel: bool = True
    action_repeat: int = 1
    time_limit: int = 200

    # Algorithm desc
    disentangle: bool = True

    # buffer desc
    capacity: int = int(1e6)
    obs_dtype: np.dtype = np.uint8
    action_dtype: np.dtype = np.float32

    # training desc
    train_steps: int = int(3e5)
    train_every: int = 5
    collect_intervals: int = 5
    batch_size: int = 20
    seq_len: int = 30
    eval_every = int(5e4)
    eval_episode: int = 3
    eval_render: bool = False
    visualize_episode: int = 3
    save_every: int = int(5e4)
    seed_episodes: int = 5
    model_dir: int = 'results'
    gif_dir: int = 'results'
    seed_steps: int = 500

    # adapting desc
    adapt_steps: int = int(1e5)
    adapt_every: int = 5

    # expanding desc
    expand_steps: int = int(5e4)

    # searching desc
    search_steps: int = int(1e2)

    # agent desc
    agent_steps: int = int(1e5)

    # latent space desc
    rssm_type: str = 'continuous'
    embedding_size: int = 100
    rssm_node_size: int = 100
    rssm_info: Dict = field(
        default_factory=lambda: {'deter_size': 30,
                                 'stoch_size': 4,
                                 'class_size': 16, 'category_size': 16, 'min_std': 0.1, 'mask_threshold': 0,
                                 'allow_mask': False})
    # objective desc
    grad_clip: float = 100.0
    discount_: float = 0.99
    lambda_: float = 0.95
    horizon: int = 8
    lr: Dict = field(default_factory=lambda: {'model': 2e-4, 'actor': 4e-5, 'critic': 5e-5})
    loss_scale: Dict = field(
        default_factory=lambda: {'kl': 0.1, 'reward': 5.0,
                                 'discount': 5.0})
    kl: Dict = field(default_factory=lambda: {'use_kl_balance': True, 'kl_balance_scale': 0.8, 'use_free_nats': False,
                                              'free_nats': 0.0})
    use_slow_target: float = True
    slow_target_update: int = 50
    slow_target_fraction: float = 1.0

    # actor critic
    actor: Dict = field(
        default_factory=lambda: {'layers': 2, 'node_size': 100, 'dist': 'one_hot', 'min_std': 1e-4, 'init_std': 5,
                                 'mean_scale': 5, 'activation': nn.ELU})
    critic: Dict = field(
        default_factory=lambda: {'layers': 2, 'node_size': 100, 'dist': 'normal', 'activation': nn.ELU})
    expl: Dict = field(
        default_factory=lambda: {'train_noise': 0.4, 'eval_noise': 0.0, 'expl_min': 0.05, 'expl_decay': 10000.0,
                                 'expl_type': 'epsilon_greedy'})
    actor_grad: str = 'reinforce'
    actor_grad_mix: int = 0.0
    actor_entropy_scale: float = 1e-3

    # learnt world-models desc
    obs_encoder: Dict = field(
        default_factory=lambda: {'layers': 2, 'node_size': 100, 'dist': None, 'activation': nn.ELU, 'kernel': 2,
                                 'depth': 16})
    obs_decoder: Dict = field(
        default_factory=lambda: {'layers': 2, 'node_size': 100, 'dist': 'normal', 'activation': nn.ELU, 'kernel': 2,
                                 'depth': 16})
    reward: Dict = field(
        default_factory=lambda: {'layers': 2, 'node_size': 100, 'dist': 'normal', 'activation': nn.ELU})
    aux_reward_1: Dict = field(
        default_factory=lambda: {'layers': 2, 'node_size': 100, 'dist': 'normal', 'activation': nn.ELU})
    aux_reward_2: Dict = field(
        default_factory=lambda: {'layers': 2, 'node_size': 100, 'dist': 'normal', 'activation': nn.ELU})
    aux_action_1: Dict = field(
        default_factory=lambda: {'layers': 2, 'node_size': 100, 'dist': 'categorical', 'activation': nn.ELU})
    aux_action_2: Dict = field(
        default_factory=lambda: {'layers': 2, 'node_size': 100, 'dist': 'categorical', 'activation': nn.ELU})

    action: Dict = field(default_factory=lambda: {'layers': 2, 'node_size': 100, 'dist': None, 'activation': nn.ELU})

    discount: Dict = field(
        default_factory=lambda: {'layers': 2, 'node_size': 100, 'dist': 'binary', 'activation': nn.ELU, 'use': True})
