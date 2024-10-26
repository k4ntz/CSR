import datetime
import gym
import numpy as np
import uuid


class EpsilonGreedyWrapper(gym.Wrapper):
    """
    Wrapper to perform a random action each step instead of the requested action, 
    with the provided probability.
    """
    def __init__(self, env, prob=0.05):
        gym.Wrapper.__init__(self, env)
        self.prob = prob
        self.num_envs = env.num_envs

    def reset(self):
        return self.env.reset()

    def step(self, action):
        if np.random.uniform()<self.prob:
            action = np.random.randint(self.env.action_space.n, size=self.num_envs)
        
        return self.env.step(action)


class EpisodeRewardWrapper(gym.Wrapper):
    def __init__(self, env):
        env.metadata = {'render.modes': []}
        env.reward_range = (-float('inf'), float('inf'))
        nenvs = env.num_envs
        self.num_envs = nenvs
        super(EpisodeRewardWrapper, self).__init__(env)

        self.aux_rewards = None
        self.num_aux_rews = None

        def reset(**kwargs):
            self.rewards = np.zeros(nenvs)
            self.lengths = np.zeros(nenvs)
            self.aux_rewards = None
            self.long_aux_rewards = None

            return self.env.reset(**kwargs)

        def step(action):
            obs, rew, done, infos = self.env.step(action)

            if self.aux_rewards is None:
                info = infos[0]
                if 'aux_rew' in info:
                    self.num_aux_rews = len(infos[0]['aux_rew'])
                else:
                    self.num_aux_rews = 0

                self.aux_rewards = np.zeros((nenvs, self.num_aux_rews), dtype=np.float32)
                self.long_aux_rewards = np.zeros((nenvs, self.num_aux_rews), dtype=np.float32)

            self.rewards += rew
            self.lengths += 1

            use_aux = self.num_aux_rews > 0

            if use_aux:
                for i, info in enumerate(infos):
                    self.aux_rewards[i,:] += info['aux_rew']
                    self.long_aux_rewards[i,:] += info['aux_rew']

            for i, d in enumerate(done):
                if d:
                    epinfo = {'r': round(self.rewards[i], 6), 'l': self.lengths[i], 't': 0}
                    aux_dict = {}

                    for nr in range(self.num_aux_rews):
                        aux_dict['aux_' + str(nr)] = self.aux_rewards[i,nr]

                    if 'ale.lives' in infos[i]:
                        game_over_rew = np.nan

                        is_game_over = infos[i]['ale.lives'] == 0

                        if is_game_over:
                            game_over_rew = self.long_aux_rewards[i,0]
                            self.long_aux_rewards[i,:] = 0

                        aux_dict['game_over_rew'] = game_over_rew

                    epinfo['aux_dict'] = aux_dict

                    infos[i]['episode'] = epinfo

                    self.rewards[i] = 0
                    self.lengths[i] = 0
                    self.aux_rewards[i,:] = 0

            return obs, rew, done, infos

        self.reset = reset
        self.step = step

def add_final_wrappers(env):
    env = EpisodeRewardWrapper(env)

    return env


class TimeLimit(gym.Wrapper):
    def __init__(self, env, duration):
        super().__init__(env)
        self._duration = duration
        self._step = None

    def step(self, action):
        assert self._step is not None, "Must reset environment."
        obs, reward, done, info = self.env.step(action)
        self._step += 1
        if self._step >= self._duration:
            done = True
            if "discount" not in info:
                info["discount"] = np.array(1.0).astype(np.float32)
            self._step = None
        return obs, reward, done, info

    def reset(self):
        self._step = 0
        return self.env.reset()


class NormalizeActions(gym.Wrapper):
    def __init__(self, env):
        super().__init__(env)
        self._mask = np.logical_and(
            np.isfinite(env.action_space.low), np.isfinite(env.action_space.high)
        )
        self._low = np.where(self._mask, env.action_space.low, -1)
        self._high = np.where(self._mask, env.action_space.high, 1)
        low = np.where(self._mask, -np.ones_like(self._low), self._low)
        high = np.where(self._mask, np.ones_like(self._low), self._high)
        self.action_space = gym.spaces.Box(low, high, dtype=np.float32)

    def step(self, action):
        original = (action + 1) / 2 * (self._high - self._low) + self._low
        original = np.where(self._mask, original, action)
        return self.env.step(original)


class OneHotAction(gym.Wrapper):
    def __init__(self, env):
        assert isinstance(env.action_space, gym.spaces.Discrete)
        super().__init__(env)
        self._random = np.random.RandomState()
        shape = (self.env.action_space.n,)
        space = gym.spaces.Box(low=0, high=1, shape=shape, dtype=np.float32)
        space.discrete = True
        self.action_space = space

    def step(self, action):
        index = np.argmax(action).astype(int)
        reference = np.zeros_like(action)
        reference[index] = 1
        if not np.allclose(reference, action):
            raise ValueError(f"Invalid one-hot action:\n{action}")
        return self.env.step(index)

    def reset(self):
        return self.env.reset()

    def _sample_action(self):
        actions = self.env.action_space.n
        index = self._random.randint(0, actions)
        reference = np.zeros(actions, dtype=np.float32)
        reference[index] = 1.0
        return reference


class RewardObs(gym.Wrapper):
    def __init__(self, env):
        super().__init__(env)
        spaces = self.env.observation_space.spaces
        if "obs_reward" not in spaces:
            spaces["obs_reward"] = gym.spaces.Box(
                -np.inf, np.inf, shape=(1,), dtype=np.float32
            )
        self.observation_space = gym.spaces.Dict(spaces)

    def step(self, action):
        obs, reward, done, info = self.env.step(action)
        if "obs_reward" not in obs:
            obs["obs_reward"] = np.array([reward], dtype=np.float32)
        return obs, reward, done, info

    def reset(self):
        obs = self.env.reset()
        if "obs_reward" not in obs:
            obs["obs_reward"] = np.array([0.0], dtype=np.float32)
        return obs


class SelectAction(gym.Wrapper):
    def __init__(self, env, key):
        super().__init__(env)
        self._key = key

    def step(self, action):
        return self.env.step(action[self._key])


class UUID(gym.Wrapper):
    def __init__(self, env):
        super().__init__(env)
        timestamp = datetime.datetime.now().strftime("%Y%m%dT%H%M%S")
        self.id = f"{timestamp}-{str(uuid.uuid4().hex)}"

    def reset(self):
        timestamp = datetime.datetime.now().strftime("%Y%m%dT%H%M%S")
        self.id = f"{timestamp}-{str(uuid.uuid4().hex)}"
        return self.env.reset()
