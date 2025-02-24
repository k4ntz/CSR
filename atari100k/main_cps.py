import argparse
import functools
import os
import pathlib
import sys
from rtpt import RTPT

os.environ["MUJOCO_GL"] = "osmesa"

import numpy as np
import ruamel.yaml as yaml

sys.path.append(str(pathlib.Path(__file__).parent))

import exploration as expl
import models
import tools
import envs.wrappers as wrappers
from parallel import Parallel, Damy

import torch
from torch import nn
from torch import distributions as torchd


to_np = lambda x: x.detach().cpu().numpy()


class CSRAgent(nn.Module):
    def __init__(self, obs_space, act_space, config, logger, dataset, expand_stoch_size, expand_deter_size):
        super(CSRAgent, self).__init__()
        self._config = config
        self._logger = logger
        self._should_log = tools.Every(config.log_every)
        batch_steps = config.batch_size * config.batch_length
        self._should_train = tools.Every(batch_steps / config.train_ratio)
        self._should_pretrain = tools.Once()
        self._should_reset = tools.Every(config.reset_every)
        self._should_expl = tools.Until(int(config.expl_until / config.action_repeat))
        self._metrics = {}
        # this is update step
        self._step = logger.step // config.action_repeat
        self._update_count = 0
        self._dataset = dataset
        self._wm = models.WorldModel(obs_space, act_space, self._step, config, expand_stoch_size, expand_deter_size)
        self._task_behavior = models.ImagBehavior(config, self._wm, expand_stoch_size, expand_deter_size)
        if (
            config.compile and os.name != "nt"
        ):  # compilation is not supported on windows
            self._wm = torch.compile(self._wm)
            self._task_behavior = torch.compile(self._task_behavior)
        reward = lambda f, s, a: self._wm.heads["reward"](f).mean()
        self._expl_behavior = dict(
            greedy=lambda: self._task_behavior,
            random=lambda: expl.Random(config, act_space),
            plan2explore=lambda: expl.Plan2Explore(config, self._wm, reward),
        )[config.expl_behavior]().to(self._config.device)

    def __call__(self, obs, reset, state=None, training=True, manager=None):
        step = self._step
        if training:
            steps = (
                self._config.pretrain
                if self._should_pretrain()
                else self._should_train(step)
            )
            for _ in range(steps):
                self._train(next(self._dataset))
                self._update_count += 1
                self._metrics["update_count"] = self._update_count
            if self._should_log(step):
                for name, values in self._metrics.items():
                    if name == 'model_loss' and manager is not None:
                        manager.tau = np.mean(values)
                    self._logger.scalar(name, float(np.mean(values)))
                    self._metrics[name] = []
                if self._config.video_pred_log:
                    openl = self._wm.video_pred(next(self._dataset))
                    self._logger.video("train_openl", to_np(openl))
                self._logger.write(fps=True)

        policy_output, state = self._policy(obs, state, training)

        if training:
            self._step += len(reset)
            self._logger.step = self._config.action_repeat * self._step
        return policy_output, state

    def _policy(self, obs, state, training):
        if state is None:
            latent = action = None
        else:
            latent, action = state
        obs = self._wm.preprocess(obs)
        embed = self._wm.encoder(obs)
        latent, _ = self._wm.dynamics.obs_step(latent, action, embed, obs["is_first"])
        if self._config.eval_state_mean:
            latent["stoch"] = latent["mean"]
        feat = self._wm.dynamics.get_actor_feat(latent)
        if not training:
            actor = self._task_behavior.actor(feat)
            action = actor.mode()
        elif self._should_expl(self._step):
            actor = self._expl_behavior.actor(feat)
            action = actor.sample()
        else:
            actor = self._task_behavior.actor(feat)
            action = actor.sample()
        logprob = actor.log_prob(action)
        latent = {k: v.detach() for k, v in latent.items()}
        action = action.detach()
        if self._config.actor["dist"] == "onehot_gumble":
            action = torch.one_hot(
                torch.argmax(action, dim=-1), self._config.num_actions
            )
        policy_output = {"action": action, "logprob": logprob}
        state = (latent, action)
        return policy_output, state

    def _train(self, data):
        metrics = {}
        post, context, mets = self._wm._train(data)
        metrics.update(mets)
        start = post
        reward = lambda f, s, a: self._wm.heads["reward"](
            self._wm.dynamics.get_feat(s)
        ).mode()
        metrics.update(self._task_behavior._train(start, reward)[-1])
        if self._config.expl_behavior != "greedy":
            mets = self._expl_behavior.train(start, context, data)[-1]
            metrics.update({"expl_" + key: value for key, value in mets.items()})
        for name, value in metrics.items():
            if not name in self._metrics.keys():
                self._metrics[name] = [value]
            else:
                self._metrics[name].append(value)

    def _weight_succeed(self, source_agent):
        self._wm._param_succeed(source_agent._wm)
        self._task_behavior._param_succeed(self._wm, source_agent._task_behavior)


class AgentManager:
    def __init__(self):
        self.expand_stoch_size = 0
        self.expand_deter_size = 0
        self.tau = None
        self.tau_star = None

    def state_space_grow(self):
        self.expand_stoch_size += 4
        self.expand_deter_size += 32


def count_steps(folder):
    return sum(int(str(n).split("-")[-1][:-4]) - 1 for n in folder.glob("*.npz"))


def make_dataset(episodes, config):
    generator = tools.sample_episodes(episodes, config.batch_length)
    dataset = tools.from_generator(generator, config.batch_size)
    return dataset


def make_env(config, id):
    suite, task = config.task.split("_", 1)
    if suite == "atari":
        import envs.atari as atari

        env = atari.Atari(
            task,
            config.action_repeat,
            config.size,
            gray=config.grayscale,
            noops=config.noops,
            lives=config.lives,
            sticky=config.stickey,
            actions=config.actions,
            resize=config.resize,
            mode=config.game_mode,
            difficulty=config.game_difficulty,
            seed=config.seed + id,
        )
        env = wrappers.OneHotAction(env)
    else:
        raise NotImplementedError(suite)
    env = wrappers.TimeLimit(env, config.time_limit)
    env = wrappers.SelectAction(env, key="action")
    env = wrappers.UUID(env)
    return env


def gen_across_tasks(config, idx, manager):
    tools.set_seed_everywhere(config.seed)
    if config.deterministic_run:
        tools.enable_deterministic_run()
    up_logdir = pathlib.Path(config.logdir).expanduser()
    logdir = pathlib.Path(config.logdir + '/task{}'.format(idx + 1)).expanduser()

    config.traindir = config.traindir or logdir / "train_eps"
    config.evaldir = config.evaldir or logdir / "eval_eps"
    config.steps //= config.action_repeat
    config.pred_steps //= config.action_repeat
    config.eval_every //= config.action_repeat
    config.log_every //= config.action_repeat
    config.time_limit //= config.action_repeat

    allow_mode_dict={
        'atari_alien': [0, 1, 2, 3],
        'atari_bank_heist': [ 0, 4, 8, 12, 16, 20, 24, 28],
        'atari_crazy_climber': [0, 1, 2, 3],
        'atari_gopher': [0, 2],
        'atari_pong': [0, 1], 
    }

    allow_diff_dict={
        'atari_alien': [0, 1, 2, 3],
        'atari_bank_heist': [0, 1, 2, 3],
        'atari_crazy_climber': [0, 1],
        'atari_gopher': [0, 1],
        'atari_pong': [0, 1],      
    }

    if idx:
        rng = np.random.default_rng(idx)
        config.game_mode = rng.choice(allow_mode_dict[config.task])
        config.game_difficulty = rng.choice(allow_diff_dict[config.task])

    print("Logdir", logdir)
    # ✅ 创建 RTPT 对象
    rtpt = RTPT(name_initials='Liu', experiment_name=f'{config.task}', max_iterations=config.steps)
    rtpt.start()
    logdir.mkdir(parents=True, exist_ok=True)
    config.traindir.mkdir(parents=True, exist_ok=True)
    config.evaldir.mkdir(parents=True, exist_ok=True)
    step = count_steps(config.traindir)
    # step in logger is environmental step
    logger = tools.Logger(logdir, config.action_repeat * step)

    print("Create envs.")
    if config.offline_traindir:
        directory = config.offline_traindir.format(**vars(config))
    else:
        directory = config.traindir
    train_eps = tools.load_episodes(directory, limit=config.dataset_size)
    if config.offline_evaldir:
        directory = config.offline_evaldir.format(**vars(config))
    else:
        directory = config.evaldir
    eval_eps = tools.load_episodes(directory, limit=1)
    make = lambda id: make_env(config, id)
    train_envs = [make(i) for i in range(config.envs)]
    eval_envs = [make(i) for i in range(config.envs)]
    if config.parallel:
        train_envs = [Parallel(env, "process") for env in train_envs]
        eval_envs = [Parallel(env, "process") for env in eval_envs]
    else:
        train_envs = [Damy(env) for env in train_envs]
        eval_envs = [Damy(env) for env in eval_envs]
    acts = train_envs[0].action_space
    print("Action Space", acts)
    config.num_actions = acts.n if hasattr(acts, "n") else acts.shape[0]

    state = None
    if not config.offline_traindir:
        prefill = max(0, config.prefill - count_steps(config.traindir))
        print(f"Prefill dataset ({prefill} steps).")
        if hasattr(acts, "discrete"):
            random_actor = tools.OneHotDist(
                torch.zeros(config.num_actions).repeat(config.envs, 1)
            )
        else:
            random_actor = torchd.independent.Independent(
                torchd.uniform.Uniform(
                    torch.Tensor(acts.low).repeat(config.envs, 1),
                    torch.Tensor(acts.high).repeat(config.envs, 1),
                ),
                1,
            )

        def random_agent(o, d, s):
            action = random_actor.sample()
            logprob = random_actor.log_prob(action)
            return {"action": action, "logprob": logprob}, None

        state = tools.simulate(
            random_agent,
            train_envs,
            train_eps,
            config.traindir,
            logger,
            limit=config.dataset_size,
            steps=prefill,
        )
        logger.step += prefill * config.action_repeat
        print(f"Logger: ({logger.step} steps).")

    print("Simulate agent.")
    train_dataset = make_dataset(train_eps, config)
    eval_dataset = make_dataset(eval_eps, config)
        
    agent = CSRAgent(
        train_envs[0].observation_space,
        train_envs[0].action_space,
        config,
        logger,
        train_dataset,
        manager.expand_stoch_size,
        manager.expand_deter_size,
    ).to(config.device)
    agent.requires_grad_(requires_grad=False)

    checkpoint_files = sorted(logdir.glob("checkpoint_*.pth"), key=os.path.getmtime, reverse=True)

    if checkpoint_files:
        latest_checkpoint = checkpoint_files[0]
        print(f"✅ Loading checkpoint: {latest_checkpoint}")
        checkpoint = torch.load(latest_checkpoint)

        agent.load_state_dict(checkpoint["agent_state_dict"])
        tools.recursively_load_optim_state_dict(agent, checkpoint["optims_state_dict"])
        agent._step = checkpoint.get("step", 0)  # 恢复训练步数
        
        # ✅ 恢复随机状态
        if "numpy_random_state" in checkpoint:
            np.random.set_state(checkpoint["numpy_random_state"])
        if "torch_random_state" in checkpoint:
            torch.set_rng_state(checkpoint["torch_random_state"])
        if "torch_cuda_random_state" in checkpoint and torch.cuda.is_available():
            torch.cuda.set_rng_state_all(checkpoint["torch_cuda_random_state"])
        # 恢复超参数
        if "hyperparameters" in checkpoint:
            config.learning_rate = checkpoint["hyperparameters"].get("learning_rate", config.model_lr)
            if hasattr(agent, "epsilon"):
                agent.epsilon = checkpoint["hyperparameters"].get("epsilon", agent.epsilon)

        # 恢复经验回放缓冲区
        if hasattr(agent, "replay_buffer") and "replay_buffer" in checkpoint:
            agent.replay_buffer.load(checkpoint["replay_buffer"])

        # 恢复环境状态
        if "env_state" in checkpoint and hasattr(train_envs[0], "load_state"):
            train_envs[0].load_state(checkpoint["env_state"])

        print(f"🔄 Checkpoint loaded, resuming from step {agent._step}")
    else:
        print("⚠️ No checkpoint found, starting training from scratch.")    
    
    if manager.tau is not None:
        print("Start determining whether to adapt or to expand.")
        while agent._step <= config.pred_steps:
            logger.write()
            state = tools.simulate(
                agent,
                train_envs,
                train_eps,
                config.traindir,
                logger,
                limit=config.dataset_size,
                steps=config.eval_every,
                state=state,
                manager=manager,
            )
        if manager.tau >= config.tau_bound * manager.tau_star:
            print('state space expansion.')
            manager.state_space_grow()
            new_agent = CSRAgent(
                train_envs[0].observation_space,
                train_envs[0].action_space,
                config,
                logger,
                train_dataset,
                manager.expand_stoch_size,
                manager.expand_deter_size,
            ).to(config.device)
            new_agent.requires_grad_(requires_grad=False)
            new_agent._weight_succeed(agent)
            agent = new_agent

            state = tools.simulate(
                random_agent,
                train_envs,
                train_eps,
                config.traindir,
                logger,
                limit=config.dataset_size,
                steps=prefill,
            )
        else:
            print('distribution shifts')
    else:
        config.pred_steps = 0
    # make sure eval will be executed once after config.steps
    
    while agent._step < config.pred_steps + config.steps + config.eval_every:
        logger.write()
        # ✅ 更新 RTPT 进度
        rtpt.step(subtitle=f"Step: {agent._step}")
        # ✅ 如果已经加载了 Checkpoint，就跳过存储
        if checkpoint_files:  
            print("🔄 Checkpoint already loaded, skipping save step.")
        else:
            if agent._step % 50000 == 0:  #save Checkpoint every 50,000 steps
                checkpoint_path = logdir / f"checkpoint_{agent._step}.pth"
                torch.save(items_to_save, checkpoint_path)
                print(f"✅ Saved checkpoint at {checkpoint_path}")


        if config.eval_episode_num > 0:
            print("Start evaluation.")
            eval_policy = functools.partial(agent, training=False)
            tools.simulate(
                eval_policy,
                eval_envs,
                eval_eps,
                config.evaldir,
                logger,
                is_eval=True,
                episodes=config.eval_episode_num,
            )
            if config.video_pred_log:
                video_pred = agent._wm.video_pred(next(eval_dataset))
                logger.video("eval_openl", to_np(video_pred))
        print("Start training.")
        state = tools.simulate(
            agent,
            train_envs,
            train_eps,
            config.traindir,
            logger,
            limit=config.dataset_size,
            steps=config.eval_every,
            state=state,
            manager=manager,
        )
        checkpoint_path = logdir / f"checkpoint_{agent._step}.pth"

        items_to_save = {
            "agent_state_dict": agent.state_dict(),  # 模型参数
            "optims_state_dict": tools.recursively_collect_optim_state_dict(agent),  # 优化器
            "step": agent._step,  # 训练步数
            "hyperparameters": {  # 记录超参数
                "learning_rate": config.model_lr,
                "epsilon": getattr(agent, "epsilon", None),  # DQN 需要保存 epsilon
            },
            # ✅ 保存随机状态
            "numpy_random_state": np.random.get_state(),
            "torch_random_state": torch.get_rng_state(),
            "torch_cuda_random_state": torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None,            
        }

        # 如果使用经验回放缓冲区（DQN/SAC）
        if hasattr(agent, "replay_buffer"):
            items_to_save["replay_buffer"] = agent.replay_buffer.save()  # 需要你的 replay buffer 有 save 方法

        # 如果环境可以保存状态（如 MuJoCo）
        if hasattr(train_envs[0], "save_state"):
            items_to_save["env_state"] = train_envs[0].save_state()

        # 存储 checkpoint
        torch.save(items_to_save, checkpoint_path)  # 存储带 step 的 Checkpoint
        torch.save(items_to_save, logdir / "latest.pt")  # 仍然保存 latest.pt 作为最新模型

        print(f"✅ Checkpoint saved at {checkpoint_path}")

    if manager.tau_star is None:
        manager.tau_star = manager.tau
            
    for env in train_envs + eval_envs:
        try:
            env.close()
        except Exception:
            pass


def main(config):
    import copy
    manager = AgentManager()
    for idx in range(4):
        tmp_config = copy.deepcopy(config)
        gen_across_tasks(tmp_config, idx, manager)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--configs", nargs="+")
    args, remaining = parser.parse_known_args()
    configs = yaml.safe_load(
        (pathlib.Path(sys.argv[0]).parent / "configs.yaml").read_text()
    )

    def recursive_update(base, update):
        for key, value in update.items():
            if isinstance(value, dict) and key in base:
                recursive_update(base[key], value)
            else:
                base[key] = value

    name_list = ["defaults", *args.configs] if args.configs else ["defaults"]
    defaults = {}
    for name in name_list:
        recursive_update(defaults, configs[name])
    parser = argparse.ArgumentParser()
    for key, value in sorted(defaults.items(), key=lambda x: x[0]):
        arg_type = tools.args_type(value)
        parser.add_argument(f"--{key}", type=arg_type, default=arg_type(value))
    main(parser.parse_args(remaining))

# python3 main.py --configs atari100k --task atari_pong --logdir ./results/pong
