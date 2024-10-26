import warnings
warnings.filterwarnings("ignore")
import wandb
import argparse
import os,sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(__file__))))
import torch
import numpy as np
import gym, random
import matplotlib.pyplot as plt
plt.rcParams['axes.linewidth'] = 1
plt.rcParams['figure.dpi'] = 600
from mymodel.utils.wrapper import OneHotAction
from myenv.cartpole.cartpole_fric import CartPoleFricEnv
from mymodel.training.config import CartPoleFricConfig
from mymodel.training.dtrainer_cartpole import CPOTrainer
from mymodel.training.evaluator import CPOEvaluator
from agent.ac import ACAgent
os.environ['SDL_VIDEODRIVER'] = 'dummy'

import logging
logging.getLogger('matplotlib.font_manager').setLevel(logging.WARNING)


def plot_scores(scores, path):
    np.save(os.path.join(path, 'scores.npy'), np.array(scores))
    plt.clf()
    plt.figure(figsize=(8,6))
    plt.plot(scores, color="blue")
    plt.xlabel("Episode", fontsize=15)
    plt.ylabel("Accumulated Reward", fontsize=15)
    plt.savefig(os.path.join(path, 'score_v_episodes.png'))
    plt.close()


def main(args):
    wandb.login()
    env_name = 'cartpolefric'
    model_name = 'Ours'
    exp_id = args.id

    '''make dir for saving results'''
    result_dir = os.path.join('results', '{}'.format(model_name), '{}'.format(exp_id))
    model_dir = os.path.join(result_dir, 'models')
    gif_dir = os.path.join(result_dir, 'visualization')
    #dir to save learnt models
    os.makedirs(model_dir, exist_ok=True)

    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    random.seed(args.seed)

    if torch.cuda.is_available() and args.device:
        device = torch.device('cuda')
        torch.cuda.manual_seed(args.seed)
        torch.cuda.manual_seed_all(args.seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
    else:
        device = torch.device('cpu')
    print('using :', device)

    """
    init env,
    easily modify gravity and masscart by setting
        env = OneHotAction(CartPoleFricEnv(gravity=g', masscart=m',  state_obs_noise=args.noise))
        test_env = OneHotAction(CartPoleFricEnv(gravity=g', masscart=m', state_obs_noise=args.noise, reward_type='sparse'))
    set diff_mode=True with state expansion
    set action_size=k to simulate action expansion, where we choose k=12 in our paper
    """
    env = OneHotAction(CartPoleFricEnv(state_obs_noise=args.noise))
    test_env = OneHotAction(CartPoleFricEnv(state_obs_noise=args.noise, reward_type='sparse'))
    obs_shape = env.observation_space.shape
    action_size = env.action_space.shape[0]
    obs_dtype = np.uint8
    action_dtype = np.float32
    batch_size = args.batch_size
    seq_len = args.seq_len
    rssm_type = args.rssm_type
    disentangle = args.disentangle
    horizon = args.horizon

    config = CartPoleFricConfig(
        env=env_name,
        obs_shape=obs_shape,
        action_size=action_size,
        obs_dtype = obs_dtype,
        action_dtype = action_dtype,
        seq_len = seq_len,
        batch_size = batch_size,
        model_dir=model_dir,
        gif_dir=gif_dir,
        rssm_type=rssm_type,
        horizon=horizon
    )

    config_dict = config.__dict__
    
    trainer = CPOTrainer(config, device, action_expansion=True)
    resume_step = 0
    if args.resume:
        resume_step = trainer.resume_training(model_dir, 200000)   
    agent = ACAgent(config, device, action_expansion=True)

    evaluator = CPOEvaluator(config, device)
    # evaluator = CPOEvaluator(config, device, 10, 4) use this when state expansion

    # prev_save_dict = torch.load('./checkpoints/models.pth') load model if needed
    # trainer.load_save_dict(prev_save_dict)    
    # prev_agent_dict = torch.load('./checkpoints/agent.pth' )
    # agent.load_save_dict(prev_agent_dict)

    # trainer._action_expansion_initialize(config) add this when action expands
    # agent._action_expansion_initialize(config)

    # agent._model_expand_initialize(config)
    
    with wandb.init(project='CartpoleExp', entity="", config=config_dict):
        """training loop"""
        
        print('...training...')
        train_metrics = {}
        trainer.collect_seed_episodes(env)
        obs, score = env.reset(), 0
        done = False
        prev_rssmstate = trainer.RSSM._init_rssm_state(1)
        prev_action = torch.zeros(1, trainer.action_size).to(trainer.device)
        print(f"Enter training iteration")
        for iter in range(1 + resume_step, trainer.config.train_steps+1+resume_step):
            if iter%trainer.config.train_every == 1:
                train_metrics = trainer.train_batch(train_metrics) # set expand_deter=False if use _model_expand_initialize
                # train_metrics = trainer.adapt_batch(train_metrics) use this when there are only distribution shifts
            if iter%trainer.config.save_every == 1:
                model_dir = trainer.save_model(iter)
            if iter%trainer.config.eval_every == 1:
                evaluator.eval_agent(test_env, trainer.RSSM, trainer.ObsEncoder, trainer.ObsDecoder, iter)
            with torch.no_grad():
                obs_tensor = torch.tensor(obs, dtype=torch.float32)
                if obs.dtype == np.uint8:
                    obs_tensor = obs_tensor.div(255).sub_(0.5)
                embed = trainer.ObsEncoder(obs_tensor.unsqueeze(0).to(trainer.device))
                _, posterior_rssm_state = trainer.RSSM.rssm_observe(embed, prev_action, not done, prev_rssmstate)
                action, action_dist = trainer.ActionModel(trainer.RSSM.get_actor_state(posterior_rssm_state, trainer.RSSM.theta_deter, trainer.RSSM.theta_stoch))
                action = trainer.ActionModel.add_exploration(action, iter).detach()

            next_obs, rew, done, _ = env.step(action.squeeze(0).cpu().numpy())
            score += rew

            if done:
                trainer.buffer.add(obs, action.squeeze(0).cpu().numpy(), rew, done)
                # wandb.log(train_metrics, step=iter)
                obs, score = env.reset(), 0
                done = False
                prev_rssmstate = trainer.RSSM._init_rssm_state(1)
                prev_action = torch.zeros(1, trainer.action_size).to(trainer.device)
            else:
                trainer.buffer.add(obs, action.squeeze(0).detach().cpu().numpy(), rew, done)
                obs = next_obs
                prev_rssmstate = posterior_rssm_state
                prev_action = action

        trainer_dir = trainer.save_model(trainer.config.train_steps)    
        model_path = os.path.join(trainer_dir, 'models_%d.pth' % (trainer.config.train_steps))    
        save_trainer_dict = torch.load(model_path)
        agent.load_trainer_dict(save_trainer_dict)

        print('...policy learning...')
        train_metrics = {}
        agent.collect_seed_episodes(env)
        obs, score = env.reset(), 0
        done = False
        prev_rssmstate = agent.RSSM._init_rssm_state(1)
        prev_action = torch.zeros(1, agent.action_size).to(agent.device)
        episode_actor_ent = []
        scores = []
        print(f"Enter policy learning iteration")
        for iter in range(1, agent.config.agent_steps+1):
            train_metrics = agent.policy_learning(train_metrics)
            agent.update_target()
            if iter%agent.config.save_every == 1:
                model_dir = agent.save_model(iter)

            with torch.no_grad():
                obs_tensor = torch.tensor(obs, dtype=torch.float32)
                if obs.dtype == np.uint8:
                    obs_tensor = obs_tensor.div(255).sub_(0.5)
                embed = agent.ObsEncoder(obs_tensor.unsqueeze(0).to(agent.device))
                _, posterior_rssm_state = agent.RSSM.rssm_observe(embed, prev_action, not done, prev_rssmstate)
                action, action_dist = agent.ActionModel(agent.RSSM.get_actor_state(posterior_rssm_state, agent.RSSM.theta_deter, agent.RSSM.theta_stoch))
                action = agent.ActionModel.add_exploration(action, iter).detach()
                action_ent = torch.mean(action_dist.entropy()).item()
                episode_actor_ent.append(action_ent)

            next_obs, rew, done, _ = env.step(action.squeeze(0).cpu().numpy())
            score += rew

            if done:
                agent.buffer.add(obs, action.squeeze(0).cpu().numpy(), rew, done)
                train_metrics['train_rewards'] = score
                train_metrics['action_ent'] =  np.mean(episode_actor_ent)
                wandb.log(train_metrics, step=iter)
                scores.append(score)
                obs, score = env.reset(), 0
                done = False
                prev_rssmstate = agent.RSSM._init_rssm_state(1)
                prev_action = torch.zeros(1, agent.action_size).to(agent.device)
                episode_actor_ent = []
                plot_scores(scores, result_dir)
            else:
                agent.buffer.add(obs, action.squeeze(0).detach().cpu().numpy(), rew, done)
                obs = next_obs
                prev_rssmstate = posterior_rssm_state
                prev_action = action

        agent_path = agent.save_model(agent.config.agent_steps)
    
    print('done.')
    # evaluating probably best model
    evaluator.eval_saved_agent(test_env, model_path, agent_path)
    

if __name__ == "__main__":
    """there are tonnes of HPs, if you want to do an ablation over any particular one, please add if here"""
    parser = argparse.ArgumentParser()
    parser.add_argument("--env", type=str, default='cartpolefric', help='mini atari env name')
    parser.add_argument("--id", type=str, default='0', help='Experiment ID')
    parser.add_argument('--seed', type=int, default=0, help='Random seed')
    parser.add_argument('--horizon', type=int, default=12, help='Random seed')
    parser.add_argument('--device', default='cuda', help='CUDA or CPU')
    parser.add_argument('--batch_size', type=int, default=20, help='Batch size')
    parser.add_argument('--seq_len', type=int, default=30, help='Sequence Length (chunk length)')
    parser.add_argument('--rssm_type', type=str, default="continuous", help='rssm_type: continuous or discrete')
    parser.add_argument('--noise', action='store_true', help='noise in the dynamics')
    parser.add_argument('--resume', action='store_true', help='noise in the dynamics')
    parser.add_argument('--distractor', action='store_true', help='distractor in the input observation')
    parser.add_argument('--disentangle', action='store_true')
    parser.add_argument('--no-noise', dest='noise', action='store_false')
    parser.add_argument('--no-distractor', dest='distractor', action='store_false')
    parser.add_argument('--no-disentangle', dest='disentangle', action='store_false')
    parser.set_defaults(noise=False)
    parser.set_defaults(distractor=False)
    parser.set_defaults(disentangle=True)
    parser.set_defaults(resume=False)
    args = parser.parse_args()
    main(args)
