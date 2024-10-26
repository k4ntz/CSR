import numpy as np
from typing import *
import torch 
import os
from mymodel.models.actor import DiscreteActionModel
from mymodel.models.drssm_cartpole import DRSSM
from mymodel.models.dense import DenseModel
from mymodel.models.pixel_cartpole import ObsDecoder, ObsEncoder
from mymodel.utils.visualize import Visualizer
from mymodel.utils.identifiability.metrics import compute_r2, test_independence

class CPOEvaluator(object):
    '''
    used this only for minigrid envs
    '''
    def __init__(
        self,
        config,
        device,
        expand_deter_size=0,
        expand_stoch_size=0,
    ):
        self.device = device
        self.config = config
        self.action_size = config.action_size
        self.visualizer = Visualizer(config)

        self.expand_deter_size = expand_deter_size
        self.expand_stoch_size = expand_stoch_size

        action_size = config.action_size
        deter_size = config.rssm_info['deter_size'] + expand_deter_size
        if config.rssm_type == 'continuous':
            stoch_size = config.rssm_info['stoch_size'] + expand_stoch_size
        elif config.rssm_type == 'discrete':
            category_size = config.rssm_info['category_size']
            class_size = config.rssm_info['class_size']
            stoch_size = category_size * class_size

        embedding_size = config.embedding_size
        theta_size = config.theta_deter_size + config.theta_stoch_size
        self.ActionModel = DiscreteActionModel(action_size, deter_size, stoch_size,
                                               embedding_size, config.actor,
                                               config.expl, theta_size).to(self.device).eval()

    def load_model(self, config, model_path, agent_path):
        saved_model_dict = torch.load(model_path)
        saved_agent_dict = torch.load(agent_path)
        obs_shape = config.obs_shape
        action_size = config.action_size
        deter_size = config.rssm_info['deter_size'] + self.expand_deter_size
        if config.rssm_type == 'continuous':
            stoch_size = config.rssm_info['stoch_size'] + self.expand_stoch_size
        elif config.rssm_type == 'discrete':
            category_size = config.rssm_info['category_size']
            class_size = config.rssm_info['class_size']
            stoch_size = category_size * class_size

        embedding_size = config.embedding_size
        rssm_node_size = config.rssm_node_size
        modelstate_size = stoch_size + deter_size
        theta_deter_size = config.theta_deter_size
        theta_stoch_size = config.theta_stoch_size
        theta_size = theta_deter_size + theta_stoch_size

        if config.pixel:
                self.ObsEncoder = ObsEncoder(obs_shape, embedding_size, config.obs_encoder).to(self.device).eval()
                self.ObsDecoder = ObsDecoder(obs_shape, modelstate_size, config.obs_decoder).to(self.device).eval()
        else:
            self.ObsEncoder = DenseModel((embedding_size,), int(np.prod(obs_shape)), config.obs_encoder).to(self.device).eval()
            self.ObsDecoder = DenseModel(obs_shape, modelstate_size, config.obs_decoder).to(self.device).eval()
               
        self.RSSM = DRSSM(action_size, rssm_node_size, embedding_size, theta_deter_size, theta_stoch_size,
                          self.device, config.rssm_type, config.rssm_info,
                          deter_size, stoch_size).to(self.device).eval()
        self.ActionModel = DiscreteActionModel(action_size, deter_size, stoch_size,
                                               embedding_size, config.actor,
                                               config.expl, theta_size).to(self.device).eval()

        self.RSSM.load_state_dict(saved_model_dict["RSSM"])
        self.ObsEncoder.load_state_dict(saved_model_dict["ObsEncoder"])
        self.ObsDecoder.load_state_dict(saved_model_dict["ObsDecoder"])
        self.ActionModel.load_state_dict(saved_agent_dict["ActionModel"])        

    def eval_saved_agent(self, env, model_path, agent_path):
        self.load_model(self.config, model_path, agent_path)
        train_step = model_path.split('/')[-1].split('.')[-2]
        print('train_step', train_step)
        return self.eval_agent(env, self.RSSM, self.ObsEncoder, self.ObsDecoder, train_step)

    def eval_visualize(self, env, model_path, interval=40, frame_save=6, visualize_episode=5, random=False):
        self.load_model(self.config, model_path)
        print("eval agent")
        for e in range(visualize_episode):
            obs, score = env.reset(), 0
            done = False
            with torch.no_grad():
                prev_rssmstate = self.RSSM._init_rssm_state(1)
                prev_action = torch.zeros(1, self.action_size).to(self.device)
            video_frames_dict = {"obs":[], "rssm_state_1234":[], "rssm_state_1": [], "rssm_state_2": [], "rssm_state_3": [], "rssm_state_4": [], "rssm_state_12": [], "rssm_state_34": []}
            first_state_flag = True
            iter_cnt = 0
            frame_cnt = 0
            while not done:
                with torch.no_grad():
                    obs_tensor = torch.tensor(obs, dtype=torch.float32)
                    if obs.dtype == np.uint8:
                        obs_tensor = obs_tensor.div(255).sub_(0.5)
                    embed = self.ObsEncoder(obs_tensor.unsqueeze(0).to(self.device))
                    _, posterior_rssm_state = self.RSSM.rssm_observe(embed, prev_action, not done, prev_rssmstate)
                    if first_state_flag:
                        first_state = posterior_rssm_state
                        first_state_flag = False
                    if e < visualize_episode and iter_cnt % interval == 0:
                        self.visualizer.collect_frames_cartpole(obs_tensor, posterior_rssm_state, first_state, self.RSSM, self.ObsDecoder, video_frames_dict)
                        frame_cnt += 1
                    action = self.ActionModel.optimal_action(self.RSSM.get_actor_state(posterior_rssm_state, self.RSSM.theta_deter, self.RSSM.theta_stoch))
                    prev_rssmstate = posterior_rssm_state
                    prev_action = action
                    action = action.squeeze(0).cpu().numpy()
                    if random:
                        action = env.action_space.sample()
                next_obs, rew, done, _ = env.step(action)
                if self.config.eval_render:
                    env.render()
                score += rew
                obs = next_obs
                iter_cnt += 1
                if frame_cnt >= frame_save:
                    break
            # print(f'episode {e}: {score}')
            first_state_flag = True
            save_dir = os.path.join(os.path.split(model_path)[0], 'eval', 'visualize')
            self.visualizer.output_picture(save_dir, e, video_frames_dict)
        
    def eval_agent(self, env, RSSM, ObsEncoder, ObsDecoder, train_step):
        eval_scores = []
        eval_episode = self.config.eval_episode
        visualize_episode = self.config.visualize_episode
        print("eval agent")
        for e in range(eval_episode):
            obs, score = env.reset(), 0
            done = False
            with torch.no_grad():
                prev_rssmstate = RSSM._init_rssm_state(1)
                prev_action = torch.zeros(1, self.action_size).to(self.device)
            video_frames_dict = {"rssm_state":[]}
            first_state_flag = True
            while not done:
                with torch.no_grad():
                    obs_tensor = torch.tensor(obs, dtype=torch.float32)
                    if obs.dtype == np.uint8:
                        obs_tensor = obs_tensor.div(255).sub_(0.5)
                    embed = ObsEncoder(obs_tensor.unsqueeze(0).to(self.device))
                    _, posterior_rssm_state = RSSM.rssm_observe(embed, prev_action, not done, prev_rssmstate)
                    if first_state_flag:
                        first_state = posterior_rssm_state
                        first_state_flag = False
                    if e < visualize_episode:
                        self.visualizer.collect_frames(obs_tensor, posterior_rssm_state, first_state, RSSM, ObsDecoder, video_frames_dict)
                    action = self.ActionModel.optimal_action(RSSM.get_actor_state(posterior_rssm_state, RSSM.theta_deter, RSSM.theta_stoch))                    
                    prev_rssmstate = posterior_rssm_state
                    prev_action = action
                next_obs, rew, done, _ = env.step(action.squeeze(0).cpu().numpy())
                if self.config.eval_render:
                    env.render()
                score += rew
                obs = next_obs
            first_state_flag = True
            eval_scores.append(score)
            if e < visualize_episode:
                self.visualizer.output_video(train_step, e, video_frames_dict)
        print('evaluation scores = ', eval_scores)
        print('average evaluation score = ' + str(np.mean(eval_scores)))
        print('std evaluation score = ' + str(np.std(eval_scores)))
        return np.mean(eval_scores)
    
    def eval_score(self, env, RSSM, ObsEncoder, ActionModel, eval_num=5):
        eval_scores = []
        print("eval agent")
        for e in range(eval_num):
            obs, score = env.reset(), 0
            done = False
            with torch.no_grad():
                prev_rssmstate = RSSM._init_rssm_state(1)
                prev_action = torch.zeros(1, self.action_size).to(self.device)
            while not done:
                with torch.no_grad():
                    obs_tensor = torch.tensor(obs, dtype=torch.float32)
                    if obs.dtype == np.uint8:
                        obs_tensor = obs_tensor.div(255).sub_(0.5)
                    embed = ObsEncoder(obs_tensor.unsqueeze(0).to(self.device))
                    _, posterior_rssm_state = RSSM.rssm_observe(embed, prev_action, not done, prev_rssmstate)
                    action = ActionModel.optimal_action(RSSM.get_actor_state(posterior_rssm_state, RSSM.theta_deter, RSSM.theta_stoch))
                    prev_rssmstate = posterior_rssm_state
                    prev_action = action
                next_obs, rew, done, _ = env.step(action.squeeze(0).cpu().numpy())
                if self.config.eval_render:
                    env.render()
                score += rew
                obs = next_obs
            first_state_flag = True
            eval_scores.append(score)
            print('Evaluation score = ' + str((score)))
        return np.mean(eval_scores), np.std(eval_scores)
