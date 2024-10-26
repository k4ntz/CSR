# xvfb-run --auto-servernum -s "-screen 0 1400x900x24" python -m world_model.collect_coinrun --configs coinrun --task coinrun --logdir ./world_model/coinrun_env


import matplotlib
matplotlib.use('Agg')  # 切换到 Agg 后端，不依赖图形界面
import matplotlib.pyplot as plt

from matplotlib import rc

# Adjust other figure settings
plt.rcParams['axes.linewidth'] = 1
plt.rcParams['figure.dpi'] = 600

def plot_images(images, logdir, step):
    # fig, axes = plt.subplots(3, 3, figsize=(9, 9))
    plt.clf()
    plt.imshow(images, interpolation='none')
    plt.axis('off')  # 移除坐标轴

    plt.savefig(logdir + '/coinrun_{}.pdf'.format(step), bbox_inches='tight', pad_inches=0)
    plt.savefig(logdir + '/coinrun_{}.png'.format(step), bbox_inches='tight', pad_inches=0)

import numpy as np
from coinrun import setup_utils, make
from q_learning.utils import Scalarize

def random_agent(num_envs=1, max_steps=100):
    setup_utils.setup_and_load(is_high_res=True, use_cmd_line_args=False)
    env = make('standard', num_envs=num_envs)
    # env = Scalarize(make('standard', num_envs=1))
    for step in range(max_steps):
        env.render()
        acts = np.array([env.action_space.sample() for _ in range(env.num_envs)])
        # action = np.random.randint(0, env.action_space.n)
        # _obs, reward, done, info = env.step(action)
        _obs, rews, _dones, _infos = env.step(acts)
        print("step", step, "rews", rews)
        # image = env.get_images()[0]
        # plot_images(image, "./world_model/coinrun_env", step)
    
    image = env.get_images()[0]
    env.close()

    return image

if __name__ == '__main__':
    image = random_agent()
    logdir = "./world_model/coinrun_env"
    plot_images(image, logdir, 1)


# from coinrun import make
# env = utils.Scalarize(make('standard', num_envs=1))
