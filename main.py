from tqdm import tqdm
from agent import Agent
from tensorboardX import SummaryWriter
import gym
import matplotlib.pyplot as plt
import numpy as np
import random
import torch

class ClipRewardEnv(gym.RewardWrapper):
    def __init__(self, env, min_reward, max_reward):
        super().__init__(env)
        self.min_reward = min_reward
        self.max_reward = max_reward
        self.reward_range = (min_reward, max_reward)
    
    def reward(self, reward):
        return np.clip(reward, self.min_reward, self.max_reward)


env = gym.make("PongNoFrameskip-v4")
env = gym.wrappers.AtariPreprocessing(env, noop_max=30, frame_skip=4, screen_size=84, terminal_on_life_loss=False, grayscale_obs=True, grayscale_newaxis=False, scale_obs=True)
#env = ClipRewardEnv(env, -1, 1)
env = gym.wrappers.FrameStack(env, 4)
#env = gym.wrappers.Monitor(env, './video/', video_callable=lambda episode_id: episode_id % 50 == 0, force=True)
writer = SummaryWriter()

mem_size = 100000
agent = Agent(mem_size, [0, 1, 2, 3, 4, 5], 1, 0.1, .01, 0.99, 1e-4)

index = 0
print("Prefilling memory buffer")
for i in tqdm(range(10000)):
    obs = env.reset()[0]
    done = False
    while not done:
        a = agent.get_action(obs)
        next_frame, reward, done, _, _ = env.step(a)
        agent.add_experience(obs, a, reward, next_frame)
        obs = next_frame
        index += 1

        if index > mem_size:
            break

losses_list, reward_list, episode_len_list, epsilon_list  = [], [], [], []
index = 0
episodes = 2000

for i in tqdm(range(episodes)):
    obs = env.reset()[0]
    done = False
    losses = 0
    ep_len = 0
    total_reward = 0
    while not done:
        ep_len += 1
        a = agent.get_action(obs)
        next_frame, reward, done, _, _ = env.step(a)
        agent.add_experience(obs, a, reward, next_frame)
        obs = next_frame
        total_reward += reward
        index += 1

        if index > 10000:
            #index = 0
            loss = agent.train(batch_size=32)
            losses += loss

    # if agent.eps > agent.eps_min:
    #     agent.eps -= agent.eps_decay

    agent.eps = agent.eps_min + (agent.eps_max - agent.eps_min) * np.exp(-1. * agent.timesteps / 1000000)

    agent.timesteps += 1
    writer.add_scalar('Loss', losses, global_step=i)
    writer.add_scalar('Reward', total_reward, global_step=i)
    writer.add_scalar('Epsilon', agent.eps, global_step=i)
    writer.add_scalar('Episode Length', ep_len, global_step=i)

    losses_list.append(losses/ep_len)
    reward_list.append(total_reward)
    episode_len_list.append(ep_len)
    epsilon_list.append(agent.eps)

torch.save(agent.model.state_dict(), "model.pt")
torch.save(agent.model_target.state_dict(), "target_model.pt")