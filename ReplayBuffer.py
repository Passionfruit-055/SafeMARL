import random
from collections import deque

import numpy as np
import torch


class RecurrentReplayBuffer(object):
    def __init__(self, agent_num, seq_length=10, buffer_size=64, dev='cuda:0'):
        self.agent_num = agent_num
        self.seq_length = seq_length
        self.buffer_size = buffer_size
        self.dev = dev
        self.buffer = {}
        self.last_episode = 0
        for i in range(agent_num):
            self.buffer['agent' + str(i)] = {}
            for e in range(buffer_size):
                self.buffer['agent' + str(i)]['episode' + str(e)] = []

    def add(self, s, u, s_next, episode, agent):
        hashe = episode % self.buffer_size
        if self.last_episode != episode:
            self.last_episode = episode
            if len(self.buffer['agent' + str(agent)]['episode' + str(hashe)]) != 0:
                self.buffer['agent' + str(agent)]['episode' + str(hashe)] = []
        experience = (s, u, s_next)
        self.buffer['agent' + str(agent)]['episode' + str(hashe)].append(experience)

    def sample_batch(self, agent, sequence_length, episodes, start_pos=0):
        # 这里会送入一个episode的列表，对其中的每一个都抽取sequence_length长度的序列
        end_pos = start_pos + sequence_length
        obs = []
        actions = []
        next_obs = []
        episodes = [e % self.buffer_size for e in episodes]
        for episode in episodes:
            memory = self.buffer['agent' + str(agent)]['episode' + str(episode)]
            minibatch = memory[start_pos:end_pos]

            ob = [d[0] for d in minibatch]
            action = [d[1] for d in minibatch]
            next_ob = [d[2] for d in minibatch]

            # 对某些未完全结束的episode，需要补充序列
            if len(memory) <= end_pos:
                for i in range(len(memory), end_pos):
                    ob.append(memory[-1][0])
                    action.append(memory[-1][1])
                    next_ob.append(memory[-1][2])

            obs.append(ob)
            actions.append(action)
            next_obs.append(next_ob)

        state = torch.tensor(obs).to(self.dev)
        action = torch.tensor(actions).to(self.dev)
        next_state = torch.tensor(next_obs).to(self.dev)

        return state, action, next_state, start_pos

    def clear(self, agent_num, episode):
        self.buffer['agent' + str(agent_num)]['episode' + str(episode)] = []


class ReplayBuffer(object):
    def __init__(self, buffer_size, dev='cuda:0'):
        self.count = 0
        self.dev = dev
        self.buffer_size = buffer_size
        self.buffer = {}
        self.last_episode = 0
        for e in range(buffer_size):
            self.buffer['episode' + str(e)] = []

    def add(self, episode, s, r, s_next):
        hashe = episode % self.buffer_size
        if self.last_episode != episode:
            self.last_episode = episode
            if len(self.buffer['episode' + str(hashe)]) != 0:
                self.buffer['episode' + str(hashe)] = []
        experience = (s, r, s_next)
        self.buffer['episode' + str(hashe)].append(experience)

    def sample_batch(self, episodes, sequence_length, start_pos):
        states = []
        rewards = []
        next_states = []
        batch_size = len(episodes)
        episodes = [e % self.buffer_size for e in episodes]
        for episode in episodes:
            # last_pos = len(self.buffer['episode' + str(episode)]) - sequence_length
            # last_pos = max(last_pos, 0)
            # start_pos = random.randint(0, last_pos)
            end_pos = start_pos + sequence_length
            memory = self.buffer['episode' + str(episode)]
            minibatch = memory[start_pos:end_pos]

            state = [d[0] for d in minibatch]
            reward = [d[1] for d in minibatch]
            next_state = [d[2] for d in minibatch]

            if len(memory) <= end_pos:
                for i in range(len(memory), end_pos):
                    state.append(memory[-1][0])
                    reward.append(memory[-1][1])
                    next_state.append(memory[-1][2])

            states.append(state)
            rewards.append(reward)
            next_states.append(next_state)

        state = torch.tensor(list(states), dtype=torch.float).view(sequence_length, batch_size, -1).to(self.dev)
        reward = torch.tensor(list(rewards), dtype=torch.float).view(sequence_length, batch_size, -1).to(self.dev)
        next_state = torch.tensor(list(next_states), dtype=torch.float).view(sequence_length, batch_size, -1).to(self.dev)

        return state, reward, next_state

    def clear(self):
        self.buffer.clear()
        self.count = 0


if __name__ == '__main__':
    arr = [1, 2, 3, 4, 5, 6, 7, 8]
    sub = arr[5:len(arr)+2]
    print(sub)
