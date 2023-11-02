import random
from collections import deque

import numpy as np
import torch
from torch.nn.utils.rnn import pad_sequence, pack_padded_sequence


class RecurrentReplayBuffer(object):
    def __init__(self, agent_num, seq_length=10, buffer_size=64, dev='cuda:0'):
        self.agent_num = agent_num
        self.seq_length = seq_length
        self.buffer_size = buffer_size
        self.dev = dev
        self.buffer = {}
        self.last_episode = [-1] * self.agent_num
        for i in range(agent_num):
            self.buffer['agent' + str(i)] = {}
            for e in range(buffer_size):
                self.buffer['agent' + str(i)]['episode' + str(e)] = []

    def add(self, s, u, s_next, episode, agent, r=None):
        experience = [s, u, s_next] if r is None else [s, u, s_next, r]
        if len(self.buffer['agent' + str(agent)]['episode' + str(episode % self.buffer_size)]) >= 0 and episode != \
                self.last_episode[agent]:
            del self.buffer['agent' + str(agent)]['episode' + str(episode % self.buffer_size)]
            self.buffer['agent' + str(agent)]['episode' + str(episode % self.buffer_size)] = []
            self.last_episode[agent] = episode
        self.buffer['agent' + str(agent)]['episode' + str(episode % self.buffer_size)].append(experience)

    def sample_batch(self, agent, seqLen, batch_size, episodes):
        # 这里会送入一个episode的列表，对其中的每一个都抽取sequence_length长度的序列
        obs = []
        actions = []
        next_obs = []
        episodes = [e % self.buffer_size for e in episodes]
        batch_length = []
        for episode in episodes:
            minibatch = self.buffer['agent' + str(agent)]['episode' + str(episode)]
            batch_length.append(len(minibatch))

            ob = [d[0] for d in minibatch]
            action = [d[1] for d in minibatch]
            next_ob = [d[2] for d in minibatch]

            ob = torch.tensor(ob, dtype=torch.float32).to(self.dev)
            action = torch.tensor(action, dtype=torch.long).to(self.dev)
            next_ob = torch.tensor(next_ob, dtype=torch.float32).to(self.dev)

            obs.append(ob)
            actions.append(action)
            next_obs.append(next_ob)

        obs = sorted(obs, key=lambda x: len(x), reverse=True)
        actions = sorted(actions, key=lambda x: len(x), reverse=True)
        next_obs = sorted(obs, key=lambda x: len(x), reverse=True)

        state = pad_sequence(obs).to(self.dev)
        action = pad_sequence(actions).to(self.dev)
        next_state = pad_sequence(next_obs).to(self.dev)

        # state = pack_padded_sequence(state, batch_length, batch_first=False).to(self.dev)
        # action = pack_padded_sequence(action, batch_length, batch_first=False).to(self.dev)
        # next_state = pack_padded_sequence(next_state, batch_length, batch_first=False).to(self.dev)

        return state, action, next_state

    def sample_fullbatch(self, agent, episodes):
        obs = []
        actions = []
        rewards = []
        next_obs = []
        episodes = [e % self.buffer_size for e in episodes]
        seqLens = []
        for episode in episodes:
            minibatch = self.buffer['agent' + str(agent)]['episode' + str(episode)]
            seqLens.append(len(minibatch))

            ob = [d[0] for d in minibatch]
            action = [d[1] for d in minibatch]
            next_ob = [d[2] for d in minibatch]
            r = [d[3] for d in minibatch]

            ob = torch.tensor(ob, dtype=torch.float32).to(self.dev)
            action = torch.tensor(action, dtype=torch.long).to(self.dev)
            next_ob = torch.tensor(next_ob, dtype=torch.float32).to(self.dev)
            r = torch.tensor(r, dtype=torch.float32).to(self.dev)

            obs.append(ob)
            actions.append(action)
            next_obs.append(next_ob)
            rewards.append(r)

        obs = sorted(obs, key=lambda x: len(x), reverse=True)
        actions = sorted(actions, key=lambda x: len(x), reverse=True)
        next_obs = sorted(obs, key=lambda x: len(x), reverse=True)
        rewards = sorted(rewards, key=lambda x: len(x), reverse=True)

        state = pad_sequence(obs).to(self.dev)
        action = pad_sequence(actions).to(self.dev)
        next_state = pad_sequence(next_obs).to(self.dev)
        reward = pad_sequence(rewards).to(self.dev)

        return state, action, reward, next_state, seqLens

    def clear(self, agent_num, episode):
        self.buffer['agent' + str(agent_num)]['episode' + str(episode % self.buffer_size)] = []


class ReplayBuffer(object):
    def __init__(self, buffer_size, dev='cuda:0'):
        self.count = 0
        self.dev = dev
        self.buffer_size = buffer_size
        self.buffer = {}
        self.last_episode = -1
        for e in range(buffer_size):
            self.buffer['episode' + str(e)] = []

    def add(self, episode, s, r, s_next):
        experience = [s, r, s_next]
        if len(self.buffer['episode' + str(episode % self.buffer_size)]) >= 0 and episode != self.last_episode:
            del self.buffer['episode' + str(episode % self.buffer_size)]
            self.buffer['episode' + str(episode % self.buffer_size)] = []
            self.last_episode = episode
        self.buffer['episode' + str(episode % self.buffer_size)].append(experience)

    def sample_batch(self, episodes):
        states = []
        rewards = []
        next_states = []
        minibatch = []
        mapped_episodes = [e % self.buffer_size for e in episodes]
        seqLen = 0
        for episode in mapped_episodes:
            memory = self.buffer['episode' + str(episode)]
            seqLen = max(seqLen, len(memory))
            minibatch.append(memory)

        # padding
        batch_length = []
        for batch in minibatch:
            state = [d[0] for d in batch]
            reward = [d[1] for d in batch]
            next_state = [d[2] for d in batch]
            batch_length.append(len(batch))

            state = torch.tensor(state, dtype=torch.float)
            reward = torch.tensor(reward, dtype=torch.float)
            next_state = torch.tensor(next_state, dtype=torch.float)

            states.append(state)
            rewards.append(reward)
            next_states.append(next_state)

        states = sorted(states, key=lambda x: len(x), reverse=True)
        rewards = sorted(rewards, key=lambda x: len(x), reverse=True)
        next_states = sorted(next_states, key=lambda x: len(x), reverse=True)
        batch_length = sorted(batch_length, key=lambda x: x, reverse=True)

        state = pad_sequence(states).to(self.dev)
        reward = pad_sequence(rewards).unsqueeze(2).to(self.dev)
        next_state = pad_sequence(next_states).to(self.dev)

        # state = pack_padded_sequence(state, batch_length, batch_first=False).to(self.dev)
        # reward = pack_padded_sequence(reward, batch_length, batch_first=False).to(self.dev)
        # next_state = pack_padded_sequence(next_state, batch_length, batch_first=False).to(self.dev)

        return state, reward, next_state, seqLen, batch_length

    def clear(self):
        self.buffer.clear()
        self.count = 0


if __name__ == '__main__':
    arr = [1, 2, 3, 4, 5, 6, 7, 8]
    sub = arr[5:len(arr) + 2]
    print(sub)
