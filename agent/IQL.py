import pdb
import random
import time
from collections import deque

import numpy as np

from network.QMIXnets import QMIXNet
from network.BaseModel import RNN
import torch
import torch.nn as nn
import torch.nn.functional as F
from ReplayBuffer import RecurrentReplayBuffer, ReplayBuffer
from utils.parser import args

torch.autograd.set_detect_anomaly(True)


class IQLagent(object):
    def __init__(self, agent_num, state_size, action_size, obs_size, logger=None, mixing_hidden_size=args['hsMixing'],
                 hyper_hidden_size=args['hsHyper'], rnn_hidden_size=args['hsRNN'], initial_epsilon=0.2,
                 final_epsilon=0.001, epoch=args['epoch'], lr=args['learning_rate'], buffer_size=args['buffer_size'],
                 batch_size=args['batch_size'], episode=args['episode'],
                 seq_length=args['seq_len'], gamma=args['gamma'], dev='cuda:0'):
        self.logger = logger
        self.agent_num = agent_num
        self.state_size = state_size
        self.action_size = action_size
        self.rnn_hidden_size = rnn_hidden_size
        self.mixing_hidden_size = mixing_hidden_size
        self.hyper_hidden_size = hyper_hidden_size
        self.obs_size = obs_size
        self.dev = dev
        self.episode = episode
        self.buffer_size = buffer_size
        self.batch_size = batch_size
        self.epoch = epoch
        self.seq_length = seq_length
        self.initial_epsilon = initial_epsilon
        self.final_epsilon = final_epsilon
        self.epsilon = initial_epsilon
        self.gamma = gamma
        self.lr = lr
        self.local_q_mode = ['sharing', 'individual'][1]

        self.param = []
        self.eval_net = []
        self.target_net = []
        net_template = RNN(obs_size, action_size, rnn_hidden_size).state_dict()
        for i in range(self.agent_num):
            self.eval_net.append(RNN(obs_size, action_size, rnn_hidden_size).to(self.dev))
            self.eval_net[i].load_state_dict(net_template)
            self.target_net.append(RNN(obs_size, action_size, rnn_hidden_size).to(self.dev))
            self.target_net[i].load_state_dict(self.eval_net[i].state_dict())
            self.param.extend(self.eval_net[i].parameters())

        self.memory = RecurrentReplayBuffer(agent_num, seq_length, buffer_size, dev=self.dev)
        # for train
        self.eval_hidden_state = [None] * self.agent_num
        self.target_hidden_state = [None] * self.agent_num
        self.hidden_state_reset()
        # for execute
        self.eval_hidden_state_exec = [None] * self.agent_num
        self.target_hidden_state_exec = [None] * self.agent_num
        self.exec_hidden_state_reset()

        self.mixing_memory = ReplayBuffer(buffer_size, dev=self.dev)

        self.optimizer = torch.optim.RMSprop(self.param, lr=self.lr)
        self.loss = nn.MSELoss()

        # target network update
        self.training_step = 0
        self.replace_cycle = args['replace_cycle']
        self.tau = args['tau']

    def target_net_update(self, logger):
        if self.training_step % self.replace_cycle == 0:
            for enet, tnet in zip(self.eval_net, self.target_net):
                tnet.load_state_dict(enet.state_dict())

    def decrement_epsilon(self, episode, batch_size):
        if episode > batch_size:
            self.epsilon = self.initial_epsilon - episode * (self.initial_epsilon - self.final_epsilon) / self.episode
        else:
            self.epsilon = 1

    def hidden_state_reset(self, batch_size=1):
        self.eval_hidden_state = torch.zeros(self.agent_num, batch_size, self.rnn_hidden_size).to(self.dev)
        self.target_hidden_state = torch.zeros(self.agent_num, batch_size, self.rnn_hidden_size).to(self.dev)

    def exec_hidden_state_reset(self, batch_size=1):
        self.eval_hidden_state_exec = torch.zeros(self.agent_num, batch_size, self.rnn_hidden_size).to(self.dev)
        self.target_hidden_state_exec = torch.zeros(self.agent_num, batch_size, self.rnn_hidden_size).to(self.dev)

    def choose_actions(self, obs):
        obs = torch.tensor(obs, dtype=torch.float64).view(self.agent_num, -1).to(self.dev)
        actions = []

        for a in range(self.agent_num):
            with torch.no_grad():
                Qvals, self.eval_hidden_state_exec[a] = self.eval_net[a](obs[a], self.eval_hidden_state_exec[a])

            if random.random() > self.epsilon:
                actions.append(torch.argmax(Qvals[0], dim=0).item())
            else:
                actions.append(random.choice(range(self.action_size)))
        return actions

    def update(self, episodes):
        assert len(episodes) > 0
        batchSize = len(episodes)

        loss_set = [deque(maxlen=args['timestep']), deque(maxlen=args['timestep'])]
        for ep in range(self.epoch):
            for a in range(self.agent_num):
                obs, us, rs, n_obs, seqLens = self.memory.sample_fullbatch(a, episodes)
                max_seqLen = max(seqLens)
                mask = torch.zeros((max_seqLen, batchSize)).to(self.dev)
                for row, seql in zip(mask, seqLens):
                    row[:seql - 1] = 1

                self.hidden_state_reset(batchSize)
                eval_net, target_net = self.eval_net[a], self.target_net[a]
                eval_hidden_state, target_hidden_state = self.eval_hidden_state[a], self.target_hidden_state[a]
                losses = loss_set[a]

                for transition in range(max_seqLen):
                    self.training_step += 1

                    # Q_eval -> (batchSize, action_size)
                    Q_evals, eval_hidden_state = eval_net(obs[transition], eval_hidden_state)
                    Q_targets, target_hidden_state = target_net(n_obs[transition], target_hidden_state)
                    # Q_eval -> (batchSize, 1)
                    Q_eval = torch.gather(Q_evals, 1, us[transition].unsqueeze(1)).squeeze(-1)
                    Q_target = torch.max(Q_targets, 1).values

                    # 处理为了对齐的padding的尾部数据
                    # last_timestep的td_target仅有reward, 终止后的td_target设置与Q_eval相同, 之间无loss不会影响反向传播

                    td_target = rs[transition] + self.gamma * torch.mul(Q_target, mask[transition].detach())

                    for batch, seql in enumerate(seqLens):
                        if seql < transition + 1:
                            td_target[batch] = Q_eval[batch]

                    self.optimizer.zero_grad()
                    loss_val = self.loss(Q_eval, td_target)
                    loss_val.backward(retain_graph=True)
                    losses.append(loss_val.detach().cpu().item())
                    self.optimizer.step()
                    self.target_net_update(self.logger)
        return loss_set

    def store_transition(self, ob, u, ob_next, episode, agent):
        self.memory.add(ob, u, ob_next, episode, agent)

    def store_local_transition(self, obs, us, rs, n_obs, episode):
        for a in range(self.agent_num):
            self.memory.add(obs[a], us[a], n_obs[a], episode, a, rs[a])

    def store_mixing_transition(self, s, r, s_next, episode):
        self.mixing_memory.add(episode, s, r, s_next)


if __name__ == '__main__':
    pass
