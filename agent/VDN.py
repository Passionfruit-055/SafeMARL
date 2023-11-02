import pdb
import random
import time

import numpy as np

from network.QMIXnets import QMIXNet
from network.BaseModel import RNN
import torch
import torch.nn as nn
import torch.nn.functional as F
from ReplayBuffer import RecurrentReplayBuffer, ReplayBuffer
from utils.parser import args

torch.autograd.set_detect_anomaly(True)


class VDNagent(object):
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
        # individual local Q-network
        if self.local_q_mode == 'individual':
            self.eval_net = []
            self.target_net = []
            net_template = RNN(obs_size, action_size, rnn_hidden_size).state_dict()
            for i in range(self.agent_num):
                self.eval_net.append(RNN(obs_size, action_size, rnn_hidden_size).to(self.dev))
                self.eval_net[i].load_state_dict(net_template)
                self.target_net.append(RNN(obs_size, action_size, rnn_hidden_size).to(self.dev))
                self.target_net[i].load_state_dict(self.eval_net[i].state_dict())
            for i in range(self.agent_num):
                self.param.extend(self.eval_net[i].parameters())
        # sharing local Q-network
        else:
            # self.obs_size += self.agent_num
            # self.state_size += self.agent_num ** 2

            self.eval_net = RNN(self.obs_size, action_size, rnn_hidden_size).to(self.dev)
            self.target_net = RNN(self.obs_size, action_size, rnn_hidden_size).to(self.dev)
            self.target_net.load_state_dict(self.eval_net.state_dict())
            self.param.extend(self.eval_net.parameters())

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
            if self.local_q_mode == 'individual':
                for enet, tnet in zip(self.eval_net, self.target_net):
                    tnet.load_state_dict(enet.state_dict())
            else:
                self.target_net.load_state_dict(self.eval_net.state_dict())

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
            if self.local_q_mode == 'individual':
                with torch.no_grad():
                    Qvals, self.eval_hidden_state_exec[a] = self.eval_net[a](obs[a], self.eval_hidden_state_exec[a])
            else:
                with torch.no_grad():
                    Qvals, self.eval_hidden_state_exec[a] = self.eval_net(obs[a], self.eval_hidden_state_exec[a])

            if random.random() > self.epsilon:
                actions.append(torch.argmax(Qvals[0], dim=0).item())
            else:
                actions.append(random.choice(range(self.action_size)))
        return actions

    def update(self, episodes):
        assert len(episodes) > 0
        batchSize = len(episodes)
        '''
           学习过程的循环不应按照episode而应该是按照一个时间序列来，
           而在每一个时刻相当于对多个episode同时进行重放，提高了数据的并行程度
           也就是说，hidden_size应有两维，(episode_num(batch_size), obs_size)
           batchSize = 每一次抽取多少个episode的数据
           seqLen = 每个episode抽取多少个时间步的数据
        '''
        states, rewards, next_states, seqLen, batch_length = self.mixing_memory.sample_batch(episodes)
        rewards = rewards.squeeze(-1).detach()
        state_size = states.shape[-1]
        # 创建mask以标记末尾状态
        mask = torch.zeros((seqLen, batchSize)).to(self.dev)
        for row, last_index in zip(mask, batch_length):
            row[:last_index] = 1

        obs_set = []
        us_set = []
        n_obs_set = []

        for a in range(self.agent_num):
            # Q值需要重新经过前向传播获得，只有这样才能在反向传播时更新到local的eval_net
            obs, us, n_obs = self.memory.sample_batch(a, seqLen, len(episodes), episodes)
            obs_set.append(obs)
            us_set.append(us)
            n_obs_set.append(n_obs)

        loss_set = []
        for ep in range(self.epoch):
            self.hidden_state_reset(batchSize)
            for transition in range(seqLen):
                self.training_step += 1

                Q_eval_l = []
                Q_target_l = []
                if self.local_q_mode == 'individual':
                    for obs, us, n_obs, eval_net, target_net, eval_hidden_state, target_hidden_state in zip(obs_set,
                                                                                                            us_set,
                                                                                                            n_obs_set,
                                                                                                            self.eval_net,
                                                                                                            self.target_net,
                                                                                                            self.eval_hidden_state,
                                                                                                            self.target_hidden_state):
                        # Q_eval -> (batchSize, action_size)
                        # with torch.autograd.set_detect_anomaly(True):
                        Q_eval, eval_hidden_state = eval_net(obs[transition], eval_hidden_state)
                        Q_target, target_hidden_state = target_net(n_obs[transition], target_hidden_state)
                        # Q_eval -> (batchSize, 1)
                        Q_eval = torch.gather(Q_eval, 1, us[transition].unsqueeze(1)).squeeze(-1)
                        Q_target = torch.max(Q_target, 1)
                        # wait to stack
                        Q_eval_l.append(Q_eval)
                        Q_target_l.append(Q_target[0])
                else:
                    for obs, us, n_obs, eval_hidden_state, target_hidden_state in zip(obs_set, us_set, n_obs_set,
                                                                                      self.eval_hidden_state,
                                                                                      self.target_hidden_state):
                        # Q_eval -> (batchSize, action_size)
                        # with torch.autograd.set_detect_anomaly(True):
                        Q_eval, eval_hidden_state = self.eval_net(obs[transition], eval_hidden_state)
                        Q_target, target_hidden_state = self.target_net(n_obs[transition], target_hidden_state)
                        # Q_eval -> (batchSize, 1)
                        Q_eval = torch.gather(Q_eval, 1, us[transition].unsqueeze(1)).squeeze(-1)
                        Q_target = torch.max(Q_target, 1)
                        # wait to stack
                        Q_eval_l.append(Q_eval)
                        Q_target_l.append(Q_target[0])

                # Q_evals -> (batchSize, agent_num)
                Q_evals = torch.stack(Q_eval_l, dim=1).view(batchSize, -1)
                Q_targets = torch.stack(Q_target_l, dim=1).view(batchSize, -1)
                Q_tot_eval = torch.sum(Q_evals, dim=1)
                Q_tot_target = torch.sum(Q_targets, dim=1)

                td_target = rewards[transition] + self.gamma * torch.mul(Q_tot_target.detach(), mask[transition].detach())

                for i, b_l in enumerate(batch_length):
                    if b_l < transition + 1:
                        td_target[i] = Q_tot_eval[i]  # padding loss = 0

                self.optimizer.zero_grad()
                loss_val = self.loss(Q_tot_eval, td_target)
                loss_val.backward()
                loss_set.append(loss_val.detach().cpu().item())
                # torch.nn.utils.clip_grad_norm_(self.param, max_norm=10, norm_type=2)
                self.optimizer.step()
                self.target_net_update(self.logger)
        return loss_set

    def store_transition(self, ob, u, ob_next, episode, agent):
        self.memory.add(ob, u, ob_next, episode, agent)

    def store_local_transition(self, obs, us, rs, n_obs, episode):
        for a in range(self.agent_num):
            self.memory.add(obs[a], us[a], n_obs[a], episode, a)

    def store_mixing_transition(self, s, r, s_next, episode):
        self.mixing_memory.add(episode, s, r, s_next)


if __name__ == '__main__':
    pass
