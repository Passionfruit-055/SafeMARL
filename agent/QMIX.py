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


class QMIXagent(object):
    def __init__(self, agent_num, state_size, action_size, obs_size, mixing_hidden_size=args['hsMixing'],
                 hyper_hidden_size=args['hsHyper'], rnn_hidden_size=args['hsRNN'], initial_epsilon=0.2,
                 final_epsilon=0.001, epoch=args['epoch'], lr=args['learning_rate'], buffer_size=args['buffer_size'],
                 batch_size=args['batch_size'], episode=args['episode'],
                 seq_length=args['seq_len'], gamma=args['gamma'], dev='cuda:0'):
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

        self.eval_net = []
        self.target_net = []
        for i in range(self.agent_num):
            self.eval_net.append(RNN(obs_size, action_size, rnn_hidden_size).to(self.dev))
            self.target_net.append(RNN(obs_size, action_size, rnn_hidden_size).to(self.dev))
            self.target_net[i].load_state_dict(self.eval_net[i].state_dict())

        # mixing net (实际上是hypernets在更新)
        self.mixing_eval_net = QMIXNet(self.agent_num, self.mixing_hidden_size, self.state_size,
                                       self.hyper_hidden_size).to(self.dev)
        self.mixing_target_net = QMIXNet(self.agent_num, self.mixing_hidden_size, self.state_size,
                                         self.hyper_hidden_size).to(self.dev)
        self.mixing_target_net.load_state_dict(self.mixing_eval_net.state_dict())

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

        self.param = []
        for i in range(self.agent_num):
            self.param.extend(self.eval_net[i].parameters())
        self.param.extend(self.mixing_eval_net.parameters())

        self.optimizer = torch.optim.Adam(self.param, lr=self.lr)
        self.loss = nn.MSELoss()

        # target network update
        self.training_step = 0
        self.replace_cycle = args['replace_cycle']
        self.tau = args['tau']

    def target_net_update(self):
        if self.training_step % self.replace_cycle == 0:
            for enet, tnet in zip(self.eval_net, self.target_net):
                tnet.load_state_dict(enet.state_dict())
            self.mixing_target_net.load_state_dict(self.mixing_eval_net.state_dict())

    def decrement_epsilon(self, episode):
        self.epsilon = self.initial_epsilon - episode * (self.initial_epsilon - self.final_epsilon) / self.episode

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
            Qvals, self.eval_hidden_state_exec[a] = self.eval_net[a](obs[a], self.eval_hidden_state_exec[a])
            if random.random() > self.epsilon:
                actions.append(torch.argmax(Qvals[0], dim=0).item())
            else:
                actions.append(random.choice(range(self.action_size)))
        return actions

    # def learn(self, episode, timestep, rand=True):
    #     if timestep == 0 or episode == 0:
    #         print('Not enough memory')
    #         return
    #     seqLen = min(timestep, self.seq_length)
    #     start_pos = random.randint(0, max(timestep - seqLen, 0)) if rand else 0
    #     # batchSize = min(self.batch_size, episode)
    #     # episodes = random.sample(range(episode + 1), batchSize) if rand else episodes
    #     batchSize = 1
    #     episodes = [episode]
    #     '''
    #        学习过程的循环不应按照episode而应该是按照一个时间序列来，
    #        而在每一个时刻相当于对多个episode同时进行重放，提高了数据的并行程度
    #        也就是说，hidden_size应有两维，(episode_num(batch_size), obs_size)
    #        batchSize = 每一次抽取多少个episode的数据
    #        seqLen = 每个episode抽取多少个时间步的数据
    #     '''
    #     # sample minibatch是为了获取Q值，与DQN中是为了获取状态等不同
    #     states, rewards, next_states = self.mixing_memory.sample_batch(episodes, seqLen, start_pos)
    #     state_size = states.shape[2]
    #
    #     obs_set = []
    #     us_set = []
    #     n_obs_set = []
    #
    #     for a in range(self.agent_num):
    #         # sample minibatch
    #         obs, us, n_obs = self.memory.sample_batch(a, seqLen, episodes, start_pos)
    #         obs = obs.view(seqLen, batchSize, -1)
    #         us = us.view(seqLen, batchSize, -1)
    #         n_obs = n_obs.view(seqLen, batchSize, -1)
    #         obs_set.append(obs)
    #         us_set.append(us)
    #         n_obs_set.append(n_obs)
    #
    #     loss_set = []
    #     self.hidden_state_reset(batchSize)
    #     for ep in range(self.epoch):
    #         for transition in range(seqLen):
    #             self.training_step += 1
    #
    #             Q_eval_l = []
    #             Q_target_l = []
    #             for obs, us, n_obs, eval_net, target_net, eval_hidden_state, target_hidden_state in zip(obs_set, us_set,
    #                                                                                                     n_obs_set,
    #                                                                                                     self.eval_net,
    #                                                                                                     self.target_net,
    #                                                                                                     self.eval_hidden_state,
    #                                                                                                     self.target_hidden_state):
    #                 # Q_eval -> (batchSize, action_size)
    #                 with torch.autograd.set_detect_anomaly(True):
    #                     Q_eval, eval_hidden_state = eval_net(obs[transition], eval_hidden_state)
    #                     Q_target, target_hidden_state = target_net(n_obs[transition], target_hidden_state)
    #                     # Q_eval -> (batchSize, 1)
    #                     Q_eval = torch.gather(Q_eval, 1, us[transition])
    #                     Q_target = torch.max(Q_target, 1)
    #                     # wait to stack
    #                     Q_eval_l.append(Q_eval)
    #                     Q_target_l.append(Q_target[0])
    #
    #             # Q_evals -> (batchSize, agent_num)
    #             Q_evals = torch.stack(Q_eval_l, dim=1).view(batchSize, -1)
    #             Q_targets = torch.stack(Q_target_l, dim=1).view(batchSize, -1)
    #             Q_tot_eval = self.mixing_eval_net(Q_evals, states[transition], self.agent_num, state_size)
    #             Q_tot_target = self.mixing_target_net(Q_targets, next_states[transition], self.agent_num,
    #                                                   state_size)
    #             td_target = rewards[transition] + self.gamma * Q_tot_target.detach()
    #
    #             self.optimizer.zero_grad()
    #             loss_val = self.loss(Q_tot_eval, td_target)
    #             # print(f"loss = {loss_val.data}")
    #             loss_set.append(loss_val.detach().cpu())
    #             with torch.autograd.set_detect_anomaly(True):
    #                 loss_val.backward(retain_graph=True)
    #             torch.nn.utils.clip_grad_norm_(self.param, max_norm=10, norm_type=2)
    #             # for i, param in enumerate(self.param):
    #             #     print(f"param{i} = {param}")
    #             #     print(f"param{i} grad = {param.grad}")
    #             self.optimizer.step()
    #             self.target_net_update()
    #     return loss_set

    def update(self, episodes):
        assert len(episodes) > 0
        start_pos = 0
        batchSize = len(episodes)
        '''
           学习过程的循环不应按照episode而应该是按照一个时间序列来，
           而在每一个时刻相当于对多个episode同时进行重放，提高了数据的并行程度
           也就是说，hidden_size应有两维，(episode_num(batch_size), obs_size)
           batchSize = 每一次抽取多少个episode的数据
           seqLen = 每个episode抽取多少个时间步的数据
        '''
        states, rewards, next_states, seqLen = self.mixing_memory.sample_batch(episodes)
        state_size = states.shape[2]

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
                for obs, us, n_obs, eval_net, target_net, eval_hidden_state, target_hidden_state in zip(obs_set, us_set,
                                                                                                        n_obs_set,
                                                                                                        self.eval_net,
                                                                                                        self.target_net,
                                                                                                        self.eval_hidden_state,
                                                                                                        self.target_hidden_state):
                    # Q_eval -> (batchSize, action_size)
                    with torch.autograd.set_detect_anomaly(True):
                        Q_eval, eval_hidden_state = eval_net(obs[transition], eval_hidden_state)
                        Q_target, target_hidden_state = target_net(n_obs[transition], target_hidden_state)
                        # Q_eval -> (batchSize, 1)
                        Q_eval = torch.gather(Q_eval, 1, us[transition])
                        Q_target = torch.max(Q_target, 1)
                        # wait to stack
                        Q_eval_l.append(Q_eval)
                        Q_target_l.append(Q_target[0])

                # Q_evals -> (batchSize, agent_num)
                Q_evals = torch.stack(Q_eval_l, dim=1).view(batchSize, -1)
                Q_targets = torch.stack(Q_target_l, dim=1).view(batchSize, -1)
                Q_tot_eval = self.mixing_eval_net(Q_evals, states[transition], self.agent_num, state_size)
                Q_tot_target = self.mixing_target_net(Q_targets, next_states[transition], self.agent_num,
                                                      state_size)
                td_target = rewards[transition] + self.gamma * Q_tot_target.detach()

                self.optimizer.zero_grad()
                loss_val = self.loss(Q_tot_eval, td_target)
                print(f"loss = {loss_val.data}")
                loss_set.append(loss_val.detach().cpu())
                with torch.autograd.set_detect_anomaly(True):
                    loss_val.backward(retain_graph=True)
                torch.nn.utils.clip_grad_norm_(self.param, max_norm=10, norm_type=2)
                # for i, param in enumerate(self.param):
                #     print(f"param{i} = {param}")
                #     print(f"param{i} grad = {param.grad}")
                self.optimizer.step()
                self.target_net_update()
        return loss_set

    def store_transition(self, ob, u, ob_next, episode, agent):
        self.memory.add(ob, u, ob_next, episode, agent)

    def store_local_transition(self, obs, us, rs, n_obs, episode):
        for a in range(self.agent_num):
            self.memory.add(obs[a], us[a], n_obs[a], episode, a)

    def store_mixing_transition(self, s, r, s_next, done, episode):
        s = np.array(s, dtype=object).flatten().tolist()
        s_next = np.array(s_next, dtype=object).flatten().tolist()
        self.mixing_memory.add(episode, s, r, s_next, done)


if __name__ == '__main__':
    agent = QMIXagent(5, 200, 200, 200)
    eval_mixing_param = agent.mixing_eval_net.state_dict()
    target_mixing_param = agent.mixing_target_net.state_dict()
    eval_param = []
    target_param = []

    eval_param.extend(list(agent.mixing_eval_net.parameters()))
    target_param.extend(list(agent.mixing_target_net.parameters()))

    agent.optimizer = torch.optim.RMSprop(eval_param, lr=agent.lr)
    for e in range(10):
        agent.optimizer.zero_grad()
        loss = nn.MSELoss()(torch.rand(0, 10), torch.rand(0, 10))
        loss.requires_grad = True
        loss.backward()
        print(f"loss = {loss}")
        agent.optimizer.step()
    pass
