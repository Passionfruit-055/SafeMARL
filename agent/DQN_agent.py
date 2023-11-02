#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import os
import random

import numpy as np
import torch as T
from torch import nn

from network.fc3 import DeepQNetwork
from agent.Experience_replay import ReplayBuffer
from datetime import datetime
now = datetime.now()
dt_string = now.strftime("_%d_%m_%H_%M")


class DQN(object):  # DQN类进行强化学习
    def __init__(self, gamma, lr, action_num, state_num,
                 buffer_size, batch_size, INITIAL_EPSILON, FINAL_EPSILON, max_episode=8000,
                 replace=800, chkpt_dir=None):
        self.gamma = gamma
        self.epsilon = INITIAL_EPSILON
        self.lr = lr
        self.n_actions = action_num
        self.state_dim = state_num
        self.batch_size = batch_size
        self.INITIAL_EPSILON = INITIAL_EPSILON
        self.FINAL_EPSILON = FINAL_EPSILON
        self.max_episode = max_episode
        self.replace_target_cnt = replace
        self.chkpt_dir = chkpt_dir
        self.action_space = [i for i in range(action_num)]
        self.learn_step_counter = 0

        self.memory = ReplayBuffer(buffer_size)
        self.risk_memory = ReplayBuffer(buffer_size)

        self.q_eval = DeepQNetwork(self.lr, output_dim=self.n_actions, input_dim=self.state_dim,
                                   chkpt_dir=self.chkpt_dir)

        self.q_next = DeepQNetwork(self.lr, output_dim=self.n_actions, input_dim=self.state_dim,
                                   chkpt_dir=self.chkpt_dir)

        self.q_next.load_state_dict(self.q_eval.state_dict())
        self.tau = 0.001
        self.loss = nn.MSELoss()

    def choose_action(self, observation):  # ε-greedy
        state = T.tensor(observation, dtype=T.float).to(self.q_eval.device)
        with T.no_grad():  # 这里仅做前向传播，不需要计算梯度
            actions = self.q_eval.forward(state).reshape(1, -1)
        if np.random.random() > self.epsilon:
            # print("values \n{}".format(actions.detach().cpu().numpy()))
            action = T.argmax(actions).detach().cpu().item()
        else:
            action = np.random.choice(self.action_space)
        return action

    def store_transition(self, state, action, reward, state_, warning=0):
        if warning == 0:
            self.memory.add(state, action, reward, state_)
        else:
            self.risk_memory.add(state, action, reward, state_)

    def sample_memory(self):
        # safe_batch
        state, action, reward, new_state = self.memory.sample_batch(self.batch_size)
        # risk_batch
        r_state, r_action, r_reward, r_new_state = self.risk_memory.sample_batch(self.batch_size)
        # mini_batch
        state.extend(r_state)
        action.extend(r_action)
        reward.extend(r_reward)
        new_state.extend(r_new_state)

        states = T.tensor(state, dtype=T.float).to(self.q_eval.device)
        rewards = T.tensor(reward, dtype=T.float).to(self.q_eval.device)
        actions = T.tensor(action).to(self.q_eval.device)
        states_ = T.tensor(new_state, dtype=T.float).to(self.q_eval.device)

        return states, actions, rewards, states_

    def replace_target_network(self):
        self.learn_step_counter += 1
        if self.learn_step_counter % self.replace_target_cnt == 0:
            self.q_next.load_state_dict(self.q_eval.state_dict())
            # self.save_models(self.chkpt_dir)

    def decrement_epsilon(self, episode):
        self.epsilon = self.INITIAL_EPSILON - episode * (self.INITIAL_EPSILON - self.FINAL_EPSILON) / self.max_episode

    def save_models(self, save_file):
        self.q_eval.save_checkpoint(save_file+'_eval')
        self.q_next.save_checkpoint(save_file+'_next')

    def load_models(self, load_file):
        self.q_eval.load_checkpoint(load_file+'_eval')
        self.q_next.load_checkpoint(load_file+'_next')

    def learn(self):
        self.decrement_epsilon(self.learn_step_counter)

        states, actions, rewards, states_ = self.sample_memory()

        # experience replay
        q_eval = self.q_eval.forward(states)
        q_next = self.q_next.forward(states_)
        max_q_value = T.max(q_next, dim=1)[0].detach()
        q_target = rewards.view(max_q_value.size()) + self.gamma * max_q_value
        q_eval_replaced = q_eval.clone()
        q_eval_replaced[T.arange(q_eval.shape[0]), actions] = q_target
        loss = self.loss(q_eval_replaced, q_eval)
        # print("loss = {}".format(loss))

        # 更新评估网络
        self.q_eval.optimizer.zero_grad()
        loss.backward()
        self.q_eval.optimizer.step()

        # 更新目标网络
        # 软更新
        for k in self.q_next.state_dict().keys():
            self.q_next.state_dict()[k] = self.tau * self.q_eval.state_dict()[k] + (1. - self.tau) * \
                                          self.q_next.state_dict()[k]
        # 硬更新
        self.replace_target_network()


if __name__ == '__main__':
    batch_size = 3
    action_dim = 5
    reward = T.full((batch_size, 1), 0.5)
    action = T.full((batch_size, 1), random.randint(0, action_dim - 1))
    print("action \n{}".format(action))
    gamma = 0.8
    q_eval = T.full((batch_size, action_dim), 2, dtype=T.float)
    q_next = T.rand((batch_size, action_dim))
    max_q_value, max_q_value_index = T.max(q_next, dim=1)
    q_target = reward.view(max_q_value.size()) + gamma * max_q_value
    q_eval_replaced = q_eval.clone()
    q_eval_replaced[T.arange(q_eval.shape[0]), action] = q_target
    loss_func = T.nn.MSELoss()
    loss = loss_func(q_eval_replaced, q_eval)

    print("loss \n{}".format(loss))
