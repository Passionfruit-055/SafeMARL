import argparse
import datetime
import random
import itertools as it
from collections import deque

from matplotlib import pyplot as plt

now = datetime.datetime.now()
formatted_date = now.strftime("%Y-%m-%d %H:%M:%S")

import os
import math

import numpy as np
import torch
import torch.nn.functional as F
from torch import optim, nn

# from env.FederatedUpload.IoVenv import IoV
from env.FederatedUpload.Models import Mnist_CNN
from env.FederatedUpload.clients import ClientsGroup
from env.FederatedUpload.model.SpinalVGG import Spinalvgg19_bn

from utils.parser import args as c_args

parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter, description="FedAvg")
parser.add_argument('-g', '--gpu', type=str, default='0', help='gpu id to use(e.g. 0,1,2,3)')
# 客户端的数量
parser.add_argument('-nc', '--num_of_clients', type=int, default=5, help='numer of the clients')
# 随机挑选的客户端的数量
parser.add_argument('-cf', '--cfraction', type=float, default=0.4,
                    help='C fraction, 0 means 1 client, 1 means total clients')
# 训练次数(客户端更新次数)
parser.add_argument('-E', '--epoch', type=int, default=1, help='local train epoch')
# batchsize大小
parser.add_argument('-B', '--batch_size', type=int, default=32, help='local train batch size')
# 模型名称
parser.add_argument('-mn', '--model_name', type=str, default='mnist_cnn', help='the model to train')
# 学习率
parser.add_argument('-lr', "--learning_rate", type=float, default=0.001, help="learning rate, \
                    use value from origin paper as default")
parser.add_argument('-lrs', "--lrs", type=float, default=0.001, help="dqn server Learning rate")
parser.add_argument('-dataset', "--dataset", type=str, default="mnist", help="需要训练的数据集")
# 模型验证频率（通信频率）
parser.add_argument('-vf', "--val_freq", type=int, default=5, help="model validation frequency(of communications)")
parser.add_argument('-sf', '--save_freq', type=int, default=20, help='global model save frequency(of communication)')
# n um_comm 表示通信次数，此处设置为10
parser.add_argument('-ncomm', '--num_comm', type=int, default=800, help='number of communications')
parser.add_argument('-sp', '--save_path', type=str, default='./checkpoints', help='the saving path of checkpoints')
parser.add_argument('-iid', '--IID', type=int, default=1, help='the way to allocate data to clients')
parser.add_argument('-seed', '--seed', type=int, default=1, help='random seed')

args = parser.parse_args()
args = args.__dict__
np.set_printoptions(precision=3, suppress=True)

os.environ['CUDA_VISIBLE_DEVICES'] = args['gpu']
os.environ['CUDA_LAUNCH_BLOCKING'] = '1'

INFINITY = float('inf')


class IoV(object):
    def __init__(self, dataset, epoch, vnum=5, cnum=5, size=30):
        self.com_round = 0
        self.memory = 100  # fraction=0.1
        # 传输时延相关
        self.power = [3, 3, 2, 1, 1]  # w
        self.gain = [3e-5, 3e-5, 1e-6, 2e-7, 1e-7]
        self.noise = [1e-14, 1e-14, 1e-13, 1e-13, 1e-13]  # pw
        self.delta = 1e6  # bit, 即0.1Mb
        self.dist = [0, 300]  # m
        self.width = 1e6  # Mhz
        # 计算时延相关
        self.dataset = dataset
        self.datasize = [0, 10, 20, 30, 40, 50]
        # self.computeSize = 1.5e4  # bit
        self.computeSize = 2.4e4 if dataset == 'cifar10' else 1.5e4  # 6.2e3
        self.computeCapacity = [4e9, 4e9, 2e9, 1e9, 1e9]  # bps, 即1-5Gbps
        self.discount = 0.9

        # 根据车辆数和信道数初始化环境
        self.vnum = vnum
        self.cnum = cnum
        self.vehicle = []
        self.chans = []
        self.fed_rate = [0.9, 0.9, 0.3, 0.2, 0.1]

        self.latency = [0.482, 0.482, 0.866, 1.376, 1.403]  # 单位是 n*10ms
        self.threshold = 0.0317 + 0.0054 * epoch if dataset == 'cifar10' else 7.5e-5 * 1.5  # 0.0317 + 0.045 * epoch

        # 初始化车的状态
        for v in range(self.vnum):
            self.vehicle.append(self.connect2car(v, size))
        # 初始化信道状态
        self.initialize_channel()

    def connect2car(self, i, size=600):
        vehicle = {'dist': random.uniform(self.dist[0], self.dist[1]), 'latency': self.latency[i], 'size': size,
                   'quality': 0, 'fed': 1 if i < 2 else 0, 'fed_cnt': 0, 'fed_rate': self.fed_rate[i],
                   'compute': self.computeCapacity[i], 'action_history':deque([0]*10, maxlen=10)}
        return vehicle

    def initialize_channel(self):
        for i, (p, h, sigma) in enumerate(zip(self.power, self.gain, self.noise)):
            chan = {'gain': h, 'noise': sigma, 'rate': self.width * math.log(1 + p * h / sigma, 2)}
            self.chans.append(chan)
        return self.chans

    def move(self):
        dist = []
        for v in self.vehicle:
            d = v['dist'] = random.uniform(self.dist[0], self.dist[1])
            dist.append(d)
        return dist

    def set_frate(self, fed, timestep):
        frates = []
        for i, (v, f) in enumerate(zip(self.vehicle, fed)):
            v['fed_cnt'] += f
            v['fed_rate'] = v['fed_cnt'] / (timestep + 1)
            frates.append(v['fed_rate'])
        return frates

    def set_latency(self, sizes, epoch):
        for i, size in enumerate(sizes):
            print(f"Vehicle{i}")
            self.vehicle[i]['size'] = size
            if self.vehicle[i]['size'] > 0:
                v = self.chans[i]['rate']
                l_t = self.delta / v
                print(f"传输时延    = {l_t * 1e3:.3f}ms")
                l_c = epoch * self.computeSize * self.vehicle[i]['size'] / self.computeCapacity[i]
                print(f"本地计算时延 = {l_c * 1e3:.3f}ms")
                self.vehicle[i]['latency'] = l_c * 1e3 + l_t
            else:
                self.vehicle[i]['latency'] = 0
                print("No local training")

    def set_quality(self, gq, fed):
        for v in range(len(self.vehicle)):
            if fed[v] == 1:
                self.vehicle[v]['quality'] = gq

    def update_action_path(self, action):
        for i, vehicle in enumerate(self.vehicle):
            vehicle['action_history'].append(action[i])
    def get_frate(self):
        return [v['fed_rate'] for v in self.vehicle]

    def get_latency(self):
        return [v['latency'] for v in self.vehicle]

    def get_quality(self):
        return [v['quality'] for v in self.vehicle]

    def get_utility(self, fed, accu, accu_p, la):
        o1 = 0.1
        o2 = 1.1
        o3 = 0.7  # pre = 0.03
        m1 = 0.4
        m2 = 1
        tau = self.threshold  # s

        par = self.get_frate()
        reward = o1 * min(par) + o2 * accu - o3 * la * 1e3  # latency*10 归一化

        risk = 0
        for v in range(self.vnum):
            late = 1 if la >= tau else 0
            risk += late * m1 * fed[v]
        diff = min(accu - accu_p, 0) * m2  # difference in double accuracy
        risk -= diff
        return reward - risk

    def get_independent_utility(self, gq):
        o1 = 0.3
        o2 = 1
        o3 = 0.5
        utilities = []
        for vehicle in self.vehicle:
            low_accu = 0.5 if vehicle['quality'] < gq else 0
            ROOT = 0.5 if vehicle['latency'] > self.threshold else 0
            utilities.append(vehicle['fed_rate'] * o1 + vehicle['quality'] * o2 - vehicle['latency'] * o3 - low_accu - ROOT)
        return utilities

    def get_team_reward(self, rewards, gq, timestep):
        return sum(rewards)

    def observe(self, gq):
        state = self.get_latency() + self.get_quality() + self.get_frate()
        state.append(gq)
        return state

    def step(self, action, gq, fed, epoch, timestep):
        # env
        self.com_round += 1
        self.move()
        # RL metrics
        self.set_frate(fed, timestep)
        self.set_latency(action, epoch)
        self.set_quality(gq, fed)
        self.update_action_path(action)
        return self.observe(gq)

    def view(self):
        print("\nCheck all vehicle")
        for i in range(self.vnum):
            v = self.vehicle[i]
            c = self.chans[i]
            print(
                "Vehicle{}\ndist = {:.3f}m\nlatency = {:.3f}ms\nsize = {}\nquality = {}\nfed = {}\nfed_cnt = {}\nfed_rate = {:.3f}".format(
                    i + 1, v[
                        'dist'], v['latency'], v['size'], v['quality'], True if v['fed'] else False, v['fed_cnt'],
                    v['fed_rate']))
            print(
                "Channel: gain = {} | noise = {} | rate = {:.3f}Mbps\n".format(c['gain'], c['noise'], c['rate'] / 1e6))

    def ROOT(self, i, size):
        v = self.chans[i]['rate']
        # 传输时延
        l_t = self.delta / v
        # 计算时延
        l_c = self.computeSize * size / self.computeCapacity[i]
        latency = l_c
        if latency > self.threshold:
            print(f'Client{i} ROOT')
            return True
        else:
            return False

    def get_local_obs(self):
        obs = []
        for i, vehicle in enumerate(self.vehicle):
            s_obs = []
            s_obs.extend([vehicle['latency'], vehicle['fed_rate'], vehicle['quality']])
            s_obs.extend(self.vehicle[i]['action_history'])
            one_hot = [0] * self.vnum
            one_hot[i] = 1
            s_obs.extend(one_hot)
            obs.append(s_obs)
        return obs


class FederatedUpload(object):
    def __init__(self):
        self.dev = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
        self.net = Spinalvgg19_bn().to(self.dev) if args['dataset'] == 'cifar10' else Mnist_CNN().to(self.dev)
        self.loss_func = nn.CrossEntropyLoss()
        self.opti = optim.SGD(self.net.parameters(), lr=args['learning_rate'], momentum=0.9)

        self.myClients = ClientsGroup(args['dataset'], args['IID'], c_args['agent_num'], self.dev)

        self.global_parameters = {}
        for key, var in self.net.state_dict().items():
            self.global_parameters[key] = var.clone()

        self.IoV = IoV(args['dataset'], args['epoch'], vnum=c_args['agent_num'])
        self.model_q = 0
        self.history = {'accuracy': [], 'utility': [], 'latency': []}
        self.team_reward = 0
        self.la_list = []

    def reset(self):
        self.net = Spinalvgg19_bn().to(self.dev) if args['dataset'] == 'cifar10' else Mnist_CNN().to(self.dev)
        self.loss_func = nn.CrossEntropyLoss()
        self.opti = optim.SGD(self.net.parameters(), lr=args['learning_rate'], momentum=0.9)

        self.myClients = ClientsGroup(args['dataset'], args['IID'], c_args['agent_num'], self.dev)

        self.global_parameters = {}
        for key, var in self.net.state_dict().items():
            self.global_parameters[key] = var.clone()

        self.IoV = IoV(args['dataset'], args['epoch'], vnum=c_args['agent_num'])
        self.model_q = 0

        del self.la_list
        self.la_list = []
        self.team_reward = 0

    def step(self, timestep, agent=None, actions=None, root_path=None):
        assert agent is not None
        local_params = []
        fed = [0 if action == 0 else 1 for action in actions]
        datasizes = [self.IoV.datasize[action] for action in actions]
        print(f'datasize = {datasizes}')
        for i, datasize in enumerate(datasizes):
            if datasize != 0:
                if self.IoV.ROOT(i, datasize):
                    continue
                local_parameters = self.myClients.clients_set[f'client{i}'].localUpdate(args['epoch'],
                                                                                        args['batch_size'],
                                                                                        self.net, self.loss_func,
                                                                                        self.opti,
                                                                                        self.global_parameters,
                                                                                        datasize)
                local_params.append(local_parameters)
                self.myClients.clients_set[f'client{i}'].local_val(self.net, i, args['batch_size'])

        # 联邦聚合本地参数
        sum_parameters = None
        if len(local_params) > 0:
            for local_parameters in local_params:
                if sum_parameters is None:
                    sum_parameters = {key: var.clone() for key, var in local_parameters.items()}
                else:
                    for key, var in local_parameters.items():
                        sum_parameters[key] = sum_parameters[key] + var.clone()

            for key, var in sum_parameters.items():
                sum_parameters[key] = sum_parameters[key] / len(local_params)
            self.global_parameters = sum_parameters
        self.net.load_state_dict(self.global_parameters, strict=True)

        # 测试本轮模型精度
        pre_q = self.model_q
        testDataLoader = self.myClients.refreshTestLoader(args['batch_size'])
        self.eval_local_accu(testDataLoader)

        latency = max(self.IoV.get_latency())
        self.la_list.append(latency)

        rewards = self.IoV.get_independent_utility(self.model_q)
        # self.team_reward = self.IoV.get_utility(fed, self.model_q, pre_q, latency)
        self.team_reward = self.IoV.get_team_reward(rewards, self.model_q, timestep)

        if timestep == c_args['timestep'] - 1:
            self.history['accuracy'].append(self.model_q)
            self.history['latency'].append(np.mean(self.la_list))
            self.update_fig(root_path)

        n_state = self.IoV.step(actions, self.model_q, fed, args['epoch'], timestep)
        n_obs = self.IoV.get_local_obs()
        return n_obs, n_state, rewards, False

    def get_team_reward(self, rewards, timestep):
        return self.team_reward

    def update_fig(self, root_path):
        cell = ['accuracy', 'latency']
        cmap = plt.cm.get_cmap('Set2')
        colors = cmap.colors
        for c, color in zip(cell, colors):
            data = self.history[c]
            plt.figure()
            plt.plot(data, color=color)
            plt.ylabel(c.title())
            plt.xlabel('Episode')
            fig_path = root_path + '/Fig/'
            plt.savefig(fig_path + f'{c}.png')
            plt.close()

    def eval_local_accu(self, testDataLoader):
        sum_accu = []
        for data, label in testDataLoader:
            data, label = data.to(self.dev), label.to(self.dev)
            preds = self.net(data)
            preds = torch.argmax(preds, dim=1)
            sum_accu.append((preds == label).float().mean().item())
        self.model_q = np.mean(sum_accu)
        print(f"Server accuracy = {self.model_q:.3f}")
