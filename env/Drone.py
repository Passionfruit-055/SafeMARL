from collections import deque
from math import sqrt

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
import random
from datetime import datetime

now = datetime.now()
dt_string = now.strftime("%Y-%m-%d_%H-%M")


class Drone(object):
    def __init__(self, x, y, height, view_range, energy, index, col, row, total, realMap, obstacle, AoI_limit,
                 report_cycle, comm_cycle):
        self.consumption = 0
        self.pos = [x, y, height]
        self.pre_pos = [x, y, height]
        self.energy = energy
        self.broken = 0
        self.view_range = view_range
        self.max_height = 3  # 设置3是为了区别有1，2两档高度的障碍物
        self.index = index
        self.col = col
        self.row = row

        self.realMap = realMap
        self.obs = np.full((col, row), -1)
        self.explored = 0

        self.shortMemSize = (1 + 2 * view_range) ** 2
        self.shortMem = deque(maxlen=self.shortMemSize)
        self.view = []

        self.movement = 0
        self.report_cycle = report_cycle
        self.comm_cycle = comm_cycle
        self.total = total  # 任务中的无人机总数
        self.partner = [[-1, -1, -1] if i != self.index else self.pos for i in range(self.total)]

        self.broken_limit = energy / 2
        self.max_trans_dist = sqrt((self.col / 2) ** 2 + (self.row / 2) ** 2 + 1 ** 2)
        self.AoI_limit = AoI_limit
        self.trans_energy_limit = self.energy / 10
        self.reward = 0  # reward三个来源：成功移动+1;更新观测每块+2;成功通信（对方接受到有用信息）, 每成功进行一次收发+1
        self.msg = None

    def reset(self, pos, energy):
        self.pos = pos
        self.msg = None
        self.reward = 0
        self.explored = 0
        self.energy = energy
        self.broken = 0
        self.obs = np.full((self.col, self.row), -1)
        self.movement = 0
        self.partner = [[-1, -1, -1] if i != self.index else self.pos for i in range(self.total)]
        self.shortMem.clear()
        self.view = []

    def move(self, action, timestep, wholemap):
        moved = False
        if self.energy <= 0:
            print(f'Drone{self.index} Energy Used Up!')
            return moved
        elif self.broken >= self.broken_limit:
            print(f'Drone{self.index} Broken!')
            return moved
        self.movement += 1
        # 这里假设所有无人机都面向同方向，不涉及到转换坐标系问题
        # 0-up 1-down 2-forward 3-backward 4-left 5-right
        broken = self.broken
        consumption = 0
        self.reward = 0

        if action == 0:
            if self.pos[2] == self.max_height:
                self.broken += 1
                print(f'Drone{self.index} Reached Max Height!')
            else:
                self.pos[2] += 1
                self.reward += 1
                consumption += 1
                moved = True
                print(f'Drone{self.index} Move up!')
        elif action == 1:
            if self.pos[2] == 1:
                self.broken += 1
                print(f'Drone{self.index} Reached Min Height!')
            else:
                self.pos[2] -= 1
                self.reward += 1
                consumption += 1
                moved = True
                print(f'Drone{self.index} Move down!')
        # 对于平行移动，先判断有没有到边界，再判断会不会撞到障碍物
        elif action == 2:
            dest = (self.pos[0], self.pos[1] + 1, self.pos[2])
            # 无人机下一步移动的目标位置信息，在观测阶段应该已经得到
            if not self.inmap(dest):
                print(f"Drone{self.index} Destination out of Map!")
            elif list(dest) in self.partner:
                print("Drone{self.index} Destination Occupied!")
            elif self.obs[dest[0]][dest[1]] == -1:
                print(f"Drone{self.index} Map Error!")
            else:
                if dest[1] >= self.row:
                    self.broken += 0.5
                    print(f'Drone{self.index} Reached up Boundary!')
                elif dest[2] <= self.obs[dest[0]][dest[1]]:
                    self.broken += 1
                    print(f'Drone{self.index} Hit an Obstacle at height {self.obs[dest[0]][dest[1]]}!')
                else:
                    self.pos[1] += 1
                    self.reward += 1
                    consumption += 1
                    moved = True
                    print(f'Drone{self.index} Move forward!')

        elif action == 3:
            dest = (self.pos[0], self.pos[1] - 1, self.pos[2])
            if not self.inmap(dest):
                print(f"Drone{self.index} Destination out of Map!")
            elif list(dest) in self.partner:
                print("Drone{self.index} Destination Occupied!")
            elif self.obs[dest[0]][dest[1]] == -1:
                print(f"Drone{self.index} Map Error!")
            else:
                if dest[1] <= 0:
                    self.broken += 0.5
                    print(f'Drone{self.index} Reached bottom Boundary!')
                elif dest[2] <= self.obs[dest[0]][dest[1]]:
                    self.broken += 1
                    print(f'Drone{self.index} Hit an Obstacle at height {self.obs[dest[0]][dest[1]]}!')
                else:
                    self.pos[1] -= 1
                    self.reward += 1
                    consumption += 1
                    moved = True
                    print(f'Drone{self.index} Move backward!')

        elif action == 4:
            dest = (self.pos[0] - 1, self.pos[1], self.pos[2])
            if not self.inmap(dest):
                print(f"Drone{self.index} Destination out of Map!")
            elif list(dest) in self.partner:
                print("Drone{self.index} Destination Occupied!")
            elif self.obs[dest[0]][dest[1]] == -1:
                print(f"Drone{self.index} Map Error!")
            else:
                if dest[0] <= 0:
                    self.broken += 0.5
                    print(f'Drone{self.index} Reached left Boundary!')
                elif dest[2] <= self.obs[dest[0]][dest[1]]:
                    self.broken += 1
                    print(f'Drone{self.index} Hit an Obstacle at height {self.obs[dest[0]][dest[1]]}!')
                else:
                    self.pos[0] -= 1
                    self.reward += 1
                    consumption += 1
                    moved = True
                    print(f'Drone{self.index} Move Left!')

        elif action == 5:
            dest = (self.pos[0] + 1, self.pos[1], self.pos[2])
            if not self.inmap(dest):
                print(f"Drone{self.index} Destination out of Map!")
            elif list(dest) in self.partner:
                print("Drone{self.index} Destination Occupied!")
            elif self.obs[dest[0]][dest[1]] == -1:
                print(f"Drone{self.index} Map Error!")
            else:
                if dest[0] >= self.col:
                    self.broken += 0.5
                    print(f'Drone{self.index} Reached right Boundary!')
                elif dest[2] <= self.obs[dest[0]][dest[1]]:
                    self.broken += 1
                    print(f'Drone{self.index} Hit an Obstacle at height {self.obs[dest[0]][dest[1]]}!')
                else:
                    self.pos[0] += 1
                    self.reward += 1
                    consumption += 1
                    moved = True
                    print(f'Drone{self.index} Move Right!')
        # 不同的飞行高度有不同的能耗
        consumption *= self.pos[2]
        self.update_local_obs()

        if timestep % self.report_cycle == 0:
            self.update_whole_map(wholemap)
            consumption += 1

        self.consumption = consumption

        self.energy = self.energy - consumption - broken

        print(f'consumption = {consumption} energy = {self.energy} broken = {self.broken}')
        return moved

    def update_local_obs(self):
        self.view = []
        # self.view_range = 1 if self.pos[2] < self.max_height else 2
        pos = self.pos
        for i in range(pos[0] - self.view_range, pos[0] + self.view_range + 1):
            for j in range(pos[1] - self.view_range, pos[1] + self.view_range + 1):
                if 0 <= i < self.col and 0 <= j < self.row:
                    # 每探索到一个新块，获得探索奖励
                    if self.obs[i][j] == -1:
                        self.reward += 3
                        # self.explored += 1
                    self.obs[i][j] = self.realMap[i][j]
                    self.shortMem.append((i, j, self.obs[i][j]))
                    self.view.append((i, j, self.obs[i][j]))
                else:
                    self.view.append((i, j, -1))

    def communicate(self, timestep, msgs):
        senders = []
        # 通信共有五种情况： 0 = 消息来自自身 1 = 成功通信并获得新消息  2 = 成功通信但未获得新消息  3 = 因距离太远未能成功通信  4 = 因信息年龄选择不接收
        msg_stat = []
        for msg in msgs:
            index = msg['index']
            if index == self.index:
                msg_stat.append(0)
            else:
                sender_loc = self.partner[index] = msg['loc']
                dist = sqrt((self.pos[0] - sender_loc[0]) ** 2 + (self.pos[1] - sender_loc[1]) ** 2 + (
                        self.pos[2] - sender_loc[2]) ** 2)
                AoI = timestep - msg['timestamp']
                if dist > self.max_trans_dist:
                    msg_stat.append(3)
                elif AoI > self.AoI_limit:
                    msg_stat.append(4)
                else:
                    self.partner[index] = msg['loc']
                    update = False
                    for loc in msg['data']:
                        if self.obs[loc[0]][loc[1]] == -1:
                            update = True
                            self.obs[loc[0]][loc[1]] = loc[2]
                    # 只要成功通信，无论是否有新消息，通信次数都要增加
                    senders.append(index)
                    if not update:
                        msg_stat.append(1)
                        # print(f'Drone{index} connected but no new information!')
                    else:
                        msg_stat.append(2)
                        # print(f'Drone{index} provides new information!')
        return senders

    def update_whole_map(self, sys_map):
        """
        聚合地图更新有两种方式
        第一是更新的map是时隙开始时的map，这样做的的缺点是可能会有重复更新
        第二是更新的map是最新更新的map，这样做意味着序号在前面的无人机会有更多的探索奖励
        这里采用第一种方式，尽管牺牲了部分效率，但是保证了智能体间的公平性
        """
        # 上一时刻无人机位置消除
        for i in range(self.col):
            for j in range(self.row):
                if sys_map[i][j] != self.obs[i][j] and self.obs[i][j] != -1:
                    # print(f"update ({i}, {j}, {self.obs[i][j]})")
                    sys_map[i][j] = self.obs[i][j]
                    self.reward += 2

    def send_message(self, timestep):
        """
        无人机可能会选择在以下几个情况下不上传：
        1. 能耗
        2. 损毁程度
        3. 隐私保护
        第三部分应该也是要和时延结合的，比如我对数据做一次本地差分隐私，但是会有能耗消耗
        如果考虑了数据扰动，意味着上传的数据很可能是有误差的，因此无人机之间可能还存在着信誉值问题
        """
        if self.energy >= self.trans_energy_limit and self.broken < self.broken_limit and timestep % self.comm_cycle == 0:
            self.msg = {'index': self.index, 'timestamp': timestep, 'data': self.shortMem, 'loc': self.pos, 'LDP': True}
            self.consumption += 1
            # print('Msg send')
            return self.msg
        else:
            # print(f'Drone{self.index} Choose not Send Message!')
            return None

    def inmap(self, dest):
        inmap = True
        if dest[0] < 0 or dest[0] >= self.col or dest[1] < 0 or dest[1] >= self.row:
            inmap = False
        return inmap

    # 以下的方法与环境无关，用于RL
    def get_reward(self):
        reward = self.reward
        return reward, self.broken

    def get_state(self):
        # state包括 探索到的总块数，位置（所处位置，周围地图信息），能量，损坏程度，时刻奖励（反映探索，通信，移动）
        # size = 1 * 4 + self.shortMemSize * 3 + last action = 32
        state = [self.explored, self.energy, self.broken, self.reward]
        for v in self.view:
            state.extend(v)
        return state


if __name__ == '__main__':
    arr1 = np.array([0, 1, 0, 2, 3, 0])
    arr2 = np.array([1, 2, 0, 0, 2, 0])
    # 将非零元素的值赋值给目标数组
    arr2[arr1 != 0] = arr1[arr1 != 0]
    print(arr2)
