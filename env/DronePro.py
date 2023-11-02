from collections import deque
from math import sqrt
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
import random
from datetime import datetime

from agent.DQN_agent import DQN as Combrain
from utils.parser import args

now = datetime.now()
dt_string = now.strftime("%Y-%m-%d_%H-%M")


class Drone(object):
    def __init__(self, pos, idn, col, row, height, env, dest):
        self.id = idn
        self.pos = pos

        self.damage = 0
        self.energy = 0

        self.arrived = False
        self.broken = False

        self.map = env
        self.map_col = col
        self.map_row = row
        self.map_height = height
        self.dest = dest

        self.utility = 0

        self.brain = Combrain(gamma=args['gamma'], lr=args['learning_rate'], INITIAL_EPSILON=0.2, FINAL_EPSILON=0.001,
                              state_num=args['com_size'], action_num=args['com_mode'], batch_size=args['batch_size'],
                              buffer_size=args['buffer_size'])
        self.com_cost = 0
        self.com_stat = np.zeros(args['com_size']).tolist()
        self.msg_s = None
        self.msg_r = None
        self.connected = []
        # 以下参数手工设定
        self.broken_limit = 100
        self.rated_energy = 100
        self.view_range = 1
        self.com_range = int(
            sqrt((self.map_col // 2) ** 2 + (self.map_row // 2) ** 2 + (self.map_height // 2) ** 2)) + 1
        self.gear = [0.0, 0.01, 0.02, 0.03]
        self.history = deque([0] * 40, maxlen=40)

    def reset(self, pos):
        self.pos = pos
        self.energy = self.rated_energy
        self.damage = 0
        self.arrived = False
        self.broken = False
        self.utility = 0
        self.connected = []
        self.msg_s = None
        self.msg_r = None
        self.com_cost = 0
        self.history.clear()
        self.history = deque([0] * 40, maxlen=40)

    def observe(self, mode, agent_num, timestep):
        # 3d的观测，考虑的是一个块，6个动作分别对应最多6个块
        obs = []
        pos = self.pos
        obs.extend(pos)
        # direction = [[0, 1, 0], [0, -1, 0], [1, 0, 0], [-1, 0, 0],
        #              [0, 0, 1], [0, 0, -1]]
        # for d in direction:
        #     dest = pos + d
        #     if not self.inmap(dest):
        #         obs.extend([dest[0], dest[1], -2])
        #     else:
        #         if pos[2] > self.map[dest[0]][dest[1]]:
        #             obs.extend([dest[0], dest[1], 0])
        #         else:
        #             obs.extend([dest[0], dest[1], -1])
        obs.extend(self.history)
        obs.extend([self.utility, self.damage, timestep])

        one_hot = [0] * agent_num
        one_hot[agent_num - (self.id + 1)] = 1
        obs.extend(one_hot)
        return obs

    def move(self, action, timestep, teammate):
        moved = False
        reward = 0
        damage = 0

        if self.energy <= 0:
            print(f'Drone{self.id} Energy Used Up!')
        elif self.damage >= self.broken_limit:
            self.broken = True
            damage = 5
            print(f'Drone{self.id} already Broken!')
        elif self.arrived:
            print(f'Drone{self.id} already arrived!')
        else:
            action += 2
            # 0-up 1-down 2-forward 3-backward 4-left 5-right
            # if action == 0:
            #     if self.pos[2] == self.map_height:
            #         damage += 0.5
            #         print(f'Drone{self.id} Reached Max Height!')
            #     else:
            #         self.pos[2] += 1
            #         moved = True
            #         reward += sqrt(self.map_row ** 2 + self.map_col ** 2) / 2 - min(
            #             [self.Euclid_dist(dest) for dest in self.dest])
            #         print(f'Drone{self.id} Move up!')
            # elif action == 1:
            #     if self.pos[2] == 1:
            #         damage += 0.5
            #         print(f'Drone{self.id} Reached Min Height!')
            #     else:
            #         self.pos[2] -= 1
            #         moved = True
            #         reward += sqrt(self.map_row ** 2 + self.map_col ** 2) / 2 - min(
            #             [self.Euclid_dist(dest) for dest in self.dest])
            #         print(f'Drone{self.id} Move down!')
            # 对于平行移动，先判断有没有到边界，再判断会不会撞到障碍物
            if action == 2:
                dest = [self.pos[0], self.pos[1] + 1, self.pos[2]]
                if dest in teammate:
                    damage += 1
                    print(f"Drone {self.id} has a collision!")
                elif self.arriveDest(dest):
                    self.arrived = True
                    self.pos = dest
                    moved = True
                    print(f'Drone{self.id} Arrived!')
                else:
                    if dest[1] >= self.map_row:
                        damage += 0.5
                        print(f'Drone{self.id} Reached North Boarder!')
                    elif dest[2] <= self.map[dest[0]][dest[1]]:
                        damage += 1
                        # print(f'Drone{self.id} Hit an Obstacle at height {self.map[dest[0]][dest[1]]}!')
                        print(f'Drone{self.id} hit an obstacle at [{dest[0]},{dest[1]}]!')
                    else:
                        self.pos = dest
                        reward += sqrt(self.map_row ** 2 + self.map_col ** 2) / 2 - min(
                            [self.Euclid_dist(dest) for dest in self.dest])
                        moved = True
                        print(f'Drone{self.id} to N!')

            elif action == 3:
                dest = [self.pos[0], self.pos[1] - 1, self.pos[2]]
                if dest in teammate:
                    damage += 1
                    print(f"Drone {self.id} has a collision!")
                elif self.arriveDest(dest):
                    self.arrived = True
                    self.pos = dest
                    moved = True
                    print(f'Drone{self.id} Arrived!')
                else:
                    if dest[1] <= 0:
                        damage += 0.5
                        print(f'Drone{self.id} Reached South Boarder!')
                    elif dest[2] <= self.map[dest[0]][dest[1]]:
                        damage += 1
                        # print(f'Drone{self.id} Hit an Obstacle at height {self.map[dest[0]][dest[1]]}!')
                        print(f'Drone{self.id} hit an obstacle at [{dest[0]},{dest[1]}]!')
                    else:
                        self.pos = dest
                        reward += sqrt(self.map_row ** 2 + self.map_col ** 2) / 2 - min(
                            [self.Euclid_dist(dest) for dest in self.dest])
                        moved = True
                        print(f'Drone{self.id} to S!')

            elif action == 4:
                dest = [self.pos[0] - 1, self.pos[1], self.pos[2]]
                if dest in teammate:
                    damage += 1
                    print(f"Drone {self.id} has a collision!")
                elif self.arriveDest(dest):
                    self.pos = dest
                    self.arrived = True
                    moved = True
                    print(f'Drone{self.id} Arrived!')
                else:
                    if dest[0] <= 0:
                        damage += 0.5
                        print(f'Drone{self.id} Reached West Boarder!')
                    elif dest[2] <= self.map[dest[0]][dest[1]]:
                        damage += 1
                        # print(f'Drone{self.id} Hit an Obstacle at height {self.map[dest[0]][dest[1]]}!')
                        print(f'Drone{self.id} hit an obstacle at [{dest[0]},{dest[1]}]!')
                    else:
                        self.pos = dest
                        reward += sqrt(self.map_row ** 2 + self.map_col ** 2) / 2 - min(
                            [self.Euclid_dist(dest) for dest in self.dest])
                        moved = True
                        print(f'Drone{self.id} to W!')

            elif action == 5:
                dest = [self.pos[0] + 1, self.pos[1], self.pos[2]]
                if dest in teammate:
                    damage += 1
                    print(f"Drone {self.id} has a collision!")
                elif self.arriveDest(dest):
                    self.arrived = True
                    self.pos = dest
                    moved = True
                    print(f'Drone{self.id} Arrived!')
                else:
                    if dest[0] >= self.map_col:
                        damage += 0.5
                        print(f'Drone{self.id} Reached East Boarder!')
                    elif dest[2] <= self.map[dest[0]][dest[1]]:
                        damage += 1
                        # print(f'Drone{self.id} Hit an Obstacle at height {self.map[dest[0]][dest[1]]}!')
                        print(f'Drone{self.id} hit an obstacle at [{dest[0]},{dest[1]}]!')
                    else:
                        self.pos = dest
                        reward += sqrt(self.map_row ** 2 + self.map_col ** 2) / 2 - min(
                            [self.Euclid_dist(dest) for dest in self.dest])
                        moved = True
                        print(f'Drone{self.id} to E!')

        cost = self.gear[self.pos[2]] if not self.arrived else 0
        utility = self.compute_utility(moved, reward, damage, cost, timestep)
        self.history.extend([self.pos[0], self.pos[1], reward, damage])
        return utility, self.arrived, self.broken

    def compute_utility(self, moved, reward, damage, cost, timestep):

        if self.arrived and not moved:
            utility = 0.5
        elif self.broken and not moved:
            utility = -0.5
        else:
            reward = 20 * (1 - timestep / args['timestep']) if self.arrived else reward
            utility = 0.1 * (reward - damage)  # - o3 * cost

        self.utility += utility
        self.damage += damage
        print(f'Utility: {utility:.2f} | Reward={reward:.2f} Damage={damage:.2f}')  # Cost={cost:.2f}')
        return utility

    def inmap(self, dest):
        inmap = True
        if dest[0] < 0 or dest[0] >= self.map_col or dest[1] < 0 or dest[1] >= self.map_row:
            inmap = False
        return inmap

    def arriveDest(self, dest):
        Arrived = False
        for r_d in self.dest:
            if [dest[0], dest[1]] == r_d:
                Arrived = True
        return Arrived

    def Euclid_dist(self, pos):
        return sqrt(
            abs(self.pos[0] - pos[0]) ** 2 + abs(self.pos[1] - pos[1]) ** 2)  # + abs(self.pos[2] - pos[2]) ** 2)

    def request(self, teammate, obs):
        self.msg_s = [obs]
        self.connected = []
        com_able = 0
        com_cost = 0
        AoI = []
        for pos in teammate:
            if pos is None:
                continue

            e_dist = self.Euclid_dist(pos)
            if e_dist < self.com_range:
                com_able += 1
                com_cost += 0.01 * e_dist
                AoI.append(e_dist)
                self.connected.append(teammate.index(pos))
            else:
                AoI.append(-1)
        self.msg_s.append(AoI)
        com_stat = obs + AoI + [self.com_cost]
        com_mode = self.brain.choose_action(self.com_stat)
        # 为了增强马尔可夫性，只要进行通信，就给予奖励，直接反映到reward中
        com_reward = com_mode + self.utility - self.com_cost
        self.brain.store_transition(self.com_stat, com_mode, com_reward, com_stat)

        self.com_cost = com_cost
        self.com_stat = com_stat
        if com_able > 0:
            print(f"Drone {self.id} can communicate with {com_able} drones in mode {com_mode}")

        local_param = self.brain.q_eval.state_dict()
        if com_mode == 2:
            self.msg_s.append(local_param)
        elif com_mode == 1:
            self.msg_s.append(None)
        else:
            self.msg_s = [None, None]

    def response(self, msgs):
        self.msg_r = []
        padding_msg = np.zeros(args['obs_size']).tolist()
        for i in range(args['agent_num']):
            if i in self.connected:
                self.msg_r.append(msgs[i])
            elif i == self.id:
                continue
            else:
                self.msg_r.append(padding_msg)
        self.msg_r = np.array(self.msg_r).flatten().tolist()


if __name__ == '__main__':
    arr1 = np.array([0, 1, 0, 2, 3, 0])
    arr2 = np.array([1, 2, 0, 0, 2, 0])
    # 将非零元素的值赋值给目标数组
    arr2[arr1 != 0] = arr1[arr1 != 0]
    print(arr2)
    drone = Drone((0, 0, 0), 0, 6, 6, 3, None, None)
    print(drone.com_range)
