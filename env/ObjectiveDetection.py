import random
from math import sqrt

from env.DronePro import Drone
from utils.logger import logger
import numpy as np
from utils.parser import args
import matplotlib.pyplot as plt
import matplotlib as mpl
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.animation as animation


class ObjectiveDetection(object):
    def __init__(self, args, logger, num, length, width, height, risklevel, energy):
        self.args = args
        self.logger = logger
        # Map
        self.map = np.zeros((length, width))
        self.map_length = length
        self.map_width = width
        self.map_height = height
        self.obstacle_num = length * width * risklevel // (height - 1)  # 每层的障碍物数量
        self.obstacle = []
        self.start_loc = [[self.map_length // 2 - 1, self.map_width // 2 - 1, 1],
                          [self.map_length // 2 - 1, self.map_width // 2 + 1, 1],
                          [self.map_length // 2 + 1, self.map_width // 2 - 1, 1],
                          [self.map_length // 2 + 1, self.map_width // 2 + 1, 1],
                          [self.map_length // 2, self.map_width // 2, 1]]
        self.dest = [[[0, self.map_width - 1], [0, 0]], [[self.map_length - 1, 0],
                     [self.map_length - 1, self.map_width - 1]]]
        self.explored = []
        # Drone
        self.agent_num = num
        self.UAVs = []

        self.loading()

    def loading(self):
        chosed = []

        # init dest
        # while len(self.dest) < self.agent_num:
        #     j = np.random.randint(self.map_length * self.map_width // 2, self.map_length * self.map_width)
        #     if j not in chosed:
        #         self.map[j // self.map_length][j % self.map_length] = self.map_height
        #         self.dest.append([j // self.map_length, j % self.map_length])
        #         chosed.append(j)

            # init agents
        assert self.agent_num <= len(self.start_loc)
        for i in range(self.agent_num):
            loc = self.start_loc[i]
            chosed.append(loc[0] * self.map_length + loc[1])
            self.UAVs.append(
                Drone(loc.copy(), i, self.map_length, self.map_width, self.map_height, self.map,
                      self.dest[i]))


        self.obstacle.append(random.sample([[0, 1], [7, 1], [1, 4], [6, 5]], args['risk_num']))
        self.obstacle.append(random.sample([[self.map_length // 2 - 2, self.map_width // 2 + 2],
                                            [self.map_length // 2 - 2, self.map_width // 2 - 2],
                                            [self.map_length // 2 + 2, self.map_width // 2 - 2],
                                            [self.map_length // 2 + 2, self.map_width // 2 + 2]], args['risk_num']))
        # for i, obst in enumerate(self.obstacle):
        #     for pos in obst:
        #         self.map[pos[0]][pos[1]] = 1

        # log map
        map_info = f"\nMap size: {self.map_length} * {self.map_width} * {self.map_height}"
        for i in range(self.map_height - 1):
            map_info += f"\nAt height {i + 1} \n"
            for loc in self.obstacle[i]:
                map_info += str(loc) + ' '
        map_info += "is dangerous\n"
        map_info += "UAVs are at:\n"
        for i in range(self.agent_num):
            map_info += str(self.UAVs[i].pos) + ' '
        map_info += "\nDest are at:\n"
        for i in range(self.agent_num):
            map_info += str(self.UAVs[i].dest) + ' '
        self.logger.info(map_info)

    def reset(self):
        for drone, pos in zip(self.UAVs, self.start_loc):
            drone.reset(pos.copy())
        self.explored = []

    def find_teammates(self):
        drones_pos = []
        for drone in self.UAVs:
            drones_pos.append(drone.pos)
        teammates = []
        for drone in self.UAVs:
            teammate = drones_pos.copy()
            index = teammate.index(drone.pos)
            teammate[index] = None
            teammates.append(teammate)
        return teammates

    def step(self, timestep, agent, actions, root_path=None):
        n_obs = []
        rewards = []
        done = True
        # actions = random.sample(range(0, 6), self.agent_num)
        for drone, action, teammate in zip(self.UAVs, actions, self.find_teammates()):
            reward, arrived, broken = drone.move(action, timestep, teammate)
            done &= (arrived or broken)
            # after execute
            if drone.pos not in self.explored:
                self.explored.append(drone.pos)
            n_obs.append(drone.observe(agent.local_q_mode, agent.agent_num, timestep))
            rewards.append(reward)
            # rewards.append(drone.utility)
        # self.render_3d(root_path)
        return n_obs, np.array(n_obs).flatten().tolist(), rewards, done

    def get_team_reward(self, rewards, timestep):
        # 每个智能体的reward可以当作行为分，而总体任务的进程可以看作表现分
        dec = np.sum(rewards).item() - np.std(rewards).item()
        # dist = self.half_dist - np.mean([min([drone.Euclid_dist(dest) for dest in drone.dest]) for drone in self.UAVs])
        # damage = np.mean([drone.damage for drone in self.UAVs])
        done = True
        for drone in self.UAVs:
            done &= drone.arrived
        tr = dec + 150 * (1 - timestep / args['timestep']) if done else 0
        return tr

    def communicate(self, obs):
        # request
        comms = 0
        msgs = []
        for drone, ob, teammate in zip(self.UAVs, obs, self.find_teammates()):
            msg = drone.request(teammate, ob)
            comms += len(drone.connected)
            msgs.append(msg)
        # receive
        com_info = []
        for drone in self.UAVs:
            com_info.append(drone.response(msgs))

        return com_info, comms / 2

    def render(self, root_path, mode=1):
        # prepare data
        current_map = np.full((self.map_length, self.map_width), -2)
        for loc in self.explored:
            current_map[loc[0]][loc[1]] = self.map[loc[0]][loc[1]]
        for drone in self.UAVs:
            current_map[drone.pos[0]][drone.pos[1]] = -1

        # plot
        plt.ion()
        colors = ['lightgrey', 'blue', 'white', 'yellow', 'red', 'black']
        cmap1 = mpl.colors.ListedColormap(colors[2:])  # 原始地图不包含探索信息
        cmap2_color = [colors[int(i + 2)] for i in np.unique(current_map)]
        cmap2 = mpl.colors.ListedColormap(cmap2_color)
        if mode == 2:
            fig, axes = plt.subplots(1, 2)
            img1 = axes[0].imshow(self.map, cmap=cmap1)
            axes[0].invert_yaxis()
            img2 = axes[1].imshow(current_map, cmap=cmap2)
            axes[1].invert_yaxis()
            fig.colorbar(img1, cmap=cmap1, fraction=0.03, pad=0.01)
        else:
            fig, axes = plt.subplots(1, 1)
            img2 = axes.imshow(current_map, cmap=cmap2)
            axes.invert_yaxis()

        cbar = fig.colorbar(img2, cmap=cmap2, fraction=0.03, pad=0.01)
        plt.pause(0.5)  # 一帧图片展示的时间
        plt.savefig(str(root_path) + 'map.png')
        plt.close()

    def render_3d(self, root_path):
        fig = plt.figure(figsize=(8, 4))
        ax1 = fig.add_subplot(111, projection='3d')

        x = []
        y = []
        top = []
        cmap = ['green', 'lightgrey', 'red', 'cyan']
        colors = []

        for i, o in enumerate(self.obstacle):
            for pos in o:
                x.append(pos[0])
                y.append(pos[1])
                top.append(i + 1)
                colors.append(cmap[i])

        dest_top = 0.02
        for pos in self.dest:
            x.append(pos[0])
            y.append(pos[1])
            top.append(dest_top)
            colors.append(cmap[2])

        bottom = np.zeros_like(top)
        bottom[np.where(np.array(top) == dest_top)] = 1.8
        prelen = len(bottom)
        bottom = bottom.tolist()
        bottom.extend(np.zeros(self.agent_num).tolist())

        drone_top = 0.2
        for i, drone in enumerate(self.UAVs):
            x.append(drone.pos[0])
            y.append(drone.pos[1])
            top.append(drone_top)
            colors.append(cmap[3])
            bottom[prelen + i] = drone.pos[2] + 0.02

        width = depth = .8

        ax1.bar3d(x, y, bottom, width, depth, top, shade=True, color=colors, zsort='average')
        ax1.set_title('Origin')

        plt.pause(0.5)  # 一帧图片展示的时间
        plt.savefig(root_path + 'Fig/map.png')
        plt.close()

    if __name__ == '__main__':
        env = ObjectiveDetection(args, logger, 2, 10, 10, 3, 0.2, 100)
