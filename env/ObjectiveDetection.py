from env.DronePro import Drone
from utils.logger import logger
import numpy as np
from utils.parser import args
import matplotlib.pyplot as plt
import matplotlib as mpl
from mpl_toolkits.mplot3d import Axes3D


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
        self.start_loc = [[0, 0, 1], [0, self.map_width - 1, 1], [self.map_length - 1, 0, 1],
                          [self.map_length - 1, self.map_width - 1, 1], [self.map_length // 2, self.map_width // 2, 1]]
        self.explored = []
        self.dest = []
        # Drone
        self.agent_num = num
        self.UAVs = []

        self.loading()

    def loading(self):
        chosed = []
        # init agents
        assert self.agent_num <= len(self.start_loc)
        for i in range(self.agent_num):
            loc = self.start_loc[i]
            chosed.append(loc[0] * self.map_length + loc[1])
            self.UAVs.append(
                Drone(loc.copy(), i, self.map_length, self.map_width, self.map_height, self.map, self.dest))
            # self.map[loc[0]][loc[1]] = -1

        # init dest
        while len(self.dest) < self.agent_num:
            j = np.random.randint(self.map_length * self.map_width // 2, self.map_length * self.map_width)
            if j not in chosed:
                self.map[j // self.map_length][j % self.map_length] = self.map_height
                self.dest.append([j // self.map_length, j % self.map_length])
                chosed.append(j)

        # 初始化障碍物
        for i in range(self.map_height - 1):
            obstacle = []
            while len(obstacle) < self.obstacle_num:
                j = np.random.randint(0, self.map_length * self.map_width)
                if j not in chosed:
                    self.map[j // self.map_length][j % self.map_length] = i + 1
                    obstacle.append([j // self.map_length, j % self.map_length])
                    chosed.append(j)
            self.obstacle.append(obstacle)

        # log map
        map_info = f"\nMap size: {self.map_length} * {self.map_width} * {self.map_height} \n"
        for i in range(self.map_height - 1):
            map_info += f"At height {i + 1} \n"
            for loc in self.obstacle[i]:
                map_info += str(loc) + ' '
            map_info += "is dangerous\n"
        map_info += "UAVs are at:\n"
        for i in range(self.agent_num):
            map_info += str(self.UAVs[i].pos) + ' '
        map_info += "\nDest are at:\n"
        for i in range(self.agent_num):
            map_info += str(self.dest[i]) + ' '
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
            teammate.remove(drone.pos)
            teammates.append(teammate)
        return teammates

    def step(self, timestep, obs, agent, root_path):
        n_obs = []
        rewards = []
        done = True
        actions = agent.choose_actions(obs)
        # actions = [np.random.randint(0, 6) for _ in range(self.agent_num)]
        for drone, action, teammate in zip(self.UAVs, actions, self.find_teammates()):
            reward, online = drone.move(action, timestep, teammate)
            # after execute
            if drone.pos not in self.explored:
                self.explored.append(drone.pos)
            done = done and online
            n_obs.append(drone.observe())
            rewards.append(reward)
        self.render(root_path, 2)
        return n_obs, actions, rewards, done

    def communicate(self, timestep, obs):
        msgs = []
        costs = []
        for drone in self.UAVs:
            # msg, cost = drone.communicate(timestep)
            msg = np.zeros(args['comm_size']).tolist()
            cost = 0
            costs.append(cost)
            msgs.append(msg)
        return msgs, costs

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

    def render_3d(self):
        fig = plt.figure(figsize=(8, 4))
        ax1 = fig.add_subplot(121, projection='3d')

        # fake data
        # _x = np.arange(self.map_length)
        # _y = np.arange(self.map_width)
        _x = []
        _y = []
        top = []
        for i, obstacle in enumerate(self.obstacle):
            for pos in obstacle:
                _x.append(pos[0])
                _y.append(pos[1])
                top.append(i + 1)

        _xx, _yy = np.meshgrid(_x, _y)
        x, y = _xx.ravel(), _yy.ravel()

        # top = x + y
        bottom = np.zeros_like(top)
        width = depth = 1

        ax1.bar3d(x, y, bottom, width, depth, top, shade=True, )
        ax1.set_title('Shaded')

        plt.show()


if __name__ == '__main__':
    env = ObjectiveDetection(args, logger, 2, 10, 10, 3, 0.2, 100)
    env.render_3d()
