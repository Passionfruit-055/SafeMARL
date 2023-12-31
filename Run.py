from collections import deque
from time import sleep

from env.Drone import Drone
import numpy as np
from utils.parser import args
from utils.logger import logger
import matplotlib.pyplot as plt
import matplotlib as mpl
from agent.QMIX import QMIXagent
import torch
import random
import pdb
import os
from datetime import datetime

now = datetime.now()
rq = now.strftime("%m.%d")
batchn = now.strftime("%H-%M")
path = 'results/' + rq + '/' + batchn + '/'
if not os.path.exists(path):
    os.makedirs(path)

info = 'use more than one episode'
with open(path + 'args.txt', 'w') as f:
    f.write(info + '\n\n')
    for key in args.keys():
        f.write(str(key) + ':' + str(args[key]) + '\n')

os.environ['CUDA_LAUNCH_BLOCKING'] = '1'
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "max_split_size_mb:32"

seed = args['seed']
seed = random.randint(0, 1000)
torch.manual_seed(seed)
torch.cuda.manual_seed_all(seed)
np.random.seed(seed)

# map
map_row = args['row']
map_col = args['col']
real_map = np.full((map_row, map_col), 0)
obs_map = np.full((map_row, map_col), -1)
loc = [(0, 0), (map_row - 1, map_col - 1), (0, map_col - 1), (map_row - 1, 0), (map_row // 2, map_col // 2)][
      0: args['agent_num']]
for l in loc:
    obs_map[l[0]][l[1]] = -2
# obstacle
risk_level = args['risklevel']
low_obstacle = []
for i in range(int(map_row * map_col * (risk_level / 2))):
    while True:
        x = np.random.randint(0, map_row)
        y = np.random.randint(0, map_col)
        if real_map[x][y] == 0 and (x, y) not in loc:
            break
    low_obstacle.append((x, y))
    real_map[x][y] = 1
high_obstacle = []
for i in range(int(map_row * map_col * (risk_level / 2))):
    while True:
        x = np.random.randint(0, map_row)
        y = np.random.randint(0, map_col)
        if real_map[x][y] == 0 and (x, y) not in loc:
            break
    high_obstacle.append((x, y))
    real_map[x][y] = 2
obstacle_num = len(low_obstacle) + len(high_obstacle)

plt.ion()
colors = ['blue', 'lightgrey', 'white', 'gold', 'red']

# drone
agent_num = args['agent_num']
energy = args['timestep'] * 4  # 3 = 行动 + 通信
AoI_limit = args['AoI_limit']
report_cycle = args['report_cycle']
comm_cycle = args['comm_cycle']
Action = ['up', 'down', 'forward', 'backward', 'left', 'right']
Drones = [Drone(loc[i][0], loc[i][1], 1, args['view_range'], energy, i, map_col, map_row, agent_num, real_map,
                obstacle_num, AoI_limit, report_cycle, comm_cycle) for i in range(agent_num)]
# agent
agent = QMIXagent(args['agent_num'], args['state_size'], args['action_size'], args['obs_size'])
# mission setting
episodes = args['episode']
timesteps = args['timestep']
epochs = args['epoch']
seqLength = args['seq_len']

reward_all = []
explorations = []
loss_all = deque(maxlen=int(1e4))

for episode in range(episodes):
    print(f"Episode {episode + 1}")
    rewards = []
    msgs = deque(maxlen=agent_num)
    phase_reward = {'reward': [i * 10 for i in range(10)], 'phase': [False for i in range(10)]}
    explored_now = 0
    finalTimestep = args['timestep']
    terminated = False
    # reset map
    obs_map = np.full((map_row, map_col), -1)
    for l in loc:
        obs_map[l[0]][l[1]] = -2
    # draw map
    fig, ax = plt.subplots(1, 2)
    fig.canvas.manager.window.setWindowTitle(f"epi{episodes}, t{timesteps}, epo{epochs}, seq{seqLength}")
    real_num = len(np.unique(real_map))
    current_num = len(np.unique(obs_map))
    cmap_whole = mpl.colors.ListedColormap(colors[2: 2 + real_num])
    cmap_dynamic = mpl.colors.ListedColormap(colors[0: current_num])
    ax1 = ax[0].imshow(real_map, cmap=cmap_whole, origin='lower')
    ax2 = ax[1].imshow(obs_map, cmap=cmap_dynamic, origin='lower')
    plt.colorbar(ax1, cax=None, ax=None, shrink=0.5)
    plt.colorbar(ax2, cax=None, ax=None, shrink=0.5)
    plt.pause(1)
    for l in loc:
        obs_map[l[0]][l[1]] = real_map[l[0]][l[1]]

    for i, drone in enumerate(Drones):
        drone.reset([loc[i][0], loc[i][1], 1], energy)

    agent.exec_hidden_state_reset()
    agent.hidden_state_reset(1)
    # 初始的action如何设置不确定，暂时设定为随机
    actions = [np.random.randint(0, 6) for i in range(agent_num)]
    for timestep in range(timesteps):
        print("\nTime", timestep)
        if timestep == 0:
            state = [0, 0]
            obs = []
            for drone, action in zip(Drones, actions):
                d_obs = drone.get_state()
                d_obs.append(action)
                obs.append(d_obs)
                state.extend(d_obs)
                state.extend(drone.pos)
        else:
            state = next_state
            obs = next_obs
        # state = np.zeros((args['state_size'])).tolist() if timestep == 0 else next_state
        # obs = np.zeros((agent_num, args['obs_size'])).tolist() if timestep == 0 else next_obs
        # QMIX
        actions= agent.choose_actions(obs, episode)
        # print(f'Action = {np.array(Action)[np.array(actions)]}')
        # move and send
        temp_maps = [obs_map.copy()] * agent_num
        onboard = 0
        for i, temp_map in enumerate(temp_maps):
            # move
            onboard += 1 if Drones[i].move(actions[i], timestep, temp_map) else 0
            # send
            msg = Drones[i].send_message(timestep)
            if msg is not None:
                msgs.append(msg)
        # 自检
        if onboard == 0:
            broken_num = 0
            energy_usedup = 0
            for drone in Drones:
                if drone.broken >= drone.broken_limit:
                    broken_num += 1
                if drone.energy <= 0:
                    energy_usedup += 1
            if broken_num == agent_num:
                print("All drones broken")
                terminated = True
            elif energy_usedup == agent_num:
                print("All drones used up energy")
                terminated = True

        # update map
        for temp_map in temp_maps:
            obs_map[temp_map != -1] = temp_map[temp_map != -1]

        comm = 0
        for i in range(agent_num):
            senders = Drones[i].communicate(timestep, msgs)
            for sender in senders:
                Drones[sender].reward += 1
                comm += 1

        # get reward and mark drones location
        rs = []
        brokens = []
        remain_energys = []
        drone_pos = []
        for drone in Drones:
            r, br = drone.get_reward()
            rs.append(r)
            pos = drone.pos
            drone_pos.append(pos)
            obs_map[pos[0], pos[1]] = -2

            remain_energys.append(drone.energy)
            brokens.append(br)
        remain_energy = sum(remain_energys) / agent_num
        broken_level = sum(brokens) / agent_num
        # evaluate exploration rate
        unique, count = np.unique(obs_map, return_counts=True)
        explored = 0
        for u, c in zip(unique, count):
            if u == -1:
                explored_pre = explored_now
                explored_now = 1 - c / (map_col * map_row)
                explored = explored_now - explored_pre
                break
        print(f"Exploration = {explored_now}, increment = {explored}")
        # plot
        cmap_dynamic = mpl.colors.ListedColormap(colors[0: len(unique)])
        plt.imshow(obs_map, cmap=cmap_dynamic, origin='lower')
        plt.colorbar()
        plt.pause(0.01)  # pause 这个可以认为是一帧图片展示的时间
        plt.clf() if timestep != timesteps - 1 else None
        # compute reward
        reward = remain_energy / energy * 10 + explored_now * 10 + np.mean(rs) - broken_level*10  # + comm - var_r
        print(f"Reward = {reward}, remain_energy = {remain_energy / energy}, explored = {explored_now*10}, r = {np.mean(rs)}, broken_level = {broken_level*10}")
        milestone = explored_now / 0.1
        if phase_reward['phase'][int(milestone)] is False:
            reward += phase_reward['reward'][int(milestone)]
            phase_reward['phase'][int(milestone)] = True
        rewards.append(reward)

        # state_sum = 各个智能体状态 + 最新探索度 + 本轮成功通信次数 + 各个智能体位置 = 1 + 1 + (32+3)*5 = 177
        next_state = [comm, explored]
        next_obs = []
        for drone, pos, action in zip(Drones, drone_pos, actions):
            d_obs = drone.get_state()
            d_obs.append(action)
            next_state.extend(d_obs)
            next_state.extend(pos)
            next_obs.append(d_obs)
            # delete mark , 这里可以直接赋真实地图中的值，因为无人机经过的地方一定已经被观测过了
            obs_map[pos[0], pos[1]] = real_map[pos[0], pos[1]]
        # store
        # obs = np.array(obs).reshape((agent_num, args['obs_size'])).tolist()
        # next_obs = np.array(next_obs).reshape((agent_num, args['obs_size'])).tolist()
        for d in range(agent_num):
            agent.store_transition(obs[d], actions[d], next_obs[d], episode, d)
        agent.store_mixing_transition(state, reward, next_state, episode)
        # learn
        losses = agent.learn(episode, timestep)
        if losses is not None:
            loss_all.extend(losses)
        # check mission success
        if explored_now >= 0.9:
            terminated = True

        if terminated:
            finalTimestep = timestep
            break

    plt.close()
    plt.plot(rewards)
    plt.pause(1)
    if episode % 3 == 0:
        plt.savefig(path + f"reward{episode}.png", format='png')
        plt.savefig(path + f"reward{episode}.pdf", format='pdf')
    plt.clf()
    plt.plot(loss_all)
    if episode % 3 == 0:
        plt.savefig(path + f"loss.png", format='png')
        plt.savefig(path + f"loss.pdf", format='pdf')
    plt.pause(1)
    plt.close()
    reward_all.append(rewards[-1])
    explorations.append(explored_now)
    plt.plot(reward_all)
    if episode % 3 == 0:
        plt.savefig(path + "reward_all.png", format='png')
        plt.savefig(path + "reward_all.pdf", format='pdf')
    plt.pause(1)
    plt.close()
    plt.plot(explorations)
    if episode % 5 == 0:
        plt.savefig(path + "explore.png", format='png')
        plt.savefig(path + "explore.pdf", format='pdf')
    plt.pause(1)
    plt.close()

# 这两个则是更高层的控件
plt.close()
plt.plot(reward_all)
plt.savefig(path + "reward_all.png", format='png')
plt.savefig(path + "reward_all.pdf", format='pdf')
plt.close()
plt.plot(explorations)
plt.savefig(path + "explore.png", format='png')
plt.savefig(path + "explore.pdf", format='pdf')
plt.ioff()
plt.show()
