from collections import deque

from env.ObjectiveDetection import ObjectiveDetection
from env.FederatedUpload.FederatedUpload import FederatedUpload
import env.DronePro as Drone
from utils.parser import args
from utils.logger import logger, addFileHandler

from agent.QMIX import QMIXagent as QMIX
from agent.QMIX_DNN import QMIXagent as QMIX_DNN
from agent.VDN import VDNagent as VDN
from agent.IQL import IQLagent as IQL

import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl

plt.style.use('seaborn')
plt.rcParams['font.family'] = 'Times New Roman'
plt.rcParams['font.size'] = 15
def make_autopct(values):
    def my_autopct(pct):
        total = sum(values)
        val = int(round(pct*total/100.0))
        # 同时显示数值和占比的饼图
        return '{p:.2f}%\n({v:d})'.format(p=pct,v=val)
    return my_autopct

import torch
import torch.nn as nn
import torch.nn.functional as f
import torch.optim as optim

import os
import random
import logging
from datetime import datetime

os.environ['CUDA_LAUNCH_BLOCKING'] = '1'


# os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "max_split_size_mb:32"


def choose_env(env_info, args):
    if env_info == 'maze':
        env = ObjectiveDetection(args, logger, args['agent_num'], args['col'], args['row'], args['height'],
                                 args['risklevel'],
                                 100)
        args_to_update = {'obs_size': 46 + args['agent_num'], 'action_size': 4,
                          'state_size': (46 + args['agent_num']) * 2, 'timestep': 300,
                          'com_size': 2}
        args = {k: v if k not in args_to_update.keys() else args_to_update[k] for k, v in args.items()}
    elif env_info == 'upload':
        env = FederatedUpload()
    return env, args


def choose_agent(net_info, args):
    if net_info == 'QMIXRNN':
        agent = QMIX(agent_num=args['agent_num'], obs_size=args['obs_size'], action_size=args['action_size'],
                      state_size=args['state_size'],
                      logger=logger)
    elif net_info == 'QMIXDNN':
        agent = QMIX_DNN(agent_num=args['agent_num'], obs_size=args['obs_size'], action_size=args['action_size'],
                         state_size=args['state_size'],
                         logger=logger)
    elif net_info == 'VDN':
        agent = VDN(agent_num=args['agent_num'], obs_size=args['obs_size'], action_size=args['action_size'],
                    state_size=args['state_size'],
                    logger=logger)
    elif net_info == 'IQL':
        agent = IQL(agent_num=args['agent_num'], obs_size=args['obs_size'], action_size=args['action_size'],
                    state_size=args['state_size'],
                    logger=logger)
    return agent


if __name__ == '__main__':

    batch_info = 'TimeSeqObs'
    env_info = ['maze', 'upload'][1]
    net_info = ['QMIXRNN', 'QMIXDNN', 'VDN', 'IQL'][0]
    info = batch_info + '_' + env_info + '_' + net_info
    logger.setLevel(logging.DEBUG)
    detailed = '' + '\n'

    now = datetime.now()
    root = now.strftime("%m.%d")
    root_path = './results/' + root + '/' + info + '/'
    for suffix in ['Data', 'Fig', 'Fig/Episode/reward']:
        path = root_path + suffix + '/'
        if not os.path.exists(path):
            os.makedirs(path)
    addFileHandler(logger, root_path + now.strftime("%H-%M"))

    args_info = '\n'
    for key in args.keys():
        if key not in ['seed']:
            args_info += str(key) + ':' + str(args[key]) + '\n'
    logger.info(detailed + args_info)

    env, args = choose_env(env_info, args)
    agent = choose_agent(net_info, args)

    team_rewards = []
    independent_rewards = [[] for _ in range(args['agent_num'])]
    performances = [0] * (args['agent_num'] + 1)
    com_sum = []
    losses = deque(maxlen=int(10))
    for e in range(args['episode']):
        env.reset()

        # seed = random.randint(0, 1000)
        seed = args['seed']
        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        np.random.seed(seed)
        logger.info('Episode ' + str(e) + ' seed: ' + str(seed))

        agent.exec_hidden_state_reset()
        agent.hidden_state_reset()
        agent.decrement_epsilon(e, args['buffer_size'])


        n_obs = []
        for i in range(args['agent_num']):
            one_hot = [0] * agent.agent_num
            one_hot[agent.agent_num - (i + 1)] = 1
            n_obs.append(np.zeros(args['obs_size'] - args['agent_num'], dtype=np.double).tolist() + one_hot)
        n_com_state = []
        n_state = np.zeros(args['state_size']).flatten().tolist()

            # com_state.append(np.zeros(args['com_size'], dtype=np.double).tolist())
            # n_com_state.append(np.zeros(args['com_size'], dtype=np.double).tolist())

        episode_reward = {f'Agent{i}': [] for i in range(args['agent_num'])}
        episode_reward['Team'] = []
        for t in range(args['timestep']):
            print(f"Timestep {t}")

            obs, state = n_obs, n_state
            actions = agent.choose_actions(obs)
            n_obs, n_state, rewards, done = env.step(t, agent, actions, root_path)
            agent.store_local_transition(obs, actions, rewards, n_obs, e)

            team_reward = env.get_team_reward(rewards, t)
            agent.store_mixing_transition(state, team_reward, n_state, e)
            for k in episode_reward.keys():
                if k == 'Team':
                    episode_reward[k].append(team_reward)
                else:
                    episode_reward[k].append(rewards[int(k[-1])])

            if done:
                logger.info('Episode ' + str(e) + ' finished after ' + str(t + 1) + ' timesteps')
                break
        cmap = plt.cm.get_cmap('Set2')
        colors = cmap.colors
        # colors = ['blue', 'black', 'red', 'green', 'yellow', 'purple']
        plt.figure()
        for i, k in enumerate(episode_reward.keys()):
            if k == 'Team':
                plt.plot(episode_reward[k], label=k, color=colors[i], linewidth=0.5)
            else:
                plt.plot(episode_reward[k], label=k, color=colors[i], linewidth=1)
        plt.xlabel(f'Timestep\n(Episode{e})')
        plt.ylabel('Reward')
        plt.legend()
        plt.tight_layout()
        plt.savefig(root_path + 'Fig/Episode/reward/' + f"reward_{e % args['buffer_size']}.png")
        plt.close()

        team_rewards.append(sum(episode_reward['Team']))

        # if e >= args['buffer_size']:
        episodes = random.sample(range(e + 1), min(e + 1, args['buffer_size']))
        loss = agent.update(episodes)
        losses.append(loss)
        # padding
        maxLen = max([len(i) for i in losses])
        for loss in losses:
            if len(loss) < maxLen:
                loss += [loss[-1]] * (maxLen - len(loss))

        fig, axes = plt.subplots(1, 2, figsize=(11, 5))
        axes[0].plot(np.mean(losses, axis=0))
        axes[0].set_xlabel(f'Timestep\n(Episode {e - 10 if e - 10 > 0 else 0} ~ {e})')
        axes[0].set_ylabel('Loss')
        axes[1].plot(team_rewards, label=net_info)
        axes[1].set_xlabel('Episode')
        axes[1].set_ylabel('Reward')
        plt.tight_layout()
        plt.savefig(root_path + 'Fig/' + 'reward.png')
        plt.close()

        # for drone, i_r in zip(env.UAVs, independent_rewards):
        #     i_r.append(drone.utility)
        # plt.figure()
        # for i in range(args['agent_num']):
        #     plt.plot(independent_rewards[i], label='Agent' + str(i), color=colors[i])
        # plt.xlabel('Episode')
        # plt.ylabel('Reward')
        # plt.legend()
        # plt.savefig(root_path + 'Fig/' + 'reward_individual.png')
        # plt.close()
        #
        # arrived = 0
        # broken = 0
        # performance = 0
        # for drone in env.UAVs:
        #     if drone.arrived:
        #         arrived += 1
        #         performance += 1
        #     if drone.broken:
        #         broken += 1
        # performances[performance] += 1
        # fig1, ax1 = plt.subplots()
        # labels = [f'{args["agent_num"]}B'] + [f'{args["agent_num"] - i}B{i}A' for i in range(1, args['agent_num'])] + [
        #     f'{args["agent_num"]}A']
        # for label, data in zip(labels, performances):
        #     label += f':{data}'
        # cmap = plt.cm.get_cmap('Pastel1')
        # ax1.pie(performances, labels=labels, startangle=90, colors=cmap.colors, autopct=make_autopct(performances))
        # ax1.axis('equal')
        # ax1.set_ylabel(f'Episode {e}')
        #
        # plt.legend()
        # plt.savefig(root_path + 'Fig/' + 'performance.png')
        # logger.info('Arrived: ' + str(arrived) + ' Broken: ' + str(broken))# + ' Communication: ' + str(np.sum(comms)))
        # plt.close()

        # com_sum.append(np.sum(comms))
        # del comms[:]
        # plt.figure()
        # plt.plot(com_sum, label='Comm')
        # plt.xlabel('Episode')
        # plt.ylabel('Communication')
        # plt.legend()
        # plt.savefig(root_path + 'Fig/' + 'communication.png')
        # plt.close()


