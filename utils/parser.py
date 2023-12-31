import argparse
import os
import random

parser = argparse.ArgumentParser()
# more constant
parser.add_argument('-lr', '--learning_rate', type=float, default=1e-3, help='learning rate for training')
parser.add_argument('-gamma', '--gamma', type=float, default=0.8, help='discount factor')
parser.add_argument('-tau', '--tau', type=float, default=0.01, help='soft update parameter')
parser.add_argument('-hsr', '--hsRNN', type=int, default=64, help='hidden size for RNN')
parser.add_argument('-hsm', '--hsMixing', type=int, default=64, help='hidden size for mixing net')
parser.add_argument('-hsh', '--hsHyper', type=int, default=64, help='hidden size for hyper net')
parser.add_argument('-re_cycle', '--replace_cycle', type=int, default=500, help='replace cycle for target net')
# agent
parser.add_argument('-an', '--agent_num', type=int, default=2, help='number of agents')
parser.add_argument('-obs', '--obs_size', type=int, default=15, help='observation size for each agent') # 21
parser.add_argument('-msgs', '--msg_size', type=int, default=27, help='message size of one agent')
parser.add_argument('-comm', '--com_size', type=int, default=32, help='communication state size for each agent')
parser.add_argument('-cm', '--com_mode', type=int, default=3, help='communication mode for each agent')
parser.add_argument('-state', '--state_size', type=int, default=7, help='state size for whole system') # 21*2
parser.add_argument('-action', '--action_size', type=int, default=6, help='action size for whole system')  #6
# train
parser.add_argument('-e', '--epoch', type=int, default=1, help='number of epochs to train for')
parser.add_argument('-buffer', '--buffer_size', type=int, default=32, help='number of max buffer size')
parser.add_argument('-batch', '--batch_size', type=int, default=32, help='batch size for training')
parser.add_argument('-seq', '--seq_len', type=int, default=100, help='sequence length for training')
parser.add_argument('-ep', '--episode', type=int, default=3000, help='number of episodes to train for')
parser.add_argument('-ts', '--timestep', type=int, default=300, help='number of timesteps for each episode')
parser.add_argument('-seed', '--seed', type=int, default=random.randint(0, 1000), help='random seed')
# env
parser.add_argument('-vr', '--view_range', type=int, default=1, help='view range for each agent')
parser.add_argument('-rcy', '--report_cycle', type=int, default=1, help='report cycle for training')
parser.add_argument('-ccy', '--comm_cycle', type=int, default=1, help='communication cycle for training')
parser.add_argument('-row', '--row', type=int, default=8, help='number of rows for map')
parser.add_argument('-col', '--col', type=int, default=8, help='number of columns for map')
parser.add_argument('-ht', '--height', type=int, default=3, help='UAV max height')
parser.add_argument('-risk', '--risklevel', type=float, default=0.1, help='obstacle density')
parser.add_argument('-rn', '--risk_num', type=float, default=3, help='obstacles num for one level in maze env')
parser.add_argument('-al', '--AoI_limit', type=int, default=3, help='AoI limit')
args = parser.parse_args()
args = args.__dict__
