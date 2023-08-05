import torch
import torch.nn as nn
import torch.nn.functional as F


class QMIXNet(nn.Module):
    def __init__(self, agent_num, mixing_hidden_size, state_size, hyper_hidden_size):
        super(QMIXNet, self).__init__()
        # 设置mixing network层数为2，则应有两组 w 和 b
        self.mixing_hidden_size = mixing_hidden_size
        # layer 1 for mixing network
        # QMIX原文中, 输出权重的超网络仅有一层，但为了保持单调性设置为非负，这里设置为两层
        self.hyper_w1 = nn.Sequential(nn.Linear(state_size, hyper_hidden_size),
                                      nn.ReLU(),
                                      # mixing network首层的输入为所有agent的q值(1,n), 对应的w1的形状则应该是(n, mixing_hidden_size)
                                      # nn.Linear(hyper_hidden_size, mixing_hidden_size)
                                      nn.Linear(hyper_hidden_size, agent_num * mixing_hidden_size)
                                      # 这里直接使用Relu是不是会引起梯度爆炸？暂时还是采用将输出的w1直接取绝对值的方法
                                      )
        # 为了降一点复杂度，b1就设置成1层
        self.hyper_b1 = nn.Linear(state_size, mixing_hidden_size)

        # layer 2 for mixing network
        self.hyper_w2 = nn.Sequential(nn.Linear(state_size, hyper_hidden_size),
                                      nn.ReLU(),
                                      # mixing network隐藏层的输入为所有agent的q值(1, mixing_hidden_size), 对应的w1的形状则应该是
                                      # (mixing_hidden_size, 1) 1即Q_tot
                                      nn.Linear(hyper_hidden_size, mixing_hidden_size)
                                      )
        # 最后一层的权重在论文中有明确说明
        self.hyper_b2 = nn.Sequential(nn.Linear(state_size, hyper_hidden_size),
                                      nn.ReLU(),
                                      nn.Linear(hyper_hidden_size, 1)
                                      )

    def forward(self, Qvals, states, agent_num, state_size):
        # 更改网络的形状，让其与输入的Qvals匹配，目前Qvals的形状暂时确定(batch_size, seq_len, agent_num), 以此为准，还是分transition 来学习吧
        Qvals = Qvals.view(-1, 1, agent_num)
        states = torch.Tensor(states).view(-1, state_size)
        # layer1
        w1 = torch.abs(self.hyper_w1(states)).view(-1, agent_num, self.mixing_hidden_size)
        b1 = self.hyper_b1(states).view(-1, 1, self.mixing_hidden_size)
        # layer2
        w2 = torch.abs(self.hyper_w2(states)).view(-1, self.mixing_hidden_size, 1)
        b2 = self.hyper_b2(states).view(-1, 1, 1)
        # mixing_network forward
        output = F.elu(torch.bmm(Qvals, w1) + b1)
        output = torch.bmm(output, w2) + b2
        return output.view(-1, 1)
