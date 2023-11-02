from time import sleep

import numpy as np
import torch
from torch.utils.data import TensorDataset, Subset
from torch.utils.data import DataLoader
from env.FederatedUpload.getData import GetDataSet


class client(object):
    def __init__(self, trainDataSet, testDataset, dev):
        self.train_ds = trainDataSet
        self.test_ds = testDataset
        self.dev = dev
        self.train_dl = None
        self.local_parameters = None

    def localUpdate(self, localEpoch, localBatchSize, Net, lossFun, opti, global_parameters, subset_size):
        """
            param: localEpoch 当前Client的迭代次数
            param: localBatchSize 当前Client的batchsize大小
            param: Net Server共享的模型
            param: LossFun 损失函数
            param: opti 优化函数
            param: global_parmeters 当前通讯中最全局参数
            return: 返回当前Client基于自己的数据训练得到的新的模型参数
        """
        # 加载当前通信中最新全局参数
        # 传入网络模型，并加载global_parameters参数的
        Net.load_state_dict(global_parameters, strict=True)
        Net.train().to(self.dev)

        # subsetSize = len(self.train_ds) // client_num
        indices = np.random.choice(len(self.train_ds), subset_size, replace=False)
        train_ds = Subset(self.train_ds, indices)
        self.train_dl = DataLoader(train_ds, batch_size=localBatchSize, shuffle=True)

        for epoch in range(localEpoch):
            for data, label in self.train_dl:
                data, label = data.to(self.dev), label.to(self.dev)
                preds = Net(data)

                loss = lossFun(preds, label.long())
                # print("local_loss = ", loss.item())
                loss.backward()
                opti.step()
                opti.zero_grad()

        self.local_parameters = Net.state_dict()
        return Net.state_dict()

    def local_val(self, net, id, batchSize):
        net.eval().to(self.dev)
        indices = np.random.choice(len(self.test_ds), 1000, replace=False)
        test_ds = Subset(self.test_ds, indices)
        test_loader = torch.utils.data.DataLoader(test_ds, batch_size=batchSize, shuffle=False)
        with torch.no_grad():
            correct = 0
            total = 0
            for data, label in test_loader:
                data, label = data.to(self.dev), label.to(self.dev)
                outputs = net(data)
                _, predicted = torch.max(outputs.data, 1)
                total += label.size(0)
                correct += (predicted == label).sum().item()
        accu = correct / total
        print(f"Client{id}: accu={accu:.3f}")


class ClientsGroup(object):
    '''
        param: dataSetName 数据集的名称
        param: isIID 是否是IID
        param: numOfClients 客户端的数量
        param: dev 设备(GPU)
        param: clients_set 客户端

    '''

    def __init__(self, dataSetName, isIID, numOfClients, dev):
        self.data_set_name = dataSetName
        self.is_iid = isIID
        self.num_of_clients = numOfClients
        self.dev = dev
        self.clients_set = {}
        self.data_set = None

        self.test_data_loader = None
        self.dataSetBalanceAllocation()

    def refreshTestLoader(self, batchSize=32):
        indices = np.random.choice(len(self.data_set.test_set), 1000, replace=False)
        test_ds = Subset(self.data_set.test_set, indices)
        self.test_data_loader = torch.utils.data.DataLoader(test_ds, batch_size=batchSize, shuffle=False)
        return self.test_data_loader

    def dataSetBalanceAllocation(self):
        # 得到已经被重新分配的数据
        DataSet = GetDataSet(self.data_set_name, self.is_iid)
        self.data_set = DataSet

        # 加载测试数据
        # test_data = torch.tensor(mnistDataSet.test_data)
        # test_label = torch.argmax(torch.tensor(mnistDataSet.test_label).view(test_data.shape[0], -1), dim=1)
        # self.test_data_loader = DataLoader(TensorDataset(test_data, test_label), batch_size=32, shuffle=True)

        indices = np.random.choice(len(DataSet.test_set), 1000, replace=False)
        test_ds = Subset(DataSet.test_set, indices)
        self.test_data_loader = torch.utils.data.DataLoader(test_ds, batch_size=32, shuffle=False)

        for i in range(self.num_of_clients):
            self.clients_set['client{}'.format(i)] = client(DataSet.train_set, DataSet.test_set, self.dev)

        # train_data = DataSet.train_data
        # train_label = DataSet.train_label
        # for i in range(self.num_of_clients):
        #     self.clients_set['client{}'.format(i)] = client(
        #         TensorDataset(torch.tensor(train_data), torch.tensor(train_label)), DataSet.test_set, self.dev)

        # for i in range(self.num_of_clients):
        #     indices = np.random.choice(len(train_data), subset_size, replace=False)
        #     data_subset = Subset(train_data, indices)
        #     label_subset = Subset(train_label, indices)
        #     subset = TensorDataset(torch.tensor(data_subset, dtype=torch.float32),
        #                            torch.tensor(label_subset, dtype=torch.float32))
        #     self.clients_set['client{}'.format(i)] = client(
        #         subset, self.dev)
        # '''
        #     然后将其划分为200组大小为300的数据切片,然后分给每个Client两个切片
        # '''
        #
        # # 60000 /100 = 600/2 = 300
        # shard_size = mnistDataSet.train_data_size // self.num_of_clients // 2
        # print("shard_size:" + str(shard_size))
        #
        # # np.random.permutation 将序列进行随机排序
        # # np.random.permutation(60000//300=200)
        # shards_id = np.random.permutation(mnistDataSet.train_data_size // shard_size)
        # # 一共200个
        # print("*" * 100)
        # print(shards_id)
        # print(shards_id.shape)
        # print("*" * 100)
        # for i in range(self.num_of_clients):
        #     ## shards_id1
        #     ## shards_id2
        #     ## 是所有被分得的两块数据切片
        #     # 0 2 4 6...... 偶数
        #     shards_id1 = shards_id[i * 2]
        #     # 0+1 = 1 2+1 = 3 .... 奇数
        #     shards_id2 = shards_id[i * 2 + 1]
        #     #
        #     # 例如shard_id1 = 10
        #     # 10* 300 : 10*300+300
        #     # 将数据以及的标签分配给该客户端
        #     data_shards1 = train_data[shards_id1 * shard_size: shards_id1 * shard_size + shard_size]
        #     data_shards2 = train_data[shards_id2 * shard_size: shards_id2 * shard_size + shard_size]
        #     label_shards1 = train_label[shards_id1 * shard_size: shards_id1 * shard_size + shard_size]
        #     label_shards2 = train_label[shards_id2 * shard_size: shards_id2 * shard_size + shard_size]
        #
        #     # np.vstack 是按照垂直方向堆叠
        #     # np.hstack: 按水平方向（列顺序）堆叠数组构成一个新的数组
        #
        #     local_data, local_label = np.vstack((data_shards1, data_shards2)), np.vstack((label_shards1, label_shards2))
        #     print("local_label.shape:" + str(local_label.shape))
        #     local_label = np.argmax(local_label, axis=1)
        #
        #     print("local_data.shape:" + str(local_data.shape))
        #     print("local_label.shape:" + str(local_label.shape))
        #
        #     # 创建一个客户端
        #     someone = client(TensorDataset(torch.tensor(local_data), torch.tensor(local_label)), self.dev)
        #     # 为每一个clients 设置一个名字
        #     # client10
        #     self.clients_set['client{}'.format(i)] = someone


if __name__ == "__main__":
    MyClients = ClientsGroup('mnist', True, 100, 0)
    print(client)
    print(MyClients.clients_set['client10'].train_ds[0:10])
    train_ids = MyClients.clients_set['client10'].train_ds[0:10]
    i = 0
    for x_train in train_ids[0]:
        print("client10 数据:" + str(i))
        print(x_train)
        i = i + 1
    print(MyClients.clients_set['client11'].train_ds[400:500])
