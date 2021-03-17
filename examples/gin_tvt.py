import os.path as osp

import torch
import torch.nn.functional as F
import torch.optim
from torch.nn import Sequential, Linear, ReLU
from torch_geometric.datasets import TUDataset
from torch_geometric.data import DataLoader
from torch_geometric.nn import GINConv, global_add_pool
import json

path = osp.join(osp.dirname(osp.realpath(__file__)), '..', 'data', 'TU')
dataset = TUDataset(path, name='MUTAG')
idxs = json.load(open('test.json', 'r'))
dataset = dataset[idxs]
fold = len(dataset) // 10
train_dataset = dataset[:fold * 8]
val_dataset = dataset[fold * 8:fold * 9]
test_dataset = dataset[fold * 9:]
test_loader = DataLoader(test_dataset, batch_size=128)
val_loader = DataLoader(val_dataset, batch_size=128)
train_loader = DataLoader(train_dataset, batch_size=128)

class Net(torch.nn.Module):
    def __init__(self):
        super(Net, self).__init__()

        num_features = dataset.num_features
        dim = 32

        nn1 = Sequential(Linear(num_features, dim), ReLU(), Linear(dim, dim))
        self.conv1 = GINConv(nn1)
        self.bn1 = torch.nn.BatchNorm1d(dim)

        nn2 = Sequential(Linear(dim, dim), ReLU(), Linear(dim, dim))
        self.conv2 = GINConv(nn2)
        self.bn2 = torch.nn.BatchNorm1d(dim)

        nn3 = Sequential(Linear(dim, dim), ReLU(), Linear(dim, dim))
        self.conv3 = GINConv(nn3)
        self.bn3 = torch.nn.BatchNorm1d(dim)

        nn4 = Sequential(Linear(dim, dim), ReLU(), Linear(dim, dim))
        self.conv4 = GINConv(nn4)
        self.bn4 = torch.nn.BatchNorm1d(dim)

        nn5 = Sequential(Linear(dim, dim), ReLU(), Linear(dim, dim))
        self.conv5 = GINConv(nn5)
        self.bn5 = torch.nn.BatchNorm1d(dim)

        self.fc1 = Linear(dim, dim)
        self.fc2 = Linear(dim, dataset.num_classes)

    def forward(self, x, edge_index, batch):
        x = F.relu(self.conv1(x, edge_index))
        x = self.bn1(x)
        x = F.relu(self.conv2(x, edge_index))
        x = self.bn2(x)
        x = F.relu(self.conv3(x, edge_index))
        x = self.bn3(x)
        x = F.relu(self.conv4(x, edge_index))
        x = self.bn4(x)
        x = F.relu(self.conv5(x, edge_index))
        x = self.bn5(x)
        x = global_add_pool(x, batch)
        x = F.relu(self.fc1(x))
        x = F.dropout(x, p=0.5, training=self.training)
        x = self.fc2(x)
        return F.log_softmax(x, dim=-1)


device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

def train(epoch):
    model.train()

    if epoch == 51:
        for param_group in optimizer.param_groups:
            param_group['lr'] = 0.5 * param_group['lr']

    loss_all = 0
    for data in train_loader:
        data = data.to(device)
        optimizer.zero_grad()
        output = model(data.x, data.edge_index, data.batch)
        loss = F.nll_loss(output, data.y)
        loss.backward()
        loss_all += loss.item() * data.num_graphs
        optimizer.step()
    return loss_all / len(train_dataset)


def test(loader):
    model.eval()

    correct = 0
    for data in loader:
        data = data.to(device)
        output = model(data.x, data.edge_index, data.batch)
        pred = output.max(dim=1)[1]
        correct += pred.eq(data.y).sum().item()
    return correct / len(loader.dataset)

tacc = []
from tqdm import tqdm
import numpy as np

with tqdm(range(100)) as t:
    for _ in t:
        model = Net().to(device)
        optimizer = torch.optim.Adam(model.parameters(), lr=0.01)

        val_acc = -1
        test_acc = 0

        for epoch in range(1, 101):
            train_loss = train(epoch)
            val_acc_now = test(val_loader)
            if val_acc < val_acc_now:
                test_acc = test(test_loader)
            #print('Epoch: {:03d}, Train Loss: {:.7f}, '
            #      'Train Acc: {:.7f}, Test Acc: {:.7f}'.format(epoch, train_loss,
            #                                                   train_acc, test_acc))
        tacc.append(test_acc)
        t.set_postfix(now=test_acc, mean=np.mean(tacc), std=0 if len(tacc) == 1 else np.std(tacc))

print(np.mean(tacc), np.std(tacc))