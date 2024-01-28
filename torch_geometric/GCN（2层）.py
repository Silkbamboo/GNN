import matplotlib
import matplotlib.pyplot as plt
import numpy as np
from sklearn.manifold import TSNE

import torch
from torch.nn import Linear
import torch.nn.functional as F

from torch_geometric.datasets import Planetoid
from torch_geometric.transforms import NormalizeFeatures

from torch_geometric.nn import GCNConv

from IPython.display import Javascript  # Restrict height of output cell.

def visualize(h, color):
    z = TSNE(n_components=2).fit_transform(h.detach().cpu().numpy())

    plt.figure(figsize=(10,10))
    plt.xticks([])
    plt.yticks([])

    plt.scatter(z[:, 0], z[:, 1], s=70, c=color, cmap="Set2")
    plt.show()


dataset = Planetoid(root='data/Planetoid', name='Cora', transform=NormalizeFeatures())

print()
print(f'Dataset: {dataset}:')
print('======================')
print(f'Number of graphs: {len(dataset)}')
print(f'Number of features: {dataset.num_features}')
print(f'Number of classes: {dataset.num_classes}')

data = dataset[0]  # Get the first graph object.

print()
print(data)
print('===========================================================================================================')

# Gather some statistics about the graph.
print(f'Number of nodes: {data.num_nodes}')
print(f'Number of edges: {data.num_edges}')
print(f'Average node degree: {data.num_edges / data.num_nodes:.2f}')
print(f'Number of training nodes: {data.train_mask.sum()}')
print(f'Training node label rate: {int(data.train_mask.sum()) / data.num_nodes:.2f}')
print(f'Has isolated nodes: {data.has_isolated_nodes()}')
print(f'Has self-loops: {data.has_self_loops()}')
print(f'Is undirected: {data.is_undirected()}')



# class GCN(torch.nn.Module):
#     def __init__(self, hidden_channels):
#         super().__init__()
#         torch.manual_seed(1234567)#随机数种子 模型中使用随机数生成器时固定随机数生成的种子，以确保结果的可重复性
#         """
#         这个种子将影响到所有基于随机性的操作,
#         例如初始化模型参数时的随机权重，数据集的随机打乱，Dropout操作中的随机丢弃等
#         设置种子后，每次运行相同的代码将产生相同的随机结果，这对于实验复现和结果的稳定性非常重要。
#         """
#         self.conv1 = GCNConv(dataset.num_features, hidden_channels)
#         self.conv2 = GCNConv(hidden_channels, dataset.num_classes)
#
#     def forward(self, x, edge_index):
#         x = self.conv1(x, edge_index)
#         x = x.relu()
#         x = F.dropout(x, p=0.5, training=self.training)
#         x = self.conv2(x, edge_index)
#         return x
#dropout随机丢弃防止模型过拟合
class GCN(torch.nn.Module):
    def __init__(self, hidden_channels):
        super().__init__()
        torch.manual_seed(1234567)

        self.conv1 = GCNConv(dataset.num_features, hidden_channels)
        self.conv2 = GCNConv(hidden_channels, hidden_channels)
        self.conv3 = GCNConv(hidden_channels, hidden_channels)
        self.conv4 = GCNConv(hidden_channels, hidden_channels)
        self.conv5 = GCNConv(hidden_channels, hidden_channels)
        self.conv6 = GCNConv(hidden_channels, hidden_channels)
        self.conv7 = GCNConv(hidden_channels, hidden_channels)
        self.conv8 = GCNConv(hidden_channels, hidden_channels)
        self.conv9 = GCNConv(hidden_channels, hidden_channels)
        self.conv10 = GCNConv(hidden_channels, dataset.num_classes)

    def forward(self, x, edge_index):
        x = self.conv1(x, edge_index)
        x = x.relu()
        x = F.dropout(x, p=0.5, training=self.training)

        x = self.conv2(x, edge_index)
        x = x.relu()
        x = F.dropout(x, p=0.5, training=self.training)

        x = self.conv3(x, edge_index)
        x = x.relu()
        x = F.dropout(x, p=0.5, training=self.training)

        x = self.conv4(x, edge_index)
        x = x.relu()
        x = F.dropout(x, p=0.5, training=self.training)

        x = self.conv5(x, edge_index)
        x = x.relu()
        x = F.dropout(x, p=0.5, training=self.training)

        x = self.conv6(x, edge_index)
        x = x.relu()
        x = F.dropout(x, p=0.5, training=self.training)

        x = self.conv7(x, edge_index)
        x = x.relu()
        x = F.dropout(x, p=0.5, training=self.training)

        x = self.conv8(x, edge_index)
        x = x.relu()
        x = F.dropout(x, p=0.5, training=self.training)

        x = self.conv9(x, edge_index)
        x = x.relu()
        x = F.dropout(x, p=0.5, training=self.training)

        x = self.conv10(x, edge_index)

        return x


model = GCN(hidden_channels=16)
optimizer = torch.optim.Adam(model.parameters(), lr=0.01, weight_decay=5e-4)
criterion = torch.nn.CrossEntropyLoss()

def train():
      model.train()
      optimizer.zero_grad()  # Clear gradients.
      out = model(data.x, data.edge_index)  # Perform a single forward pass.
      # Compute the loss solely based on the training nodes.
      loss = criterion(out[data.train_mask], data.y[data.train_mask])
      loss.backward()   # Derive gradients.
      optimizer.step()  # Update parameters based on gradients.
      return loss

def test():
      model.eval()
      out = model(data.x, data.edge_index)
      pred = out.argmax(dim=1)  # Use the class with highest probability.
      test_correct = pred[data.test_mask] == data.y[data.test_mask]  # Check against ground-truth labels.
      test_acc = (test_correct.sum()) / (data.test_mask.sum())  # Derive ratio of correct predictions.
      return test_acc

e, l, acc = [], [], []
for epoch in range(1, 501):
    loss = train()
    a = test()
    e.append(epoch)
    l.append(loss.detach().numpy())
    acc.append(a.detach().numpy())
    print(f'Epoch: {epoch:03d}, Loss: {loss:.4f}')

test_acc = test()
print(f'Test Accuracy: {test_acc:.4f}')


model.eval()

out = model(data.x, data.edge_index)
visualize(out, color=data.y)

#ACC=0.8100 epoch=1000


fig=plt.figure()
ax1=fig.add_subplot(111)
plt.plot(e, l,c=np.array([255, 71, 90]) / 255.)
plt.ylabel('Loss')

ax2=fig.add_subplot(111, sharex=ax1, frameon=False)
plt.plot(e, acc, c=np.array([79, 179, 255]) / 255.)
ax2.yaxis.tick_right()  # 开启右边的y坐标

ax2.yaxis.set_label_position("right")
plt.ylabel('ValAcc')

plt.xlabel('Epoch')
plt.title('Training Loss & Validation Accuracy')

plt.legend()

plt.show()

#acc=0.81

#epoch200 0.8030
#epoch500 0.8140