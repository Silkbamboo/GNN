import ssl
import argparse
import os.path as osp
import numpy as np
import torch
import torch.nn.functional as F
from sklearn.manifold import TSNE
import torch_geometric.transforms as T
from matplotlib import pyplot as plt
from torch_geometric.datasets import Planetoid
from torch_geometric.logging import init_wandb, log
from torch_geometric.nn import GATConv


parser = argparse.ArgumentParser()
parser.add_argument('--dataset', type=str, default='Cora')
parser.add_argument('--hidden_channels', type=int, default=8)
parser.add_argument('--heads', type=int, default=8)
parser.add_argument('--lr', type=float, default=0.005)
parser.add_argument('--epochs', type=int, default=200)
parser.add_argument('--wandb', action='store_true', help='Track experiment')
args = parser.parse_args()

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
init_wandb(name=f'GAT-{args.dataset}', heads=args.heads, epochs=args.epochs,
           hidden_channels=args.hidden_channels, lr=args.lr, device=device)



path = osp.join(osp.dirname(osp.realpath(__file__)), '..', 'data', 'Planetoid')
dataset = Planetoid(path, args.dataset, transform=T.NormalizeFeatures())
data = dataset[0].to(device)


class GAT(torch.nn.Module):
    def __init__(self, in_channels, hidden_channels, out_channels, heads):
        super().__init__()
        self.conv1 = GATConv(in_channels, hidden_channels, heads, dropout=0.6)# out==8*8
        # On the Pubmed dataset, use `heads` output heads in `conv2`.
        self.conv2 = GATConv(hidden_channels * heads, out_channels, heads=1,
                             concat=False, dropout=0.6)

    def forward(self, x, edge_index):
        x = F.dropout(x, p=0.6, training=self.training)#(2708, 1433)
        x = F.elu(self.conv1(x, edge_index)) #(2708, 64) 64=8*8，8是因为 head==8注意力；
        x = F.dropout(x, p=0.6, training=self.training)#(2708, 64)
        x = self.conv2(x, edge_index)#(2708, 7) 总共7类；
        return x
#使用了heads注意力头

model = GAT(dataset.num_features, args.hidden_channels, dataset.num_classes,
            args.heads).to(device)
optimizer = torch.optim.Adam(model.parameters(), lr=0.05, weight_decay=5e-4)


def train():
    model.train()
    optimizer.zero_grad()
    out = model(data.x, data.edge_index)
    loss = F.cross_entropy(out[data.train_mask], data.y[data.train_mask])
    loss.backward()
    optimizer.step()
    return float(loss)


@torch.no_grad()
def test():
    model.eval()
    pred = model(data.x, data.edge_index).argmax(dim=-1)

    accs = []
    for mask in [data.train_mask, data.val_mask, data.test_mask]:
        accs.append(((pred[mask] == data.y[mask]).sum()) / (mask.sum()))
    return accs

# 创建空列表以存储损失和验证准确度
losses = []
val_accs = []

best_val_acc = final_test_acc = 0
for epoch in range(1, args.epochs + 1):
    loss = train()
    train_acc, val_acc, tmp_test_acc = test()
    if val_acc > best_val_acc:
        best_val_acc = val_acc
        test_acc = tmp_test_acc
    log(Epoch=epoch, Loss=loss, Train=train_acc, Val=val_acc, Test=test_acc)
    # 保存损失和验证准确度
    losses.append(loss)
    val_accs.append(val_acc)

    log(Epoch=epoch, Loss=loss, Train=train_acc, Val=val_acc, Test=test_acc)

test_acc=test()
print(f'Validation Accuracy: {test_acc[1]:.4f}')
#test_acc 是一个包含三个元素的列表，分别对应训练集、验证集和测试集的准确度。
#验证集的准确度通常是列表中的第二个元素

def visualize(h, color):
    z = TSNE(n_components=2).fit_transform(h)

    plt.figure(figsize=(10, 10))
    plt.scatter(z[:, 0], z[:, 1], s=70, c=color, cmap="Set2")
    plt.show()

out = model(data.x, data.edge_index).cpu().detach().numpy()
visualize(out, color=data.y.cpu().numpy())




# plt.figure(figsize=(12, 6))
#
# # 绘制损失曲线（红色）
# plt.plot(range(1, args.epochs + 1), losses, label='Loss', color='red', marker='o')
# plt.xlabel('Epoch')
# plt.ylabel('Loss')
# plt.legend()
#
# # 在同一个子图上绘制验证准确度曲线（蓝色）
# plt.plot(range(1, args.epochs + 1), [acc.cpu().numpy() for acc in val_accs], label='Validation Accuracy', color='blue', marker='o')
# plt.ylabel('Validation Accuracy')
# plt.legend()
#
# plt.show()

# 创建平滑窗口大小
smooth_window = 10

# 对损失进行移动平均
smoothed_losses = [np.mean(losses[i:i+smooth_window]) for i in range(len(losses) - smooth_window + 1)]

# 对验证准确度进行移动平均
smoothed_val_accs = [torch.mean(torch.tensor(val_accs[i:i+smooth_window]).cpu()).item() for i in range(len(val_accs) - smooth_window + 1)]


# plt.figure(figsize=(12, 6))
# plt.plot(range(1, args.epochs + 1 - smooth_window + 1), smoothed_losses, label='Loss', color='red', marker='o')
# plt.plot(range(1, args.epochs + 1 - smooth_window + 1), smoothed_val_accs, label='Validation Accuracy', color='blue', marker='o')
# #使用与原始数据相同的长度来绘制平滑曲线
# # 这将确保平滑曲线的长度与原始数据的长度相匹配，解决了 "ValueError: x and y must have the same first dimension" 错误
# plt.xlabel('Epoch')
# plt.ylabel('Loss and Validation Accuracy')
# plt.ylim(0.0, 1.0)  # 设置 y 轴范围，上限为 1.0
# plt.legend()
#
# plt.show()
fig=plt.figure()
ax1 = fig.add_subplot(111)  # 指的是将plot界面分成1行1列
ax1.plot(range(1, args.epochs + 1 - smooth_window + 1), smoothed_losses,
             c=np.array([255, 71, 90]) / 255.)  # c为颜色
plt.ylabel('Loss')

ax2 = fig.add_subplot(111, sharex=ax1, frameon=False)  # 其本质就是添加坐标系，设置共享ax1的x轴，ax2背景透明
ax2.plot(range(1, args.epochs + 1 - smooth_window + 1), smoothed_val_accs,
             c=np.array([79, 179, 255]) / 255.)
ax2.yaxis.tick_right()  # 开启右边的y坐标

ax2.yaxis.set_label_position("right")
plt.ylabel('ValAcc')

plt.xlabel('Epoch')
plt.title('Training Loss & Validation Accuracy')

plt.legend()#添加图例
plt.show()