#%%
from torch_geometric.datasets import Planetoid, TUDataset
import torch
import torch.nn.functional as F
from torch_geometric.nn import GCNConv
from torch_geometric.data import DataLoader
from tqdm import tqdm

#from torch.utils.data.dataloader import DataLoader
#%%
dataset = Planetoid(root='./tmp_Cora', name='Cora')
#dataset = TUDataset(root='./tmp/ENZYMES', name='ENZYMES')
data_list = [dataset[0]] * 200
loader = DataLoader(data_list, batch_size=32, shuffle=True)


#%%
class Net(torch.nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.conv1 = GCNConv(dataset.num_features, 16)
        self.conv2 = GCNConv(16, dataset.num_classes)

    def forward(self, data):
        x, edge_index = data.x, data.edge_index

        x = self.conv1(x, edge_index)
        x = F.relu(x)
        x = F.dropout(x, training=self.training)
        x = self.conv2(x, edge_index)

        return F.log_softmax(x, dim=1)

class ArgNet(torch.nn.Module):
    def __init__(self):
        super(ArgNet, self).__init__()
        self.conv1 = GCNConv(dataset.num_features, 16)
        self.conv2 = GCNConv(16, 16)

    def forward(self, data):
        # GNN的一个batch是把所有点排列成一排，而不是分成batch_size个图
        x, edge_index = data.x, data.edge_index

        x = self.conv1(x, edge_index)  # x.shape[点数量，feature数量]
        x = F.relu(x)
        x = F.dropout(x, training=self.training)
        x = self.conv2(x, edge_index)  # out.shape[点数量，卷积核out值数量]

        return x


#%%
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = ArgNet().to(device)
data = dataset[0].to(device)
optimizer = torch.optim.Adam(model.parameters(), lr=0.01, weight_decay=5e-4)

#%%
model.train()
for epoch in tqdm(range(200), desc="epoch:"):
    for batch in loader:
        optimizer.zero_grad()
        out = model(batch)
        loss = F.nll_loss(out[batch.train_mask], data.y[batch.train_mask])
        loss.backward()
        optimizer.step()


#%%
model.eval()
_, pred = model(data).max(dim=1)
correct = pred[data.test_mask].eq(data.y[data.test_mask]).sum().item()
acc = correct / data.test_mask.sum().item()
print('Accuracy: {:.4f}'.format(acc))