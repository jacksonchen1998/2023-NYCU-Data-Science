import torch
import numpy as np
import torch.nn as nn
import torch.optim as optim
from torch_geometric.nn import GCNConv
from tqdm import tqdm

# Define the Graph Neural Network model
# concatenate with its own node feature
class GCNAnomalyDetector(nn.Module):
    def __init__(self, num_features, hidden_size):
        super(GCNAnomalyDetector, self).__init__()
        self.conv1 = GCNConv(num_features, hidden_size)
        self.conv2 = GCNConv(hidden_size, hidden_size*2)
        self.fc = nn.Linear(hidden_size*2+10, 1)

    def forward(self, x, edge_index):
        x_1 = x
        x = self.conv1(x, edge_index)
        x = torch.relu(x)
        x = self.conv2(x, edge_index)
        x_2 = torch.relu(x)
        x = torch.cat((x_1, x_2), dim=1)
        x = self.fc(x).squeeze(1)
        return x

# Load the training dataset
train_data = torch.load('train_sub-graph_tensor.pt') # torch_geometric.data.data.Data
train_edge_index = train_data.edge_index # 2, 6784824
train_feature = train_data.feature.float() # Convert to double
train_label = train_data.label # 15742

# Load the testing dataset
test_data = torch.load('test_sub-graph_tensor_noLabel.pt') # torch_geometric.data.data.Data
test_edge_index = test_data.edge_index # 2, 7000540
test_feature = test_data.feature.float() # Convert to double

# Create a mask to specify training nodes
# torch.Tensor, 39357
train_mask = torch.from_numpy(np.load('train_mask.npy')).bool()

# Create a mask to specify testing nodes
# torch.Tensor, 39657
test_mask = torch.from_numpy(np.load('test_mask.npy')).bool()

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Create the model and optimizer
num_epochs = 10000
model = GCNAnomalyDetector(num_features=train_feature.shape[1], hidden_size=32).to(device)
optimizer = optim.AdamW(model.parameters(), lr=5e-4, weight_decay=5e-4)
scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=num_epochs, eta_min=1e-5)

# Training loop
model.train()
pbar = tqdm(total=num_epochs, desc='Training')
for epoch in range(num_epochs):
    optimizer.zero_grad()
    train_feature = train_feature.to(device)
    train_edge_index = train_edge_index.to(device)
    train_label = train_label.to(device)
    logits = model(train_feature, train_edge_index)
    loss = nn.CrossEntropyLoss()(logits[train_mask], train_label.float())
    loss.backward()
    optimizer.step()
    scheduler.step()
    pbar.update(1)
    pbar.set_postfix({'Epoch': epoch+1, 'Loss': loss.item()})
pbar.close()

# Save the trained model
torch.save(model.state_dict(), 'gcn_anomaly_model.pt')

# Testing
# node idx, node anomaly score
# save the anomaly score for each node into a numpy array

# initialize the storage with np.zeros
anomaly_score = np.zeros(test_feature.shape[0], dtype=np.float64)

# load the trained model
model.load_state_dict(torch.load('gcn_anomaly_model.pt'))

model.eval()
with torch.no_grad():
    test_feature = test_feature.to(device)
    test_edge_index = test_edge_index.to(device)
    logits = model(test_feature, test_edge_index)
    # calculate the probability of each node being anomalous
    prob = torch.sigmoid(logits)
    
# save into csv file called anomaly_score.csv
# there are two columns: node idx, node anomaly score

idx = 0

with open('anomaly_score.csv', 'w') as f:
    f.write('node idx,node anomaly score\n')
    for i in test_mask:
        if i == True:
            # write the node idx and anomaly score into the csv line by line
            f.write(str(idx) + ',' + str(prob[idx].item()) + '\n')
        idx += 1
