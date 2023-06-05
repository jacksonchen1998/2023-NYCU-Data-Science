# Graph Anomaly Detection

## Objective

[Kaggle Competition](https://www.kaggle.com/competitions/graph-anomaly-detection/overview)

In this homework, you need to implement any kind of Graph Neural Networks to find out the anomaly nodes.

You will need to solve the problem based on the given graph with the specific node indexes.

## Dataset

### Dataset Description

- `train_sub-graph_tensor.pt`: the graph-based training set (using pytorch_geometric), including edge_index, feature, and label.
- `train_mask.npy`: the node indexes for the graph-based training set, use this to get the correct index of the node label in the training graph.
- `test_sub-graph_tensor_noLabel.pt`: the graph-based testing set (using pytorch_geometric), including edge_index, feature.
- `test_mask.npy`: the node indexes for the graph-based testing set, use this to get the correct index of the node, and predict its node label in the testing graph.
- `sample_submission.csv`: a sample submission file in the correct format.
    - `node index`: the index of the target node
    - `node anomaly score`: the probability of the anomaly label

### Dataset Format

Edge Index: [2, num_of_edges]
- 2 means the edge that connect two nodes (node_A – node_B)
- There is no edge weight or edge attribute/feature in this simple graph.

Feature: [num_of_nodes, dim_of_node_feature]
- Each node means a transaction with 10-dimension feature (We won’t know the exact meaning for each dimensions)

Label: [num_of_nodes]
- Anomaly: 1 , Normal: 0

## Libraries installation

Because we need to use `torch_geometric`, here're some tips to install those packages.

[Stackoverflow Q/A](https://stackoverflow.com/questions/74794453/torch-geometric-module-not-found/76394639#76394639)

## Method

I use Graph Convolutional Network (GCN) to solve this problem.

The network architecture is shown below:

```
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
```

In this architecture:

- The first layer is a GCN layer with 10 input features and 32 output features.
- The second layer is a GCN layer with 32 input features and 64 output features.
- The third layer is a fully connected layer with 64+10 input features and 1 output feature.

## Reference

- [Pytorch_geometric](https://github.com/pyg-team/pytorch_geometric)
- [DGL (Deep Graph Library)](https://github.com/dmlc/dgl)
