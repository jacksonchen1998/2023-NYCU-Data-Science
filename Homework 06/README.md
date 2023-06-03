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
GCNAnomalyDetector(
  (conv1): GCNConv(in_channels=num_features, out_channels=hidden_size)
  (conv2): GCNConv(in_channels=hidden_size, out_channels=hidden_size)
  (fc): Linear(in_features=hidden_size, out_features=1)
)
```

In this architecture:
- The input features of each node are passed through the first GCNConv layer (`conv1`), which takes `num_features` input channels and produces `hidden_size` output channels. The output is then passed through a ReLU activation function.
- The output from the first GCNConv layer is fed into the second GCNConv layer (`conv2`), which has `hidden_size` input channels and `hidden_size` output channels. Again, the output is passed through a ReLU activation function.
- Finally, the output from the second GCNConv layer is fed into a linear layer (`fc`), which maps the `hidden_size` features to a single output value. The output is then squeezed to remove the extra dimension and returned.

This architecture allows the model to capture and propagate information through the graph structure, ultimately predicting the anomaly scores for each node.

## Reference

- [Pytorch_geometric](https://github.com/pyg-team/pytorch_geometric)
- [DGL (Deep Graph Library)](https://github.com/dmlc/dgl)
