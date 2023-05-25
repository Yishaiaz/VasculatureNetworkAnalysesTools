import torch
import torch.nn.functional as F
from torch.nn import Linear, Sequential, BatchNorm1d, ReLU, Dropout
from torch_geometric.nn import GCNConv, GINConv
from torch_geometric.nn import global_mean_pool, global_add_pool
from torch import sum, max_pool1d
from PyTorchCustomUtilitiesForCluster import get_number_of_node_features

class GINWithDynamicLayersNumber(torch.nn.Module):
    """GIN"""

    def __init__(self, dim_h, ds, output_dim, n_hops: int = 3,
                 linear_head_dropout: float = 0.5):
        super(GINWithDynamicLayersNumber, self).__init__()
        ds_num_node_features = get_number_of_node_features(ds)
        self.linear_head_dropout = linear_head_dropout
        self.conv1 = GINConv(
            Sequential(Linear(ds_num_node_features, dim_h),
                       BatchNorm1d(dim_h), ReLU(),
                       Linear(dim_h, dim_h), ReLU()))
        self.gin_conv_layers = [self.conv1]
        for hop in range(n_hops - 1):
            self.gin_conv_layers.append(GINConv(
                Sequential(Linear(dim_h, dim_h), BatchNorm1d(dim_h), ReLU(),
                           Linear(dim_h, dim_h), ReLU())))

        self.lin1 = Linear(dim_h * n_hops, dim_h * n_hops)
        self.lin2 = Linear(dim_h * n_hops, output_dim)

    def forward(self, x, edge_index, batch):
        # Node embeddings
        prev_layer_output = self.conv1(x, edge_index)
        gin_conv_layers_output_mean = [global_mean_pool(prev_layer_output, batch)]
        for gin_conv_layer in self.gin_conv_layers[1:]:
            gin_conv_layer_out = gin_conv_layer(prev_layer_output, edge_index)
            gin_conv_layer_mean = global_mean_pool(gin_conv_layer_out, batch)
            gin_conv_layers_output_mean.append(gin_conv_layer_mean)
            prev_layer_output = gin_conv_layer_out

            # Concatenate graph embeddings
        # h = torch.cat((h1_sum, h2_sum, h3_sum), dim=1)
        h = torch.cat(tuple(gin_conv_layers_output_mean), dim=1)
        # Classifier
        h = self.lin1(h)
        h = h.relu()
        h = F.dropout(h, p=self.linear_head_dropout, training=self.training)
        h = self.lin2(h)

        return h, F.log_softmax(h, dim=1)


class FixedGIN(torch.nn.Module):
    """GIN"""

    def __init__(self, dim_h, ds, output_dim, *args, **kwargs):
        super(FixedGIN, self).__init__()
        ds_num_node_features = get_number_of_node_features(ds)
        self.conv1 = GINConv(
            Sequential(Linear(ds_num_node_features, dim_h),
                       BatchNorm1d(dim_h), ReLU(),
                       Linear(dim_h, dim_h), ReLU()))
        self.conv2 = GINConv(
            Sequential(Linear(dim_h, dim_h), BatchNorm1d(dim_h), ReLU(),
                       Linear(dim_h, dim_h), ReLU()))
        self.conv3 = GINConv(
            Sequential(Linear(dim_h, dim_h), BatchNorm1d(dim_h), ReLU(),
                       Linear(dim_h, dim_h), ReLU()))
        self.lin1 = Linear(dim_h * 3, dim_h * 3)
        self.lin2 = Linear(dim_h * 3, output_dim)

    def forward(self, x, edge_index, batch):
        # Node embeddings
        h1 = self.conv1(x, edge_index)
        h2 = self.conv2(h1, edge_index)
        h3 = self.conv3(h2, edge_index)

        # # Graph-level readout
        h1_mean = global_mean_pool(h1, batch)
        h2_mean = global_mean_pool(h2, batch)
        h3_mean = global_mean_pool(h3, batch)
        # h1_sum = sum(h1)
        # h2_sum = sum(h2)
        # h3_sum = sum(h3)
        # Concatenate graph embeddings
        # h = torch.cat((h1_sum, h2_sum, h3_sum), dim=1)
        h = torch.cat((h1_mean, h2_mean, h3_mean), dim=1)
        # Classifier
        h = self.lin1(h)
        h = h.relu()
        h = F.dropout(h, p=0.5, training=self.training)
        h = self.lin2(h)

        return h, F.log_softmax(h, dim=1)
