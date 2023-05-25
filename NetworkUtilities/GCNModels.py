import torch
import torch.nn.functional as F
from torch.nn import Linear, Sequential, BatchNorm1d, ReLU, Dropout
from torch_geometric.nn import GCNConv, GINConv, EdgeConv, DynamicEdgeConv
from torch_geometric.nn import global_mean_pool, global_add_pool
from torch import sum, max_pool1d
from PyTorchCustomUtilities import get_number_of_node_features


class GCNWithDynamicLayersNumber(torch.nn.Module):
    """GIN"""

    def __init__(self, dim_h, ds, output_dim, n_hops: int,
                 linear_head_dropout: float = 0.5):
        super(GCNWithDynamicLayersNumber, self).__init__()
        # ds_num_node_features = get_number_of_node_features(ds) # no need for node feature in GCN
        self.n_hops = n_hops
        self.linear_head_dropout = linear_head_dropout
        self.conv1 = GCNConv(
            in_channels=-1,
            out_channels=dim_h,
            improved=False,
            add_self_loops=True,
            normalize=True
            )
        self.gcn_conv_layers = [self.conv1]
        for hop in range(n_hops - 1):
            self.gcn_conv_layers.append(GCNConv(
              in_channels=dim_h,
              out_channels=dim_h
            ))

        self.lin1 = Linear(dim_h * self.n_hops, dim_h * self.n_hops)
        self.lin2 = Linear(dim_h * self.n_hops, output_dim)

    def forward(self, x, edge_index, batch):
        # Node embeddings
        prev_layer_output = self.conv1(x, edge_index)
        gcn_conv_layers_output_mean = [global_mean_pool(prev_layer_output, batch)]
        for gcn_conv_layer in self.gcn_conv_layers[1:]:
            print(f"prev_layer_output device: {prev_layer_output.get_device()}\n"
                  f"edge_index device: {edge_index.get_device()}\n")
            gcn_conv_layer_out = gcn_conv_layer(prev_layer_output, edge_index)
            gcn_conv_layer_mean = global_mean_pool(gcn_conv_layer_out, batch)
            gcn_conv_layers_output_mean.append(gcn_conv_layer_mean)
            prev_layer_output = gcn_conv_layer_out

            # Concatenate graph embeddings
        # h = torch.cat((h1_sum, h2_sum, h3_sum), dim=1)
        h = torch.cat(tuple(gcn_conv_layers_output_mean), dim=1)
        # Classifier
        h = self.lin1(h)
        h = h.relu()
        h = F.dropout(h, p=self.linear_head_dropout, training=self.training)
        h = self.lin2(h)

        return h, F.log_softmax(h, dim=1)


class FixedGCN(torch.nn.Module):
    """GIN"""

    def __init__(self, dim_h, ds, output_dim, linear_head_dropout: float = 0.5,
                 *args, **kwargs):
        super(FixedGCN, self).__init__()
        fixed_n_hops = 3
        ds_num_node_features = get_number_of_node_features(ds)
        self.linear_head_dropout = linear_head_dropout
        self.conv1 = GCNConv(
            in_channels=-1,
            out_channels=dim_h,
            improved=False,
            add_self_loops=True,
            normalize=True
            )
        self.gcn_conv_layers = [self.conv1]
        for n_hops in range(fixed_n_hops - 1):
            self.gcn_conv_layers.append(GCNConv(
              in_channels=dim_h,
              out_channels=dim_h
            ))

        self.lin1 = Linear(dim_h * fixed_n_hops, dim_h * fixed_n_hops)
        self.lin2 = Linear(dim_h * fixed_n_hops, output_dim)

    def forward(self, x, edge_index, batch):
        # Node embeddings
        prev_layer_output = self.conv1(x, edge_index)
        gcn_conv_layers_output_mean = [global_mean_pool(prev_layer_output, batch)]
        for gcn_conv_layer in self.gcn_conv_layers[1:]:
            gcn_conv_layer_out = gcn_conv_layer(prev_layer_output, edge_index)
            gcn_conv_layer_mean = global_mean_pool(gcn_conv_layer_out, batch)
            gcn_conv_layers_output_mean.append(gcn_conv_layer_mean)
            prev_layer_output = gcn_conv_layer_out

            # Concatenate graph embeddings
        # h = torch.cat((h1_sum, h2_sum, h3_sum), dim=1)
        h = torch.cat(tuple(gcn_conv_layers_output_mean), dim=1)
        # Classifier
        h = self.lin1(h)
        h = h.relu()
        h = F.dropout(h, p=self.linear_head_dropout, training=self.training)
        h = self.lin2(h)

        return h, F.log_softmax(h, dim=1)