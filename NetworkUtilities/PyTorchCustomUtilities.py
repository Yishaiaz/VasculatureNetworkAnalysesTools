import copy
import os
import warnings
from typing import *
import numpy as np
import torch
import tqdm
from torch import Tensor
from torch.nn.functional import normalize
from torch_geometric.data import Data, HeteroData, Dataset
from torch_geometric.data.feature_store import FeatureStore, TensorAttr
from torch_geometric.data.graph_store import GraphStore
from torch_geometric.loader import DataLoader, NeighborLoader
from torch_geometric.loader.base import DataLoaderIterator
from torch.utils.tensorboard import SummaryWriter
from sklearn.metrics import confusion_matrix
import seaborn as sn
import pandas as pd
from ClusterLogger import CallableLogger
from torch_geometric.loader.utils import (
    filter_custom_store,
    filter_data,

    filter_hetero_data,
)
from torch.utils.data import Subset
from torch_geometric.utils import k_hop_subgraph
from torch_geometric.utils import mask_to_index, index_to_mask
from torch_geometric.loader import DataLoader, ImbalancedSampler
from functools import reduce
from torch_geometric.utils.convert import to_networkx
from torch_geometric.utils.convert import from_networkx
import matplotlib.pyplot as plt
import networkx as nx


print(torch.cuda.is_available())

class MicroEnvironmentDataset(Dataset):
    """
    A custom PyTorch Geometric Dataset that generates subgraphs based on a specified number of hops
    and a distance constraint. The loader creates subgraphs centered around each node in the input graph,
    and filters nodes based on their distance from the center node.

    Args:
        data (torch_geometric.data.Data): The input graph as a PyTorch Geometric Data object.
        num_hops (int): The number of hops to include in the subgraphs.
        loc_attr_name (str): The name of the node attribute that represents the 3D location of each node.
        max_distance (float): The maximum allowed distance between nodes in the subgraphs.
        input_nodes (torch.Tensor, optional): The indices of nodes for which subgraphs are generated.
                                              Can be either a torch.LongTensor or torch.BoolTensor.
                                              If set to None, all nodes will be considered. Defaults to None.
        batch_size (int, optional): The number of subgraphs to process in a batch. Defaults to 1.
        shuffle (bool, optional): Whether to shuffle the subgraphs before processing. Defaults to True.
        **kwargs: Additional keyword arguments for the DataLoader.

    Example:
        data = ...  # Your PyTorch Geometric Data object
        num_hops = 2
        loc_attr_name = "location"  # Change this to match the name of the location attribute in your data
        max_distance = 5.0
        input_nodes = torch.tensor([0, 2, 5])  # Indices of nodes for which subgraphs are generated

        loader = SubgraphLoader(data, num_hops, loc_attr_name, max_distance, input_nodes=input_nodes, batch_size=4)

        for batch in loader:
            # batch is a list of Data objects, each containing a subgraph based on the specified number of hops
            # and the distance constraint
            print(batch)
    """

    def __init__(self, data, num_hops, loc_attr_name, max_distance,
                 input_nodes=None, calc_items_label_func: Callable = None,
                 k_subgraph_directed: bool = False, transform: Union[List[Callable], None] = None,
                 device=None, root: Union[os.PathLike, str] = "./MicroEnvironmentDatasets/",
                 include_node_attributes: bool = True,
                 include_edge_attributes: bool = True,
                 logger: Callable = print,
                 **kwargs):
        super().__init__(root, transform)
        self.org_graph = copy.copy(data)
        self.num_hops = num_hops
        self.loc_attr_name = loc_attr_name
        self.k_subgraph_directed = k_subgraph_directed
        if max_distance < float('inf'):
            raise NotImplementedError(f"Constraining subgraphs-nodes distances is not supported "
                                      f"yet due to a bug in the filtering method")

        self.max_distance = max_distance
        self.input_nodes = input_nodes
        self.num_nodes = self.org_graph.num_nodes

        if input_nodes is None:
            input_nodes = torch.arange(data.num_nodes)
        elif isinstance(input_nodes, torch.BoolTensor):
            input_nodes = input_nodes.nonzero().squeeze()
        self.device = device if device is not None else 'cuda:0' if torch.cuda.is_available() else 'cpu'
        self.input_nodes = input_nodes
        self.root_dir = root
        self.verbose = kwargs.get('verbose', 0)
        self.calc_items_label_func = calc_items_label_func
        self.relabel_nodes_in_k_hop = kwargs.get('relabel_nodes_in_k_hop', True)
        self.label_distribution = Counter()
        self.label_distribution[0] = 0
        self.label_distribution[1] = 0
        self.include_node_attributes = include_node_attributes
        self.include_edge_attributes = include_edge_attributes
        if self.include_node_attributes:
            self.org_graph.x = torch.zeros_like(self.org_graph.x, device=self.org_graph.x.device)
        if self.include_edge_attributes:
            self.org_graph.x = torch.zeros_like(self.org_graph.edge_attr,
                                                device=self.org_graph.edge_attr.device)

        self.logger = logger
    # @property
    # def raw_file_names(self):
    #     self.logger"raw_file_names?")
    #     return ['2.', 'd1']
    #
    # @property
    # def processed_file_names(self):
    #     self.logger"processed_file_names?")
    #     return ['2.', 'd1']
    #
    # def download(self):
    #     self.logger"downloading?")
    #
    # def process(self):
    #     self.logger"processing?")
    def __iter__(self):
        self.__iter_counter = 0
        return self

    def __next__(self):
        if self.__iter_counter < self.num_nodes:
            to_return =  self.get(self.__iter_counter)
            self.__iter_counter += 1
            return to_return
        raise StopIteration

    def __len__(self):
        return len(self.input_nodes)

    def len(self):
        return self.__len__()

    def get(self, item):
        return self.__getitem__(item)

    def __getitem__(self, item):
        rand_node_index = item  # torch.randint(0, self.num_nodes, (1, ))
        if not isinstance(item, Tensor):
            rand_node_index = torch.Tensor([rand_node_index])
        if not len(rand_node_index.shape):
            rand_node_index.unsqueeze(0)
        if self.device != 'cpu':
            rand_node_index.to(self.device)
        if rand_node_index.dtype != torch.long or \
                rand_node_index.dtype != torch.bool:
            rand_node_index = rand_node_index.to(torch.long)

        subgraph_node_indices, edge_index, mapping, edge_mask = k_hop_subgraph(rand_node_index, self.num_hops,
                                                                               self.org_graph.edge_index,
                                                                               relabel_nodes=self.relabel_nodes_in_k_hop,
                                                                               directed=self.k_subgraph_directed)
        while len(subgraph_node_indices) <= 1:
            # subgraph is only a single node, selecting a new random node as center node of subgraph
            if self.verbose > 1:
                self.logger(f"node {rand_node_index} has only {len(subgraph_node_indices)} neighbors! "
                      f"Randomly sampling a new center node")
            rand_node_index = torch.randint(0, self.num_nodes, (1,)).to(torch.long)
            subgraph_node_indices, edge_index, mapping, edge_mask = k_hop_subgraph(rand_node_index, self.num_hops,
                                                                                   self.org_graph.edge_index,
                                                                                   relabel_nodes=self.relabel_nodes_in_k_hop)

        # filtering the nodes and edges that are further than the max_distance constraint is not implemented correctly.
        # todo: fix it.
        if self.max_distance < float('inf') and self.loc_attr_name is not None:
            raise NotImplementedError(f"Constraining subgraphs-nodes distances is not supported "
                                      f"yet due to a bug in the filtering method")
            #
            # center_node_loc = self.org_graph[self.loc_attr_name][rand_node_index].unsqueeze(0)
            # subgraph_node_locs = self.org_graph[self.loc_attr_name][subgraph_node_indices]
            # distances = torch.norm(subgraph_node_locs - center_node_loc, dim=-1)
            # filtered_subgraph_node_indices = subgraph_node_indices[(distances <= self.max_distance).squeeze()]
            #
            # # removed_subgraph_nodes = subgraph_node_indices[(distances > self.max_distance).squeeze()]
            # all_edges_with_filtered_nodes = []
            # for node_idx in filtered_subgraph_node_indices:
            #     edges_to_keep = torch.argwhere(node_idx == edge_index).to(torch.long)
            #     all_edges_with_filtered_nodes.append(edges_to_keep)
            # # all_edges_with_filtered_nodes = Tensor(all_edges_with_filtered_nodes).to(torch.bool)
            # edge_index = torch.index_select(edge_index, dim=1, index=torch.cat(all_edges_with_filtered_nodes)[:, 1].unique())
            # subgraph = self.org_graph.subgraph(filtered_subgraph_node_indices)
        else:
            filtered_subgraph_node_indices = subgraph_node_indices
            # Extract node and edge attributes for the subgraph
        subgraph_node_attr = self.org_graph.x[filtered_subgraph_node_indices] if self.org_graph.x is not None else None
        subgraph_edge_attr = self.org_graph.edge_attr[edge_index] if self.org_graph.edge_attr is not None else None
        subgraph_data = Data(x=subgraph_node_attr, edge_index=edge_index, edge_attr=subgraph_edge_attr,
                             num_nodes=filtered_subgraph_node_indices.size(0))
        if self.loc_attr_name is not None:
            subgraph_pos_attr = self.org_graph.pos[
                filtered_subgraph_node_indices] if self.org_graph.edge_attr is not None else None
            subgraph_data.pos = subgraph_pos_attr
        if self.transform is not None:
            for tran_func in self.transform:
                subgraph_data = tran_func(subgraph_data)
        if self.calc_items_label_func is not None:
            if self.org_graph.get('single_node_label') is None:
                raise RuntimeError(
                    f"No single node label was found in the subgraph! the name of the attribute should be 'single_node_label'")
            subgraph_single_node_labels = self.org_graph.single_node_label[filtered_subgraph_node_indices]
            subgraph_data.single_node_label = subgraph_single_node_labels
            subgraph_data = self.calc_items_label_func(subgraph_data)
            self.label_distribution[int(subgraph_data.y)] += 1
            return subgraph_data.to(device=self.device)

        self.label_distribution[int(subgraph_data.y)] += 1
        return subgraph_data.to(device=self.device)

    def get_distribution(self):
        label_distribution = np.array(list(self.label_distribution.values()))
        label_distribution_norm = label_distribution / label_distribution.max()
        return label_distribution_norm, label_distribution

    @staticmethod
    def validate_subgraph_label_type(subgraph, required_label_type = torch.long):
        if subgraph.y.dtype != required_label_type:
            subgraph.y = subgraph.y.to(torch.long)
        return subgraph

    @staticmethod
    def calc_subgraph_label_by_bio_marker_presence(subgraph: Data,
                                                   agg_func: Callable = lambda x: torch.max(x).to(
                                                       torch.bool)) -> Tensor:
        total_subgraph_y = agg_func(subgraph.single_node_label)
        if not total_subgraph_y.shape:
            total_subgraph_y = Tensor([total_subgraph_y]).to(torch.int64)
        subgraph.y = total_subgraph_y
        # for sampler compatibility
        subgraph = MicroEnvironmentDataset.validate_subgraph_label_type(subgraph)
        return subgraph

    @staticmethod
    def calc_subgraph_label_by_bio_marker_majority(subgraph: Data) -> Tensor:
        labels_values_counts = torch.unique(subgraph.single_node_label, return_counts=True)

        if len(labels_values_counts[0]) == 1:
            total_subgraph_y = labels_values_counts[0][0]
            if not total_subgraph_y.shape:
                total_subgraph_y = Tensor([total_subgraph_y]).to(torch.long)
            subgraph.y = total_subgraph_y
        else:
            if labels_values_counts[1][0] >= labels_values_counts[1][1]:
                total_subgraph_y = labels_values_counts[0][0]
                if not total_subgraph_y.shape:
                    total_subgraph_y = Tensor([total_subgraph_y])
                subgraph.y = total_subgraph_y
            else:
                total_subgraph_y = labels_values_counts[0][1]
                if not total_subgraph_y.shape:
                    total_subgraph_y = Tensor([total_subgraph_y])
                subgraph.y = total_subgraph_y
        # for sampler compatibility
        subgraph = MicroEnvironmentDataset.validate_subgraph_label_type(subgraph)
        return subgraph

    @staticmethod
    def calc_subgraph_label_by_bio_marker_full_presence(subgraph):
        labels_values_counts = torch.unique(subgraph.single_node_label, return_counts=True)
        if len(labels_values_counts[0]) > 1:
            subgraph.y = Tensor([0])
        else:
            subgraph.y = Tensor([int(labels_values_counts[0][0]) == 1])

        # for sampler compatibility
        subgraph = MicroEnvironmentDataset.validate_subgraph_label_type(subgraph)
        return subgraph

    def calc_target_statistics(self):
        target_values_counter = Counter()
        for item in self:
            target_values_counter[item.y] += 1

        return target_values_counter


def get_hist_image_as_np_array(values: Iterable,
                               x_ticks_labels: List[str] = ['0', '1'],
                               x_label: str = 'Label',
                               y_label: str = 'Count',
                               title: str = 'Labels Histogram'):
    from matplotlib.backends.backend_agg import FigureCanvasAgg as FigureCanvas
    from matplotlib.figure import Figure
    fig, axis = plt.subplots()
    canvas = FigureCanvas(fig)
    axis.bar(np.arange(0, len(values), 1), values)
    axis.set_xticks(np.arange(0, len(x_ticks_labels), 1))
    axis.set_xticklabels(x_ticks_labels)
    axis.set_xlabel(x_label)
    axis.set_ylabel(y_label)
    axis.set_title(title)
    canvas.draw()  # draw the canvas, cache the renderer
    width, height = fig.get_size_inches() * fig.get_dpi()
    width, height = int(width), int(height)
    image = np.frombuffer(canvas.tostring_rgb(), dtype='uint8').reshape((height, width, 3))
    return image


def train(model, train_loader, train_ds, validation_loader, test_loader,
          epochs_num: int = 100,
          tensorboard_writer: SummaryWriter = None,
          logger: Callable = print):
    criterion = torch.nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(),
                                 lr=0.01,
                                 weight_decay=0.01)
    model.train()
    # conf_mat_info = np.zeros(
    #     (2,2), dtype=float
    # )
    for epoch in range(epochs_num + 1):
        print(f"INFO - progress: #epoch={epoch}/{epochs_num}]")
        logger(f"Initiated epoch #{epoch}")
        total_loss = 0
        acc = 0
        val_loss = 0
        val_acc = 0

        y_true = []
        y_pred = []
        # Train on batches
        logger("training!")
        if isinstance(logger, CallableLogger):
            iter = train_loader
        else:
            iter = tqdm.tqdm(train_loader)
        for data in iter:
            optimizer.zero_grad()
            _, out = model(data.x, data.edge_index, data.batch)
            data_y_as_onehot = torch.nn.functional.one_hot(data.y.to(torch.int64), num_classes=2).reshape(1, -1)
            loss = criterion(out, data_y_as_onehot.to(torch.float32))
            total_loss += loss
            acc += accuracy(out[0].argmax(), data.y.to(torch.int64))
            y_true.append(data.y.to(torch.int32))
            y_pred.append(out[0].argmax())
            loss.backward()
            optimizer.step()
            # break

        cf_matrix = confusion_matrix([x[0] for x in y_true], y_pred)
        df_cm = pd.DataFrame(cf_matrix,  # / np.sum(cf_matrix, axis=1)[:, None],
                             index=[i for i in ('0', '1')],
                             columns=[i for i in ('0', '1')])
        plt.figure(figsize=(12, 7))
        conf_mat_as_img = sn.heatmap(df_cm, annot=True).get_figure()
        if tensorboard_writer is not None:
            tensorboard_writer.add_scalar('totalLoss/train', total_loss, epoch)
            tensorboard_writer.add_scalar('meanLoss/train', total_loss / len(train_loader), epoch)
            tensorboard_writer.add_scalar('Accuracy/train', acc / len(train_loader), epoch)
            tensorboard_writer.add_figure("Train confusion matrix", conf_mat_as_img, epoch)
            # get distribution of labels in graph
            if epoch == 0:
                labels_distribution_norm, labels_distribution = train_ds.get_distribution()
                tmp_df = pd.DataFrame(
                    {'0': [labels_distribution[0]], '1': [labels_distribution[1]]})
                tensorboard_writer.add_figure("Train labels distributions", sn.barplot(tmp_df).get_figure())
                # tensorboard_writer.add_figure("Train labels distributions normalize", sn.barplot(labels_distribution_norm).get_figure())

        # Validation
        val_loss, val_acc, conf_mat = test(model, validation_loader,
                                           logger=logger)
        df_cm = pd.DataFrame(conf_mat,  # / np.sum(conf_mat, axis=1)[:, None],
                             index=[i for i in ('0', '1')],
                             columns=[i for i in ('0', '1')])
        plt.figure(figsize=(12, 7))
        conf_mat_as_img = sn.heatmap(df_cm, annot=True).get_figure()

        if tensorboard_writer is not None:
            tensorboard_writer.add_scalar('meanLoss/val', val_loss, epoch)
            tensorboard_writer.add_scalar('Accuracy/val', float(f"{val_acc:.2f}"), epoch)
            tensorboard_writer.add_figure("Validation confusion matrix", conf_mat_as_img, epoch)
            if epoch == 0:
                labels_distribution_norm, labels_distribution = validation_loader.get_distribution()
                tmp_df = pd.DataFrame(
                    {'0': [labels_distribution[0]], '1': [labels_distribution[1]]})
                tensorboard_writer.add_figure("Validation labels distributions", sn.barplot(tmp_df).get_figure())
                # tensorboard_writer.add_figure("Validation labels distributions normalize", sn.barplot(labels_distribution_norm).get_figure())

        # Print metrics every N epochs
        if epoch % 1 == 0:
            total_loss = total_loss / len(train_loader)
            acc = acc / len(train_loader)
            if len(acc.shape):
                acc = acc[0]
            logger(f'Epoch {epoch:>3} | Train Loss: {total_loss:.2f} '
                  f'| Train Acc: {acc * 100:>5.2f}% '
                  f'| Val Loss: {val_loss:.2f} '
                  f'| Val Acc: {val_acc * 100:.2f}%')

    test_loss, test_acc, conf_mat = test(model, test_loader,
                                         logger=logger)
    df_cm = pd.DataFrame(conf_mat,  # / np.sum(conf_mat, axis=1)[:, None],
                         index=[i for i in ('0', '1')],
                         columns=[i for i in ('0', '1')])
    plt.figure(figsize=(12, 7))
    conf_mat_as_img = sn.heatmap(df_cm, annot=True).get_figure()
    if tensorboard_writer is not None:
        tensorboard_writer.add_text('meanLoss/test', f"{test_loss / len(test_loader)}")
        tensorboard_writer.add_text('Accuracy/test', f"{test_acc / len(test_loader)}")
        tensorboard_writer.add_figure("Test confusion matrix", conf_mat_as_img)
        labels_distribution_norm, labels_distribution = test_loader.get_distribution()
        tmp_df = pd.DataFrame(
            {'0': [labels_distribution[0]], '1': [labels_distribution[1]]})
        tensorboard_writer.add_figure("Test labels distributions", sn.barplot(tmp_df).get_figure())
        # tensorboard_writer.add_figure("Test labels distributions normalize", sn.barplot(labels_distribution_norm).get_figure())

    logger(f'Test Loss: {test_loss:.2f} | Test Acc: {test_acc * 100:.2f}%')

    return model, test_acc


@torch.no_grad()
def test(model, loader, logger: Callable = print):
    criterion = torch.nn.CrossEntropyLoss()
    model.eval()
    loss = 0
    acc = 0
    y_true, y_pred = [], []

    logger("testing!")
    if isinstance(logger, CallableLogger):
        iter = loader
    else:
        iter = tqdm.tqdm(loader)
    for data in iter:
        _, out = model(data.x, data.edge_index, data.batch)
        data_y_as_onehot = torch.nn.functional.one_hot(data.y.to(torch.int64), num_classes=2).reshape(1, -1)
        loss += criterion(out, data_y_as_onehot.to(torch.float32))
        acc += accuracy(out[0].argmax(), data.y.to(torch.int64))
        y_true.append(data.y.to(torch.int32))
        y_pred.append(out[0].argmax())
        # break
    cf_matrix = confusion_matrix([x[0] for x in y_true], y_pred)

    # if the tensor is not a scalar
    if acc.shape:
        acc = acc[0]

    return loss / len(loader), acc / len(loader), cf_matrix


def accuracy(pred_y, y):
    """Calculate accuracy."""
    return pred_y == y  # ((pred_y == y).sum() / (len(y)) if y.shape else 1).item()


import torch
import torch.nn.functional as F
from torch.nn import Linear, Sequential, BatchNorm1d, ReLU, Dropout
from torch_geometric.nn import GCNConv, GINConv
from torch_geometric.nn import global_mean_pool, global_add_pool
from torch import sum, max_pool1d


def get_number_of_node_features(input_dataset: Union[Subset, Dataset]):
    ds_to_query = input_dataset
    if isinstance(input_dataset, Subset):
        ds_to_query = ds_to_query.dataset
    return ds_to_query.num_node_features


def partition_graph_by_regions(org_graph: Data,
                               n_bins_per_dim: int = 2,
                               val_bin_idx: Union[int, None] = None,
                               test_bin_idx: int = 0) -> Union[Tuple[Data, Data], Tuple[Data, Data, Data]]:
    """
    graph must have spatial coordinates in its pos attribute.
    :param org_graph:
    :type org_graph:
    :return:
    :rtype:
    """
    entire_graph_spatial_coordinates = org_graph.pos
    entire_graph_spatial_coordinates_norm = entire_graph_spatial_coordinates / torch.max(
        entire_graph_spatial_coordinates, dim=0).values

    assert n_bins_per_dim ** n_bins_per_dim > test_bin_idx
    assert val_bin_idx is None or n_bins_per_dim ** n_bins_per_dim > val_bin_idx
    # spatial_coordinates_quantiles = torch.quantile(entire_graph_spatial_coordinates, q=percentile, dim=0)
    # min_spatial_coordinates, max_spatial_coordinates = torch.min(entire_graph_spatial_coordinates, dim=0).values, torch.max
    # (entire_graph_spatial_coordinates, dim=0).values
    # len(torch.where(torch.greater_equal(entire_graph_spatial_coordinates,spatial_coordinates_quantiles)))/len(entire_graph_spatial_coordinates)
    dims_bin_edges = {}
    for t_dim in range(entire_graph_spatial_coordinates_norm.shape[1]):
        t_dim_hist = torch.histogram(entire_graph_spatial_coordinates_norm[:, t_dim], n_bins_per_dim)
        dims_bin_edges[t_dim] = t_dim_hist.bin_edges

    spatial_coordinates_indices_masks_by_bins = {}
    spatial_coordinates_indices_by_bins = {}
    all_sum = 0
    spatial_bin_idx = 0
    for x_bin_idx in range(n_bins_per_dim):
        for y_bin_idx in range(n_bins_per_dim):
            for z_bin_idx in range(n_bins_per_dim):
                bin_x_min, bin_x_max = tuple(dims_bin_edges[0][x_bin_idx: x_bin_idx + 2])
                bin_y_min, bin_y_max = tuple(dims_bin_edges[1][y_bin_idx: y_bin_idx + 2])
                bin_z_min, bin_z_max = tuple(dims_bin_edges[2][z_bin_idx: z_bin_idx + 2])
                x = torch.logical_and(entire_graph_spatial_coordinates_norm[:, 0] > bin_x_min,
                                      entire_graph_spatial_coordinates_norm[:, 0] <= bin_x_max)
                y = torch.logical_and(entire_graph_spatial_coordinates_norm[:, 1] > bin_y_min,
                                      entire_graph_spatial_coordinates_norm[:, 1] <= bin_y_max)
                z = torch.logical_and(entire_graph_spatial_coordinates_norm[:, 2] > bin_z_min,
                                      entire_graph_spatial_coordinates_norm[:, 2] <= bin_z_max)
                all = torch.logical_and(x, y)
                all = torch.logical_and(all, z)
                all_sum += torch.unique(all, return_counts=True)[1][1]
                spatial_coordinates_indices_masks_by_bins[spatial_bin_idx] = all
                spatial_coordinates_indices_by_bins[spatial_bin_idx] = mask_to_index(all)
                spatial_bin_idx += 1

    test_bin_subgraph = org_graph.subgraph(spatial_coordinates_indices_by_bins[test_bin_idx])
    if val_bin_idx is not None:
        val_bin_subgraph = org_graph.subgraph(spatial_coordinates_indices_by_bins[val_bin_idx])

    train_spatial_coordinates_indices_masks_by_bins_as_list = []
    for bin_idx in range(spatial_bin_idx):
        if bin_idx == test_bin_idx or (val_bin_idx is not None and bin_idx == val_bin_idx):
            continue
        train_spatial_coordinates_indices_masks_by_bins_as_list.append(
            spatial_coordinates_indices_masks_by_bins[bin_idx])

    train_of_nodes_mask = reduce(lambda prev, curr: torch.logical_or(prev, curr),
                                 train_spatial_coordinates_indices_masks_by_bins_as_list[1:],
                                 train_spatial_coordinates_indices_masks_by_bins_as_list[0])
    train_of_bins_subgraph = org_graph.subgraph(
        index_to_mask(train_of_nodes_mask, size=len(train_of_nodes_mask)))
    if val_bin_idx is not None:
        return train_of_bins_subgraph, val_bin_subgraph, test_bin_subgraph

    return train_of_bins_subgraph, test_bin_subgraph


def draw_subgraphs(subgraphs_dataset, draw_lim_num: int == 2):
    for idx, subgraph in enumerate(subgraphs_dataset):
        fig, axis = plt.subplots()
        nx_graph = to_networkx(subgraph)
        nx.draw(nx_graph, ax=axis)
        axis.set_title(f"subgraph y={subgraph.y}")
        if idx == draw_lim_num:
            return
        plt.tight_layout()
        plt.show()
        plt.clf()

