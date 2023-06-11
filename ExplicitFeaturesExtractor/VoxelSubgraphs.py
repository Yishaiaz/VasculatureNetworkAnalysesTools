import graph_tool as gt
from graph_tool import stats
import numpy as np
import matplotlib.pyplot as plt
import copy
import pickle
from tqdm import tqdm
import os
import sys
import shutil
import math
import time
import warnings
from typing import *
from enum import Enum
import datetime
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.spatial import cKDTree
from graph_tool.all import *
import networkx as nx

from NetworkUtilities.GraphToolFormatConverters import *

class NetworkVoxelSearcher:
    def __init__(self, complete_network: Union[gt.Graph, nx.Graph], **kwargs):
        # working only with nx networks, if a gt network, converts it to nx network
        self.__complete_network = copy.deepcopy(
            self.__validate_network_data_structure_concurency(
                _network=complete_network,
                **kwargs))
        self.__network_nodes_coordinates = np.stack([x for x in nx.get_node_attributes(
            self.__complete_network,
            kwargs.get('vertex_coordinates_attribute_name', 'coordinates')).values()])
        self.kd_struct_network = cKDTree(self.__network_nodes_coordinates)

    @staticmethod
    def __validate_network_data_structure_concurency(_network: Union[gt.Graph, nx.Graph], **kwargs):
        if isinstance(_network, gt.Graph):
            print(f"converting gt graph data structure network to networkx graph data structure")
            _network = convert_gt_to_nx(
                gt_graph=_network,
                vertices_property_names_to_add=[kwargs.get('vertex_coordinates_attribute_name', 'coordinates')]
            )
            print(f"finished converting network to networkx graph data structure")
        return _network

    def find_nodes_within_distance_kdtree(self, target_node_idx: int,
                                          distance_in_px: Union[int, float],
                                          exclude_target_node_index_from_results:bool = True):
        """
        Finds all nodes in the network that are within a specified Euclidean distance of a target node,
        using a kd-tree data structure for efficient search.

        Args:
            target_node_idx (tuple): The index of the target node as it is in the original network (index is incremental).
            distance_in_px (float, int): The maximum Euclidean distance allowed between the target node and any nodes returned.
            exclude_target_node_index_from_results (bool): whether to include the target node index in the returned list
        Returns:
            list of integers: The nodes indices in the network that are within the specified Euclidean distance of the target node.
        """
        target_coords = self.__network_nodes_coordinates[target_node_idx]

        # Query the kd-tree to find all nodes within the specified distance
        nodes_within_distance_idx = self.kd_struct_network.query_ball_point(target_coords, distance_in_px)
        if exclude_target_node_index_from_results:
            nodes_within_distance_idx.remove(target_node_idx)

        return nodes_within_distance_idx


def convert_gt_to_nx(gt_graph: gt.Graph,
                     vertices_property_names_to_add: List[str] = [],
                     edges_property_names_to_add: List[str] = []) -> nx.Graph:
    nx_graph = nx.Graph()
    nx_graph.add_nodes_from(gt_graph.get_vertices())
    gt_v_props_map = gt_graph.vp
    gt_e_props_map = gt_graph.ep
    for property_name in vertices_property_names_to_add:
        print(f"adding property: {property_name} to nx graph nodes")
        target_property_dict = convert_gt_vertices_property_map_to_dict(
            gt_graph_vertices=gt_graph.get_vertices(),
            gt_v_property_map=gt_v_props_map[property_name])
        nx.set_node_attributes(nx_graph, target_property_dict, property_name)

    for property_name in edges_property_names_to_add:
        print(f"adding property: {property_name} to nx graph nodes")
        target_property_dict = convert_gt_edges_property_map_to_dict(
            gt_graph=gt_graph,
            gt_graph_edges=gt_graph.get_edges(),
            gt_e_property_map=gt_e_props_map[property_name]
        )
        nx.set_edge_attributes(nx_graph, target_property_dict, property_name)

    return nx_graph


if __name__ == '__main__':
    # this demo validates performance, and compares execution speed of both methods.
#    graph_file = "/Users/leahbiram/Desktop/vasculature_data/firstGBMscanGraph.gt"
#    gt_graph = gt.load_graph(graph_file)
#    nx_graph_1 = convert_gt_to_nx(gt_graph,
#                                vertices_property_names_to_add=["coordinates", "radii"],
#                                edges_property_names_to_add=["length", "radii"])

    nx_graph = nx.read_gpickle("../ExtractedFeatureFiles/full_graph.gpickle")
    # large network (to compare runtime of methods on a large scale):
    v_searcher = NetworkVoxelSearcher(complete_network=nx_graph, vertex_coordinates_attribute_name="coordinates")
    dist_thresh = 50
    st_time = time.time()
    kdtree_found_near_vertices_ids = []
    for node in tqdm(nx_graph.nodes):
        kdtree_found_near_vertices_ids.append(np.array(v_searcher.find_nodes_within_distance_kdtree(node, dist_thresh, False)))
        #all_subgraphs.append(GraphView(gt_graph, vfilt=lambda v: v in kdtree_found_near_vertices_ids))
    kd_end_time = time.time() - st_time
    print(
        f"KDtree-based execution time: {kd_end_time}\n")
    with open('../ExtractedFeatureFiles/full_voxel_subgraphs_50.pkl', 'wb') as file:
        # A new file will be created
        pickle.dump(kdtree_found_near_vertices_ids, file)

