import copy
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
from GraphToolFormatConverters import *


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

    def find_nodes_within_distance_linear(self,
                                          target_node_idx: int,
                                          distance_in_px: Union[int, float],
                                          exclude_target_node_index_from_results:bool = True):
        """
        Finds all nodes in the network that are within a specified Euclidean distance of a target node.

        Args:
            target_node_idx (tuple): The index of the target node as it is in the original network (index is incremental).
            distance_in_px (float, int): The maximum Euclidean distance allowed between the target node and any nodes returned.
            exclude_target_node_index_from_results (bool): whether to include the target node index in the returned list
        Returns:
            list of integers: The nodes indices in the network that are within the specified Euclidean distance of the target node.
        """
        target_coords = self.__network_nodes_coordinates[target_node_idx]
        nodes_indices_within_distance = []
        for node_idx, node_coords in enumerate(self.__network_nodes_coordinates):
            if math.sqrt(sum((a - b) ** 2 for a, b in zip(target_coords, node_coords))) <= distance_in_px:
                nodes_indices_within_distance.append(node_idx)

        if exclude_target_node_index_from_results:
            nodes_indices_within_distance.remove(target_node_idx)

        return nodes_indices_within_distance


if __name__ == '__main__':
    # this demo validates performance, and compares execution speed of both methods.
    # small network (to validate the truthness of the response):
    t_sm_graph = nx.Graph()
    coordinates = [(0, 0, 0),
                   (1, 1, 1),
                   (-1, -1, -1),
                   (2, 2, 2),
                   (-2, -2, -2),
                   (3, 3, 3),
                   (-3, -3, -3),
                   (4, 4, 4),
                   (-4, -4, -4),
                   (5, 5, 5),
                   (-5, -5, -5),
                   ]
    print(f"Small graph test with #{len(coordinates)} nodes")
    node_indices = np.arange(0, 11, 1)
    coordinates_as_attr = {
        node_idx: coordinates[node_idx] for node_idx in node_indices
    }
    t_sm_graph.add_nodes_from(node_indices)
    nx.set_node_attributes(t_sm_graph, coordinates_as_attr, "coordinates")
    searcher = NetworkVoxelSearcher(complete_network=t_sm_graph, vertex_coordinates_attribute_name="coordinates")
    dist_threh = 5.2
    st_time = time.time()
    linear_found_near_vertices_ids = np.array(searcher.find_nodes_within_distance_linear(0, dist_threh))
    linear_end_time = time.time() - st_time
    st_time = time.time()
    kdtree_found_near_vertices_ids = np.array(searcher.find_nodes_within_distance_kdtree(0, dist_threh))
    kd_end_time = time.time() - st_time
    assert np.array_equal(linear_found_near_vertices_ids.sort(), kdtree_found_near_vertices_ids.sort()), f"nodes found in methods are different!"
    print(f"Linear execution time: {linear_end_time}, KDtree-based execution time: {kd_end_time}\nFound vertices {linear_found_near_vertices_ids}")
    # large network (to compare runtime of methods on a large scale):
    t_lg_graph = nx.Graph()
    coordinates = np.random.random_integers(-1000, 1000, (2**16, 3))
    node_indices = np.arange(0, 2**16, 1)
    print(f"Large graph test with #{len(coordinates)} nodes")
    coordinates_as_attr = {
        node_idx: coordinates[node_idx] for node_idx in node_indices
    }
    t_lg_graph.add_nodes_from(node_indices)
    nx.set_node_attributes(t_lg_graph, coordinates_as_attr, "coordinates")
    searcher = NetworkVoxelSearcher(complete_network=t_lg_graph, vertex_coordinates_attribute_name="coordinates")
    dist_threh = 100
    st_time = time.time()
    linear_found_near_vertices_ids = np.array(searcher.find_nodes_within_distance_linear(0, dist_threh))
    linear_end_time = time.time() - st_time
    st_time = time.time()
    kdtree_found_near_vertices_ids = np.array(searcher.find_nodes_within_distance_kdtree(0, dist_threh))
    kd_end_time = time.time() - st_time
    assert np.array_equal(linear_found_near_vertices_ids.sort(),
                          kdtree_found_near_vertices_ids.sort()), f"nodes found in methods are different!"
    print(
        f"Linear execution time: {linear_end_time}, KDtree-based execution time: {kd_end_time}\nFound vertices {linear_found_near_vertices_ids}")