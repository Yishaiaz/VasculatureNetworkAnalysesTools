import os
import sys
import shutil
import math
import warnings
from typing import *
from enum import Enum
import datetime
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import graph_tool.all as gt
import networkx as nx


def convert_gt_vertices_property_map_to_dict(gt_graph_vertices: np.array,
                                             gt_v_property_map: gt.PropertyMap) -> Dict[int, Any]:
    return {v_idx: gt_v_property_map[v_idx] for v_idx in np.arange(0, len(gt_graph_vertices), 1)}


def convert_gt_edges_property_map_to_dict(gt_graph: gt.Graph,
                                          gt_graph_edges: np.array,
                                          gt_e_property_map: gt.PropertyMap) -> Dict[Tuple[int, int], Any]:
    return {tuple(gt_graph_edges[e_idx]): gt_e_property_map[gt_graph.edge(*gt_graph_edges[e_idx])] for e_idx in np.arange(0, len(gt_graph_edges), 1)}


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
    gt_graph_path = "/Data/GBM_Tumor_Graphs/graph_annotated.gt"
    gt_graph = gt.load_graph(gt_graph_path)
    nx_graph = convert_gt_to_nx(gt_graph,
                     vertices_property_names_to_add=["coordinates", "radii"],
                     edges_property_names_to_add=["length", "radii"])
    # fig, ax = plt.subplots()
    # nx.draw(nx_graph, pos=nx.get_node_attributes(nx_graph, "coordinates"), ax=ax)
    # plt.show()