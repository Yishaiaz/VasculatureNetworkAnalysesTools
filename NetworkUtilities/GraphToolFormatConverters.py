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
from functools import reduce

import tqdm


def convert_gt_vertices_property_map_to_dict(gt_graph_vertices: np.array,
                                             gt_v_property_map: gt.PropertyMap) -> Dict[int, Any]:
    gt_v_property_map_as_dict = {}
    for v_idx in tqdm.tqdm(range(0, len(gt_graph_vertices))):
        v_values = gt_v_property_map[v_idx]
        if isinstance(v_values, Iterable):
            v_values = [int(elem) if isinstance(elem, np.int) else float(elem) if isinstance(elem, np.float) else str(elem) for elem in v_values]
        elif isinstance(v_values, np.int):
            v_values = int(v_values)
        elif isinstance(v_values, np.float):
            v_values = float(v_values)
        else:
            v_values = str(v_values)
        gt_v_property_map_as_dict[v_idx] = v_values

    return gt_v_property_map_as_dict


def convert_gt_edges_property_map_to_dict(gt_graph: gt.Graph,
                                          gt_graph_edges: np.array,
                                          gt_e_property_map: gt.PropertyMap) -> Dict[Tuple[int, int], Any]:
    gt_e_property_map_as_dict = {}
    for e_idx in tqdm.tqdm(range(0, len(gt_graph_edges))):
        gt_e = gt_graph.edge(*gt_graph_edges[e_idx])
        gt_e_as_int_tuple = tuple([int(i) for i in gt_e])

        e_values = gt_e_property_map[gt_e]

        if isinstance(e_values, Iterable):
            e_values = [
                int(elem) if isinstance(elem, np.int) else float(elem) if isinstance(elem, np.float) else str(elem) for
                elem in e_values]
        elif isinstance(e_values, np.int):
            e_values = int(e_values)
        elif isinstance(e_values, np.float):
            e_values = float(e_values)
        else:
            e_values = str(e_values)
        gt_e_property_map_as_dict[gt_e_as_int_tuple] = e_values

    return gt_e_property_map_as_dict



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
        print(f"adding property: {property_name} to nx graph edges")
        target_property_dict = convert_gt_edges_property_map_to_dict(
            gt_graph=gt_graph,
            gt_graph_edges=gt_graph.get_edges(),
            gt_e_property_map=gt_e_props_map[property_name]
        )
        for edge in gt_graph.get_edges():
            v_1, v2 = edge
            nx_graph.add_edge(v_1, v2, **{property_name: target_property_dict[(v_1, v2)]})
        # nx.set_edge_attributes(nx_graph, target_property_dict, property_name)

    return nx_graph

def properties_stringizer(property:Iterable):
    if isinstance(property, Iterable):
        property_as_str = reduce(lambda prev,curr: f"{prev}-{curr}", property, "")
        return property_as_str
    else:
        return str(property)
        # raise ValueError("property is not iterable")


if __name__ == '__main__':
    gt_graph_path = "/Users/yishaiazabary/PycharmProjects/University/BrainVasculatureGraphs/Data/GBM_Tumor_Graphs/CD31-graph_reduced.gt"
    gt_graph = gt.load_graph(gt_graph_path)
    nx_graph = convert_gt_to_nx(gt_graph,
                     vertices_property_names_to_add=["coordinates", "radii",'artery_binary', 'artery_raw'],
                     edges_property_names_to_add=["length", "radii",'artery_binary', 'artery_raw'])
    nx.write_gml(nx_graph, "/Users/yishaiazabary/PycharmProjects/University/BrainVasculatureGraphs/Data/GBM_Tumor_Graphs/CD31-graph_reduced_as_nx.gml.gz", stringizer=properties_stringizer)
    nx_graph = nx.read_gml("/Users/yishaiazabary/PycharmProjects/University/BrainVasculatureGraphs/Data/GBM_Tumor_Graphs/CD31-graph_reduced_as_nx.gml.gz")
    assert len(nx_graph.edges) > 0, f"adding edges failed!"
    # fig, ax = plt.subplots()
    # nx.draw(nx_graph, pos=nx.get_node_attributes(nx_graph, "coordinates"), ax=ax)
    # plt.show()