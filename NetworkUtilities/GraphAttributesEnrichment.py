import os
import sys
import shutil
import math
from typing import *
from enum import Enum
import datetime
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import graph_tool.all as gt
import tqdm


def add_vertex_distance_to_closest_target_object(_g: gt.Graph, target_object_coordinates_array: np.array,
                                                 **kwargs) -> Union[str, Tuple[gt.Graph, str]]:
    """
    Adds the minimum distance of each vertex to target objects as a new vertex property of the graph.

        Usage example - calculating the minimal distance of each vertex to all cancer cells:
            graph = gt.load_graph(graph_path)
            cancer_cells_array = np.load(cancer_cells_array_path)
            # filter the cancer cells to only cancer cells in the subgraph
            cancer_cells_in_graph_array = np.array(list(filter(lambda x: x[-1].lower()==graph_name.lower(), cancer_cells_array)))
            # collect only cancer cells coordinates
            cancer_cells_in_graph_array_only_coordinates = np.array(list(map(lambda x: tuple(x)[0:3], cancer_cells_in_graph_array)))
            add_vertex_distance_to_closest_target_object(graph, target_object_coordinates_array=cancer_cells_in_graph_array_only_coordinates)

    Additional keyword argumens:
    1. "in_place" - whether to update the vertex property map of the provided graph (True) or return an enriched
        copy of it, default = True. If False is passed, returns a tuple with the new graph instance and the name of the
            new vp. if True, only returns the vp name.
    2. "new_prop_map_name" - the key (i.e., name) of the new property map, default = "min_dist_to_cancer_cells"

    :param _g: gt.Graph, target graph to enrich
    :param target_object_coordinates_array: np.array, the coordinates of the target objects to find the minimum
        distance to.
    :param kwargs: Dict, additional keyword arguments.
    :return:
    """

    distance_prop_map = _g.new_vertex_property(value_type='double')
    for vertex in tqdm.tqdm(_g.iter_vertices(), desc="Calculating vertices min distance from target objects"):
        vertex_coordinates_as_np = _g.vp['coordinates'][vertex].a
        vertex_distances_from_target_objects = np.linalg.norm(vertex_coordinates_as_np - target_object_coordinates_array, axis=1)
        vertex_min_distance_from_target_objects = vertex_distances_from_target_objects.min()
        distance_prop_map[vertex] = vertex_min_distance_from_target_objects

    if not kwargs.get("in_place", True):
        _g_cpy = _g.copy()
        _g_cpy.vp[kwargs.get("new_prop_map_name", "min_dist_to_cancer_cells")] = distance_prop_map
        return _g_cpy, kwargs.get("new_prop_map_name", "min_dist_to_cancer_cells")

    _g.vp[kwargs.get("new_prop_map_name", "min_dist_to_cancer_cells")] = distance_prop_map
    return kwargs.get("new_prop_map_name", "min_dist_to_cancer_cells")

