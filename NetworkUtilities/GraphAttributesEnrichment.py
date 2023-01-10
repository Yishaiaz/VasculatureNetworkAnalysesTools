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


def calc_vertex_min_distance_to_target_objects(_v: gt.Vertex,
                                               _vp_map: gt.PropertyMap,
                                               target_object_coordinates_array: np.array,
                                               **kwargs) -> float:
    vertex_coordinates_as_np = _vp_map[kwargs.get('vertex_coordinates_key', 'coordinates')][_v].a
    vertex_distances_from_target_objects = np.linalg.norm(vertex_coordinates_as_np - target_object_coordinates_array,
                                                          axis=1)
    return np.min(vertex_distances_from_target_objects)


def calc_edge_volume(_e: gt.Edge,
                     _ep_map: gt.PropertyMap,
                     **kwargs):
    edge_len = _ep_map[kwargs.get("edge_length_key", 'length')][_e]
    edge_radius = _ep_map[kwargs.get("edge_radius_key", 'radii')][_e]
    edge_opening_area = (np.pi * edge_radius**2)
    edge_volume = edge_opening_area * edge_len
    return edge_volume


def add_vertex_property_to_target_object(_g: gt.Graph,
                                         target_object_coordinates_array: np.array,
                                         property_calc_function: Callable = calc_vertex_min_distance_to_target_objects,
                                         **kwargs) -> Tuple[gt.Graph, str]:
    """
    Adds a vertex property using an HOF structure:
        default behavior is calculating the minimum distance of each vertex to target objects as a
        new vertex property of the graph (using the calc_vertex_min_distance_to_target_objects function).

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
    3. "save_graph_with_new_property" - boolean whether to save the new enriched graph.
    4. "graph_name" - the graph region/file name (only used if "save_graph_with_new_property" is True).
    5. "dir_to_save_graph_path" - the path to the directory you wish to save the enriched graph in (only used if
        "save_graph_with_new_property" is True).



    :param property_calc_function: a function with the following signature: func(_v: gt.Vertex,
                                               _vp_map: gt.PropertyMap,
                                               target_object_coordinates_array: np.array,
                                               **kwargs) -> Any
    :param _g: gt.Graph, target graph to enrich
    :param target_object_coordinates_array: np.array, the coordinates of the target objects to find the minimum
        distance to.
    :param kwargs: Dict, additional keyword arguments.
    :return:
    """

    distance_prop_map = _g.new_vertex_property(value_type='double')

    for vertex in tqdm.tqdm(_g.iter_vertices(), desc=f"Calculating vertices {kwargs.get('new_prop_map_name', 'min_dist_to_cancer_cells')} from target objects"):
        vertex_min_distance_from_target_objects = property_calc_function(
            _v=vertex,
            _vp_map=_g.vp,
            target_object_coordinates_array=target_object_coordinates_array,
            **kwargs
        )
        distance_prop_map[vertex] = vertex_min_distance_from_target_objects

    if not kwargs.get("in_place", True):
        _g = _g.copy()

    _g.vp[kwargs.get("new_prop_map_name", "min_dist_to_cancer_cells")] = distance_prop_map

    if kwargs.get("save_graph_with_new_property", False):
        if any([kwargs.get("graph_name", None) is None, kwargs.get("dir_to_save_graph_path", None) is None]):
            raise ValueError(f"To save the enriched graph with the new property you must provide "
                             f"the graph name (but got: {kwargs.get('graph_name')})and the "
                             f"path to the directory to save the graph in (but got: {kwargs.get('dir_to_save_graph_path')})!")
        org_graph_name = kwargs.get('graph_name')
        to_add_to_graph_name_str = f'_enriched_with_{kwargs.get("new_prop_map_name", "min_dist_to_cancer_cells")}.gt'
        if org_graph_name.endswith('.gt'):
            new_graph_name = org_graph_name.replace('.gt', to_add_to_graph_name_str)
        else:
            new_graph_name = org_graph_name + to_add_to_graph_name_str

        new_graph_path = os.path.join(kwargs.get("dir_to_save_graph_path"), new_graph_name)
        _g.save(new_graph_path)

    return _g, kwargs.get("new_prop_map_name", "min_dist_to_cancer_cells")


def add_edge_property_to_target_object(_g: gt.Graph,
                                       property_calc_function: Callable = calc_edge_volume,
                                       **kwargs) -> Tuple[gt.Graph, str]:
    """
    Adds an edge property using an HOF structure:
        default behavior is calculating the edges volumes and additing it as a new edge property.

        Usage example - calculating the minimal distance of each vertex to all cancer cells:
            todo

    Additional keyword argumens:
    1. "in_place" - whether to update the vertex property map of the provided graph (True) or return an enriched
        copy of it, default = True. If False is passed, returns a tuple with the new graph instance and the name of the
            new vp. if True, only returns the vp name.
    2. "new_prop_map_name" - the key (i.e., name) of the new property map, default = "min_dist_to_cancer_cells"
    3. "save_graph_with_new_property" - boolean whether to save the new enriched graph.
    4. "graph_name" - the graph region/file name (only used if "save_graph_with_new_property" is True).
    5. "dir_to_save_graph_path" - the path to the directory you wish to save the enriched graph in (only used if
        "save_graph_with_new_property" is True).

    :param property_calc_function: a function with the following signature: func(_e: gt.Edge,
                                               _ep_map: gt.PropertyMap,
                                               **kwargs) -> Any
    :param _g: gt.Graph, target graph to enrich
    :param kwargs: Dict, additional keyword arguments.
    :return:
    """

    new_prop_map = _g.new_edge_property(value_type='double')

    for edge in tqdm.tqdm(_g.iter_edges(), desc="Calculating edges new property"):
        edge = _g.edge(*edge)
        edge_volume = property_calc_function(
            _e=edge,
            _ep_map=_g.ep,
            **kwargs
        )
        new_prop_map[edge] = edge_volume

    if not kwargs.get("in_place", True):
        _g = _g.copy()

    _g.ep[kwargs.get("new_prop_map_name", "edge_volume")] = new_prop_map

    if kwargs.get("save_graph_with_new_property", False):
        if any([kwargs.get("graph_name", None) is None, kwargs.get("dir_to_save_graph_path", None) is None]):
            raise ValueError(f"To save the enriched graph with the new property you must provide "
                             f"the graph name (but got: {kwargs.get('graph_name')})and the "
                             f"path to the directory to save the graph in (but got: {kwargs.get('dir_to_save_graph_path')})!")
        org_graph_name = kwargs.get('graph_name')
        to_add_to_graph_name_str = f'_enriched_with_{kwargs.get("new_prop_map_name", "edge_volume")}.gt'
        if org_graph_name.endswith('.gt'):
            new_graph_name = org_graph_name.replace('.gt', to_add_to_graph_name_str)
        else:
            new_graph_name = org_graph_name + to_add_to_graph_name_str

        new_graph_path = os.path.join(kwargs.get("dir_to_save_graph_path"), new_graph_name)
        _g.save(new_graph_path)

    return _g, kwargs.get("new_prop_map_name", "edge_volume")


if __name__ == '__main__':
    # region_name = "amygdalar capsule"
    gt_graph_path = f"/Users/yishaiazabary/PycharmProjects/University/BrainVasculatureGraphs/Data/GBM_Tumor_Graphs/graph_annotated.gt"
    gt_graph = gt.load_graph(gt_graph_path)
    # cancer_cells_array = np.load('/Users/yishaiazabary/PycharmProjects/University/BrainVasculatureGraphs/Data/1_cells.npy')
    # # get all cancer cells coordinates in region
    # cancer_cells_in_graph_array = np.array(
    #     list(filter(lambda x: x[-1].lower() == region_name.lower(), cancer_cells_array)))
    # # collect only cancer cells coordinates
    # cancer_cells_in_graph_array_only_coordinates = np.array(
    #     list(map(lambda x: tuple(x)[0:3], cancer_cells_in_graph_array)))
    #
    # add_vertex_property_to_target_object(
    #     _g=gt_graph,
    #     target_object_coordinates_array=cancer_cells_in_graph_array_only_coordinates,
    #     dir_to_save_graph_path='.',
    #     graph_name='just a test.gt',
    #     save_graph_with_new_property=True
    # )
    add_edge_property_to_target_object(
        gt_graph,
        property_calc_function=calc_edge_volume,
        save_graph_with_new_property=True,
        graph_name="gbm_graph",
        dir_to_save_graph_path='/Users/yishaiazabary/PycharmProjects/University/BrainVasculatureGraphs/Data/GBM_Tumor_Graphs'

    )
    # dir_path_to_all_graphs = '/Users/yishaiazabary/PycharmProjects/University/BrainVasculatureGraphs/Data/MiceBrainSubgraphs'
    # for graph_fname in tqdm.tqdm(list(filter(lambda x: x.endswith('.gt'),os.listdir(dir_path_to_all_graphs)))):
    #     region_name = graph_fname.split('_')[-1]
    #     region_name = region_name.replace('.gt', '')
    #     # get all cancer cells coordinates in region
    #     cancer_cells_in_graph_array = np.array(
    #         list(filter(lambda x: x[-1].lower() == region_name.lower(), cancer_cells_array)))
    #     # collect only cancer cells coordinates
    #     cancer_cells_in_graph_array_only_coordinates = np.array(
    #         list(map(lambda x: tuple(x)[0:3], cancer_cells_in_graph_array)))
    #
    #     gt_fpath = os.path.join(dir_path_to_all_graphs, graph_fname)
    #
    #     gt_graph = gt.load_graph(gt_fpath)
    #     # gt_graph, _ = add_vertex_property_to_target_object(
    #     #     _g=gt_graph,
    #     #     target_object_coordinates_array=cancer_cells_in_graph_array_only_coordinates,
    #     #     save_graph_with_new_property=False
    #     # )
    #
    #     gt_graph, _ = add_edge_property_to_target_object(
    #         gt_graph,
    #         property_calc_function=calc_edge_volume,
    #         save_graph_with_new_property=True,
    #         graph_name=graph_fname,
    #         dir_to_save_graph_path='/Users/yishaiazabary/PycharmProjects/University/BrainVasculatureGraphs/Data/MiceBrainSubgraphs/WithEdgeVolumeAndMinVertexDistToCancer'
    # )
