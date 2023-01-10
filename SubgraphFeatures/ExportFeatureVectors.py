import numpy as np
import os
from SubgraphFeatures import NodeFeatureExtractor as fe
from NetworkUtilities.GraphAttributesEnrichment import add_vertex_distance_to_closest_target_object
import pandas as pd
from tqdm import tqdm
import graph_tool as gt
from graph_tool import Graph


def get_all_subgraph_features(_g: Graph, e_features: list[str], v_features: list[str], sg_type: str, _d: int) \
        -> pd.DataFrame:
    """
    The function iterates over the vertices in the graph (_v), ןf the vertex is disconnected it is ignored.
    If sg_type is "depth", it gets the subgraph on depth n from vertex v using the ego_net_depth_N function.
    If sg_type is "distance", it gets the subgraph on distance n from vertex v using the ego_net_distance_N function.
    The function then creates a dictionary "dict_features" and populates it with the features of the
    subgraph using the analyze_properties function. It then adds the number of vertices, number of edges,
    and ratio of vertices to edges in the subgraph to the dictionary. It then creates a dataframe from the dictionary
    called "dict_2_row" and concatenates it with the "features_df" dataframe, ignoring the index. Finally,
    the function returns the "features_df" dataframe.
    :param v_features: list of vertex features
    :param _g: the original graph object
    :param e_features: list of edge features
    :param sg_type: string representing the type of subgraph to create
    :param _d: integer representing the depth or the distnace from vertex of vertices and edges to include in subgraph
    :return: subgraph features dataframe
    """
    features_df = pd.DataFrame()
    print("proccessing graph features in subgraphs " + sg_type + " " + str(_d) + " :")
    for _v in tqdm(_g.get_vertices()):
        if fe.vertex_disconnected(_g, _v):
            continue
        if sg_type == "depth":
            eg = fe.ego_net_depth_N(_g, _v, _d)  # get subgraph on depth n from vertex v
        elif sg_type == "distance":
            eg = fe.ego_net_distance_N(_g, _v, _d)
        dict_features = {}
        dict_features = fe.analyze_properties(eg, sg_type + str(_d), 'e', e_features)
        dict_features.update(fe.analyze_properties(eg, sg_type + str(_d), 'v', v_features))
        dict_features[sg_type + str(_d) + "_n_vertices"] = eg.num_vertices()
        dict_features[sg_type + str(_d) + "_n_edges"] = eg.num_edges()
        dict_features[sg_type + str(_d) + "_v_e_ratio"] = eg.num_vertices() / eg.num_edges()

        dict_2_row = pd.DataFrame([dict_features])
        features_df = pd.concat([features_df, dict_2_row], ignore_index=True)
    return features_df


def add_full_graph_properties(_g: Graph, cancer_array_path: str, region_name: str):
    """
    Calculates and adds new properties to the input graph object.
    The new properties are 'prolif', 'x_direction', 'y_direction', 'z_direction',
    and the distance of each vertex to the closest target object.

    @param _g: input graph object
    @param cancer_array_path: file path for cancer coordinates
    @param region_name: name of brain region
    """
    # calculating new properties
    proliferation = fe.graph_edge_proliferation(_g)
    directions = fe.graph_edge_direction(_g)
    min_angles = fe.graph_vertex_min_angle(_g)
    cancer_coords = fe.get_cancer_coords(cancer_array_path, region_name)

    # adding new features to graph
    add_vertex_distance_to_closest_target_object(_g, cancer_coords)
    fe.add_v_graph_float_property(_g, "min_angle", min_angles)
    fe.add_e_graph_float_property(_g, "prolif", proliferation)
    fe.add_e_graph_float_property(_g, "x_direction", [i[0] for i in directions])
    fe.add_e_graph_float_property(_g, "y_direction", [i[1] for i in directions])
    fe.add_e_graph_float_property(_g, "z_direction", [i[2] for i in directions])


def main(
        graph_path="/Users/leahbiram/Desktop/vasculature_data/SubGraphsByRegion/subgraph_area_pyramid.gt",
        graph_name: str = "pyramid",
        cancer_cells_array_path="/Users/leahbiram/Desktop/vasculature_data/CancerCellsArray.npy"):
    """
    Loads a graph object, adds new properties to it, extracts subgraph features, and saves the resulting data to a file.

    basic args: graph_path:str, cancer_cells_array_path, subgraph_type:str, dists/depths:list[int],
    # edge/vertex_features:list[str]
    """

    # loading graphs
    gs = gt.load_graph(graph_path)  # firstGBMscanGraph/subgraph_area_*_Hypothalamus/ Striatum/ _Isocortex
    g = fe.preprocess_graph(gs)

    subgraph_type = "depth"  # depth/ distance
    dists = [1, 2]  # for max depth or distance
    edge_features = ["radii", "length", "artery_binary", "prolif", "x_direction", "y_direction", "z_direction"]
    vertex_features = ["radii", "min_angle"]

    add_full_graph_properties(g, cancer_cells_array_path, graph_name)  # prolif, direction, cancer dist

    all_features_df = pd.DataFrame()
    for d in dists:
        subgraph_features = get_all_subgraph_features(g, edge_features, vertex_features, subgraph_type, d)
        all_features_df = pd.concat([all_features_df, subgraph_features], axis=1)

    #adding specific features not from subgrphs
    cancer_dists = g.vertex_properties["min_dist_to_cancer_cells"].fa
    all_features_df["min_dist_to_cancer_cells"] = [c_dist for c_dist, cv in zip(cancer_dists, g.vertices())
                                                   if not fe.vertex_disconnected(g, cv)]

    # exporting all features to file
    os.makedirs('../ExtractedFeatureVectors', exist_ok=True)
    all_features_df.to_pickle("../ExtractedFeatureVectors/" + graph_name + ".csv")


if __name__ == "__main__":
    main()
