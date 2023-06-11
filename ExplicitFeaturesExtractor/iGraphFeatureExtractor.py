import numpy as np
import pandas as pd
import igraph as ig
import graph_tool.all as gt
import warnings
import pickle
from tqdm import tqdm
import networkx as nx
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D


def convert_graph_tool_to_igraph(graph_tool_graph: gt.Graph,
                                 include_coordinates: bool = False,
                                 **kwargs) -> ig.Graph:

    all_gt_edges = [(int(_edge.source()), int(_edge.target())) for _edge in graph_tool_graph.edges()]
    igraph_graph = ig.Graph(all_gt_edges, directed=graph_tool_graph.is_directed())

    for prop in graph_tool_graph.vertex_properties.keys():
        print(f"adding vertex property '{prop}' to iGraph")
        if prop == "coordinates" and include_coordinates:
            igraph_graph.vs["x"] = [c[0] for c in graph_tool_graph.vertex_properties[prop]]
            igraph_graph.vs["y"] = [c[1] for c in graph_tool_graph.vertex_properties[prop]]
            igraph_graph.vs["z"] = [c[2] for c in graph_tool_graph.vertex_properties[prop]]
        else:
            igraph_graph.vs[prop] = graph_tool_graph.vertex_properties[prop].get_array()#.tolist()

    for prop in graph_tool_graph.edge_properties.keys():
        print(f"adding edge property '{prop}' to iGraph")
        igraph_graph.es[prop] = graph_tool_graph.edge_properties[prop].get_array()# .tolist()

    return igraph_graph


def get_voxel_subgraphs(ig_graph, voxel_file):
    with open(voxel_file, 'rb') as file:
        voxel_subs = pickle.load(file)
    print("creating voxel subgraphs... ")
    subgraphs=[]
    for vox_inds in tqdm(voxel_subs):
        subgraphs.append(ig_graph.subgraph(vox_inds))
    #subgraphs = [ig_graph.subgraph(vox_inds) for vox_inds in tqdm(voxel_subs)]
    return subgraphs


def get_hops_subgraphs(ig_graph, n_hops=2):
    print(f"creating {n_hops}-hops subgraphs... ")
    subgraphs = [ig_graph.subgraph(ig_graph.neighborhood(node, order=n_hops)) for node in tqdm(ig_graph.vs)]
    return subgraphs


def calc_3d_dist(c_1: list[float], c_2: list[float]) -> int:
    """
    calculate distnace between two 3-d points with x,y,z
    :param c_1: list of 3 floats that are x,y,z coordinates
    :param c_2: list of 3 floats that are x,y,z coordinates
    :return: float that is the distance
    """
    return np.sqrt((c_1[0] - c_2[0]) ** 2 + (c_1[1] - c_2[1]) ** 2 + (c_1[2] - c_2[2]) ** 2)


def extract_features(ig_graph, n_hops=0, output_features_path="./", voxel_file="./", save_to_file = True):

    v_df = ig_graph.get_vertex_dataframe()
    v_df[['x', 'y', 'z']] = pd.DataFrame(v_df.coordinates.tolist(), index= v_df.index)
    e_df = ig_graph.get_edge_dataframe()

    e_df.drop('edge_geometry_indices', axis=1)

    dummy_df = (e_df.merge(v_df, left_on='source',
                           right_on='vertex ID'))  # .reindex(columns=['id', 'store', 'address', 'warehouse']))
    e_df = (dummy_df.merge(v_df, left_on='target', right_on='vertex ID', suffixes=['_source', '_target']))

    e_df['distance'] = e_df.apply(lambda row: calc_3d_dist([row.x_source, row.y_source, row.z_source],
                                                           [row.x_target, row.y_target, row.z_target]), axis=1)

    ig_graph.es['prolif'] = e_df['distance'] / e_df['length']
    ig_graph.es['x_direction'] = (e_df['x_target'] - e_df['x_source']) / e_df['distance']
    ig_graph.es['y_direction'] = (e_df['y_target'] - e_df['y_source']) / e_df['distance']
    ig_graph.es['z_direction'] = (e_df['z_target'] - e_df['z_source']) / e_df['distance']
    ig_graph.es['distance'] = e_df['distance']

    if n_hops == 0:
        subgraphs = get_voxel_subgraphs(ig_graph, voxel_file)
    else:
        subgraphs = (get_hops_subgraphs(ig_graph, n_hops))

    average_lengths, average_radii, average_distance = [], [], []
    average_x_dir, average_y_dir, average_z_dir = [], [], []
    std_lengths, std_radii, std_distance = [], [], []
    std_x_dir, std_y_dir, std_z_dir = [], [], []
    edge_count, vertex_count, marker_in_subgraph = [], [], []

    print("calculating average and std of features...")
    for subgraph in tqdm(subgraphs):
        average_lengths.append(np.average(subgraph.es["length"]))
        std_lengths.append(np.average(subgraph.es["length"]))

        average_radii.append(np.average(subgraph.es["radii"]))
        std_radii.append(np.average(subgraph.es["radii"]))

        average_distance.append(np.average(subgraph.es["distance"]))
        std_distance.append(np.average(subgraph.es["distance"]))

        average_x_dir.append(np.average(subgraph.es["x_direction"]))
        std_x_dir.append(np.average(subgraph.es["x_direction"]))
        average_y_dir.append(np.average(subgraph.es["y_direction"]))
        std_y_dir.append(np.average(subgraph.es["y_direction"]))
        average_z_dir.append(np.average(subgraph.es["z_direction"]))
        std_z_dir.append(np.average(subgraph.es["z_direction"]))

        marker_in_subgraph.append(int(any(x == 1 for x in subgraph.es["artery_binary"])))
        edge_count.append(subgraph.ecount())
        vertex_count.append(subgraph.vcount())

    # all features into one dataframe
    ig_graph.vs["length_avg"] = average_lengths
    ig_graph.vs["radii_avg"] = average_radii
    ig_graph.vs["distance_avg"] = average_distance
    ig_graph.vs["x_dir_avg"] = average_x_dir
    ig_graph.vs["y_dir_avg"] = average_y_dir
    ig_graph.vs["z_dir_avg"] = average_z_dir

    ig_graph.vs["length_std"] = std_lengths
    ig_graph.vs["radii_std"] = std_radii
    ig_graph.vs["distance_std"] = std_distance
    ig_graph.vs["x_dir_std"] = std_x_dir
    ig_graph.vs["y_dir_std"] = std_y_dir
    ig_graph.vs["z_dir_std"] = std_z_dir

    ig_graph.vs["marker_in_subgraph"] = marker_in_subgraph
    ig_graph.vs["e_count"] = edge_count
    ig_graph.vs["v_count"] = vertex_count
    ig_graph.vs['degree'] = ig_graph.degree()

    # TODO optional add minimum angle, or other angle cals from direction?
    feats_df = ig_graph.get_vertex_dataframe()

    if save_to_file:
        feats_df.to_csv(output_features_path)

    return feats_df


if __name__ == '__main__':
    #    gt_graph_path = "/Users/leahbiram/Desktop/vasculature_data/firstGBMscanGraph.gt"
    #    gt_graph = gt.load_graph(gt_graph_path)
    #    ig_graph = convert_graph_tool_to_igraph(gt_graph, True)

    nx_graph = nx.read_gpickle("../ExtractedFeatureFiles/GoodTrainTestGraphs/test_graph_10hop_gap.gpickle")
    ig_graph = ig.Graph.from_networkx(nx_graph)

    #range(0,1) for voxel instead of hop
    for n_hops in range(1,11):
        output_features_path = f"../ExtractedFeatureFiles/GoodTrainTestGraphs/firstGBMFeatures_{n_hops}_Hop_test.csv"
        extract_features(ig_graph, n_hops, output_features_path)# voxel_file='../ExtractedFeatureFiles/full_voxel_subgraphs_150.pkl') #voxel_file='../ExtractedFeatureFiles/full_voxel_subgraphs_200.pkl'

# TODO convert to pyTorch Geometric.
# TODO possibly check out PyDGN
# TODO try implementing over Yishaia's implementation
# TODO add each nodes distance in subgraph
# TODO artery_binary for full subgraph
