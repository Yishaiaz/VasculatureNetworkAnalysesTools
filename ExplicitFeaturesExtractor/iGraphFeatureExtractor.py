import numpy as np
import graph_tool.all as gt
import igraph as ig
import graph_tool.all as gt
from igraph import Graph
import warnings
import pickle
import matplotlib.pyplot as plt


def convert_graph_tool_to_igraph(graph_tool_graph: gt.Graph,
                                 include_coordinates: bool = False,
                                 **kwargs) -> ig.Graph:

    all_gt_edges = [(int(_edge.source()), int(_edge.target())) for _edge in graph_tool_graph.edges()]
    igraph_graph = ig.Graph(all_gt_edges, directed=graph_tool_graph.is_directed())
    if graph_tool_graph.num_edges() != igraph_graph.ecount():
        warnings.warn(f"Number of edges is not identical after conversion! # in gt graph: {graph_tool_graph.num_edges()}!= # in igraph graph: {igraph_graph.ecount()}")
    if graph_tool_graph.num_vertices() != igraph_graph.vcount():
        warnings.warn(f"Number of vertices is not identical after conversion! # in gt graph: {graph_tool_graph.num_vertices()}!= # in igraph graph: {igraph_graph.vcount()}")

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


def calc_3d_dist(c_1: list[float], c_2: list[float]) -> int:
    """
    calculate distnace between two 3-d points with x,y,z
    :param c_1: list of 3 floats that are x,y,z coordinates
    :param c_2: list of 3 floats that are x,y,z coordinates
    :return: float that is the distance
    """
    return np.sqrt((c_1[0] - c_2[0]) ** 2 + (c_1[1] - c_2[1]) ** 2 + (c_1[2] - c_2[2]) ** 2)


if __name__ == '__main__':
    gt_graph_path = "/Users/leahbiram/Desktop/vasculature_data/firstGBMscanGraph.gt"
    gt_graph = gt.load_graph(gt_graph_path)
    ig_graph = convert_graph_tool_to_igraph(gt_graph, True)

    v_df = ig_graph.get_vertex_dataframe()
    e_df = ig_graph.get_edge_dataframe()

    e_df.drop('edge_geometry_indices', axis = 1)

    dummy_df = (e_df.merge(v_df, left_on='source', right_on='vertex ID')) #.reindex(columns=['id', 'store', 'address', 'warehouse']))
    e_df = (dummy_df.merge(v_df, left_on='target', right_on='vertex ID', suffixes=['_source', '_target']))

    e_df['distance'] = e_df.apply(lambda row: calc_3d_dist([row.x_source, row.y_source, row.z_source],[row.x_target, row.y_target, row.z_target]), axis=1)
    e_df['prolif'] = e_df['distance']/e_df['length']
    e_df['x_direction'] = (e_df['x_target'] - e_df['x_source'])/e_df['distance']
    e_df['y_direction'] = (e_df['y_target'] - e_df['y_source'])/e_df['distance']
    e_df['z_direction'] = (e_df['z_target'] - e_df['z_source'])/e_df['distance']

    with open('voxel_subgraphs_50.pkl', 'rb') as file:
        voxel_subs = pickle.load(file)

    edges = []
    for vox_inds in voxel_subs:
        pruned_vs = ig_graph.subgraph(vox_inds)
        edges.append(pruned_vs.ecount())
# TODO add number of nodes and number of edges

    #add minimum angle
    print(ig_graph.edge_attributes())

# TODO work with dataframes to save as graphs and convert to pyTorch Geometric.
# TODO possibly check out PyDGN
# TODO try implementing over Yishaia's implementation
# TODO clean repository pull and push

    new_g = Graph.DataFrame(e_df, directed=False)