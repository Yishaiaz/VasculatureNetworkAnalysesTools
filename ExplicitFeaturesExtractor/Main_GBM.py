import graph_tool.all as gt
import networkx as nx
import pandas as pd
import igraph as ig
from iGraphFeatureExtractor import extract_features
from GBMPredictions import run_compare_multiple_dfs, run_leave1_out
import ast

def convert_graph_tool_to_igraph(graph_tool_graph: gt.Graph,
                                 include_coordinates: bool = False,
                                 **kwargs) -> ig.Graph:

    all_gt_edges = [(int(_edge.source()), int(_edge.target())) for _edge in graph_tool_graph.edges()]
    igraph_graph = ig.Graph(all_gt_edges, directed=graph_tool_graph.is_directed())

    for prop in graph_tool_graph.vertex_properties.keys():
        print(f"adding vertex property '{prop}' to iGraph")
        if prop == "coordinates" and include_coordinates:
            igraph_graph.vs[prop] = [graph_tool_graph.vp[prop][v_] for v_ in graph_tool_graph.vertices()]
        else:
            igraph_graph.vs[prop] = graph_tool_graph.vertex_properties[prop].get_array()#.tolist()

    for prop in graph_tool_graph.edge_properties.keys():
        print(f"adding edge property '{prop}' to iGraph")
        igraph_graph.es[prop] = graph_tool_graph.edge_properties[prop].get_array()# .tolist()

    return igraph_graph


def split_graph(graph, split_ratio, save_as_graph, gap=0):

    start_node = 5
    # Split the graph based on the ratio
    n_vertices = graph.vcount()
    split_size = int(n_vertices * split_ratio)

    for hop in range(1, 200):
        subgraph_indices = graph.neighborhood(start_node, order=hop)
        if len(subgraph_indices) > split_size:
            test_indices = graph.neighborhood(start_node, order=hop-gap)
            train_indices = [i for i in range(0, n_vertices) if i not in subgraph_indices]
            print("----- train-test graph split -----")
            print(f"train test gap: {gap}")
            print(f"hops in test: {hop-gap}")
            print(f"Total vertices: {n_vertices}, Test: {len(test_indices)}, "
                  f"Train: {len(train_indices)}, left out: {n_vertices-len(train_indices)-len(test_indices)}")
            break

    if save_as_graph:
        graph1 = graph.subgraph(train_indices)
        graph2 = graph.subgraph(test_indices)

        train_graph = graph1.to_networkx()
        test_graph = graph2.to_networkx()

        nx.write_gpickle(train_graph, f"../ExtractedFeatureFiles/GoodTrainTestGraphs/train_graph_{gap}hop_gap.gpickle")
        nx.write_gpickle(test_graph, f"../ExtractedFeatureFiles/GoodTrainTestGraphs/test_graph_{gap}hop_gap.gpickle")

    return train_indices, test_indices


def main():
    orig_graph_file = "/Users/leahbiram/Desktop/vasculature_data/firstGBMscanGraph.gt"
    split_ratio = 0.3  # train-test split estimate before gap
    save_train_test_graphs = False  # saving plit train and test graphs after splitting
    split_gap = 18  # size of gap between test and train sets to leave out (in hops)
    feature_extractor_activate = False  # True if features were not calculated or new features were added
    include_counts = False  # including degree, v count and e count in features
    threshold = 0.3  # decision threshold for prediction probability

    # Load the graph from a GT file
    g = gt.load_graph(orig_graph_file)
    ig_graph = convert_graph_tool_to_igraph(g, True)

    train_indices, test_indices = split_graph(ig_graph, split_ratio, save_train_test_graphs, split_gap)

    features_dfs = []
    for n_hops in range(8, 9):  # range(0,1) for voxel instead of hop
        if feature_extractor_activate:
            output_features_path = f"../ExtractedFeatureFiles/firstGBMFeatures_{n_hops}_Hop.csv"
            df_ = extract_features(ig_graph, n_hops, output_features_path, save_to_file=False)
        else:
            df_ = pd.read_csv(f"../ExtractedFeatureFiles/firstGBMFeatures_{n_hops}_Hop.csv")
        features_dfs.append([df_.iloc[train_indices], df_.iloc[test_indices]])

    run_compare_multiple_dfs(features_dfs,
                             model_name="hops",
                             threshold=threshold,
                             include_counts=include_counts,
                             scatter_predictions=True)


if __name__ == "__main__":
    main()

