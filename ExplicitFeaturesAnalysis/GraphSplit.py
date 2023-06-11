import graph_tool.all as gt
import networkx as nx
import igraph as ig


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


# Load the graph from a GT file
g = gt.load_graph("/Users/leahbiram/Desktop/vasculature_data/firstGBMscanGraph.gt")
graph = convert_graph_tool_to_igraph(g, True)

# Define the ratio for splitting the graph
split_ratio = 0.3

# Split the graph based on the ratio
n_vertices = graph.vcount()
split_size = int(n_vertices * split_ratio)

for hop in range(1, 100):
    subgraph_indices = graph.neighborhood(1, order=hop)
    if len(subgraph_indices) > split_size:
        test_indices = graph.neighborhood(1, order=hop-10)
        train_indices = [i for i in range(0, n_vertices) if i not in subgraph_indices]
        break

graph1 = graph.subgraph(train_indices)
graph2 = graph.subgraph(test_indices)

train_graph = graph1.to_networkx()
test_graph = graph2.to_networkx()

nx.write_gpickle(train_graph, "../ExtractedFeatureFiles/GoodTrainTestGraphs/train_graph_10hop_gap.gpickle")
nx.write_gpickle(test_graph, "../ExtractedFeatureFiles/GoodTrainTestGraphs/test_graph_10hop_gap.gpickle")