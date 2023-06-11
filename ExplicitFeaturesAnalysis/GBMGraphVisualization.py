import igraph as ig
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt
import graph_tool as gt


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


def main():

    gt_graph_path = "/Users/leahbiram/Desktop/vasculature_data/firstGBMscanGraph.gt"
    gt_graph = gt.load_graph(gt_graph_path)
    #ig_graph = convert_graph_tool_to_igraph(gt_graph, True)
    ig_graph = ig.from_graph_tool(gt_graph)

    ig_graph["coordinates"] = gt_graph.vp['coordinates']

    # Obtain the coordinates from the graph
    #ig_layout = ig.Layout(coords=gt_graph.vp['coordinates'], dim=3)
    layout = ig_graph.layout(layout_name="coordinates")

    # Create a 3D plot
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')

    # Plot the graph
    ig.plot(ig_graph, layout=layout, target=ax)

    # Show the plot
    plt.show()


if __name__ == '__main__':
    main()

