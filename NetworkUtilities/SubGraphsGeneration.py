import os
from graph_tool.all import *
import tqdm

MAIN_DATA_DIR = '~/BrainVasculatureGraphData'
# MAIN_DATA_DIR = '/Users/yishaiazabary/PycharmProjects/University/BrainVasculatureGraphs/Data'
RAW_GRAPH_FNAME = 'data_graph.gt'
# RAW_GRAPH_FNAME = 'test.gt'
RAW_GRAPH_PATH = os.path.join(MAIN_DATA_DIR, RAW_GRAPH_FNAME)
resize_factors = (50, 10, 5, 2)


if __name__ == '__main__':
    print("Starting....")
    g = load_graph(RAW_GRAPH_PATH)
    n_vertices = g.num_vertices()
    for resize_factor in tqdm.tqdm(resize_factors):
        print(f"generating subgraph with a resize factor={resize_factor}")
        n_vertices_to_copy = n_vertices // resize_factor
        g_small = Graph()
        g_small.set_directed(g.is_directed())
        eprop_color = g_small.new_edge_property("double")
        g_small.edge_properties['e_color'] = eprop_color

        summary = f"Directed: {g.is_directed()}\n"
        summary += f"n_vertices: {g.num_vertices()}\n"
        summary += f"n_edges: {g.num_edges()}\n"
        print(f"Graph_properties:")
        g.list_properties()
        n_version = 0
        for v in g.iter_vertices():
            for v_neighbor in g.iter_out_neighbors(v):
                e = g_small.add_edge(v, v_neighbor)
                eprop_color[e] = 'red' if g.properties[('e', 'artery')][e] else 'blue'
                g_small.save(os.path.join(MAIN_DATA_DIR, f"data_graph_{resize_factor}_#vertices:{g_small.num_vertices()}.gt"))
                print(f"Saved version #{n_version}, current state: #vertices={g_small.num_vertices()}, #edges={g_small.num_edges()}")
                graph_draw(g_small,
                           pos=g_small.properties[('v', 'coordinates')],
                           vertex_size=g_small.properties[('v', 'radii')],
                           edge_marker_size=g_small.properties[('e', 'radii')],
                           vertex_fill_color=g_small.properties[('v', 'artery_raw')],
                           edge_color=eprop_color,
                           output=f'graph_test_{n_version}.pdf')
                n_version += 1
                if g_small.num_vertices() >= n_vertices_to_copy:
                    print(f"Actually Done! Summary:\n{summary}")
                    exit(0)
        g_small.save(os.path.join(MAIN_DATA_DIR, f"test_{resize_factor}_#vertices:{g_small.num_vertices()}.gt"))
        print(f"Read Everything! Summary:\n{summary}")
        del g_small
