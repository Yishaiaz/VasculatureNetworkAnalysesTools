import matplotlib.pyplot as plt
import graph_tool as gt
import numpy as np
from graph_tool import topology, draw
import matplotlib.pyplot as plt
import random
from graph_tool import Graph, Vertex, EdgePropertyMap


def calc_3d_dist(c_1: list[float], c_2: list[float]) -> int:
    '''
    calculate distnace between two 3-d points with x,y,z
    :param c_1: list of 3 floats that are x,y,z coordinates
    :param c_2: list of 3 floats that are x,y,z coordinates
    :return: float that is the distance
    '''
    return np.sqrt((c_1[0] - c_2[0]) ** 2 + (c_1[1] - c_2[1]) ** 2 + (c_1[2] - c_2[2]) ** 2)


def graph_edge_proliferation(_g: Graph, loops: bool = True) -> list[float]:
    """
    For each edge in graph calculate geometric distance between end nodes coordinates and divide by edge length
    :return:
    List of distance/length : list<float>
    """
    prolif = []
    for _e in _g.edges():
        s_coords = _g.vp['coordinates'][_e.source()]
        t_coords = _g.vp['coordinates'][_e.target()]
        dist = calc_3d_dist(s_coords, t_coords)
        _len = _g.ep['length'][_e]
        prolif.append(float(dist / _len))
    return prolif


def graph_edge_direction(_g: Graph) -> list[float]:
    """
    For each edge in graph calculate geometric direction
    :return:
    List of 3D coords : list<list<float>>
    """
    directions = []
    for _e in _g.edges():
        c_1, c_2 = _g.vp['coordinates'][_e.source()], _g.vp['coordinates'][_e.target()]
        dist = calc_3d_dist(c_2, c_1)
        if dist != 0:
            directions.append((c_2 - c_1)/dist)  # normalized
        else:
            directions.append([0.0, 0.0, 0.0])  # self loop
    return directions


def get_node_attributes_dist_N(v: Vertex) -> dict[str, any]:
    pass


def ego_net_depth_N(_g: Graph, ego: Vertex, n: int) -> Graph:
    d = gt.topology.shortest_distance(_g, ego, max_dist=n)
    ego_g = gt.GraphView(_g, vfilt=d.a < _g.num_vertices())
    return ego_g


def draw_graph_coordinated(_g: Graph, output_path: str):
    g_positions = _g.vp['coordinates']
    draw.graph_draw(_g, pos=g_positions, output=output_path)


def add_node_property(_g: Graph, prop_lis: list):
    new_prop = _g.new_edge_property('vector<double>')
    _g.edge_properties['new_prop'] = new_prop


def calculate_basic_properties(_g: Graph):
    '''
    return num edges, num of vertices, loops, num artery,
    :param _g:
    :return:
    '''
    print(_g.graph_properties)
    basic_feat = {}
    basic_feat["n_edges"] = _g.num_edges()
    basic_feat["n_vertices"] = _g.num_vertices()
    return basic_feat


def calculate_graph_properties(_g: Graph,
                               properties_to_analyze: dict[str, list[str]]) \
        -> dict[str, dict[str, float]]:
    '''
    function from @Yishaia
    :param _g: graph to extract features from
    :param properties_to_analyze: dictionary of key 'edge' or 'vertex' and value as list of property names
    :return: a dictionary with keys 'edge' and 'vertex'
    '''
    component_to_properties_values_dict = {}
    for component, component_properties_to_analyze in properties_to_analyze.items():
        component_to_properties_values_dict[component] = {}
        component_rep = 'e' if 'edge' in component.lower() else 'v'
        # component_g_props = _g.properties[component_rep]
        for property_name in component_properties_to_analyze:
            property_values = _g.properties[(component_rep, property_name)].a
            property_mean = np.mean(property_values)
            property_std = np.std(property_values)
            component_to_properties_values_dict[component][f'{property_name}'] = {}
            component_to_properties_values_dict[component][f'{property_name}']['mean'] = property_mean
            component_to_properties_values_dict[component][f'{property_name}']['std'] = property_std
    return component_to_properties_values_dict


def preprocess_graph(_g: Graph, inplace:bool = False) -> Graph:
    """
    MUST use on each network before all other analysis!
    removes duplicate edges and loops. Uses and returns a preprocessed copy of the graph.
    :param _g:
    :return: Graph
    """
    if not inplace:
        _g_cpy = _g.copy()
    else:
        _g_cpy = _g

    gt.stats.remove_parallel_edges(_g_cpy)
    gt.stats.remove_self_loops(_g_cpy)

    return _g_cpy


# loading graphs
gs = gt.load_graph("/Users/leahbiram/Desktop/vasculature_data/subgraph_area_Striatum.gt")  # _Hypothalamus/ _Isocortex
print("loaded graphs (Striatum):")

g = preprocess_graph(gs) # preproccess does not copy properties!
#sub = gt.Graph(eg, directed=False, prune=True)

proliferation = graph_edge_proliferation(g)
directions = graph_edge_direction(g)

# calculating new properties

g.edge_properties["prolif"] = g.new_edge_property("float")
g.edge_properties["direction"] = g.new_edge_property("vector<double>")
for i, e in enumerate(g.edges()):
    g.edge_properties["prolif"][e] = proliferation[i]
    g.edge_properties["direction"][e] = directions[i]

g_features = []
for v in g.get_vertices():
    eg = ego_net_depth_N(g, v, 4) #get subgraph on depth n from vertex v
    features_dict = {"vertex": ["radii"],
                     "edge": ["radii", "length", "artery_binary", "prolif"]}  # mean_binary artery should give fraction
    g_features.append(calculate_graph_properties(eg, features_dict))
    # calculate_basic_properties()

#plt.subplot(3,1,1)
plt.hist(proliferation, bins="auto")
plt.show()


# graph draw options
#draw_graph_coordinated(g, "../GraphVisualization/graph-ego4_proc.pdf")
# draw.graph_draw(g, pos=g_positions, edge_pen_width=edge_prolif, output="graph-Str2.pdf")
#            ,edge_control_points = control)  # some curvy edges
