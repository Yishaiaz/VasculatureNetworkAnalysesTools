import graph_tool as gt
import numpy as np
from graph_tool import topology, draw
from graph_tool import Graph, Vertex, EdgePropertyMap
from tqdm import tqdm


def calc_3d_dist(c_1: list[float], c_2: list[float]) -> int:
    """
    calculate distnace between two 3-d points with x,y,z
    :param c_1: list of 3 floats that are x,y,z coordinates
    :param c_2: list of 3 floats that are x,y,z coordinates
    :return: float that is the distance
    """
    return np.sqrt((c_1[0] - c_2[0]) ** 2 + (c_1[1] - c_2[1]) ** 2 + (c_1[2] - c_2[2]) ** 2)


def graph_edge_proliferation(_g: Graph, loops: bool = True) -> list[float]:
    """
    For each edge in graph calculate geometric distance between end nodes coordinates and divide by edge length
    :return:
    List of distance/length : list<float>
    """
    prolif = []
    print("calculating graph proliferation: ")
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
    directs = []
    print("calculating graph direction: ")
    num_e = _g.num_edges()
    for _e in tqdm(_g.edges(), total=num_e):
        c_1, c_2 = _g.vp['coordinates'][_e.source()], _g.vp['coordinates'][_e.target()]
        dist = calc_3d_dist(c_2, c_1)
        if dist != 0:
            directs.append([(ax_c1 - ax_c2) / dist for ax_c1, ax_c2 in zip(c_1, c_2)])  # normalized
        else:
            directs.append([0.0, 0.0, 0.0])  # self loop
            print("found self loop when calculating edge direction")
    return directs


def ego_net_distance_N(_g: Graph, ego: Vertex, d: int) -> Graph:
    sd = gt.topology.shortest_distance(_g, ego, weights=_g.edge_properties["length"], max_dist=d)
    ego_g = gt.GraphView(_g, vfilt=sd.a < _g.num_vertices())
    return ego_g


def ego_net_depth_N(_g: Graph, ego: Vertex, n: int) -> Graph:
    d = gt.topology.shortest_distance(_g, ego, max_dist=n)
    ego_g = gt.GraphView(_g, vfilt=d.a < _g.num_vertices())
    return ego_g


def draw_graph_coordinated(_g: Graph, output_path: str):
    g_positions = _g.vp['coordinates']
    draw.graph_draw(_g, pos=g_positions, output=output_path)


def calculate_basic_graph_properties(_g: Graph):
    """
    return num edges, num of vertices, loops, num artery,
    :param _g:
    :return:
    """
    basic_feat = {"n_edges": _g.num_edges(), "n_vertices": _g.num_vertices()}
    return basic_feat


def analyze_properties(_g: Graph, graph_name: str, component: str, properties_to_analyze: list[str]) \
        -> dict[str, float]:
    """
    function from @Yishaia
    :param component: 'v' if vertex property or 'e' if edge property
    :param _g: graph to extract features from
    :param properties_to_analyze: dictionary of key 'edge' or 'vertex' and value as list of property names
    :return: a dictionary with keys 'edge' and 'vertex'
    """
    properties_values_dict = {}
    for property_name in properties_to_analyze:
        property_values = _g.properties[(component, property_name)].fa  # filtered attributes only for subgraph
        if len(property_values) != 0:
            property_mean = float(np.mean(property_values))
            property_std = float(np.std(property_values))
            properties_values_dict[graph_name + '_' + component + '_' + f'{property_name}' + '_mean'] = property_mean
            properties_values_dict[graph_name + '_' + component + '_' + f'{property_name}' + '_std'] = property_std
    return properties_values_dict


def preprocess_graph(_g: Graph, inplace: bool = False) -> Graph:
    """
    MUST use on each network before all other analysis!
    removes duplicate edges and loops. Uses and returns a preprocessed copy of the graph.
    :param inplace:
    :param _g:
    :return: Graph
    """
    if not inplace:
        _g_cpy = _g.copy()
    else:
        _g_cpy = _g

    gt.stats.remove_parallel_edges(_g_cpy)
    gt.stats.remove_self_loops(_g_cpy)

    # Iterate over the vertices and check if they are connected
    for ve in _g_cpy.vertices():
        if ve == _g_cpy.num_vertices():
            break
        # Check if the vertex has any incoming or outgoing edges
        if ve.in_degree() == 0 and ve.out_degree() == 0:
            _g_cpy.remove_vertex(ve)
    _g_cpy = Graph(_g_cpy, prune=True)
    return _g_cpy


def vertex_disconnected(_g: Graph, _v: int) -> bool:
    """
    checking if vertex v is connected to any other vertices in graph by an edge
    :param _g:  the graph
    :param _v: the vertex index in graph
    :return: bool, true if disconnected, false if connected
    """
    return _g.vertex(_v).in_degree() == 0 and _g.vertex(_v).out_degree() == 0


def add_e_graph_float_property(_g: Graph, p_name: str, p_list: list[float]):
    _g.edge_properties[p_name] = _g.new_edge_property("float")
    for i, e in enumerate(_g.edges()):
        _g.edge_properties[p_name][e] = p_list[i]
