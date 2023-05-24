import numpy as np
import graph_tool as gt
from graph_tool import topology, draw
from graph_tool import Graph, Vertex, EdgePropertyMap
from tqdm import tqdm
import math


def unit_vector(vector):
    """ Returns the unit vector of the vector.  """
    return vector / np.linalg.norm(vector)


def graph_edge_volume(_g: Graph) -> list[float]:
    """
    For each edge in graph calculate geometric distance between end nodes coordinates and divide by edge length
    :return:
    List of distance/length : list<float>
    """
    vols = []
    print("calculating graph volume: ")
    for _e in _g.edges():
        vols.append(_g.ep['length'][_e] * math.pi * _g.ep['radii'][_e] ** 2)
    return vols


def angle_between(_g, vec1, vec2):
    """
     Returns the angle in radians between vectors 'v1' and 'v2'::
    """
    vec1_u = unit_vector(vec1)
    vec2_u = unit_vector(vec2)
    return np.arccos(np.clip(np.dot(vec1_u, vec2_u), -1.0, 1.0))


def combinations(test_list):
    return[(a, b) for idx, a in enumerate(test_list) for b in test_list[idx + 1:]]


def graph_vertex_min_angle(_g: Graph):
    min_angle = []
    print("calculating graph min angle: ")
    for _v in tqdm(_g.vertices(), total=_g.num_vertices()):
        source_coords = _g.vp["coordinates"][_v]
        nei = _g.get_all_neighbours(_v)
        edge_combs = combinations(nei)
        angles = []
        for comb in edge_combs:
            coords_0 = _g.vp["coordinates"][comb[0]]
            coords_1 = _g.vp["coordinates"][comb[1]]
            vector0 = [ta - so for ta, so in zip(coords_0, source_coords)]
            vector1 = [ta - so for ta, so in zip(coords_1, source_coords)]
            angles.append(angle_between(_g, vector0, vector1))
        min_angle.append((min(angles, default=0)))
    return min_angle


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
    """
    Returns a subgraph of the original graph based on distance from the ego node.

    @param _g: graph object
    @param ego: ego node
    @param d: distance from ego node
    @return: subgraph of original graph
    """
    sd = gt.topology.shortest_distance(_g, ego, weights=_g.edge_properties["length"], max_dist=d)
    ego_g = gt.GraphView(_g, vfilt=sd.a < _g.num_vertices())
    return ego_g


def ego_net_depth_N(_g: Graph, ego: Vertex, n: int) -> Graph:
    """
    The function calculates the shortest distance from the ego node to all other nodes in the graph using the
    shortest_distance function from the topology module. It then creates a graph view of the original graph using the
    GraphView function, filtering the vertices by those that have a shortest distance.
    Finally, it returns the graph view as the subgraph
    :param _g: graph object
    :param ego: vertex object representing the ego node
    :param n: depth of the subgraph
    :return: graph object of the subgraph
    """
    d = gt.topology.shortest_distance(_g, ego, max_dist=n)
    ego_g = gt.GraphView(_g, vfilt=d.a < _g.num_vertices())
    return ego_g


def ego_net_voxel(_g: Graph, ego: Vertex, all_voxels:list) -> Graph:
    """
    The function calculates the ___. It then creates a graph view of the original graph using the
    GraphView function, filtering the vertices by those that . Finally, it returns the graph view as the subgraph
    :param _g: graph object
    :param ego: vertex object representing the ego node
    :param n: depth of the subgraph
    :return: graph object of the subgraph
    """

    indices_in_voxel = all_voxels[ego]
    ego_g = gt.GraphView(_g, vfilt=lambda v: v in indices_in_voxel)
    return ego_g


def draw_graph_coordinated(_g: Graph, output_path: str):
    """
    The function gets the vertex positions of the graph from the 'coordinates' vertex property of the graph. It then
    uses the graph_draw function from the draw module to draw the graph and save it to the specified output path,
    using the vertex positions as the layout for the graph,
    :param _g: graph property
    :param output_path: string representing the output path for the graph image
    """
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
    v_to_remove = []
    for ve in _g_cpy.vertices():
        if ve == _g_cpy.num_vertices():
            break
        # Check if the vertex has any incoming or outgoing edges
        if ve.in_degree() == 0 and ve.out_degree() == 0:
            v_to_remove.append(ve)
    [_g_cpy.remove_vertex(_ve) for _ve in reversed(v_to_remove)]
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
    """
    The function adds a new edge property to the graph object with the name specified by the p_name parameter and of
    type "float". It then iterates over the edges in the graph and assigns the values in the p_list to the edge
    property of each edge.
    :param _g: graph property
    :param p_name: string representing name of property
    :param p_list:  list of floats for property values
    """
    _g.edge_properties[p_name] = _g.new_edge_property("float")
    for i, e in enumerate(_g.edges()):
        _g.edge_properties[p_name][e] = p_list[i]


def add_v_graph_float_property(_g: Graph, p_name: str, p_list: list[float]):
    """
    The function adds a new vertex property to the graph object with the name specified by the p_name parameter and of
    type "float". It then iterates over the vertices in the graph and assigns the values in the p_list to the vertex
    property of each edge.
    :param _g: graph property
    :param p_name: string representing name of property
    :param p_list:  list of floats for property values
    """
    _g.vertex_properties[p_name] = _g.new_vertex_property("float")
    for i, e in enumerate(_g.vertices()):
        _g.vertex_properties[p_name][e] = p_list[i]


def get_cancer_coords(cancer_cells_array_path: str, graph_region:str):
    """
    Returns the coordinates of cancer cells in a specified region.

    @param cancer_cells_array_path: file path for cancer coordinates
    @param graph_region: name of region for cancer coordinates
    @return: numpy array of cancer cell coordinates and info
    """
    cancer_cells_info = np.load(cancer_cells_array_path)
    region_cancer_info = np.array(list(filter(lambda x: x[-1].lower() == graph_region.lower(), cancer_cells_info)))
    clist = [canc[-1] for canc in cancer_cells_info]
    regions = list(set(clist))
    cancer_coords = np.array(list(map(lambda x: tuple(x)[0:3], region_cancer_info)))
    return cancer_coords
