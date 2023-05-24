import numpy as np
from scipy.spatial import KDTree
import graph_tool.all as gt

def add_distance_attributes(graph_file, point_cloud_file, threshold):
    # Load graph and point cloud from file
    G = gt.load_graph(graph_file)
    point_cloud_coords = np.load(point_cloud_file, allow_pickle=True)

    # Build KD-tree from point cloud coordinates
    tree = KDTree(point_cloud_coords)

    # Calculate minimum distance and count number of points within threshold for each vertex
    min_distances = np.zeros(G.num_vertices())
    num_points_within_threshold = np.zeros(G.num_vertices())
    for i, v in enumerate(G.vertices()):
        v_pos = G.vp.pos[v]
        dist, idx = tree.query(v_pos, k=1)
        min_distances[i] = dist
        num_points_within_threshold[i] = np.sum(tree.query_ball_point(v_pos, threshold))

    # Add attributes to graph vertices
    G.vertex_properties['min_distance'] = G.new_vertex_property('float')
    G.vertex_properties['num_points_within_threshold'] = G.new_vertex_property('int')
    for i, v in enumerate(G.vertices()):
        G.vertex_properties['min_distance'][v] = min_distances[i]
        G.vertex_properties['num_points_within_threshold'][v] = num_points_within_threshold[i]

    return G


# Load graph and point cloud from file
graph_file = "/Users/leahbiram/Desktop/vasculature_data/CD31-graph_reduced.gt"
marker_point_cloud_file = "/Users/leahbiram/Desktop/vasculature_data/CD3-stitched.npy"
threshold = 0.5

# Add distance attributes to graph
G = add_distance_attributes(graph_file, marker_point_cloud_file, threshold)

# Save graph to file or use it for further analysis
G.save('gbm2_with_marker_attributes.gt')
