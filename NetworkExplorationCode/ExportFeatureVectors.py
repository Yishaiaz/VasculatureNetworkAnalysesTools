import NodeFeatureExtractor as fe
import pandas as pd
from tqdm import tqdm
import graph_tool as gt
from graph_tool import Graph


def get_all_subgraph_features(_g: Graph, edge_features: list[str], vertex_features: list[str], sg_type: str, _d: int) \
        -> pd.DataFrame:
    features_df = pd.DataFrame()
    num_v = _g.num_vertices()
    print("proccessing graph features: ")
    for _v in tqdm(_g.get_vertices()):
        if fe.vertex_disconnected(_g, _v):
            continue
        if sg_type == "depth":
            eg = fe.ego_net_depth_N(_g, _v, _d)  # get subgraph on depth n from vertex v
        elif sg_type == "distance":
            eg = fe.ego_net_distance_N(_g, _v, _d)
        dict_features = {}
        dict_features = fe.analyze_properties(eg, sg_type + '_4', 'e', edge_features)
        dict_features["n_vertices"] = eg.num_vertices()
        dict_features["n_edges"] = eg.num_edges()

        dict_2_row = pd.DataFrame([dict_features])
        features_df = pd.concat([features_df, dict_2_row], ignore_index=True)
    return features_df


# basic args
graph_name = "subgraph_area_corpus_callosum"

# loading graphs
gs = gt.load_graph(
    "/Users/leahbiram/Desktop/vasculature_data/" + graph_name + ".gt")  # firstGBMscanGraph/subgraph_area_*_Hypothalamus/ Striatum/ _Isocortex
print("loaded graph " + graph_name)
g = fe.preprocess_graph(gs)

# calculating new properties
proliferation = fe.graph_edge_proliferation(g)
directions = fe.graph_edge_direction(g)

# adding new features to graph
fe.add_e_graph_float_property(g, "prolif", proliferation)
fe.add_e_graph_float_property(g, "x_direction", [i[0] for i in directions])
fe.add_e_graph_float_property(g, "y_direction", [i[1] for i in directions])
fe.add_e_graph_float_property(g, "z_direction", [i[2] for i in directions])

subgraph_type = "depth"  # depth/ distance
d = 3  # for max depth or distance
edge_features = ["radii", "length", "artery_binary", "prolif", "x_direction", "y_direction", "z_direction"]
vertex_features = ["radii"]
subgraph_features = get_all_subgraph_features(g, edge_features, vertex_features, subgraph_type, d)

all_features_df = pd.DataFrame()
all_features_df = pd.concat([all_features_df, subgraph_features], ignore_index=True)

# exporting all features to file
all_features_df.to_pickle("../ExtractedFeatureVectors/" + graph_name + ".csv")
print("finished")

