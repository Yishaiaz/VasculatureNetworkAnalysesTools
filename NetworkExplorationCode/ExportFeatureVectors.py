import NodeFeatureExtractor as fe
import pandas as pd
from tqdm import tqdm
import graph_tool as gt


#basic args
graph_name = "subgraph_area_corpus_callosum"
subgraph_type = "depth"  # depth/ distance
d = 4  # for max depth or distance

# loading graphs
gs = gt.load_graph("/Users/leahbiram/Desktop/vasculature_data/"+graph_name+".gt")  # firstGBMscanGraph/subgraph_area_*_Hypothalamus/ Striatum/ _Isocortex
print("loaded graph " + graph_name)

g = fe.preprocess_graph(gs)  # preproccess does not copy properties!

proliferation = fe.graph_edge_proliferation(g)
directions = fe.graph_edge_direction(g)

# calculating new properties

g.edge_properties["prolif"] = g.new_edge_property("float")
# g.edge_properties["direction"] = g.new_edge_property("vector<double>")
g.edge_properties["x_direction"] = g.new_edge_property("float")
g.edge_properties["y_direction"] = g.new_edge_property("float")
g.edge_properties["z_direction"] = g.new_edge_property("float")
for i, e in enumerate(g.edges()):
    g.edge_properties["prolif"][e] = proliferation[i]
    g.edge_properties["x_direction"][e] = directions[i][0]
    g.edge_properties["y_direction"][e] = directions[i][1]
    g.edge_properties["z_direction"][e] = directions[i][2]


features_df = pd.DataFrame()
edge_features = ["radii", "length", "artery_binary", "prolif", "x_direction", "y_direction", "z_direction"]
vertex_features = ["radii"]
num_v = g.num_vertices()
print("proccessing graph features: ")
for v in tqdm(g.get_vertices()):
    if fe.vertex_disconnected(g, v):
        continue
    if subgraph_type == "depth":
        eg = fe.ego_net_depth_N(g, v, d)  # get subgraph on depth n from vertex v
    elif subgraph_type == "distance":
        eg = fe.ego_net_distance_N(g, v, d)
    dict_features = {}
    dict_features = fe.analyze_properties(eg, 'ego_depth_4', 'e', edge_features)
    dict_features["n_vertices"] = eg.num_vertices()
    dict_features["n_edges"] = eg.num_edges()

    dict_2_row = pd.DataFrame([dict_features])
    features_df = pd.concat([features_df, dict_2_row], ignore_index=True)

# exporting all features to file
features_df.to_pickle("../ExtractedFeatureVectors/"+graph_name+".csv")

#features_df.hist(figsize=(10, 10), bins=100, color="purple", ec="purple")
print("finished")
#plt.subplot(3, 1, 1)
#plt.title("proliferation in vasc network")
#plt.hist(proliferation, bins="auto")
#plt.show()
