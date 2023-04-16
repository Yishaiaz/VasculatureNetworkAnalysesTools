import numpy as np
import graph_tool as gt
from NetworkUtilities.EfficientVoxelSearch import NetworkVoxelSearcher
from ExplicitFeaturesExtractor import ExplicitFeatures
import pandas as pd
from tqdm import tqdm


def gbm_main():
    gbm_path = "/Users/leahbiram/Desktop/vasculature_data/firstGBMscanGraph.gt"
    gbm_features_path = "/Users/leahbiram/Desktop/vasculature_data/firstGBMscanGraph.csv"
    g = gt.load_graph(gbm_path)
    gbm_gt = ExplicitFeatures.preprocess_graph(g)

    features_df = pd.read_pickle(gbm_features_path)
    if len(features_df.index) != gbm_gt.num_vertices():
        raise Exception("Number of nodes in graph is different from number of nodes in features vectors")

    searcher_20 = NetworkVoxelSearcher(complete_network=gbm_gt, vertex_coordinates_attribute_name="coordinates")
    near_vertices_20px = []
    for indx in tqdm(features_df.index, total=len(features_df.index)):
        dist_thresh = 20
        kdtree_found_near_vertices_ids = np.array(searcher_20.find_nodes_within_distance_kdtree(indx, dist_thresh))
        near_vertices_20px.append(len(kdtree_found_near_vertices_ids))

    searcher_100 = NetworkVoxelSearcher(complete_network=gbm_gt, vertex_coordinates_attribute_name="coordinates")
    near_vertices_100px = []
    for indx in tqdm(features_df.index, total=len(features_df.index)):
        dist_thresh = 100
        kdtree_found_near_vertices_ids = np.array(searcher_100.find_nodes_within_distance_kdtree(indx, dist_thresh))
        near_vertices_100px.append(len(kdtree_found_near_vertices_ids))

    new_features = pd.DataFrame({'near_vertices_20px':near_vertices_20px,'near_vertices_100px':near_vertices_100px})
    all_features_df = pd.concat([features_df, new_features], axis=1)
    all_features_df.to_pickle(gbm_features_path)


if __name__ == '__main__':
    gbm_main()
