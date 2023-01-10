import os
import sys
import shutil
import math
import json
import warnings
from typing import *
from enum import Enum
import datetime
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import graph_tool.all as gt
from node2vec.node2vec import *
from node2vec.edges import *
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
import argparse
from NetworkUtilities.GraphAttributesEnrichment import *
from PhuzzyMotifEmbedding import *


LOGS_DIR_PATH = '~/LOGS'
os.makedirs(LOGS_DIR_PATH, exist_ok=True)

LOG_file_PATH = os.path.join(LOGS_DIR_PATH, f'log_{datetime.datetime.now().strftime("%d|%m|%Y_%H|%M|%S")}.txt')
log_text = f'LogEntry: {datetime.datetime.now().strftime("%d/%m/%Y %H:%M:%S")}\n'
with open(LOG_file_PATH, 'w') as f:
    f.write(log_text)

parser = argparse.ArgumentParser(prog='Node2VecEmbeddingsAndCancerCellsDistanceExploration', description='')
parser.add_argument('--job_id', type=str,
                    help='job id as string')

args = parser.parse_args()

cluster_job_id = args.job_id

def get_cancer_cells_data(
        cancer_cells_array_path: Union[str, os.PathLike] =
        '/Users/yishaiazabary/PycharmProjects/University/BrainVasculatureGraphs/Data/1_cells.npy',
        **kwargs
):
    global LOG_file_PATH
    cancer_cells_array = np.load(cancer_cells_array_path)
    cancer_cells_atlas_as_np = np.stack([single_instance[-1] for single_instance in cancer_cells_array])
    cancer_cells_regions_counts = np.unique(cancer_cells_atlas_as_np, return_counts=True)
    cancer_cells_regions_counts_sorted = sorted(zip(cancer_cells_regions_counts[0], cancer_cells_regions_counts[1]),
                                                key=lambda x: x[1], reverse=True)
    return cancer_cells_regions_counts_sorted


def calc_num_cancer_cells_per_graph(cancer_cells_regions_counts_sorted: List[Tuple[str, int]],
                                    regions_graphs_dir_path: Union[str, os.PathLike],
                                    path_to_save_results_dir: Union[str, os.PathLike] = "Results/MiceBrain/",
                                    **kwargs):
    global LOG_file_PATH
    filtered_cancer_cells_regions_counts_sorted = {}
    for region_descriptor in cancer_cells_regions_counts_sorted[1:]:
        region_name, region_n_cancer_cells = region_descriptor
        region_graph_path = os.path.join(regions_graphs_dir_path, f"subgraph_area_{region_name}.gt")
        if os.path.exists(region_graph_path):
            region_graph = gt.load_graph(region_graph_path)
            filtered_cancer_cells_regions_counts_sorted[region_name] = {
                'n_cancer_cells': int(region_n_cancer_cells),
                'region_n_vertices': int(region_graph.num_vertices()),
                'region_n_edges': int(region_graph.num_edges())
            }
        elif kwargs.get('verbose', 0) > 2:
            with open(LOG_file_PATH, 'a') as f:
                log_text = f"Could not find graph {region_name}, skipping.."
                f.write(log_text)
    # saving the results for later use:
    os.makedirs(path_to_save_results_dir, exist_ok=True)
    with open(
            os.path.join(path_to_save_results_dir, f"MiceBrainRegionsCancerCellsCountsAndGraphSizes.json"),
            'w') as f:
        json.dump(filtered_cancer_cells_regions_counts_sorted, f)

    return filtered_cancer_cells_regions_counts_sorted


def calc_min_cancer_cells_distance_and_node2vec_embeddings(
        filtered_cancer_cells_regions_counts_sorted: Dict[str, Dict[str, int]],
        embeddings_columns_names: List,
        regions_graphs_dir_path: Union[str, os.PathLike] = "/Users/yishaiazabary/PycharmProjects/University/BrainVasculatureGraphs/Data/MiceBrainSubgraphs",
        cancer_cells_array_path: Union[str, os.PathLike] =
        '/Users/yishaiazabary/PycharmProjects/University/BrainVasculatureGraphs/Data/1_cells.npy',
        embeddings_results_directory_path: Union[str, os.PathLike] =
        "/Users/yishaiazabary/PycharmProjects/University/BrainVasculatureGraphs/UnsupervisedGraphTraversal/vertex_embeddings/",
        **kwargs
                                                           ):
    global LOG_file_PATH
    # cancer cells coordinates
    cancer_cells_array = np.load(cancer_cells_array_path)
    # embeddings directory
    os.makedirs(embeddings_results_directory_path, exist_ok=True)

    all_regions_vertices_embeddings_df = None

    for region_name, cancer_descriptor in filtered_cancer_cells_regions_counts_sorted.items():
        single_region_vertices_embeddings_df = pd.DataFrame(
            columns=['region_name'] +
                    embeddings_columns_names +
                    ['pc_0', 'pc_1', 'tsne_0', 'tsne_1', 'min_dist_to_cancer_cells']
        )
        with open(LOG_file_PATH, 'a') as f:
            log_text = f"Processing region: {region_name} with #CancerCells={cancer_descriptor['n_cancer_cells']} and #Vertices= {cancer_descriptor['region_n_vertices']} and #Edges={cancer_descriptor['region_n_edges']}"
            f.write(log_text)

        # get all cancer cells coordinates in region
        cancer_cells_in_graph_array = np.array(
            list(filter(lambda x: x[-1].lower() == region_name.lower(), cancer_cells_array)))
        # collect only cancer cells coordinates
        cancer_cells_in_graph_array_only_coordinates = np.array(
            list(map(lambda x: tuple(x)[0:3], cancer_cells_in_graph_array)))

        # read the region's graph:
        gt_graph_path = os.path.join(regions_graphs_dir_path, f"subgraph_area_{region_name}.gt")
        region_graph = gt.load_graph(gt_graph_path)
        # defining results directories and validating it exists.
        regions_embeddings_results_dir_path = os.path.join(embeddings_results_directory_path, region_name)
        os.makedirs(regions_embeddings_results_dir_path, exist_ok=True)
        regions_embeddings_results_file_path = os.path.join(regions_embeddings_results_dir_path,
                                                            f"node2vec_vertices_embeddings_dimSize{embeddings_dim_size}_walkSize{walk_len}_nWalks{n_walks}.npy")

        # calculate region's vertices' embeddings and saving them:
        regions_vertices_embeddings = calculate_gt_graph_node_embeddings(
            region_graph,
            edges_embedders_method=None,  # edges_embedders_method,
            save_node_embeddings_path=regions_embeddings_results_file_path,
            vertices_embedding_module=Node2Vec,
            embd_dims=embeddings_dim_size,
            walks_len=walk_len,
            num_of_walks=n_walks
        )
        # update the single region dataframe with vertices embeddings:
        single_region_vertices_embeddings_df[embeddings_columns_names] = regions_vertices_embeddings
        single_region_vertices_embeddings_df[region_name] = region_name
        single_region_vertices_embeddings_df[region_name] = region_name

        # calculating dim reduction for region's embeddings
        pca_module = PCA(n_components=2)
        tsne_module = TSNE(n_components=2, verbose=kwargs.get('verbose', 0), perplexity=40, n_iter=300)
        if kwargs.get('verbose', 0) > 1:
            with open(LOG_file_PATH, 'a') as f:
                log_text = "calculating PCA"
                f.write(log_text)

        vertices_embeddings_pca = pca_module.fit_transform(regions_vertices_embeddings)
        if kwargs.get('verbose', 0) > 1:
            with open(LOG_file_PATH, 'a') as f:
                log_text = "calculating tSNE"
                f.write(log_text)
        if kwargs.get('verbose', 0) > 1:
            with open(LOG_file_PATH, 'a') as f:
                log_text = "Calculating distances to target objects"
                f.write(log_text)

        new_vp_key = add_vertex_property_to_target_object(_g=region_graph,
                                                          target_object_coordinates_array=cancer_cells_in_graph_array_only_coordinates)
        vertices_min_dist_to_cancer_cells = region_graph.vp[new_vp_key].a
        if kwargs.get('verbose', 0) > 2:
            with open(LOG_file_PATH, 'a') as f:
                log_text = "Normalizing distances to target objects"
                f.write(log_text)

        vertices_min_dist_to_cancer_cells_norm = np.divide(vertices_min_dist_to_cancer_cells,
                                                           vertices_min_dist_to_cancer_cells.max())

        vertices_embeddings_tsne = tsne_module.fit_transform(regions_vertices_embeddings)
        single_region_vertices_embeddings_df['pc_0'] = vertices_embeddings_pca[:, 0]
        single_region_vertices_embeddings_df['pc_1'] = vertices_embeddings_pca[:, 1]
        single_region_vertices_embeddings_df['tsne_0'] = vertices_embeddings_tsne[:, 0]
        single_region_vertices_embeddings_df['tsne_1'] = vertices_embeddings_tsne[:, 1]
        single_region_vertices_embeddings_df.loc[:,
        ['min_dist_to_cancer_cells']] = vertices_min_dist_to_cancer_cells_norm
        # saving mid calculation results
        regions_embeddings_and_dim_reduction_results_df_file_path = regions_embeddings_results_file_path.replace('.npy',
                                                                                                                 '.csv')
        regions_embeddings_and_dim_reduction_results_df_file_path = regions_embeddings_and_dim_reduction_results_df_file_path.replace(
            "node2vec_vertices_embeddings",
            f"{region_name}_node2vec_vertices_embeddings_and_reduced_dimensions_with_min_cancer_dist")
        single_region_vertices_embeddings_df.to_csv(regions_embeddings_and_dim_reduction_results_df_file_path)

        if all_regions_vertices_embeddings_df is None:
            all_regions_vertices_embeddings_df = single_region_vertices_embeddings_df.copy()
        else:
            all_regions_vertices_embeddings_df = pd.concat([all_regions_vertices_embeddings_df,
                                                           single_region_vertices_embeddings_df])

        del single_region_vertices_embeddings_df, region_graph


if __name__ == '__main__':
    embeddings_dim_size = 128
    walk_len = 30
    n_walks = 200
    verbose = 3
    embeddings_columns_names = [f"embd_{i}" for i in range(embeddings_dim_size)]

    cancer_cells_array_path = "/Data/1_cells.npy"
    regions_graphs_dir_path = "/Data/MiceBrainSubgraphs"
    embeddings_results_directory_path = "/UnsupervisedGraphTraversal/vertex_embeddings"
    cancer_cells_mid_results_dir_path = "/Results/MiceBrain"
    cancer_cells_regions_counts_sorted = get_cancer_cells_data(cancer_cells_array_path=cancer_cells_array_path,
                                                               verbose=verbose
                                                               )
    filtered_cancer_cells_regions_counts_sorted = calc_num_cancer_cells_per_graph(
        cancer_cells_regions_counts_sorted=cancer_cells_regions_counts_sorted,
        regions_graphs_dir_path=regions_graphs_dir_path,
        path_to_save_results_dir=cancer_cells_mid_results_dir_path,
        verbose=verbose

    )

    calc_min_cancer_cells_distance_and_node2vec_embeddings(
        filtered_cancer_cells_regions_counts_sorted=filtered_cancer_cells_regions_counts_sorted,
        embeddings_columns_names=embeddings_columns_names,
        regions_graphs_dir_path=regions_graphs_dir_path,
        cancer_cells_array_path=cancer_cells_array_path,
        embeddings_results_directory_path=embeddings_results_directory_path,
        verbose=verbose
    )
    with open(LOG_file_PATH, 'a') as f:
        log_text = f"Finished run at {datetime.datetime.now().strftime('%d/%m/%Y %H:%M:%S')}"
        f.write(log_text)