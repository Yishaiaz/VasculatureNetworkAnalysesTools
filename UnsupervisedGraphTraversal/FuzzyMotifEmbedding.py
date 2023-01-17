import json
import os
import sys
import shutil
import math
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
sys.path.append("/Users/yishaiazabary/PycharmProjects/University/BrainVasculatureGraphs")
from NetworkUtilities.GraphAttributesEnrichment import *


def get_cancer_cells_counts_in_regions(cancer_cells_array_path: Union[
    str, os.PathLike] = '/Users/yishaiazabary/PycharmProjects/University/BrainVasculatureGraphs/Data/1_cells.npy',
                                       return_sorted: bool = True,
                                       **kwargs):
    cancer_cells_array = np.load(cancer_cells_array_path)

    cancer_cells_atlas_as_np = np.stack([single_instance[-1] for single_instance in cancer_cells_array])

    cancer_cells_regions_counts = np.unique(cancer_cells_atlas_as_np, return_counts=True)

    if return_sorted:
        return sorted(zip(cancer_cells_regions_counts[0], cancer_cells_regions_counts[1]),
                      key=lambda x: x[1], reverse=True)
    return zip(cancer_cells_regions_counts[0], cancer_cells_regions_counts[1])


def calculate_gt_graph_node_embeddings(graph: gt.Graph,
                                       vertices_embedding_module: callable,
                                       edges_embedders_method: EdgeEmbedder = None,
                                       embd_dims: int = 64,
                                       walks_len: int = 30,
                                       num_of_walks: int = 200,
                                       prl_workers: int = 4,
                                       **kwargs
                                       ):
    if not isinstance(graph, gt.Graph):
        raise TypeError(f"Currently support graph-tool graphs instances only!, but got graph with type: {type(graph)}")

    node_embeddings_module = vertices_embedding_module(graph,
                                                       dimensions=embd_dims,
                                                       walk_length=walks_len,
                                                       num_walks=num_of_walks,
                                                       workers=prl_workers)

    node_embed_model = node_embeddings_module.fit(window=10, min_count=1, batch_words=4)
    if edges_embedders_method is not None:
        edges_embedders_method = edges_embedders_method(keyed_vectors=node_embed_model.wv)

    nodes_embeddings = []

    for vertex in graph.iter_vertices():
        vertex_id = int(vertex)
        nodes_embeddings.append(node_embed_model.wv.get_vector(vertex_id))

    nodes_embeddings = np.vstack(nodes_embeddings)
    if kwargs.get("save_node_embeddings_path", None):
        dir_to_save_in = os.path.join(*kwargs.get("save_node_embeddings_path").split(os.sep)[:-1])
        os.makedirs(dir_to_save_in, exist_ok=True)
        if str(kwargs.get("save_node_embeddings_path")).endswith('npy'):
            np.save(kwargs.get("save_node_embeddings_path"), nodes_embeddings)
        elif str(kwargs.get("save_node_embeddings_path")).endswith('csv'):
            df_to_save = pd.DataFrame(columns=['vertex_id'] +
                                              [f"{str(vertices_embedding_module)}_{i}" for i in range(embd_dims)])
            df_to_save.loc[:, ['vertex_id']] = list(graph.vertices())
            df_to_save.iloc[:, 1:] = nodes_embeddings
            df_to_save.set_index('vertex_id')
            df_to_save.to_csv(kwargs.get("save_node_embeddings_path"))
            del df_to_save
        else:
            raise ValueError(f"Automatic save format inference failed!\nCurrently supports saving the embeddings"
                             f" only as numpy array (path should end with 'npy') OR"
                             f" as pandas dataframe (path should end with 'csv'), but got: "
                             f"{kwargs.get('save_node_embeddings_path')} ")
    return nodes_embeddings


def collect_regions_with_cancer_cells_info(cancer_cells_regions_counts_sorted: List[Tuple[str, int]],
                                           path_to_save_regions_cancer_cells_descriptor: Union[
                                               str, os.PathLike] = f"/Users/yishaiazabary/PycharmProjects/University/BrainVasculatureGraphs/Results/MiceBrain/MiceBrainRegionsCancerCellsCountsAndGraphSizes.json",
                                           regions_graphs_dir_path: Union[
                                               str, os.PathLike] = "/Users/yishaiazabary/PycharmProjects/University/BrainVasculatureGraphs/Data/MiceBrainSubgraphs/",
                                           **kwargs
                                           ):
    filtered_cancer_cells_regions_counts_sorted = {}
    for region_descriptor in cancer_cells_regions_counts_sorted[1:]:
        region_name, region_n_cancer_cells = region_descriptor
        # new_region_descriptor = [region_name, region_n_cancer_cells, None, None]
        region_graph_path = os.path.join(regions_graphs_dir_path, f"subgraph_area_{region_name}.gt")
        if os.path.exists(region_graph_path):
            region_graph = gt.load_graph(region_graph_path)

            filtered_cancer_cells_regions_counts_sorted[region_name] = {
                'n_cancer_cells': int(region_n_cancer_cells),
                'region_n_vertices': int(region_graph.num_vertices()),
                'region_n_edges': int(region_graph.num_edges())
            }
            del region_graph
    # saving the results for later use:
    with open(
            path_to_save_regions_cancer_cells_descriptor,
            'w') as f:
        json.dump(filtered_cancer_cells_regions_counts_sorted, f)

    return filtered_cancer_cells_regions_counts_sorted


def calc_embeddings_df_for_regions_lst(
        filtered_cancer_cells_regions_counts_sorted: Dict[str, Dict[str, int]],
        embeddings_columns_names: List[str],
        embeddings_dim_size: int,
        walk_len: int,
        n_walks: int,
        cancer_cells_array_path: Union[
            str, os.PathLike] = '/Users/yishaiazabary/PycharmProjects/University/BrainVasculatureGraphs/Data/1_cells.npy',
        embeddings_results_directory_path: Union[
            str, os.PathLike] = "/Users/yishaiazabary/PycharmProjects/University/BrainVasculatureGraphs/UnsupervisedGraphTraversal/vertex_embeddings/",
        all_regions_embeddings_df_fname: str = 'graph_embeddings/all_mice_brain_regions_vertices_embeddings_and_dim_reductions_with_cancer_cells_min_distance.csv',
        **kwargs
):
    verbose = kwargs.get("verbose", 0)
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
        print(
            f"Processing region: {region_name} with #CancerCells={cancer_descriptor['n_cancer_cells']} and #Vertices= {cancer_descriptor['region_n_vertices']} and #Edges={cancer_descriptor['region_n_edges']}")
        # get all cancer cells coordinates in region
        cancer_cells_in_graph_array = np.array(
            list(filter(lambda x: x[-1].lower() == region_name.lower(), cancer_cells_array)))
        # collect only cancer cells coordinates
        cancer_cells_in_graph_array_only_coordinates = np.array(
            list(map(lambda x: tuple(x)[0:3], cancer_cells_in_graph_array)))

        # read the region's graph:
        gt_graph_path = f"/Users/yishaiazabary/PycharmProjects/University/BrainVasculatureGraphs/Data/MiceBrainSubgraphs/subgraph_area_{region_name}.gt"
        region_graph = gt.load_graph(gt_graph_path)
        # defining results directories and validating it exists.
        regions_embeddings_results_dir_path = os.path.join(embeddings_results_directory_path, region_name)
        os.makedirs(regions_embeddings_results_dir_path, exist_ok=True)
        regions_embeddings_results_file_path = os.path.join(regions_embeddings_results_dir_path,
                                                            f"node2vec_vertices_embeddings_dimSize{embeddings_dim_size}_walkSize{walk_len}_nWalks{n_walks}.npy")
        if os.path.isfile(regions_embeddings_results_file_path):
            if verbose > 2:
                print(f"Reading previously calculated vertices embeddings of region: {region_name}")
            regions_vertices_embeddings = np.load(regions_embeddings_results_file_path)
        else:
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
            np.save(regions_embeddings_results_file_path, regions_vertices_embeddings)
        # update the single region dataframe with vertices embeddings:
        single_region_vertices_embeddings_df[embeddings_columns_names] = regions_vertices_embeddings
        single_region_vertices_embeddings_df['region_name'] = region_name

        # defining path for df saving:
        regions_embeddings_and_dim_reduction_results_df_file_path = regions_embeddings_results_file_path.replace('.npy',
                                                                                                                 '.csv')
        regions_embeddings_and_dim_reduction_results_df_file_path = regions_embeddings_and_dim_reduction_results_df_file_path.replace(
            "node2vec_vertices_embeddings",
            f"{region_name}_node2vec_vertices_embeddings_and_reduced_dimensions_with_min_cancer_dist")

        # look for previously calculated results and load instead of recalculating:
        if os.path.isfile(regions_embeddings_and_dim_reduction_results_df_file_path):
            print(f"found previously calculated dataframe of {region_name}, skipping...")
            single_region_vertices_embeddings_df = pd.read_csv(
                regions_embeddings_and_dim_reduction_results_df_file_path)
        else:
            # calculating dim reduction for region's embeddings
            pca_module = PCA(n_components=2)
            tsne_module = TSNE(n_components=2, verbose=1, perplexity=40, n_iter=300)
            if verbose > 1:
                print("calculating PCA")
            vertices_embeddings_pca = pca_module.fit_transform(regions_vertices_embeddings)
            if verbose > 1:
                print("calculating tSNE")

            if verbose > 1:
                print("Calculating distances to target objects")
            new_vp_key = add_vertex_property_to_target_object(_g=region_graph,
                                                              target_object_coordinates_array=cancer_cells_in_graph_array_only_coordinates)
            vertices_min_dist_to_cancer_cells = region_graph.vp[new_vp_key].a
            if verbose > 2:
                print("Normalizing distances to target objects")
            vertices_min_dist_to_cancer_cells_norm = np.divide(vertices_min_dist_to_cancer_cells,
                                                               vertices_min_dist_to_cancer_cells.max())

            vertices_embeddings_tsne = tsne_module.fit_transform(regions_vertices_embeddings)
            single_region_vertices_embeddings_df['pc_0'] = vertices_embeddings_pca[:, 0]
            single_region_vertices_embeddings_df['pc_1'] = vertices_embeddings_pca[:, 1]
            single_region_vertices_embeddings_df['tsne_0'] = vertices_embeddings_tsne[:, 0]
            single_region_vertices_embeddings_df['tsne_1'] = vertices_embeddings_tsne[:, 1]
            single_region_vertices_embeddings_df.loc[:,
            ['min_dist_to_cancer_cells_norm']] = vertices_min_dist_to_cancer_cells_norm
            single_region_vertices_embeddings_df.loc[:,
            ['min_dist_to_cancer_cells']] = vertices_min_dist_to_cancer_cells

            # saving mid calculation results
            single_region_vertices_embeddings_df.to_csv(regions_embeddings_and_dim_reduction_results_df_file_path)

        # concatenating with other regions DFs
        if all_regions_vertices_embeddings_df is None:
            all_regions_vertices_embeddings_df = single_region_vertices_embeddings_df.copy()
        else:
            all_regions_vertices_embeddings_df = pd.concat(
                [all_regions_vertices_embeddings_df, single_region_vertices_embeddings_df])

        del single_region_vertices_embeddings_df, region_graph

    path_to_save_all_regions_embeddings_df = os.path.join(embeddings_results_directory_path,
                                                          all_regions_embeddings_df_fname)
    all_regions_vertices_embeddings_df.to_csv(path_to_save_all_regions_embeddings_df)


if __name__ == '__main__':
    # shared_mem_temp_dir_path = './tmp_node2vec_cache_mem'
    # os.makedirs(shared_mem_temp_dir_path, exist_ok=True)
    # verbose = 3
    #
    # cancer_cells_array_path = '/Users/yishaiazabary/PycharmProjects/University/BrainVasculatureGraphs/Data/1_cells.npy'
    #
    # cancer_cells_regions_counts_sorted = get_cancer_cells_counts_in_regions(cancer_cells_array_path,
    #                                                                         return_sorted=True)
    #
    # filtered_cancer_cells_regions_counts_sorted = collect_regions_with_cancer_cells_info(
    #     cancer_cells_regions_counts_sorted=cancer_cells_regions_counts_sorted
    # )
    #
    embeddings_dim_size = 128
    walk_len = 30
    n_walks = 200
    #
    # embeddings_columns_names = [f"embd_{i}" for i in range(embeddings_dim_size)]
    #
    # calc_embeddings_df_for_regions_lst(
    #     filtered_cancer_cells_regions_counts_sorted=filtered_cancer_cells_regions_counts_sorted,
    #     embeddings_columns_names=embeddings_columns_names,
    #     embeddings_dim_size=embeddings_dim_size,
    #     walk_len=walk_len,
    #     n_walks=n_walks,
    #     verbose=verbose
    # )

    gbm_graph = gt.load_graph("/Users/yishaiazabary/PycharmProjects/University/BrainVasculatureGraphs/Data/GBM_Tumor_Graphs/graph_annotated.gt")
    gbm_graph_cpy = gbm_graph.copy()
    v_filter_prop_map = gbm_graph_cpy.new_vertex_property('bool')
    for _v in tqdm.tqdm(gbm_graph_cpy.iter_vertices()):
        v_filter_prop_map[_v] = int(_v) < 5000
    gbm_graph_cpy.vp['sample_filter'] = v_filter_prop_map
    gbm_graph_cpy.set_vertex_filter(gbm_graph_cpy.vp['sample_filter'])

    for _v in tqdm.tqdm(gbm_graph_cpy.iter_vertices()):
        _v = gbm_graph_cpy.vertex(_v)
        v_filter_prop_map[_v] = _v.in_degree()+_v.out_degree() != 0
    gbm_graph_cpy.vp['sample_filter'] = v_filter_prop_map
    gbm_graph_cpy.set_vertex_filter(gbm_graph_cpy.vp['sample_filter'])

    gt.graph_draw(gbm_graph_cpy, output='test.pdf')
    calculate_gt_graph_node_embeddings(
        graph=gbm_graph,
        edges_embedders_method=None,
        save_node_embeddings_path=f"/Users/yishaiazabary/PycharmProjects/University/BrainVasculatureGraphs/UnsupervisedGraphTraversal/vertex_embeddings/GBM_1st_Graph/SAMPLE_GBM_node2vec_vertices_embeddings_and_dimSize{embeddings_dim_size}_walkSize{walk_len}_nWalks{n_walks}.csv",
        vertices_embedding_module=Node2Vec,
        embd_dims=embeddings_dim_size,
        walks_len=walk_len,
        num_of_walks=n_walks
        )


