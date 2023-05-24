import torch
import torch.nn.functional as F
from torch.nn import Linear, Sequential, BatchNorm1d, ReLU, Dropout
from torch_geometric.nn import GCNConv, GINConv
from torch_geometric.nn import global_mean_pool, global_add_pool
from torch import sum, max_pool1d
from PyTorchCustomUtilitiesForCluster import *
from GINModels import GINWithDynamicLayersNumber, FixedGIN


if __name__ == '__main__':
    import argparse
    computation_device = 'cuda:0' if torch.cuda.is_available() else 'cpu'
    print(f"Using computation device: {computation_device}")
    parser = argparse.ArgumentParser(prog='TorchGNN', description='')
    parser.add_argument('--job_id', type=str,
                        help='job id as string')
    args = parser.parse_args()
    cluster_job_id = args.job_id
    # logger = CallableLogger(
    #     stream_output_fpath=f'./log_{cluster_job_id}.txt',
    #     log_entry_prefix='log:'
    # ) # for cluster use
    if cluster_job_id is None or cluster_job_id == '':
        # working locally
        logger = print # for local use
        cluster_job_id = "local_test"
        scrum_working_dir = os.path.join(
            os.sep, "Users", "yishaiazabary", "PycharmProjects",
            "University", "BrainVasculatureGraphs",
            "Data", "GBM_Tumor_Graphs"
        )
    else:
        # working on cluster
        logger = CallableLogger(
            stream_output_fpath=f'./log_{cluster_job_id}.txt',
            log_entry_prefix='log:'
        ) # for cluster use
        scrum_working_dir = f"/scratch/yishaiaz@auth.ad.bgu.ac.il/{cluster_job_id}"
        logger(f"scrum_working_dir = {scrum_working_dir}")
    logger(f"working on device: {computation_device}")
    torch_geometric_graph_data = torch.load(
        os.path.join(scrum_working_dir, "torch_data_graph_annotated_subgraph_with_labels"))

    logger(f"Read Graph, Metadata: #Nodes={torch_geometric_graph_data.num_nodes} | "
          f"#Edges={len(torch_geometric_graph_data.edge_index[0])}")
    # todo: convert all parameters to arguments with default values
    # gin_type = GINWithDynamicLayersNumber
    use_node_features = True
    use_edge_features = True
    use_sampler = True
    n_epochs = 50
    n_spatial_bins_per_dim = 2
    val_spatial_bin_idx, test_spatial_bin_idx = 0, 1
    best_model_score = 0
    gin_latent_space_size = 32
    for gin_type in (GINWithDynamicLayersNumber, FixedGIN):
        for k_hops in (5, 7, 10, 20):
            for type_of_label in ('single', 'majority'):
                os.makedirs(os.path.join(scrum_working_dir, 'GIN_experiments'), exist_ok=True)
                gin_exp_name = f'DL_{type(gin_type).__name__}_GBM_BioMarker_Presence_Prediction_' \
                                                    f'used_sampler_{use_sampler}_' \
                                                    f'by_label_{type_of_label}_' \
                                                    f'k_hops={k_hops}_' \
                                                    f'gin_latent_space_size={gin_latent_space_size}_' \
                                                    f'n_spatial_bins_per_dim={n_spatial_bins_per_dim}' \
                                                    f'val_spatial_bin_idx={val_spatial_bin_idx}' \
                                                    f'test_spatial_bin_idx={test_spatial_bin_idx}' \
                                                    f'use_node_features={use_node_features}' \
                                                    f'use_edge_features={use_edge_features}' \
                                                    f'#EPOCHS={n_epochs}'
                tensor_board_log_dir = os.path.join(scrum_working_dir,
                                                    f'GIN_experiments',
                                                    gin_exp_name)

                if type_of_label == "majority":
                    calc_items_label_func = MicroEnvironmentDataset.calc_subgraph_label_by_bio_marker_majority
                elif type_of_label == 'single':
                    calc_items_label_func = MicroEnvironmentDataset.calc_subgraph_label_by_bio_marker_presence
                elif type_of_label == 'all':
                    calc_items_label_func = MicroEnvironmentDataset.calc_subgraph_label_by_bio_marker_full_presence
                else:
                    raise NameError(f"no such method to calc microenvironment subgraph label, got= {type_of_label}")

                train_graphs, \
                val_graphs, \
                test_graphs = \
                    partition_graph_by_regions(torch_geometric_graph_data,
                                               n_bins_per_dim=n_spatial_bins_per_dim,
                                               val_bin_idx=val_spatial_bin_idx,
                                               test_bin_idx=test_spatial_bin_idx)
                # microenv_dataset = MicroEnvironmentDataset(
                #     data=torch_geometric_graph_data, num_hops=k_hops, loc_attr_name="pos", max_distance=float('inf'),device=computation_device,verbose=0,
                #     calc_items_label_func=calc_items_label_func
                # )
                train_ds = MicroEnvironmentDataset(data=train_graphs,
                                                   num_hops=k_hops,
                                                   loc_attr_name="pos",
                                                   max_distance=float('inf'),
                                                   device=computation_device, verbose=0,
                                                   calc_items_label_func=calc_items_label_func,
                                                   include_node_attributes=use_node_features,
                                                   include_edge_attributes=use_edge_features,
                                                   logger=logger
                                                   )
                if use_sampler:
                    sampler = ImbalancedSampler(train_ds)
                    loader = DataLoader(train_ds, batch_size=1, sampler=sampler)
                else:
                    loader = train_ds
                draw_subgraphs(train_ds, draw_lim_num=2)
                val_ds = MicroEnvironmentDataset(data=val_graphs,
                                                 num_hops=k_hops,
                                                 loc_attr_name="pos",
                                                 max_distance=float('inf'),
                                                 device=computation_device, verbose=0,
                                                 calc_items_label_func=calc_items_label_func,
                                                 include_node_attributes=use_node_features,
                                                 include_edge_attributes=use_edge_features,
                                                 logger=logger
                                                 )
                draw_subgraphs(val_ds, draw_lim_num=2)
                test_ds = MicroEnvironmentDataset(data=test_graphs,
                                                  num_hops=k_hops,
                                                  loc_attr_name="pos",
                                                  max_distance=float('inf'),
                                                  device=computation_device, verbose=0,
                                                  calc_items_label_func=calc_items_label_func,
                                                  include_node_attributes=use_node_features,
                                                  include_edge_attributes=use_edge_features,
                                                  logger=logger
                                                  )
                # some preliminary statistics
                draw_subgraphs(test_ds, draw_lim_num=2)
                logger(f"train_ds #nodes={len(train_ds)}")
                logger(f"val_ds #nodes={len(val_ds)}")
                logger(f"test_ds #nodes={len(test_ds)}")
                #
                # train_ds = Subset(microenv_dataset, np.arange(0, int(len(microenv_dataset)*0.7), 1))
                # val_ds = Subset(microenv_dataset, np.arange(int(len(microenv_dataset)*0.7), int(len(microenv_dataset)*0.9), 1))
                # test_ds = Subset(microenv_dataset, np.arange(int(len(microenv_dataset)*0.9), len(microenv_dataset), 1))

                # microenv_dataset_loader = DataLoader(microenv_dataset, batch_size=2, collate_fn=lambda x: x)
                # logger(f"presplit target values count = {microenv_dataset.calc_target_statistics()}")

                gin = gin_type(dim_h=gin_latent_space_size, ds=train_ds, output_dim=2)
                gin.to(computation_device)

                tensorboard_summary_writer = SummaryWriter(log_dir=tensor_board_log_dir)
                gin, test_acc = train(gin, train_loader=loader,
                                      train_ds=train_ds,
                                      validation_loader=val_ds, epochs_num=n_epochs,
                                      tensorboard_writer=tensorboard_summary_writer,
                                      test_loader=test_ds,
                                      logger=logger)

                if best_model_score < test_acc:
                    best_model_score = test_acc
                    with open(f"model_{gin_exp_name}.model", 'w') as f:
                        torch.save(gin, f)
