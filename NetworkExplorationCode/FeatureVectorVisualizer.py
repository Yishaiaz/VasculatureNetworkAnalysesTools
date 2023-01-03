import pandas as pd
from sklearn.decomposition import PCA
import plotly.express as px
from sklearn.manifold import TSNE
import plotly.figure_factory as ff
import numpy as np
from scipy.stats import ks_2samp


def exclude_one_feature(data: pd.DataFrame, exclude_f: str) -> list[str]:
    #all features except one (label)
    features_list = list(data.columns)
    return features_list.remove(exclude_f)


def dis_plots(dfs: list[pd.DataFrame], features: list[str], graph_names: list[str]):
    # Create distplot with custom bin_size
    for f in features:
        hist_data = [_df[f] for _df in dfs]
        fig = ff.create_distplot(hist_data, graph_names, bin_size=.2)
        fig.update_layout(title=f + " feature distribution")
        fig.show()


def vasc_3d_scatter(scatter_data: pd.DataFrame, features: list[str]):
    fig = px.scatter_3d(
        merged_df, x=features[0], y=features[1], z=features[2],
        color=scatter_data.brain_region, labels={'color': 'brain_region'}
    )
    fig.show()


def vasc_box_hist(scatter_data: pd.DataFrame):
    fig = px.histogram(scatter_data, x="ego_depth_4_e_prolif_mean", color='brain_region',
                       marginal="box",  # or violin, rug
                       histnorm='probability',
                       hover_data=df.columns)
    fig.show()


def vasc_scatter(scatter_data: pd.DataFrame, features: list[str]):
    fig = px.scatter_matrix(scatter_data, dimensions=features, color="brain_region")
    fig.update_traces(marker={'size': 2})
    fig.show()


def vasc_tsne(data: pd.DataFrame, features: list[str]):
    tsne = TSNE(n_components=2, random_state=0)
    projections = tsne.fit_transform(data[features])

    fig = px.scatter(
        projections, x=0, y=1,
        color=data.brain_region, labels={'color': 'brain_region'}
    )
    fig.show()


def vasc_pca(data: pd.DataFrame, features: list[str]):
    pca = PCA()
    components = pca.fit_transform(data[features])
    labels = {
        str(i): f"PC {i+1} ({var:.1f}%)"
        for i, var in enumerate(pca.explained_variance_ratio_ * 100)
    }

    fig = px.scatter_matrix(
        components,
        labels=labels,
        dimensions=range(2),
        color=data["brain_region"]
    )
    fig.update_traces(diagonal_visible=False)
    fig.show()


g_file_names = ["subgraph_area_corpus_callosum"] #"subgraph_area_Striatum", "subgraph_area_Hypothalamus", "subgraph_area_pyramid","subgraph_area_Claustrum", "subgraph_area_corpus_callosum", "subgraph_area_Isocortex",firstGBMscanGraph
features_dfs = []
for graph_name in g_file_names:
    df = pd.read_pickle("../ExtractedFeatureVectors/" + graph_name + ".csv")
    df["brain_region"] = graph_name
    df["v_e_ratio"] = df["n_vertices"]/df["n_edges"]
    features_dfs.append(df)
merged_df = pd.concat(features_dfs)
all_features = exclude_one_feature(merged_df, "brain_region")
chosen_features = ["ego_depth_4_e_prolif_mean"]

#stats = ks_2samp(features_dfs[0][feature], features_dfs[1][feature]) #graph regions 0 and 3
#print(stats)

vasc_box_hist(merged_df)
#dis_plots(features_dfs, chosen_features, g_file_names, )


# "v_e_ratio", "n_edges", "n_vertices", "ego_depth_4_e_prolif_mean", "ego_depth_4_e_radii_mean״, "ego_depth_4_e_length_mean״
# "ego_depth_4_e_x_direction_mean", "ego_depth_4_e_y_direction_mean", "ego_depth_4_e_z_direction_mean"
