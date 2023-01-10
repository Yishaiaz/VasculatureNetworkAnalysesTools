import numpy as np
import pandas as pd
import os
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
import plotly.express as px
import plotly.figure_factory as ff
import seaborn as sns


def seaborn_pairplot(data: pd.DataFrame):
    sns.set_theme(style="ticks")
    sns.pairplot(data, hue="brain_region", plot_kws={"s": 4})
    plt.show()


def exclude_one_feature(data: pd.DataFrame, exclude_f: str) -> list[str]:
    """
    Returns a list of features from the input dataframe, excluding one specified feature.

    @param data: input dataframe
    @param exclude_f: feature to exclude
    @return: list of features
    """
    # get all features except one (label)
    features_list = list(data.columns)
    features_list.remove(exclude_f)
    return features_list


def dis_plots(dfs: list[pd.DataFrame], features: list[str], graph_names: list[str]):
    """
    Plots histograms of the specified features for each dataframe in the input list.

    @param dfs: list of dataframes
    @param features: list of features to plot
    @param graph_names: list of names for the histograms
    """
    # Create distplot with custom bin_size
    for f in features:
        hist_data = [_df[f] for _df in dfs]
        fig = ff.create_distplot(hist_data, graph_names, bin_size=.2)
        fig.update_layout(title=f + " feature distribution")
        fig.show()


def vasc_3d_scatter(scatter_data: pd.DataFrame, features: list[str]):
    """
    Plots a 3D scatter plot of the specified features from the input dataframe, colored by the 'brain_region' column.

    @param scatter_data: input dataframe
    @param features: list of 3 features to plot
    """
    fig = px.scatter_3d(
        scatter_data, x=features[0], y=features[1], z=features[2],
        color=scatter_data.brain_region, labels={'color': 'brain_region'}
    )
    fig.show()


def vasc_box_hist(scatter_data: pd.DataFrame, feature:str):
    """
    Plots a box plot and histogram of the 'feature' column from the input dataframe,
    colored by the 'brain_region' column.

    @param scatter_data: input dataframe
    @param feature: string representing feature to plot
    """
    fig = px.histogram(scatter_data, x=feature, color='brain_region',
                       marginal="box",  # or violin, rug
                       histnorm='probability',
                       hover_data=scatter_data.columns)
    fig.show()


def vasc_scatter(scatter_data: pd.DataFrame, features: list[str]):
    """
    Plots a scatter matrix of the specified features from the input dataframe, colored by the 'brain_region' column.

    @param scatter_data: input dataframe
    @param features: list of features to plot
    """
    fig = px.scatter_matrix(scatter_data, dimensions=features, color="brain_region")
    fig.update_traces(marker={'size': 2})
    fig.show()


def vasc_tsne(data: pd.DataFrame, features: list[str]):
    """
    Plots a 2D scatter plot of the specified features from the input dataframe, colored by the 'brain_region' column.
    The features are first transformed using t-SNE.

    @param data: input dataframe
    @param features: list of features to plot
    """
    tsne = TSNE(n_components=2, random_state=0)
    projections = tsne.fit_transform(data[features])

    fig = px.scatter(
        projections, x=0, y=1,
        color=data.brain_region, labels={'color': 'brain_region'}
    )
    fig.show()


def vasc_pca(data: pd.DataFrame, features: list[str], color_feature: str):
    """
    Plots a scatter matrix of the specified features from the input dataframe, colored by the 'brain_region' column.
    The features are first transformed using PCA.

    @param data: input dataframe
    @param features: list of features to plot
    """
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
        color=data[color_feature]
    )

    ind = np.argpartition(pca.components_[0], 4)[:4]
    top4 = pca.components_[0][ind]
    top4f = [features[i] for i in ind]
    print(top4)
    print(top4f)

    ind = np.argpartition(pca.components_[1], 4)[:4]
    top4 = pca.components_[1][ind]
    top4f = [features[i] for i in ind]
    print(top4)
    print(top4f)

    fig.update_traces(diagonal_visible=False)
    fig.show()


def features_heatmap(data: pd.DataFrame):
    fig = px.imshow(data.corr())
    fig.show()


def main():
    features_dfs = []
    dir_path = "/Users/leahbiram/Desktop/vasculature_data/ExtractedFeatureVectors/"
    for filename in os.listdir(dir_path):
        df = pd.read_pickle(dir_path+filename)
        df["brain_region"] = filename
        features_dfs.append(df)
    merged_df = pd.concat(features_dfs)
    features_heatmap(merged_df)
 #   chosen_features = ["brain_region", "min_dist_to_cancer_cells", 'depth2_e_prolif_mean', 'depth2_v_min_angle_mean']
 #   seaborn_pairplot(merged_df[chosen_features])
    print("loaded")

def main0():
    g_file_names = ["Striatum", "pyramid", "Claustrum"] #"subgraph_area_Striatum", "subgraph_area_Hypothalamus", "subgraph_area_pyramid","subgraph_area_Claustrum", "subgraph_area_corpus_callosum", "subgraph_area_Isocortex",firstGBMscanGraph
    label = "min_dist_to_cancer_cells"
    features_dfs = []
    for graph_name in g_file_names:
        df = pd.read_pickle("../ExtractedFeatureVectors/" + graph_name + ".csv")
        df["brain_region"] = graph_name
        features_dfs.append(df)
    merged_df = pd.concat(features_dfs)
    all_features = exclude_one_feature(merged_df, label)
    all_features.remove( "brain_region")
    chosen_features = ["brain_region", "min_dist_to_cancer_cells", 'depth3_e_length_mean', 'depth3_e_radii_mean', 'depth3_e_prolif_mean', 'depth1_v_e_ratio']

    merged_df = merged_df.reset_index()
    seaborn_pairplot(merged_df[chosen_features])
    #vasc_box_hist(merged_df, "min_dist_to_cancer_cells")
    #dis_plots(features_dfs, chosen_features, g_file_names, )
    #features_heatmap(merged_df[chosen_features])
    #vasc_pca(merged_df, all_features, label)


if __name__ == "__main__":
    main()


# "v_e_ratio", "n_edges", "n_vertices", "ego_depth_4_e_prolif_mean", "ego_depth_4_e_radii_mean״, "ego_depth_4_e_length_mean״
# "ego_depth_4_e_x_direction_mean", "ego_depth_4_e_y_direction_mean", "ego_depth_4_e_z_direction_mean"
