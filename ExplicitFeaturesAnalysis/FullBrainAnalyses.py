import numpy as np
import pandas as pd
import os
from scipy.stats import zscore
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC, NuSVC
import plotly.express as px
import plotly.figure_factory as ff
import seaborn as sns
from sklearn.neural_network import MLPClassifier as mlp
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
from xgboost import XGBClassifier
from statistics import median, mean
from scipy.stats import wasserstein_distance, kstest


def seaborn_pairplot(data: pd.DataFrame):
    sns.set_theme(style="ticks")
    sns.pairplot(data, hue="is_close", plot_kws={"s": 4})
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


def vasc_box_hist(scatter_data: pd.DataFrame, feature: str):
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


def vasc_tsne(data: pd.DataFrame, features: list[str], plot_name: str):
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
        color=data.brain_region, labels={'color': 'brain_region'},
        title=plot_name
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
        str(i): f"PC {i + 1} ({var:.1f}%)"
        for i, var in enumerate(pca.explained_variance_ratio_ * 100)
    }
    #   components['topfeat']
    fig = px.scatter_matrix(
        components,
        labels=labels,
        dimensions=range(3),
        color=data[color_feature]
    )

    ind = np.argpartition(pca.components_[0], -4)[-4:]
    top4 = pca.components_[0][ind]
    top4f = [features[i] for i in ind]
    print(top4)
    print(top4f)
    # cancer dist best for brain region

    fig.update_traces(diagonal_visible=False)
    fig.show()


def features_heatmap(data: pd.DataFrame):
    fig = px.imshow(data.corr())
    fig.show()


def sample_regions_equally(_df: pd.DataFrame, sample_size: int):
    return _df.groupby("brain_region").apply(lambda x: x.sample(sample_size))


def main_tsne_cancer_dist():
    dir_path = "/Users/leahbiram/Desktop/vasculature_data/ExtractedFeatureVectors/"
    dfs = []
    for filename in os.listdir(dir_path):
        if not filename.endswith(".csv"):
            continue
        df = pd.read_pickle(dir_path + filename)
        df["brain_region"] = filename
        if len(df) > 50000:
            dfs.append(df)
    merged_df = pd.concat(dfs)
    sampled_df = sample_regions_equally(merged_df, 1000)
    all_features = exclude_one_feature(merged_df, "brain_region")
    vasc_tsne(sampled_df, all_features, "Tsne Over All Regions")


def predict_acc(_df: pd.DataFrame, _features, label_name: str, thresh:int):
    feats = _df.loc[:, _features].values
    labels = _df.loc[:, [label_name]].values
    x_train, x_test, y_train, y_test = train_test_split(feats, labels, test_size=0.1, shuffle=True, stratify=labels)

#    mlp_model = mlp(hidden_layer_sizes=(128, 64, 32, 16, 8, 4, 2, 1),
#                     random_state=42, n_iter_no_change=10,
#                     verbose=True)
    logit_reg = LogisticRegression(max_iter=10000)
    svc = SVC(degree=3)
#    nusvc = NuSVC()
    xgb = XGBClassifier()
    for m in [logit_reg, xgb, svc]:
        m.fit(x_train, y_train)
        y_pred = m.predict(x_test)
        cm = confusion_matrix(y_test, y_pred)
        print(classification_report(y_test, y_pred))
        disp = ConfusionMatrixDisplay(confusion_matrix=cm)
        disp.plot(cmap=plt.cm.Blues)
        plt.title(type(m).__name__ + " on GBM predicting distance from cells, on threshold = "+str(int(thresh)))
        plt.show()


def wasserstein(df_0, df_1, _features):
    wds = []
    for _feature in _features:
        col_0 = df_0[_feature]
        col_1 = df_1[_feature]
       #wd = wasserstein_distance(col_0, col_1)
        wd = kstest(col_0, col_1)
        print(_feature + " wasserstein distance: " + str(wd))
        wds.append(wd)
        """
        if wd > 1:
            plt.title(_feature)
            df_0[_feature].plot.kde()
            df_1[_feature].plot.kde()
            plt.show()
            """
    return wds


def wasserstein_median_cancer_dist(file_path):
#    file_path = "/Users/leahbiram/Desktop/vasculature_data/ExtractedFeatureFiles/Inferior colliculus.csv"
    df = pd.read_pickle(file_path)
    proximity_cancer_cells_threshold = median(df['min_dist_to_cancer_cells'])
    df['is_close'] = df['min_dist_to_cancer_cells'].apply(
        lambda x: x <= proximity_cancer_cells_threshold)
    all_features = list(df.columns)
    all_features.remove('min_dist_to_cancer_cells')
    all_features.remove('is_close')
    #    predict_acc(df, all_features, "is_close",proximity_cancer_cells_threshold)
    #    sampled_df = df.sample(500)
    df_0 = df[df['is_close'] == 0]
    df_1 = df[df['is_close'] == 1]
    return wasserstein(df_0, df_1, all_features), all_features

def main_wass_analysis():
    ws_inferior_col, fs = wasserstein_median_cancer_dist("/Users/leahbiram/Desktop/vasculature_data/ExtractedFeatureVectors/Inferior colliculus.csv")
    ws_pons, fs = wasserstein_median_cancer_dist("/Users/leahbiram/Desktop/vasculature_data/ExtractedFeatureVectors/Pons.csv")
#    ws_stria = wasserstein_analysis("/Users/leahbiram/Desktop/vasculature_data/ExtractedFeatureFiles/Striatum.csv")
    zipped = zip(ws_inferior_col, ws_pons, fs)
 #   szip = sorted(zipped)
    zip_ic, zip_pons, zipf = zip(*zipped)
    plt.figure(figsize=(20, 10))
    plt.subplots_adjust(bottom=0.25)
    plt.title(" ks-test on features distributions partitioned by median distance to cancer cells ", fontsize=30)
    plt.ylabel("p-value", fontsize=20)
    plt.scatter([k-0.2 for k in range(len(zipf))], [item.pvalue for item in zip_ic], label = "Inferior Colliculus")
    plt.scatter(range(len(zipf)), [item.pvalue for item in zip_pons], label = "Pons")
#    plt.plot(ws_stria)
    plt.xticks(np.arange(len(zipf)), zipf, rotation=90)
    plt.legend()
    plt.savefig('../ExtractedFeatureFiles/kstest.eps', format="eps")
    plt.show()

    plt.figure(figsize=(20, 10))
    plt.title(" ks-test on features distributions partitioned by median distance to cancer cells ")
    plt.ylabel("statistic")
    plt.subplots_adjust(bottom=0.25)
    plt.scatter([k-0.5 for k in range(len(zipf))], [item.statistic for item in zip_ic], label="Inferior Colliculus")
    plt.scatter(range(len(zipf)), [item.statistic for item in zip_pons], label="Pons")
    #    plt.plot(ws_stria)
    plt.xticks(rotation=90)
    plt.legend()
    plt.show()

    '''

        dict = {'inf_col':[item.pvalue for item in ws_inferior_col],'pons':[item.pvalue for item in ws_pons],'fs':fs}
        df_ws0 = pd.DataFrame(dict)
        df_melt = pd.melt(df_ws0, id_vars=['fs'], var_name='region', value_name='pvalue')
    #    fig = px.histogram(df_melt, x="fs", y="pvalue",color='region', barmode='group')
        fig = px.bar(df_melt, x="fs", color="region",
                     y='pvalue',
                     title="A Grouped Bar Chart With Plotly Express in Python",
                     barmode='group',
                     height=600
                     )

        fig.show()

    '''



def tsne_cancer_dist_single_region():
    features_dfs = []
    filename = "pons.csv"
    file_path = "/Users/leahbiram/Desktop/vasculature_data/ExtractedFeatureFiles/" +filename
    df = pd.read_pickle(file_path)
    all_features = exclude_one_feature(df, "min_dist_to_cancer_cells")
    #sampled_df = sample_regions_equally(df, 5000)
    vasc_tsne(df, all_features, filename)

def main_prediction_models():
    file_path = "/Users/leahbiram/Desktop/vasculature_data/ExtractedFeatureVectors/Pons.csv"
    df = pd.read_pickle(file_path)
    proximity_cancer_cells_threshold = median(df['min_dist_to_cancer_cells'])
    df['is_close'] = df['min_dist_to_cancer_cells'].apply(
        lambda x: x <= proximity_cancer_cells_threshold)
    all_features = list(df.columns)
    all_features.remove('min_dist_to_cancer_cells')
    all_features.remove('is_close')
    predict_acc(df, all_features, "is_close",proximity_cancer_cells_threshold)


def one_region_analysis():
    file_path = "/Users/leahbiram/Desktop/vasculature_data/ExtractedFeatureVectors/Pons.csv"
    df = pd.read_pickle(file_path)
#    features_heatmap(df)
    plt.scatter(df["depth1_e_volume_mean"])
    plt.show()


def all_region_analysis():
    dir_path = "/Users/leahbiram/Desktop/vasculature_data/ExtractedFeatureVectors/"
    dfs = []
    for filename in os.listdir(dir_path):
        if not filename.endswith(".csv"):
            continue
        df = pd.read_pickle(dir_path + filename)
        df["brain_region"] = filename.removesuffix(".csv")
 #       vol_df[filename.removesuffix(".csv")] = df["depth1_e_volume_mean"]
        if len(df) > 5000:
            dfs.append(df)
#    features_heatmap(vol_df)

    merged_df = pd.concat(dfs)
    sampled_df = sample_regions_equally(merged_df, 1000)
 #   sns.displot(sampled_df, x="depth1_e_volume_mean", hue="brain_region", kind="kde")
    plt.figure(figsize=(10, 4))
    ax = sns.barplot(data=sampled_df, x="brain_region", y="depth2_e_volume_mean")
    ax.set_xticklabels(ax.get_xticklabels(), rotation=90, ha="right")
    plt.tight_layout()

    ax2 = ax.twinx()
    sns.lineplot(data=sampled_df, x="brain_region", y="depth2_e_radii_mean")
    plt.show()

#    all_features = exclude_one_feature(merged_df, "brain_region")
#    vasc_tsne(sampled_df, all_features, "Tsne Over All Regions")


def main():
#    one_region_analysis()
    all_region_analysis()


if __name__ == "__main__":
    main()

# "v_e_ratio", "n_edges", "n_vertices", "ego_depth_4_e_prolif_mean", "ego_depth_4_e_radii_mean״, "ego_depth_4_e_length_mean״
# "ego_depth_4_e_x_direction_mean", "ego_depth_4_e_y_direction_mean", "ego_depth_4_e_z_direction_mean"
