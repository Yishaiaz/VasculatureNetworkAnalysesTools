import numpy as np
import pandas as pd
from sklearn.manifold import TSNE
import plotly.express as px
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC, NuSVC
import matplotlib.pyplot as plt
from statistics import median, mean
from xgboost import XGBClassifier
from sklearn.preprocessing import MinMaxScaler
import seaborn as sns


def exclude_immune_features(_df):
    all_features = list(_df.columns)
    return [feat for feat in all_features if ("artery" not in feat and "immune" not in feat)]


def predict_acc(_df: pd.DataFrame, _features, label_name: str, thresh:int):
    features = _df.loc[:, _features].values
    labels = _df.loc[:, [label_name]].values
    x_train, x_test, y_train, y_test = train_test_split(features, labels, test_size=0.8, shuffle = False) # shuffle=True, stratify=labels)

#    mlp_model = mlp(hidden_layer_sizes=(128, 64, 32, 16, 8, 4, 2, 1),
#                     random_state=42, n_iter_no_change=10,
#                     verbose=True)
#    logit_reg = LogisticRegression(max_iter=10000)
    svc = SVC(degree=3)
#    nusvc = NuSVC()
    xgb = XGBClassifier()
    y_preds = []
    for m in [xgb]:
        m.fit(x_train, y_train)
        y_pred = (m.predict(x_test))
        y_preds.append(y_pred)
        cm = confusion_matrix(y_test, y_pred)
        print(classification_report(y_test, y_pred))
        disp = ConfusionMatrixDisplay(confusion_matrix=cm)
        disp.plot(cmap=plt.cm.Blues)
        plt.title(type(m).__name__ + " on GBM predicting immune cells, on threshold = "+str(thresh)+"\u03BCm")
        plt.show()
   # all_node_preds = np.zeros(len(features))
    all_node_preds = np.empty(len(features))
    all_node_preds.fill(-1)
    all_node_preds[len(features)-len(y_test):] = y_pred
    return all_node_preds


def scatter(data: pd.DataFrame, features: list[str]):
    fig = px.scatter(data, x=features[0], y=features[1])
    fig.show()


def vasc_tsne(data: pd.DataFrame, features: list[str], plot_name: str = "tsne"):
    """
    Plots a 2D scatter plot of the specified features from the input dataframe, colored by the 'brain_region' column.
    The features are first transformed using t-SNE.

    @param data: input dataframe
    @param features: list of features to plot
    @param plot_name: title of plot
    """
    tsne = TSNE(n_components=2, random_state=0)
    projections = tsne.fit_transform(data[features])

    fig = px.scatter(
        projections, x=0, y=1,
        color=data.log_immune, labels={'color': 'log_immune'},
        title=plot_name
    )
    fig.show()


def features_heatmap(data: pd.DataFrame):
    fig = px.imshow(data.corr(), title="GBM all features correlation")
    fig.show()


def add_immune_analysis(df: pd.DataFrame):
    proximity_immune_cells_threshold = median(df['depth1_e_artery_raw_mean'])
    df['has_immune_median'] = df['depth1_e_artery_raw_mean'].apply(
        lambda x: x <= proximity_immune_cells_threshold)
    df['has_immune'] = df['depth1_e_artery_binary_mean'].apply(
        lambda x: x > 0)
    df['log_immune'] = np.log2(df['depth1_e_artery_raw_mean'])
    return df


def add_loop_features(df: pd.DataFrame):
    df["depth1_loops"] = 1 + df["depth1_n_edges"] - df["depth1_n_vertices"]
    df["depth2_loops"] = 1 + df["depth2_n_edges"] - df["depth2_n_vertices"]
    return df


def main():
    analysis = "predict"  # [features_heatmap, vasc_tsne, predict_acc]
    features = []  # if left empty, all features will be taken
    sample = 0  # 0 == use full dataset, otherwise num of nodes to sample for analysis
    scale = True
    extract_output = True  # False

    file_path = "/Users/leahbiram/Desktop/vasculature_data/firstGBMscanGraph.csv"
    df = pd.read_pickle(file_path)
    df = add_immune_analysis(df)
    if len(features) == 0 and analysis in ["tsne", "predict"]:
        features = exclude_immune_features(df)
    print(features)

    if scale:
        scaler = MinMaxScaler()
        scaler = scaler.fit(df)
        scaler.transform(df)

    sampled_df = df if sample == 0 else df.sample(sample)

    if analysis == "heatmap": features_heatmap(sampled_df)
    if analysis == "tsne": vasc_tsne(sampled_df, features, "GBM tsne by log raw immune")
    if analysis == "predict": predictions = predict_acc(sampled_df, features, "has_immune", 100)
    if analysis == "scatter": scatter(sampled_df, features)
    if analysis == "bar plot": print("not implemented")

    if extract_output:
        df.drop("predictions", axis=1)
        #remove previous predictions
        all_features_df = pd.concat([df, pd.DataFrame({"predictions":predictions})], axis=1)
        all_features_df.to_pickle(file_path)



if __name__ == "__main__":
    main()


#depth1_e_volume_mean', 'depth1_e_volume_std', 'depth2_e_volume_mean', 'depth2_e_volume_std', 'depth1_e_radii_mean',
# 'depth1_e_radii_std', 'depth1_e_length_mean', 'depth1_e_length_std', 'depth1_e_prolif_mean', 'depth1_e_prolif_std',
# 'depth1_e_x_direction_mean', 'depth1_e_x_direction_std', 'depth1_e_y_direction_mean', 'depth1_e_y_direction_std',
# 'depth1_e_z_direction_mean', 'depth1_e_z_direction_std', 'depth1_v_radii_mean', 'depth1_v_radii_std',
# 'depth1_v_min_angle_mean', 'depth1_v_min_angle_std', 'depth1_n_vertices', 'depth1_n_edges', 'depth1_v_e_ratio',
# 'depth2_e_radii_mean', 'depth2_e_radii_std', 'depth2_e_length_mean', 'depth2_e_length_std', 'depth2_e_prolif_mean',
# 'depth2_e_prolif_std', 'depth2_e_x_direction_mean', 'depth2_e_x_direction_std', 'depth2_e_y_direction_mean',
# 'depth2_e_y_direction_std', 'depth2_e_z_direction_mean', 'depth2_e_z_direction_std', 'depth2_v_radii_mean',
# 'depth2_v_radii_std', 'depth2_v_min_angle_mean', 'depth2_v_min_angle_std', 'depth2_n_vertices', 'depth2_n_edges',
# 'depth2_v_e_ratio', 'near_vertices_20px', 'near_vertices_100px']
