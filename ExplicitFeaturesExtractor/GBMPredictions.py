import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from xgboost import XGBClassifier
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
from sklearn.metrics import classification_report
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
from scipy.stats import ttest_rel
import plotly.express as px
import ast


def run_xgb(df_train, df_test, leave_out_feats, label, threshold=0.5, print_cm=False, title=""):
    y_train = df_train[label]
    y_test = df_test[label]

    # normalization
    x_combined = pd.concat([df_train, df_test], axis=0)
    x_combined = x_combined.drop(leave_out_feats, axis=1)
    scaler = StandardScaler()
    x_combined_normalized = scaler.fit_transform(x_combined)
    x_train_normalized = x_combined_normalized[:len(df_train)]
    x_test_normalized = x_combined_normalized[len(df_train):]

    xgb = XGBClassifier()
    bst = xgb.fit(x_train_normalized, y_train)

    #y_pred = (xgb.predict(x_test_normalized))
    y_pred = (xgb.predict_proba(x_test_normalized)[:, 1] >= threshold).astype(int)
    y_pred_prob = xgb.predict_proba(x_test_normalized)
    print(classification_report(y_test, y_pred))
    if print_cm:
        cm = confusion_matrix(y_test, y_pred, labels=xgb.classes_)
        disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=xgb.classes_)
        ax = disp.plot()
        disp.ax_.set_title(title)
        plt.show()

    return y_test, y_pred, y_pred_prob, xgb


def calculate_metrics(y_true, y_pred, y_pred_prob):
    accuracy = accuracy_score(y_true, y_pred)
    precision = precision_score(y_true, y_pred)
    recall = recall_score(y_true, y_pred)
    f1 = f1_score(y_true, y_pred)
    roc_auc = roc_auc_score(y_true, y_pred_prob[:, 1])

    return accuracy, precision, recall, f1, roc_auc


def perform_statistical_test(metric_values):
    num_features = len(metric_values)
    p_values = []
    for i in range(num_features):
        for j in range(i + 1, num_features):
            _, p_value = ttest_rel(metric_values[i], metric_values[j])
            p_values.append(p_value)

    return p_values


def visualize_importance(importance, feats):
    # Create a box plot
    plt.boxplot(list(zip(*importance)))

    # Set labels and title
    plt.xticks(range(1, len(feats) + 1), feats, rotation=90)
    plt.xlabel("Feature Importance Lists")
    plt.ylabel("Feature Importance")
    plt.title("Feature Importance 1-10 hops models")
    plt.tight_layout()
    # Show the plot
    plt.show()

    for imp in importance:
        sorted_idx = np.argsort(imp)[::-1]
        for index in sorted_idx:
            print([feats[index], imp[index]])


def visualize_metrics(feature_metrics, n_models=1, LOO=False, loo_feats=[], threshold=0.5):
    # Create lists of metric names and values for plotting
    metric_names = ['Accuracy', 'Precision', 'Recall', 'F1-Score', 'AUC-ROC']
    metric_values = list(zip(*feature_metrics))

    # Plotting bar graphs
    plt.figure(figsize=(10, 6))
    for i, metric in enumerate(metric_values):
        plt.bar([x + 0.1 * (i - 2) for x in range(len(metric))], metric, 0.1, label=metric_names[i])

    if LOO:
        plt.xticks(range(len(loo_feats)), loo_feats, rotation=90)
    else:
        plt.xticks(range(n_models), [f'{i}-hop' for i in range(1, n_models + 1)])
    plt.xlabel('Models')
    plt.ylabel('Metric Score')
    plt.title(f'Comparison of Evaluation Metrics, threshold={threshold}')
    plt.legend()
    plt.tight_layout()
    plt.show()

    # Perform statistical tests
 #   p_values = perform_statistical_test(metric_values)
 #   print(p_values)


def get_df_data(file_p, drop_cols, pref):
    data_df = pd.read_csv(file_p)
    data_df = data_df.drop(drop_cols, axis=1)
    data_df.rename(columns=lambda x: pref + x.replace(pref, '', 1), inplace=True)
    return data_df

def run_vox_and_hop():
    f_metrics = []
    f_importance = []
    leave_out_feats = ['vox_artery_binary', 'vox_marker_in_subgraph', 'hop_artery_binary',
                       'hop_coordinates', 'vox_coordinates', 'hop_marker_in_subgraph']
    label = 'vox_marker_in_subgraph'

    data_train_vox = get_df_data(f"../ExtractedFeatureFiles/firstGBMFeatures_100_Vox_train.csv",
                                 ['vertex ID', 'artery_raw', '_nx_name'], 'vox_')
    data_train_hop = get_df_data(f"../ExtractedFeatureFiles/firstGBMFeatures_10_hop_train.csv",
                                 ['vertex ID', 'artery_raw', '_nx_name'], 'hop_')
    data_test_vox = get_df_data(f"../ExtractedFeatureFiles/firstGBMFeatures_100_Vox_test.csv",
                                 ['vertex ID', 'artery_raw', '_nx_name'], 'vox_')
    data_test_hop = get_df_data(f"../ExtractedFeatureFiles/firstGBMFeatures_10_hop_test.csv",
                                 ['vertex ID', 'artery_raw', '_nx_name'], 'hop_')

    combined_train_df = pd.concat([data_train_hop, data_train_vox], axis=1)
    combined_test_df = pd.concat([data_test_hop, data_test_vox], axis=1)

    y_test, y_pred, y_pred_prob, model = run_xgb(combined_train_df, combined_test_df, leave_out_feats, label)

    f_metrics.append(calculate_metrics(y_test, y_pred, y_pred_prob))
    f_importance.append(model.feature_importances_)

    visualize_metrics(f_metrics)
    feature_names = [s.replace('vox_', '') for s in combined_train_df.drop(leave_out_feats, axis=1).columns]
    visualize_importance(f_importance, feature_names)


def run_vox():
    f_metrics = []
    f_importance = []
    leave_out_feats = ['vox_artery_binary', 'vox_marker_in_subgraph', 'vox_coordinates']
    label = 'vox_marker_in_subgraph'

    data_train_vox = get_df_data(f"../ExtractedFeatureFiles/firstGBMFeatures_200_vox_train.csv",
                                 ['vertex ID', 'artery_raw', '_nx_name'], 'vox_')

    data_test_vox = get_df_data(f"../ExtractedFeatureFiles/firstGBMFeatures_200_vox_test.csv",
                                 ['vertex ID', 'artery_raw', '_nx_name'], 'vox_')


    y_test, y_pred, model = run_xgb(data_train_vox, data_test_vox, leave_out_feats, label)

    f_metrics.append(calculate_metrics(y_test, y_pred))
    f_importance.append(model.feature_importances_)

    visualize_metrics(f_metrics)
    feature_names = [s.replace('vox_', '') for s in data_train_vox.drop(leave_out_feats, axis=1).columns]
    visualize_importance(f_importance, feature_names)


def run_different_hops(n_models = 10):
    f_metrics = []
    f_importance = []
    leave_out_feats = ['vertex ID', 'artery_binary', 'artery_raw', 'marker_in_subgraph',
                       'coordinates', '_nx_name','degree', 'e_count', 'v_count']
    label = 'marker_in_subgraph'
    for hop in range(1, n_models + 1):
        data_train = pd.read_csv(f"../ExtractedFeatureFiles/GoodTrainTestGraphs/firstGBMFeatures_{hop}_Hop_train.csv") #200_Vox.csv")
        data_test = pd.read_csv(f"../ExtractedFeatureFiles/GoodTrainTestGraphs/firstGBMFeatures_{hop}_Hop_test.csv")
        #data_train, data_test = train_test_split(data_df, test_size=0.2)
        print(f"running model with {hop}-hops features")
        y_test, y_pred, y_pred_prob, model = run_xgb(data_train, data_test, leave_out_feats, label)

        f_metrics.append(calculate_metrics(y_test, y_pred, y_pred_prob))
        f_importance.append(model.feature_importances_)

    visualize_metrics(f_metrics, n_models)
    feature_names = [s.replace('hop_', '') for s in data_train.drop(leave_out_feats, axis=1).columns]
    visualize_importance(f_importance, feature_names)


def convert_string_list_to_integers(str_coord):
    cleaned_string = str_coord.replace('array(', '').replace(')', '')
    float_list = ast.literal_eval(cleaned_string)
    return [int(num) for num in float_list]


def scatter_points_preds(data_test, y_real, y_pred):
    assert len(data_test) == len(y_real)
    coordinates = data_test["coordinates"].tolist()
    labels = []
    for y_r, y_p in zip(y_real, y_pred):
        if y_p == 1:
            if y_r == 1:
                labels.append("TP")
            else:
                labels.append("FP")
        else:
            if y_r == 0:
                labels.append("TN")
            else:
                labels.append("FN")

    int_coords = [convert_string_list_to_integers(c) for c in coordinates]
    fig = px.scatter_3d(
        x=[coord[0] for coord in int_coords],
        y=[coord[1] for coord in int_coords],
        z=[coord[2] for coord in int_coords],
        color=labels,
        color_discrete_sequence=["green", "red", "blue", "orange"],
    )
    fig.update_traces(marker_size=3)
    fig.show()




def run_compare_multiple_dfs(dfs, model_name, threshold=0.5, include_counts=False, scatter_predictions=False):
    f_metrics = []
    f_importance = []
    leave_out_feats = ['vertex ID', 'artery_binary', 'artery_raw', 'marker_in_subgraph',
                       'coordinates', '_nx_name']
    if not include_counts:
        leave_out_feats.extend(['degree', 'e_count', 'v_count'])
    label = 'marker_in_subgraph'
    for i, train_test in enumerate(dfs):
        data_train = train_test[0]
        data_test = train_test[1]
        print(f"running {i} - {model_name} model")
        y_test, y_pred, y_pred_prob, model = run_xgb(data_train, data_test, leave_out_feats, label, threshold=threshold)
        if scatter_predictions:
            scatter_points_preds(data_test, y_test, y_pred)
        f_metrics.append(calculate_metrics(y_test, y_pred, y_pred_prob))
        f_importance.append(model.feature_importances_)

    visualize_metrics(f_metrics, len(dfs), threshold=threshold)
    feature_names = [s.replace(f'{model_name}_', '') for s in data_train.drop(leave_out_feats, axis=1).columns]
    visualize_importance(f_importance, feature_names)


def run_leave1_out():
    f_metrics = []
    f_importance = []
    loo_feats = []

    leave_out_feats = ['vertex ID', 'artery_binary', 'artery_raw',
                       'marker_in_subgraph', 'coordinates', '_nx_name', 'degree', 'e_count', 'v_count']
    label = 'marker_in_subgraph'
    data_train = pd.read_csv(f"../ExtractedFeatureFiles/GoodTrainTestGraphs/firstGBMFeatures_5_Hop_train.csv")  # 200_Vox.csv")
    data_test = pd.read_csv(f"../ExtractedFeatureFiles/GoodTrainTestGraphs/firstGBMFeatures_5_Hop_test.csv")

    feature_names = data_train.drop(leave_out_feats, axis=1).columns
    data_test.drop(leave_out_feats, axis=1)

    for f in feature_names:
        print(f"running leave-one-out for '{f}' feature")
        y_test, y_pred, y_pred_prob, model = run_xgb(data_train, data_test, leave_out_feats + [f], label)

        f_metrics.append(calculate_metrics(y_test, y_pred, y_pred_prob))
        f_importance.append(model.feature_importances_)

    #print(f"running leave-one-out for '{['degree', 'e_count', 'v_count']}' feature")
    #y_test, y_pred, y_pred_prob, model = run_xgb(data_train, data_test, leave_out_feats + ['degree', 'vox_e_count', 'vox_v_count'], label)

    #f_metrics.append(calculate_metrics(y_test, y_pred, y_pred_prob))
    #f_importance.append(model.feature_importances_)

    visualize_metrics(f_metrics, LOO=True, loo_feats=feature_names)
    visualize_importance(f_importance, feature_names)


if __name__ == "__main__":
    #main()
    run_leave1_out()
    #run_different_hops()
    #run_vox_and_hop()
    #run_vox()

    # TODO Kfold cross validation
    # TODO split with spare nodes
