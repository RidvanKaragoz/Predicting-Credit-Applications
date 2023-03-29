import pandas as pd
import numpy as np
import shap
import matplotlib.pyplot as plt
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    average_precision_score,
    balanced_accuracy_score,
    cohen_kappa_score,
    matthews_corrcoef,
    zero_one_loss,
    brier_score_loss,
    log_loss,
    roc_curve,
    auc,
    precision_recall_curve,
)


def get_curve_plots(y_true, y_score):
    fpr, tpr, thresholds = roc_curve(y_true, y_score)
    roc_auc = auc(fpr, tpr)

    precision, recall, thresholds = precision_recall_curve(y_true, y_score)
    avg_precision = average_precision_score(y_true, y_score)

    # ROC-AUC curve
    plt.figure()
    plt.plot(
        fpr, tpr, color="darkorange", lw=2, label="ROC curve (area = %0.2f)" % roc_auc
    )
    plt.plot([0, 1], [0, 1], color="navy", lw=2, linestyle="--")
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title("Receiver operating characteristic curve")
    plt.legend(loc="lower right")
    plt.savefig("results/figures/Receiver-operating-characteristic-curve.png", dpi=700)

    # PR curve
    plt.figure()
    plt.step(recall, precision, color="b", alpha=0.2, where="post")
    plt.fill_between(recall, precision, step="post", alpha=0.2, color="b")
    plt.xlabel("Recall")
    plt.ylabel("Precision")
    plt.ylim([0.0, 1.05])
    plt.xlim([0.0, 1.0])
    plt.title("Precision-Recall curve: AP={0:0.2f}".format(avg_precision))
    plt.savefig("results/figures/Precision-Recall-curve.png", dpi=700)


def binary_classification_metrics(y_true, y_pred):
    y_class = y_pred > 0.5

    metrics = {}
    metrics["accuracy"] = accuracy_score(y_true, y_class)
    metrics["balanced_accuracy"] = balanced_accuracy_score(y_true, y_class)
    metrics["average_precision"] = average_precision_score(y_true, y_pred)
    metrics["roc_auc"] = roc_auc_score(y_true, y_pred)
    metrics["f1"] = f1_score(y_true, y_class, zero_division=0)
    metrics["precision"] = precision_score(y_true, y_class, zero_division=0)
    metrics["recall"] = recall_score(y_true, y_class)
    # metrics['confusion_matrix'] = confusion_matrix(y_true, y_class)
    metrics["cohen_kappa"] = cohen_kappa_score(y_true, y_class)
    metrics["matthews_corrcoef"] = matthews_corrcoef(y_true, y_class)
    metrics["zero_one_loss"] = zero_one_loss(y_true, y_class)
    metrics["brier_score_loss"] = brier_score_loss(y_true, y_pred)
    metrics["log_loss"] = log_loss(y_true, y_pred)
    return metrics


def X_y_split(df, label="label", features=None):
    if features is not None:
        X = df[features]
    else:
        X = df.drop(label, axis=1)

    y = df[label]

    return X, y


def ts_train_test_split(df, train_percent=0.8):
    df.sort_index(inplace=True)
    dates = df.index.get_level_values("date")

    min_date = dates.min()
    max_date = dates.max()

    cutoff_date = min_date + train_percent * (max_date - min_date)

    train_df = df.loc[:cutoff_date]
    test_df = df.loc[cutoff_date + pd.Timedelta(days=1) :]

    return train_df, test_df


def get_fold_dates(dates, k=5):
    dates = pd.date_range(dates.min(), dates.max(), freq="MS")[::-1]

    fold_size = len(dates) // (k)

    fold_dates = []
    for i in range(0, k):
        start_index = (i + 1) * fold_size
        end_index = i * fold_size

        start_date = dates[start_index - 1]
        end_date = dates[end_index]

        if i == k - 1:
            start_date = dates[-1]

        fold_dates.append((start_date, end_date))

    return fold_dates


def get_folds(df, k=5):
    dates = df.index.get_level_values("date")

    fold_dates = get_fold_dates(dates, k)

    for i in range(k - 1):
        train_df = df[: fold_dates[i + 1][1]]
        valid_df = df[fold_dates[i][0] : fold_dates[i][1]]

        X_train, y_train = X_y_split(train_df)
        X_valid, y_valid = X_y_split(valid_df)

        yield X_train, y_train, X_valid, y_valid


def evaluate_model(model, X_train, y_train, X_valid, y_valid, plots=False):
    model.fit(X_train, y_train)
    y_pred = model.predict_proba(X_valid)[:, 1]
    metrics = binary_classification_metrics(y_valid, y_pred)

    if plots:
        get_curve_plots(y_valid, y_pred)

    return metrics


def cross_val_scores(df, model, metric="average_precision"):
    validation_metrics = []

    for X_train, y_train, X_valid, y_valid in get_folds(df, k=5):
        metrics = evaluate_model(model, X_train, y_train, X_valid, y_valid)

        validation_metrics.append(metrics)

    if metric is not None:
        score = np.mean([metrics[metric] for metrics in validation_metrics])
        return validation_metrics, score
    else:
        return validation_metrics


def explain(model, X):
    explainer = shap.TreeExplainer(model)
    shap_values = explainer.shap_values(X)

    return shap_values
