import pandas as pd
from models import Baseline
from model_selection import (
    ts_train_test_split,
    X_y_split,
    cross_val_scores,
    evaluate_model,
    explain,
    get_curve_plots,
)
from feature_selection import select_features
from optuna.visualization import plot_slice, plot_optimization_history
import pickle
import shap
import optuna
import xgboost as xgb
import matplotlib.pyplot as plt
import json
import warnings

warnings.filterwarnings("ignore")

base_params = {
    "objective": "binary:logistic",
    "random_state": 42,
    "n_jobs": -1,
    "tree_method": "hist",
    # 'booster': 'dart',
}


def load_data(input_path="data/features/features.pkl"):
    # Load the dataset
    df = pd.read_pickle(input_path)

    return df


def objective(trial, df):
    # Define the hyperparameters to be tuned
    params = {
        "n_estimators": trial.suggest_int("n_estimators", 10, 100),
        "max_depth": trial.suggest_int("max_depth", 3, 12),
        "learning_rate": trial.suggest_float("learning_rate", 1e-4, 1, log=True),
        "reg_alpha": trial.suggest_float("reg_alpha", 1e-5, 10.0, log=True),
        "reg_lambda": trial.suggest_float("reg_lambda", 1e-5, 10.0, log=True),
        "scale_pos_weight": trial.suggest_float(
            "scale_pos_weight", 0.1, 10000, log=True
        ),
        "min_child_weight": trial.suggest_float("min_child_weight", 1, 1000, log=True),
        "subsample": trial.suggest_float("subsample", 0.1, 1, log=True),
        # 'num_parallel_tree': trial.suggest_int('num_parallel_tree', 1, 3),
        # 'colsample_bytree': trial.suggest_float('colsample_bytree',  0.1, 1, log=True),
        # 'colsample_bylevel': trial.suggest_float('colsample_bylevel', 0.1, 1, log=True),
        # 'colsample_bynode': trial.suggest_float('colsample_bynode',  0.1, 1, log=True),
    }

    base_params.update(params)

    # Create the XGBoost classifier with the chosen hyperparameters
    model = xgb.XGBClassifier(**params)

    # Evaluate the model using cross-validation with AUC-PR score
    _, score = cross_val_scores(df, model, metric="average_precision")

    # Return the mean score as the objective value
    return score


def optimize(df, n_trials=100, timeout=30):
    func = lambda trial: objective(trial, df)

    # Create an Optuna study and optimize the objective function
    study = optuna.create_study(direction="maximize")
    study.optimize(func, n_trials=n_trials, timeout=timeout)

    return study


def fit_best(X, y, study):
    params = study.best_params
    base_params.update(params)
    model = xgb.XGBClassifier(**params)
    model.fit(X, y)

    return model


def save_feature_names(
    selected_feature_names, path="data/features/selected_feature_names.pkl"
):
    with open(path, "wb") as file:
        # A new file will be created
        pickle.dump(selected_feature_names, file)


def split_and_select_features(df):
    train_df, test_df = ts_train_test_split(df, train_percent=0.8)

    test_df.to_pickle("data/dummy/dummy_features.pkl")

    selected_feature_names = select_features(train_df, max_features=50)

    save_feature_names(selected_feature_names)

    selected_feature_names.append("label")

    train_df = train_df[selected_feature_names]
    test_df = test_df[selected_feature_names]

    return train_df, test_df


def get_optimization_plots(study):
    # plot the optimization history of the objective function
    plot_optimization_history(study).write_html("results/reports/optuna_results.html")

    # plot the parameter relationship with the objective function
    plot_slice(study).write_html("results/reports/optuna_slice.html")


def get_shap_plots(model, X):
    shap_values = explain(model, X)

    # Plot and save the summary plot of shap values
    summary_plot = shap.summary_plot(shap_values, X, show=False)
    plt.savefig("results/figures/shap_summary.png", dpi=700)


def save_metrics(
    random_score,
    baseline_test_metrics,
    xbg_test_metrics,
    path="results/metrics/performance.json",
):
    performance = {}
    performance["random_score"] = random_score
    performance["baseline_test_metrics"] = baseline_test_metrics
    performance["xbg_test_metrics"] = xbg_test_metrics

    with open(path, "w") as fp:
        json.dump(performance, fp, sort_keys=False, indent=4)


def main():
    df = load_data(input_path="data/features/features.pkl")

    train_df, test_df = split_and_select_features(df)

    X_train, y_train = X_y_split(train_df)
    X_test, y_test = X_y_split(test_df)

    study = optimize(df, n_trials=100)

    model = fit_best(X_train, y_train, study)
    model.save_model("models/xgb_model.json")

    get_optimization_plots(study)

    xbg_validation_metrics = cross_val_scores(train_df, model, metric=None)
    xbg_test_metrics = evaluate_model(
        model, X_train, y_train, X_test, y_test, plots=True
    )
    xbg_score = xbg_test_metrics["average_precision"]

    print(f"xbg_score: {xbg_score}")

    get_shap_plots(model, X_test)

    random_score = y_test.mean()
    print(f"random_score: {random_score}")

    baseline_model = Baseline()
    baseline_validation_metrics = cross_val_scores(
        train_df, baseline_model, metric=None
    )
    baseline_test_metrics = evaluate_model(
        baseline_model, X_train, y_train, X_test, y_test
    )
    baseline_score = baseline_test_metrics["average_precision"]
    print(f"baseline_score: {baseline_score}")

    save_metrics(random_score, baseline_test_metrics, xbg_test_metrics)


if __name__ == "__main__":
    main()
