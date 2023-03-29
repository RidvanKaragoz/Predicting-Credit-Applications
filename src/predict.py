import pandas as pd
import pickle
from model_selection import explain
import xgboost as xgb
import warnings

warnings.filterwarnings("ignore")


def load_model(path="models/xgb_model.json"):
    model = xgb.XGBClassifier()
    model.load_model(path)

    return model


def load_feature_names(path="data/features/selected_feature_names.pkl"):
    with open(path, "rb") as file:
        selected_feature_names = pickle.load(file)
        return selected_feature_names


def predict(model, X):
    y_pred = model.predict_proba(X)[:, 1]

    shap_values = explain(model, X)

    prediction_df = X.copy()
    prediction_df[:] = shap_values

    return prediction_df


def main():
    X = pd.read_pickle("data/dummy/dummy_features.pkl")

    selected_feature_names = load_feature_names()

    X = X[selected_feature_names]

    model = load_model()

    prediction_df = predict(model, X).add_suffix("_shap")

    output_path = "data/predictions/predictions.csv"
    prediction_df.to_csv(output_path)


if __name__ == "__main__":
    main()
