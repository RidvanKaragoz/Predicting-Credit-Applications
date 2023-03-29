import pandas as pd


def load_data(input_path="data/processed/processed.pkl"):
    # Load the dataset
    df = pd.read_pickle(input_path)

    return df


def clean_client_data(df):
    df = df.set_index(["date"], verify_integrity=True).asfreq(freq="MS")
    df.sort_index()
    df["CRG"] = df["CRG"].max()
    df.fillna(0, inplace=True)

    return df


def add_features(df):
    df["month"] = df.index.month
    df["total_volume_trx"] = df["volume_credit_trx"] + df["volume_debit_trx"]
    df["balance_change"] = df["volume_credit_trx"] - df["volume_debit_trx"]
    df["balance_range"] = df["max_balance"] - df["min_balance"]
    df["average_credit_trx"] = (df["volume_credit_trx"] / df["nr_credit_trx"]).fillna(0)
    df["average_debit_trx"] = (df["volume_debit_trx"] / df["nr_debit_trx"]).fillna(0)

    return df


def add_temporal_features(df):
    temporal_features = df.drop(["month", "CRG", "client_nr"], axis=1)

    lag_features = temporal_features.shift(1).add_prefix("lag_")
    diff_features = temporal_features.diff().add_prefix("diff_")
    rolling_features = temporal_features.rolling(window=12, min_periods=0).agg(
        ["mean", "std", "median", "max", "min"]
    )
    rolling_features.columns = [
        "_".join(col) for col in rolling_features.columns.to_flat_index()
    ]
    bernoulli_p = (
        temporal_features["nr_credit_applications"]
        .expanding()
        .mean()
        .rename("bernoulli_p")
    )

    df = pd.concat(
        [df, lag_features, diff_features, rolling_features, bernoulli_p], axis=1
    )

    return df


def add_labels(df, time_till_application=False):
    df["label"] = df["credit_application"].shift(-1)

    if time_till_application:
        df["time_till_application"] = df.groupby(
            df["credit_application"].cumsum()
        ).cumcount(ascending=False)

    return df


def process_client_data(client_data):
    client_data = clean_client_data(client_data)

    client_data = add_features(client_data)

    client_data = add_temporal_features(client_data)

    client_data = add_labels(client_data)

    client_data.dropna(inplace=True)

    return client_data


def create_features(df):
    """
    Create features from the merged dataframe.
    """

    grouping = df.groupby("client_nr")
    client_data_list = []

    for _, client_data in grouping:
        client_data = process_client_data(client_data)

        client_data_list.append(client_data)

    feature_df = pd.concat(client_data_list, axis=0)
    feature_df.set_index("client_nr", append=True, inplace=True)
    feature_df.sort_index(inplace=True)

    return feature_df


def main():
    df = load_data()
    df = create_features(df)
    df.to_pickle("data/features/features.pkl")


if __name__ == "__main__":
    main()
