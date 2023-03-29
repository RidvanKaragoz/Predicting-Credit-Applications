import pandas as pd
from mrmr import mrmr_classif
from mrmr.pandas import parallel_df
import functools
import numpy as np
from model_selection import X_y_split


def add_shadow_features(df):
    shadow_df = df.apply(lambda x: x.sample(frac=1).values).add_prefix("shadow_")
    df = pd.concat([df, shadow_df], axis=1)

    return df


def rank_correlation(target_column, features, X, n_jobs):
    def _correlation(X, y):
        return np.abs(X.corrwith(y, method="kendall")).fillna(0.0)

    return parallel_df(
        _correlation, X.loc[:, features], X.loc[:, target_column], n_jobs=n_jobs
    )


def score_features(X, y, n_jobs=-1, max_features=100):
    max_features = min(len(y), max_features)
    redundancy_func = functools.partial(rank_correlation, n_jobs=n_jobs)
    scored_features = mrmr_classif(
        X=X,
        y=y,
        K=max_features,
        relevance="rf",
        redundancy=redundancy_func,
        return_scores=True,
    )
    return scored_features


def select_features(df, label="label", max_features=100, return_scores=False):
    df = add_shadow_features(df)

    X, y = X_y_split(df, label)
    scored_features = score_features(X, y, n_jobs=-1, max_features=max_features)

    selected_features = scored_features[1]
    max_shadow = selected_features[
        selected_features.index.str.startswith("shadow")
    ].max()

    selected_features = selected_features[selected_features > max_shadow]

    if return_scores:
        return selected_features
    else:
        return selected_features.index.values.tolist()
