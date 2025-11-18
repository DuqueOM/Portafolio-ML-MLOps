import pandas as pd
from data.preprocess import PreprocessConfig, make_features_and_target


def test_make_features_and_target_synthetic():
    # Synthetic tiny dataset with minimal columns
    df = pd.DataFrame(
        {
            "name": ["A", "B", "C", "D"],
            "platform": ["PS4", "PS4", "XOne", "PC"],
            "year_of_release": [2015, 2016, 2014, 2012],
            "genre": ["Action", "Sports", "Action", "Strategy"],
            "critic_score": [80, 82, 75, None],
            "user_score": [8.0, 7.5, 8.5, 7.0],
            "rating": ["M", "E", "M", None],
            "na_sales": [0.6, 0.1, 0.3, 0.01],
            "eu_sales": [0.2, 0.05, 0.1, 0.01],
            "jp_sales": [0.05, 0.02, 0.0, 0.0],
            "other_sales": [0.05, 0.03, 0.02, 0.0],
        }
    )
    df["total_sales"] = df[["na_sales", "eu_sales", "jp_sales", "other_sales"]].sum(
        axis=1
    )

    cfg = PreprocessConfig(target_threshold_million=1.0)
    X, y = make_features_and_target(df, config=cfg)

    assert "platform" in X.columns and "genre" in X.columns
    assert len(X) == len(y) == 4
    assert set(y.unique()).issubset({0, 1})
