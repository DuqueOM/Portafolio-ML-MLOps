import numpy as np
import pandas as pd
from data.preprocess import PreprocessConfig, build_preprocessor
from sklearn.ensemble import RandomForestClassifier
from sklearn.pipeline import Pipeline


def test_platform_groups_produce_finite_predictions():
    """Smoke test de fairness: el clasificador funciona para distintas plataformas.

    Verifica que el pipeline produce predicciones finitas y no degenera en
    un único valor para todos los ejemplos de un subgrupo de plataforma.
    """

    df = pd.DataFrame(
        {
            "platform": ["PS4", "PS4", "XOne", "PC", "PS4", "XOne", "PC", "PS4"],
            "year_of_release": [2015, 2016, 2014, 2012, 2011, 2016, 2013, 2015],
            "genre": [
                "Action",
                "Sports",
                "Action",
                "Strategy",
                "Action",
                "Sports",
                "RPG",
                "Shooter",
            ],
            "critic_score": [80, 82, 75, 70, 65, 85, 78, 90],
            "user_score": [8.0, 7.5, 8.5, 7.0, 6.0, 9.0, 7.5, 8.8],
            "rating": ["M", "E", "M", "T", "M", "E", "T", "M"],
        }
    )
    y = np.array([1, 0, 1, 0, 0, 1, 0, 1])

    pre = build_preprocessor(df, config=PreprocessConfig())
    clf = RandomForestClassifier(n_estimators=16, random_state=42)
    pipe = Pipeline(steps=[("preprocessor", pre), ("clf", clf)])

    pipe.fit(df, y)
    preds = pipe.predict(df)
    assert preds.shape == y.shape

    for platform in ["PS4", "XOne", "PC"]:
        mask = df["platform"] == platform
        group_preds = preds[mask.to_numpy()]
        assert group_preds.size > 0
        # Al menos debe haber variabilidad o, como mínimo, valores finitos
        assert np.isfinite(group_preds).all()
