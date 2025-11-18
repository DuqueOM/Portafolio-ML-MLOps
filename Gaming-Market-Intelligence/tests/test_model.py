import numpy as np
import pandas as pd
from data.preprocess import PreprocessConfig, build_preprocessor
from evaluate_business import compute_investment_kpis
from sklearn.ensemble import RandomForestClassifier
from sklearn.pipeline import Pipeline


def test_training_pipeline_fits():
    df = pd.DataFrame(
        {
            "platform": ["PS4", "PS4", "XOne", "PC", "PS4", "XOne"],
            "year_of_release": [2015, 2016, 2014, 2012, 2011, 2016],
            "genre": ["Action", "Sports", "Action", "Strategy", "Action", "Sports"],
            "critic_score": [80, 82, 75, 70, 65, 85],
            "user_score": [8.0, 7.5, 8.5, 7.0, 6.0, 9.0],
            "rating": ["M", "E", "M", "T", "M", "E"],
        }
    )
    y = np.array([1, 0, 1, 0, 0, 1])

    pre = build_preprocessor(df, config=PreprocessConfig())
    clf = RandomForestClassifier(n_estimators=10, random_state=42)
    pipe = Pipeline(steps=[("preprocessor", pre), ("clf", clf)])

    pipe.fit(df, y)
    preds = pipe.predict(df)
    assert preds.shape == y.shape


def test_investment_kpis_basic_coherence():
    """KPIs de inversión deben ser coherentes y estar en rangos razonables."""

    y_true = np.array([1, 1, 0, 0, 1, 0, 1, 0])
    y_pred = np.array([1, 1, 0, 0, 1, 0, 0, 0])

    kpis = compute_investment_kpis(y_true, y_pred)

    assert kpis["total_projects"] == float(len(y_true))
    assert 0 <= kpis["success_rate_model"] <= 1
    assert 0 <= kpis["failure_rate_model"] <= 1
    # El modelo no debería generar más fallos que la estrategia de invertir en todo
    assert kpis["baseline_failed_if_invest_all"] >= kpis["model_failed_if_invest_pred1"]
    # Fallos evitados no negativos
    assert kpis["estimated_failures_avoided"] >= 0
