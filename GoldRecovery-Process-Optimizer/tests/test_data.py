import numpy as np
import pandas as pd
from data.preprocess import (
    basic_clean,
    compute_recovery,
    create_features,
    fill_missing_with_median,
)


def test_compute_recovery_basic():
    feed = pd.Series([10.0, 15.0, 20.0])
    conc = pd.Series([30.0, 25.0, 22.0])
    tail = pd.Series([1.0, 2.0, 3.0])

    rec = compute_recovery(feed, conc, tail)

    assert rec.isna().sum() == 0
    assert (rec >= -1e3).all()  # rangos razonables, sin NaN/inf


def test_basic_clean_and_features():
    df = pd.DataFrame(
        {
            "date": pd.to_datetime(["2020-01-01 00:00:00", "2020-01-01 01:00:00"]),
            "final.output.recovery": [90.0, 95.0],
            "rougher.output.concentrate_au": [2.0, 3.0],
            "primary_cleaner.output.concentrate_au": [4.0, 6.0],
            "rougher.output.concentrate_ag": [1.5, 2.5],
            "primary_cleaner.output.concentrate_ag": [2.0, 3.0],
            "some_null": [np.nan, np.nan],
        }
    )

    dfc = basic_clean(df)
    dff = create_features(dfc)
    dff = fill_missing_with_median(dff)

    assert "au_recovery_ratio" in dff.columns
    assert "ag_recovery_ratio" in dff.columns
    assert dff["au_recovery_ratio"].notna().all()
    assert dff["ag_recovery_ratio"].notna().all()
