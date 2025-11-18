from __future__ import annotations

import json
from pathlib import Path

import pandas as pd
import yaml
from scipy import stats


def load_dataset_and_config(
    cfg_path: str | Path = "configs/config.yaml",
) -> tuple[pd.DataFrame, dict]:
    cfg_path = Path(cfg_path)
    with cfg_path.open("r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)
    ds_path = Path(cfg["paths"]["dataset_path"])
    df = pd.read_csv(ds_path)
    return df, cfg


def export_hypothesis_tests_summary(
    cfg_path: str | Path = "configs/config.yaml",
) -> dict:
    df, _ = load_dataset_and_config(cfg_path)

    summary: dict[str, dict] = {}

    # Platform performance: ANOVA sobre global_sales para plataformas con suficientes juegos
    if {"platform", "global_sales"}.issubset(df.columns):
        platform_counts = df["platform"].value_counts()
        major_platforms = platform_counts[platform_counts >= 50].index.tolist()
        groups = [
            df.loc[df["platform"] == p, "global_sales"].dropna().to_numpy()
            for p in major_platforms
        ]
        groups = [g for g in groups if g.size > 0]
        if len(groups) >= 2:
            f_stat, p_val = stats.f_oneway(*groups)
            means = (
                df[df["platform"].isin(major_platforms)]
                .groupby("platform")["global_sales"]
                .mean()
                .sort_values(ascending=False)
            )
            summary["platform_performance"] = {
                "test": "one_way_anova",
                "p_value": float(p_val),
                "significant": bool(p_val < 0.05),
                "f_stat": float(f_stat),
                "top_platforms_by_mean_sales": means.head(5).to_dict(),
            }

    # Genre performance: ANOVA sobre global_sales por género
    if {"genre", "global_sales"}.issubset(df.columns):
        genre_counts = df["genre"].value_counts()
        major_genres = genre_counts[genre_counts >= 50].index.tolist()
        groups = [
            df.loc[df["genre"] == g, "global_sales"].dropna().to_numpy()
            for g in major_genres
        ]
        groups = [g for g in groups if g.size > 0]
        if len(groups) >= 2:
            f_stat, p_val = stats.f_oneway(*groups)
            means = (
                df[df["genre"].isin(major_genres)]
                .groupby("genre")["global_sales"]
                .mean()
                .sort_values(ascending=False)
            )
            summary["genre_performance"] = {
                "test": "one_way_anova",
                "p_value": float(p_val),
                "significant": bool(p_val < 0.05),
                "f_stat": float(f_stat),
                "top_genres_by_mean_sales": means.head(5).to_dict(),
            }

    # Correlación critic_score vs user_score
    if {"critic_score", "user_score"}.issubset(df.columns):
        mask = df[["critic_score", "user_score"]].notna().all(axis=1)
        if mask.any():
            r, p_val = stats.pearsonr(
                df.loc[mask, "critic_score"].to_numpy(),
                df.loc[mask, "user_score"].to_numpy(),
            )
            summary["critic_user_correlation"] = {
                "test": "pearsonr",
                "r": float(r),
                "p_value": float(p_val),
                "significant": bool(p_val < 0.05),
            }

    # Tendencia temporal de ventas medias por año
    if {"year_of_release", "global_sales"}.issubset(df.columns):
        year_sales = (
            df.dropna(subset=["year_of_release", "global_sales"])
            .groupby("year_of_release")["global_sales"]
            .mean()
        )
        years = year_sales.index.to_numpy()
        vals = year_sales.to_numpy()
        if years.size >= 3:
            rho, p_val = stats.spearmanr(years, vals)
            summary["temporal_trend"] = {
                "test": "spearmanr",
                "rho": float(rho),
                "p_value": float(p_val),
                "significant": bool(p_val < 0.05),
            }

    # Guardar a artifacts/hypothesis_tests_summary.json
    artifacts_dir = Path("artifacts")
    artifacts_dir.mkdir(parents=True, exist_ok=True)
    out_path = artifacts_dir / "hypothesis_tests_summary.json"
    with out_path.open("w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)

    return summary


def main() -> None:
    summary = export_hypothesis_tests_summary()
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
