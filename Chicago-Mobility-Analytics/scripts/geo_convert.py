from __future__ import annotations

import argparse
from pathlib import Path

import geopandas as gpd
import pandas as pd
from shapely.geometry import Point


def csv_to_geoparquet(csv_path: Path, lat_col: str, lon_col: str, out_path: Path) -> None:
    df = pd.read_csv(csv_path)
    if lat_col not in df.columns or lon_col not in df.columns:
        raise SystemExit(f"Columns {lat_col}/{lon_col} not in CSV")
    gdf = gpd.GeoDataFrame(df, geometry=[Point(xy) for xy in zip(df[lon_col], df[lat_col])], crs="EPSG:4326")
    out_path.parent.mkdir(parents=True, exist_ok=True)
    gdf.to_parquet(out_path)


def main() -> None:
    ap = argparse.ArgumentParser(description="Convert CSV with lat/lon to GeoParquet")
    ap.add_argument("--csv", required=True, help="Input CSV path")
    ap.add_argument("--lat", default="lat")
    ap.add_argument("--lon", default="lon")
    ap.add_argument("--out", default="data/processed/points.geoparquet")
    args = ap.parse_args()
    csv_to_geoparquet(Path(args.csv), args.lat, args.lon, Path(args.out))


if __name__ == "__main__":
    main()
