#!/usr/bin/env python3
"""Curate a firewalled NOAA GHCN-Daily HCN station graph and time split."""

from __future__ import annotations

import argparse
import hashlib
import json
import tarfile
from datetime import date, datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd
import yaml
from sklearn.neighbors import NearestNeighbors


EARTH_RADIUS_KM = 6371.0088


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def parse_station_line(line: str) -> dict[str, object]:
    """Parse the official fixed-width GHCNd station metadata record."""
    return {
        "station_id": line[0:11].strip(),
        "latitude": float(line[12:20]),
        "longitude": float(line[21:30]),
        "elevation_m": float(line[31:37]),
        "state": line[38:40].strip(),
        "station_name": line[41:71].strip(),
    }


def parse_dly_lines(lines: list[str], minimum_year: int, maximum_year: int) -> pd.DataFrame:
    """Return valid paired TMAX/TMIN daily means from one station file."""
    values: dict[tuple[str, date], dict[str, float]] = {}
    for line in lines:
        if len(line) < 269:
            continue
        station_id = line[0:11]
        year, month, element = int(line[11:15]), int(line[15:17]), line[17:21]
        if year < minimum_year or year > maximum_year or element not in {"TMAX", "TMIN"}:
            continue
        for day in range(1, 32):
            offset = 21 + (day - 1) * 8
            raw_value = int(line[offset : offset + 5])
            quality_flag = line[offset + 6 : offset + 7]
            if raw_value == -9999 or quality_flag.strip():
                continue
            try:
                observed_date = date(year, month, day)
            except ValueError:
                continue
            values.setdefault((station_id, observed_date), {})[element] = raw_value / 10.0
    records = []
    for (station_id, observed_date), pair in values.items():
        if pair.keys() >= {"TMAX", "TMIN"}:
            records.append(
                {
                    "station_id": station_id,
                    "date": pd.Timestamp(observed_date),
                    "tmean_c": 0.5 * (pair["TMAX"] + pair["TMIN"]),
                }
            )
    return pd.DataFrame.from_records(records, columns=["station_id", "date", "tmean_c"])


def assign_partition(year: int, config: dict) -> str | None:
    split = config["time_split"]
    for partition, key in [
        ("train", "train_years"),
        ("validation", "validation_years"),
        ("test", "test_years"),
    ]:
        if year in {int(item) for item in split[key]}:
            return partition
    return None


def build_station_graph(stations: pd.DataFrame, config: dict) -> pd.DataFrame:
    graph = config["graph"]
    k = int(graph["k"])
    if len(stations) <= k:
        raise ValueError(f"Need more than {k} stations to build the frozen kNN graph")
    coordinates = np.radians(stations[["latitude", "longitude"]].to_numpy(dtype=float))
    model = NearestNeighbors(n_neighbors=k + 1, metric="haversine").fit(coordinates)
    distances, indices = model.kneighbors(coordinates)
    station_ids = stations["station_id"].tolist()
    maximum_distance = float(graph["maximum_edge_distance_km"])
    edge_map: dict[tuple[str, str], float] = {}
    for source_index in range(len(stations)):
        for distance_rad, target_index in zip(distances[source_index, 1:], indices[source_index, 1:]):
            distance_km = float(distance_rad * EARTH_RADIUS_KM)
            if distance_km > maximum_distance:
                continue
            lo, hi = sorted((station_ids[source_index], station_ids[int(target_index)]))
            edge_map[(lo, hi)] = min(distance_km, edge_map.get((lo, hi), float("inf")))
    if not edge_map:
        raise RuntimeError("Frozen graph construction produced no edge")
    median_distance = float(np.median(list(edge_map.values())))
    records = [
        {
            "station_id_lo": lo,
            "station_id_hi": hi,
            "distance_km": distance,
            "weight": float(np.exp(-distance / median_distance)),
        }
        for (lo, hi), distance in sorted(edge_map.items())
    ]
    return pd.DataFrame.from_records(records)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--station-limit", type=int)
    parser.add_argument("--smoke", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.smoke != (args.station_limit is not None):
        raise SystemExit("Use --smoke and --station-limit together")
    config = yaml.safe_load(args.config.read_text(encoding="utf-8"))
    source = config["source"]
    archive_path, stations_path = Path(source["archive_path"]), Path(source["stations_path"])
    if archive_path.stat().st_size != int(source["archive_bytes"]):
        raise RuntimeError("NOAA archive byte count mismatch")
    if sha256_file(archive_path) != str(source["archive_sha256"]):
        raise RuntimeError("NOAA archive SHA-256 mismatch")
    if stations_path.stat().st_size != int(source["stations_bytes"]):
        raise RuntimeError("NOAA station metadata byte count mismatch")
    if sha256_file(stations_path) != str(source["stations_sha256"]):
        raise RuntimeError("NOAA station metadata SHA-256 mismatch")

    # Preserve the official byte-defined fixed-width columns even when a station
    # name contains an extended character.
    station_rows = [
        parse_station_line(line)
        for line in stations_path.read_text(encoding="latin-1").splitlines()
    ]
    stations = pd.DataFrame.from_records(station_rows)
    curation = config["curation"]
    stations = stations[
        stations["latitude"].between(*map(float, curation["latitude_range"]))
        & stations["longitude"].between(*map(float, curation["longitude_range"]))
        & np.isfinite(stations["elevation_m"])
        & (stations["elevation_m"] > -999.0)
    ].copy()
    eligible_metadata_ids = set(stations["station_id"])
    years = [int(year) for key in ["train_years", "validation_years", "test_years"] for year in config["time_split"][key]]

    frames: list[pd.DataFrame] = []
    with tarfile.open(archive_path, mode="r:gz") as archive:
        members = [
            member for member in archive.getmembers()
            if member.isfile() and Path(member.name).stem in eligible_metadata_ids
        ]
        if args.station_limit is not None:
            members = sorted(members, key=lambda item: (-item.size, item.name))[: args.station_limit]
        else:
            members = sorted(members, key=lambda item: item.name)
        for member in members:
            handle = archive.extractfile(member)
            if handle is None:
                continue
            lines = handle.read().decode("ascii").splitlines()
            parsed = parse_dly_lines(lines, min(years), max(years))
            if not parsed.empty:
                frames.append(parsed)
    observations = pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()
    if observations.empty:
        raise RuntimeError("No paired valid NOAA temperature observations were parsed")
    observations["partition"] = observations["date"].dt.year.map(lambda year: assign_partition(int(year), config))
    observations = observations[observations["partition"].notna()].drop_duplicates(
        ["station_id", "date"], keep="first"
    )

    expected_counts = {
        partition: sum(pd.Timestamp(f"{year}-12-31").dayofyear for year in config["time_split"][key])
        for partition, key in [("train", "train_years"), ("validation", "validation_years"), ("test", "test_years")]
    }
    counts = observations.groupby(["station_id", "partition"]).size().unstack(fill_value=0)
    for partition in expected_counts:
        if partition not in counts:
            counts[partition] = 0
        counts[f"{partition}_coverage"] = counts[partition] / expected_counts[partition]
    threshold = float(curation["minimum_partition_coverage"])
    counts["eligible"] = np.logical_and.reduce(
        [counts[f"{partition}_coverage"] >= threshold for partition in expected_counts]
    )
    eligible_ids = set(counts.index[counts["eligible"]])
    minimum_stations = 10 if args.smoke else int(curation["minimum_eligible_stations"])
    if len(eligible_ids) < minimum_stations:
        raise RuntimeError(
            f"Only {len(eligible_ids)} stations meet coverage, below required {minimum_stations}"
        )
    observations = observations[observations["station_id"].isin(eligible_ids)].copy()
    stations = stations[stations["station_id"].isin(eligible_ids)].sort_values("station_id").reset_index(drop=True)
    edges = build_station_graph(stations, config)

    args.output_dir.mkdir(parents=True, exist_ok=True)
    stations.to_parquet(args.output_dir / "stations.parquet", index=False)
    edges.to_parquet(args.output_dir / "graph_edges.parquet", index=False)
    observations[["station_id", "date", "partition"]].sort_values(
        ["date", "station_id"]
    ).to_parquet(args.output_dir / "partitions.parquet", index=False)
    observations[["station_id", "date", "tmean_c"]].sort_values(
        ["date", "station_id"]
    ).to_parquet(args.output_dir / "labels_private.parquet", index=False)
    counts.reset_index().sort_values("station_id").to_csv(
        args.output_dir / "station_coverage_audit.csv", index=False
    )
    manifest = {
        "dataset": config["name"],
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "source_version": source["version"],
        "archive_sha256": source["archive_sha256"],
        "stations_sha256": source["stations_sha256"],
        "archive_members_read": len(members),
        "eligible_stations": len(stations),
        "graph_edges": len(edges),
        "partition_rows": int(len(observations)),
        "smoke": args.smoke,
        "station_limit": args.station_limit,
        "response_summary_exposed": False,
    }
    (args.output_dir / "curation_manifest.json").write_text(
        json.dumps(manifest, indent=2, sort_keys=True), encoding="utf-8"
    )
    print(json.dumps(manifest, indent=2))


if __name__ == "__main__":
    main()
