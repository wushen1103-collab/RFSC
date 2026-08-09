from __future__ import annotations

import pandas as pd

from scripts.prepare_noaa_ghcnd_hcn import build_station_graph, parse_dly_lines, parse_station_line


def _dly_line(station: str, year: int, month: int, element: str, values: dict[int, tuple[int, str]]) -> str:
    fields = []
    for day in range(1, 32):
        value, qflag = values.get(day, (-9999, " "))
        fields.append(f"{value:5d} {qflag} ")
    return f"{station:<11}{year:04d}{month:02d}{element}{''.join(fields)}"


def test_station_fixed_width_parser() -> None:
    line = f"{'USC00000001':<11} {35.125:8.4f} {-80.5:9.4f} {123.0:6.1f} NC {'TEST STATION':<30}"
    observed = parse_station_line(line)
    assert observed["station_id"] == "USC00000001"
    assert observed["latitude"] == 35.125
    assert observed["longitude"] == -80.5
    assert observed["elevation_m"] == 123.0


def test_station_parser_accepts_extended_name_bytes_without_shifting_coordinates() -> None:
    station_name = "PE\u00d1A STATION"
    line = f"{'USC00000002':<11} {40.25:8.4f} {-105.75:9.4f} {999.0:6.1f} CO {station_name:<30}"
    observed = parse_station_line(line)
    assert observed["latitude"] == 40.25
    assert observed["longitude"] == -105.75
    assert observed["station_name"] == station_name


def test_dly_requires_both_temperatures_and_blank_quality_flags() -> None:
    station = "USC00000001"
    tmax = _dly_line(station, 2024, 1, "TMAX", {1: (100, " "), 2: (120, "X")})
    tmin = _dly_line(station, 2024, 1, "TMIN", {1: (20, " "), 2: (40, " ")})
    observed = parse_dly_lines([tmax, tmin], 2024, 2024)
    assert observed["date"].tolist() == [pd.Timestamp("2024-01-01")]
    assert observed["tmean_c"].tolist() == [6.0]


def test_station_graph_has_no_self_edges_and_respects_distance_cap() -> None:
    stations = pd.DataFrame(
        {
            "station_id": [f"S{i}" for i in range(4)],
            "latitude": [35.0, 35.1, 35.2, 35.3],
            "longitude": [-80.0, -80.1, -80.2, -80.3],
        }
    )
    config = {"graph": {"k": 2, "maximum_edge_distance_km": 500.0}}
    edges = build_station_graph(stations, config)
    assert (edges["station_id_lo"] != edges["station_id_hi"]).all()
    assert (edges["distance_km"] <= 500.0).all()
    assert edges[["station_id_lo", "station_id_hi"]].duplicated().sum() == 0
