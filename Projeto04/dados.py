import io
import os
import urllib.request
import zipfile

import numpy as np
import pandas as pd
import torch
from numpy.lib.stride_tricks import sliding_window_view

STATION_KEY = "A728_TAUBATE"
URL = "https://portal.inmet.gov.br/uploads/dadoshistoricos/{year}.zip"
YEARS = range(2012, 2026)
DATA_DIR = "dados"
RAW_DIR = f"{DATA_DIR}/inmet_raw"
CSV_PATH = f"{DATA_DIR}/taubate_horario.csv"
START, END = "2012-01-01 00:00", "2025-12-31 23:00"
UTC_OFFSET_HOURS = -3
HORIZON = 24
MAX_GAP = 6
TRAIN_END, VAL_END = "2019-12-31 23:00", "2022-12-31 23:00"
TRAIN_STRIDE = 4
BASE_FEATURES = ["temp", "pressao", "umidade", "chuva", "vento_x", "vento_y"]
CYCLIC_FEATURES = ["hora_sin", "hora_cos", "dia_sin", "dia_cos"]
FEATURES = BASE_FEATURES + CYCLIC_FEATURES
VALID_RANGES = {"temp": (-5.0, 45.0), "pressao": (900.0, 1000.0), "umidade": (0.0, 100.0), "chuva": (0.0, 200.0),
                "vel": (0.0, 40.0), "dir": (0.0, 360.0)}
COLUMN_KEYS = {"chuva": "PRECIPITA", "pressao": "PRESSAO ATMOSFERICA AO NIVEL", "temp": "BULBO SECO",
               "umidade": "UMIDADE RELATIVA DO AR", "vel": "VELOCIDADE", "dir": "DIRE"}


def fetch_raw() -> None:
    os.makedirs(RAW_DIR, exist_ok=True)
    for year in YEARS:
        out = f"{RAW_DIR}/{year}.csv"
        if os.path.exists(out):
            continue
        request = urllib.request.Request(URL.format(year=year), headers={"User-Agent": "Mozilla/5.0"})
        with urllib.request.urlopen(request, timeout=300) as response:
            archive = zipfile.ZipFile(io.BytesIO(response.read()))
        name = next(n for n in archive.namelist() if STATION_KEY in n and n.upper().endswith(".CSV"))
        with open(out, "wb") as f:
            f.write(archive.read(name))


def parse_year(path: str) -> pd.DataFrame:
    text = open(path, "rb").read().decode("latin-1")
    df = pd.read_csv(io.StringIO(text), sep=";", skiprows=8, decimal=",", na_values=["-9999"])
    df = df.loc[:, ~df.columns.str.startswith("Unnamed")]
    date = df.iloc[:, 0].astype(str).str.replace("/", "-")
    hour = df.iloc[:, 1].astype(str).str.replace(":", "").str[:4]
    stamp = pd.to_datetime(date + " " + hour, format="%Y-%m-%d %H%M") + pd.Timedelta(hours=UTC_OFFSET_HOURS)
    out = {}
    for name, key in COLUMN_KEYS.items():
        column = next(c for c in df.columns if key in c)
        out[name] = pd.to_numeric(df[column], errors="coerce").to_numpy()
    return pd.DataFrame(out, index=stamp)


def fill_short_gaps(series: pd.Series, max_gap: int) -> pd.Series:
    missing = series.isna()
    run_id = (missing != missing.shift()).cumsum()
    run_len = missing.groupby(run_id).transform("sum")
    filled = series.interpolate(limit_area="inside")
    filled[missing & (run_len > max_gap)] = np.nan
    return filled


def build_hourly() -> None:
    fetch_raw()
    raw = pd.concat([parse_year(f"{RAW_DIR}/{y}.csv") for y in YEARS]).sort_index()
    raw = raw[~raw.index.duplicated(keep="first")]
    raw = raw.reindex(pd.date_range(START, END, freq="h"))
    for name, (low, high) in VALID_RANGES.items():
        raw.loc[(raw[name] < low) | (raw[name] > high), name] = np.nan
    angle = np.deg2rad(raw["dir"])
    hourly = pd.DataFrame({"temp": raw["temp"], "pressao": raw["pressao"], "umidade": raw["umidade"],
                           "chuva": np.log1p(raw["chuva"]), "vento_x": raw["vel"] * np.cos(angle),
                           "vento_y": raw["vel"] * np.sin(angle)})
    hourly = hourly.apply(lambda s: fill_short_gaps(s, MAX_GAP))
    hourly.index.name = "data_local"
    hourly.to_csv(CSV_PATH)


def load_hourly() -> pd.DataFrame:
    if not os.path.exists(CSV_PATH):
        build_hourly()
    return pd.read_csv(CSV_PATH, index_col=0, parse_dates=True).sort_index()


def build_frame(hourly: pd.DataFrame) -> pd.DataFrame:
    frame = hourly.copy()
    hour = frame.index.hour.to_numpy()
    day = frame.index.dayofyear.to_numpy()
    frame["hora_sin"], frame["hora_cos"] = np.sin(2 * np.pi * hour / 24), np.cos(2 * np.pi * hour / 24)
    frame["dia_sin"], frame["dia_cos"] = np.sin(2 * np.pi * day / 365.25), np.cos(2 * np.pi * day / 365.25)
    frame["target"] = frame["temp"].shift(-HORIZON)
    return frame


def split_of(dates: pd.DatetimeIndex) -> np.ndarray:
    return np.where(dates <= pd.Timestamp(TRAIN_END), 0, np.where(dates <= pd.Timestamp(VAL_END), 1, 2))


def make_windows(window: int, device: torch.device) -> dict:
    frame = build_frame(load_hourly())
    dates = frame.index
    n = len(frame)
    split = split_of(dates)
    feats = frame[FEATURES].to_numpy(dtype=np.float64)
    valid = frame[BASE_FEATURES].notna().all(axis=1).to_numpy()

    fit_rows = (split == 0) & valid
    mean, std = feats[fit_rows].mean(axis=0), feats[fit_rows].std(axis=0)
    mean[len(BASE_FEATURES):], std[len(BASE_FEATURES):] = 0.0, 1.0
    feats = np.nan_to_num((feats - mean) / std, nan=0.0)

    ends = np.arange(window - 1, n - HORIZON)
    invalid_cum = np.concatenate([[0], np.cumsum(~valid)])
    window_ok = invalid_cum[ends + 1] - invalid_cum[ends + 1 - window] == 0
    target = frame["target"].to_numpy()
    ends = ends[window_ok & ~np.isnan(target[ends]) & (split[ends] == split[ends + HORIZON])]
    which = split[ends]
    windows = sliding_window_view(feats, window, axis=0).transpose(0, 2, 1)
    y_mean, y_std = target[ends][which == 0].mean(), target[ends][which == 0].std()

    out = {"window": window, "y_mean": float(y_mean), "y_std": float(y_std), "frame": frame}
    for idx, name in enumerate(["train", "val", "test"]):
        sel = ends[which == idx]
        if name == "train":
            sel = sel[::TRAIN_STRIDE]
        out[f"x_{name}"] = torch.from_numpy(windows[sel - window + 1].astype(np.float32)).to(device)
        out[f"y_{name}"] = torch.from_numpy(((target[sel] - y_mean) / y_std).astype(np.float32)).to(device)
        out[f"meta_{name}"] = pd.DataFrame({"date": dates[sel], "target_date": dates[sel + HORIZON],
                                            "temp_now": frame["temp"].to_numpy()[sel], "target": target[sel],
                                            "index": sel})
    return out


def get_device() -> torch.device:
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")
