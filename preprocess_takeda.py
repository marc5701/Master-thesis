#!/usr/bin/env python3
"""
Takeda (TAK-994-0001) actigraphy preprocessing (aligned to Niels as much as possible + Andreas alignment):

Core preprocessing (Niels-like):
1) Load 3-axis CSV -> time-indexed df
2) Dropout (x=y=z=0) -> NaN
3) Low-pass filter cutoff = fs_out//2  (matches Niels cutoff_rate=resample_freq//2)
4) Resample to fs_out (default 30 Hz)
5) Gravity calibration (actipy) with Niels parameters
6) Non-wear flagging (actipy.flag_nonwear) -> set non-wear to NaN (keeps sampling)

Takeda-specific workflow (UPDATED per professor):
A) Compute day/night missingness using ONLY raw-quality missingness:
   dropout OR detected_nonwear (but NOT protocol masking, NOT QC exclusions)
B) DO NOT apply PSG/MWT protocol exclusion as hard NaN anymore.
   Instead:
   - compute the PSG/MWT protocol windows correctly (local days -> UTC via align_lag)
   - store them as a mask: masks/excluded_by_protocol_psg_mwt
   - store local + UTC protocol windows in meta_json for auditing
C) DO NOT hard-exclude nights/days with >50% missing; instead write masks/tables for downstream.

Time alignment (Andreas):
- Use TAK-994-0001_alignment_tab.csv which contains a per-subject shift (minutes) from UTC.
- This shift is used ONLY to:
  (1) interpret local-clock PSG times in lights_0001.csv
  (2) compute local day boundaries and convert them to UTC exclusion windows
- IMPORTANT: We DO NOT shift actigraphy timestamps; actigraphy index remains true UTC (tz-naive representing UTC).

Outputs (UPDATED to match professor's required model format):
- ONE .h5 per CSV containing the whole recording, resampled to fs_out (default 30 Hz)
- REQUIRED: data stored at "data/accelerometry" with shape (3, N), dtype float32
- REQUIRED: root attribute "start_time" = "YYYY-MM-DD HH:MM:SS" (this string represents TRUE UTC)
- No time_ns dataset is needed for the model; time is implied by start_time + 30 Hz.
- Masks stored under masks/: dropout, detected_nonwear, protocol_psg_mwt, night_missing_gt_50, day_missing_gt_50
- QC tables stored under tables/ as embedded CSV bytes
- If calibration failed (CalibOK==0): write to exclusion_dir (like Niels)
"""

from __future__ import annotations

import os
import re
import json
import argparse
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
import h5py
import actipy

# ----------------------------
# Utilities: Takeda CSV parsing
# ----------------------------

HEADER_SAMPLE_RATE_RE = re.compile(r"^#Sample Rate:\s*([0-9]+)\s*$")
HEADER_START_DATE_RE = re.compile(r"^#Start Date:\s*(.+?)\s*UTC\s*$", re.IGNORECASE)

EXPECTED_ACTIGRAPHY_HEADER = "Timestamp UTC,Accelerometer X,Accelerometer Y,Accelerometer Z"


def first_non_comment_line(path: str | Path, max_lines: int = 200) -> str | None:
    """Returns the first non-empty, non-# line (stripped) from a file, or None."""
    with open(path, "r", encoding="utf-8", errors="ignore") as f:
        for _ in range(max_lines):
            line = f.readline()
            if not line:
                break
            s = line.strip()
            if not s:
                continue
            if s.startswith("#"):
                continue
            return s
    return None


def is_actigraphy_csv(path: str | Path) -> bool:
    """True only for raw actigraphy CSVs (not the TAK-994 tables)."""
    line = first_non_comment_line(path)
    return line == EXPECTED_ACTIGRAPHY_HEADER


def read_takeda_centerpoint_csv(
    csv_path: str | Path,
    default_sample_rate: int = 32,
) -> tuple[pd.DataFrame, dict]:
    """
    Reads CenterPoint/CP3 accelerometer CSV with comment header lines.

    Returns:
      df with columns x,y,z and DatetimeIndex (UTC naive, representing UTC)
      meta dict with SampleRate and parsed header fields
    """
    csv_path = str(csv_path)
    sample_rate = None
    start_date_utc = None

    # Read header lines (lines starting with "#")
    with open(csv_path, "r", encoding="utf-8", errors="ignore") as f:
        header_lines = []
        while True:
            pos = f.tell()
            line = f.readline()
            if not line:
                break
            if line.startswith("#"):
                header_lines.append(line.strip())
            else:
                f.seek(pos)
                break

    for line in header_lines:
        m = HEADER_SAMPLE_RATE_RE.match(line)
        if m:
            sample_rate = int(m.group(1))
        m = HEADER_START_DATE_RE.match(line)
        if m:
            start_date_utc = pd.to_datetime(m.group(1), utc=True)

    inferred = False
    if sample_rate is None:
        sample_rate = int(default_sample_rate)
        inferred = True

    df = pd.read_csv(
        csv_path,
        comment="#",
        dtype={
            "Accelerometer X": "float64",
            "Accelerometer Y": "float64",
            "Accelerometer Z": "float64",
        },
    )

    required_cols = {"Timestamp UTC", "Accelerometer X", "Accelerometer Y", "Accelerometer Z"}
    if not required_cols.issubset(df.columns):
        raise ValueError(f"CSV does not look like raw actigraphy (missing required cols) in {csv_path}")

    # Parse timestamps as UTC-aware first, then drop tz to store as UTC-naive
    ts = pd.to_datetime(df["Timestamp UTC"], errors="coerce", utc=True)
    if ts.isna().any():
        raise ValueError(f"Some timestamps could not be parsed in {csv_path}")

    ts_values = ts.astype("int64").to_numpy()
    is_monotonic = np.all(np.diff(ts_values) >= 0)
    dup_frac = 1.0 - (pd.Index(ts_values).is_unique)
    rebuild = (not is_monotonic) or (dup_frac > 0.01)

    if rebuild:
        anchor = ts.iloc[0]
        dt_ns = int(round(1e9 / sample_rate))
        new_ns = anchor.value + np.arange(len(df), dtype=np.int64) * dt_ns
        ts = pd.to_datetime(new_ns, utc=True)
        rebuild_reason = {
            "rebuild_timeline": True,
            "is_monotonic_original": bool(is_monotonic),
            "duplicate_fraction_original": float(dup_frac),
            "anchor_timestamp": str(anchor),
        }
    else:
        rebuild_reason = {"rebuild_timeline": False}

    out = pd.DataFrame(
        {
            "x": df["Accelerometer X"].to_numpy(dtype="float64"),
            "y": df["Accelerometer Y"].to_numpy(dtype="float64"),
            "z": df["Accelerometer Z"].to_numpy(dtype="float64"),
        },
        index=pd.DatetimeIndex(ts).tz_convert("UTC").tz_localize(None),  # UTC-naive representing UTC
    )
    out.index.name = "time"

    meta = {
        "SampleRate": int(sample_rate),
        "SampleRate_inferred_default_32hz": bool(inferred),
        "StartDateUTC_header": str(start_date_utc) if start_date_utc is not None else None,
        **rebuild_reason,
    }
    return out, meta


# ----------------------------
# Alignment tab (Andreas)
# ----------------------------

def load_alignment_tab(alignment_csv: str | Path) -> pd.DataFrame:
    """
    Expected columns: subjid, align_lag (minutes).
    align_lag is the shift (in minutes) to add to raw UTC to obtain local-clock time:
        local = utc + align_lag
    """
    al = pd.read_csv(alignment_csv, dtype={"subjid": str})
    if "subjid" not in al.columns or "align_lag" not in al.columns:
        raise ValueError("alignment_tab must contain columns: subjid, align_lag")
    al["subjid"] = al["subjid"].astype(str).str.strip()
    al["align_lag"] = pd.to_numeric(al["align_lag"], errors="coerce")
    if al["align_lag"].isna().any():
        bad = al.loc[al["align_lag"].isna(), "subjid"].tolist()
        raise ValueError(f"alignment_tab has non-numeric align_lag for subjid(s): {bad}")
    return al


def get_align_shift_minutes(alignment_tab: pd.DataFrame, subjid: str) -> int:
    subjid = str(subjid).strip()
    row = alignment_tab[alignment_tab["subjid"] == subjid]
    if row.empty:
        warnings.warn(f"[alignment_tab] No align_lag found for subjid={subjid}; using 0 minutes.")
        return 0
    return int(round(float(row.iloc[0]["align_lag"])))


# --------------------------------------------
# Core preprocessing (aligned to Niels)
# --------------------------------------------

def preprocess_actigraphy_df(
    data_df: pd.DataFrame,
    fs_in: int,
    fs_out: int = 30,
    calib_cube: float = 0.3,          # match Niels
    nonwear_patience_min: int = 90,   # match Niels
    nonwear_window: str = "10s",      # match Niels
    nonwear_stdtol: float = 0.013,    # match Niels
    verbose: bool = True,
) -> tuple[pd.DataFrame, dict, pd.Series]:
    """
    Returns:
      df_out: processed df with NaNs for dropout + nonwear
      info: dict with diagnostics
      mask_nonwear_added: boolean Series where NaNs were introduced by flag_nonwear specifically
    """
    if not isinstance(data_df.index, pd.DatetimeIndex):
        raise ValueError("data_df must have DatetimeIndex")

    info: dict = {
        "input_fs": int(fs_in),
        "output_fs": int(fs_out),
        "lowpass_cutoff_hz": int(fs_out // 2),
        "calib_cube": float(calib_cube),
        "nonwear_patience_min": int(nonwear_patience_min),
        "nonwear_window": str(nonwear_window),
        "nonwear_stdtol": float(nonwear_stdtol),
        "calib_min_samples": 50,
        "calib_window": "10s",
        "calib_stdtol": 0.013,
        "calib_chunksize": int(1e6),
    }

    df = data_df[["x", "y", "z"]].copy()

    # Dropout: all axes == 0
    force = np.linalg.norm(df[["x", "y", "z"]].values, axis=1)
    dropout = (force == 0)
    info["dropout_points"] = int(dropout.sum())
    if info["dropout_points"] > 0:
        df.loc[dropout, ["x", "y", "z"]] = np.nan

    # Low-pass filter (Niels: cutoff_rate=resample_freq//2)
    cutoff = int(fs_out // 2)
    if verbose:
        print(f"  Low-pass filtering: fs_in={fs_in}, cutoff={cutoff} Hz")
    try:
        df, filter_info = actipy.processing.lowpass(
            data=df,
            data_sample_rate=fs_in,
            cutoff_rate=cutoff,
        )
        info.update({f"filter_{k}": v for k, v in filter_info.items()})
        info["filter_ok"] = True
    except Exception as e:
        warnings.warn(f"Lowpass failed: {e}. Proceeding without filter.")
        info["filter_ok"] = False
        info["filter_error"] = str(e)

    # Resample
    if verbose:
        print(f"  Resampling to {fs_out} Hz")
    try:
        df, resample_info = actipy.processing.resample(data=df, sample_rate=fs_out)
        info.update({f"resample_{k}": v for k, v in resample_info.items()})
        info["resample_ok"] = True
    except Exception as e:
        raise RuntimeError(f"Resample failed (required step): {e}")

    # Gravity calibration (Niels params)
    if verbose:
        print("  Gravity calibration (actipy)")
    try:
        df_cal, calib_diag = actipy.processing.calibrate_gravity(
            data=df,
            calib_cube=calib_cube,
            calib_min_samples=50,
            window="10s",
            stdtol=0.013,
            chunksize=int(1e6),
        )
        info.update(calib_diag)
        info["CalibOK"] = int(calib_diag.get("CalibOK", 0))
        if info["CalibOK"] == 1:
            df = df_cal
    except Exception as e:
        warnings.warn(f"Calibration failed: {e}. Marking CalibOK=0 and proceeding.")
        info["CalibOK"] = 0
        info["calib_error"] = str(e)

    # Snapshot BEFORE nonwear so we can identify NaNs introduced by flag_nonwear
    nan_before = df["x"].isna()

    # Non-wear flagging -> NaN
    if verbose:
        print(
            "  Non-wear flagging (actipy.flag_nonwear): "
            f"patience={nonwear_patience_min}m window={nonwear_window} stdtol={nonwear_stdtol}"
        )
    try:
        df, nonwear_info = actipy.processing.flag_nonwear(
            df,
            patience=f"{int(nonwear_patience_min)}m",
            window=str(nonwear_window),
            stdtol=float(nonwear_stdtol),
        )
        info.update(nonwear_info)
        info["nonwear_ok"] = True
    except Exception as e:
        warnings.warn(f"Non-wear flagging failed: {e}. Proceeding without nonwear.")
        info["nonwear_ok"] = False
        info["nonwear_error"] = str(e)

    nan_after = df["x"].isna()
    mask_nonwear_added = (nan_after & (~nan_before)).astype(bool)

    return df, info, mask_nonwear_added


# ----------------------------
# lights_0001 handling + PSG/MWT mask (NO HARD EXCLUSION)
# ----------------------------

def _parse_local_clock_series(series: pd.Series) -> pd.Series:
    """
    Parse timestamps as LOCAL clock times (tz-naive).
    Strip trailing 'Z' if present, then parse without utc=True.
    """
    s = series.astype(str).str.strip()
    s = s.replace({"": np.nan, "nan": np.nan, "NaN": np.nan})
    s = s.str.replace(r"Z$", "", regex=True)
    t = pd.to_datetime(s, errors="coerce")

    # If any timezone-aware values slipped through, drop tz
    try:
        if hasattr(t.dt, "tz") and t.dt.tz is not None:
            t = t.dt.tz_convert(None)
    except Exception:
        pass

    # Ensure tz-naive
    try:
        t = t.dt.tz_localize(None)
    except Exception:
        t = t.apply(lambda x: x.tz_localize(None) if hasattr(x, "tzinfo") and x.tzinfo is not None else x)

    return t


def load_lights(lights_csv: str | Path) -> pd.DataFrame:
    """Parse lights as LOCAL-clock (tz-naive), stripping trailing 'Z' if present."""
    lights = pd.read_csv(lights_csv, dtype={"subjid": str})
    if "subjid" not in lights.columns:
        raise ValueError("lights_0001.csv must contain 'subjid' column")
    lights["subjid"] = lights["subjid"].astype(str).str.strip()

    for c in ["lights_off", "lights_on", "sleep_start"]:
        if c in lights.columns:
            lights[c] = _parse_local_clock_series(lights[c])

    return lights


def psg_mwt_exclusion_windows_utc_from_lights_local_days(
    lights: pd.DataFrame,
    subjid: str,
    align_lag_min: int,
) -> tuple[list[tuple[pd.Timestamp, pd.Timestamp, str]], list[tuple[pd.Timestamp, pd.Timestamp, str]]]:
    """
    Define PSG/MWT windows (but do NOT apply them):
    - Exclude full LOCAL calendar days for PSG dates + the following MWT day
    - Convert those local-day windows to UTC using utc = local - align_lag
    Returns windows in UTC-naive (representing UTC) and local-naive (for audit).
    """
    subjid = str(subjid).strip()
    sub = lights[lights["subjid"] == subjid].copy()
    sub = sub.dropna(subset=["lights_off"]).sort_values("lights_off")
    if sub.empty:
        return [], []

    lights_off_dt = pd.to_datetime(sub["lights_off"])
    psg_dates = set(lights_off_dt.dt.date.tolist())
    first_psg_date = lights_off_dt.iloc[0].date()
    mwt_date = (pd.Timestamp(first_psg_date) + pd.Timedelta(days=1)).date()
    exclude_dates = sorted(psg_dates | {mwt_date})

    windows_local: list[tuple[pd.Timestamp, pd.Timestamp, str]] = []
    windows_utc: list[tuple[pd.Timestamp, pd.Timestamp, str]] = []

    shift = pd.Timedelta(minutes=int(align_lag_min))

    for d in exclude_dates:
        local_start = pd.Timestamp(d)  # local naive midnight
        local_end = local_start + pd.Timedelta(days=1)

        # Convert to UTC naive: utc = local - align_lag
        utc_start = local_start - shift
        utc_end = local_end - shift

        windows_local.append((local_start, local_end, "exclude_psg_or_mwt_local_day"))
        windows_utc.append((utc_start, utc_end, "exclude_psg_or_mwt_local_day_as_utc_window"))

    return windows_utc, windows_local


def build_mask_from_windows_naive(
    index: pd.DatetimeIndex,
    windows: list[tuple[pd.Timestamp, pd.Timestamp, str]],
) -> pd.Series:
    """
    Build a boolean mask over `index` for windows, without modifying data.

    Assumes `index` and `windows` are tz-naive and represent the SAME time basis.
    Here we use UTC windows against a UTC-naive index representing UTC.
    """
    mask = pd.Series(False, index=index)
    for start, end, _label in windows:
        start_naive = pd.to_datetime(start)
        end_naive = pd.to_datetime(end)
        sel = (index >= start_naive) & (index < end_naive)
        if sel.any():
            mask.loc[sel] = True
    return mask


# ----------------------------
# Day/Night missingness (QC) - compute masks/tables ONLY (do not apply)
# ----------------------------

def _build_night_day_windows_from_lights(
    lights_sub: pd.DataFrame | None,
    idx: pd.DatetimeIndex,
) -> tuple[list[dict], list[dict]]:
    """
    Build nights/days in LOCAL clock time (tz-naive) using lights when possible.
    Fallback: fixed 21:00-09:00 nights, 09:00-21:00 days in the same naive basis.
    """
    nights: list[dict] = []
    days: list[dict] = []

    if lights_sub is not None and not lights_sub.empty:
        ls = lights_sub.dropna(subset=["lights_off", "lights_on"]).sort_values("lights_off")

        for _, r in ls.iterrows():
            nights.append(
                {
                    "subj_day": r.get("subj_day", None),
                    "start": pd.to_datetime(r["lights_off"]),
                    "end": pd.to_datetime(r["lights_on"]),
                }
            )

        if len(ls) >= 2:
            for i in range(len(ls) - 1):
                a = ls.iloc[i]
                b = ls.iloc[i + 1]
                days.append(
                    {
                        "start": pd.to_datetime(a["lights_on"]),
                        "end": pd.to_datetime(b["lights_off"]),
                    }
                )
        return nights, days

    start_day = idx.min().normalize()
    end_day = idx.max().normalize()
    cur = start_day
    while cur <= end_day:
        n0 = cur + pd.Timedelta(hours=21)
        n1 = cur + pd.Timedelta(days=1, hours=9)
        nights.append({"subj_day": str(cur.date()), "start": n0, "end": n1})

        d0 = cur + pd.Timedelta(hours=9)
        d1 = cur + pd.Timedelta(hours=21)
        days.append({"start": d0, "end": d1})

        cur += pd.Timedelta(days=1)

    return nights, days


def compute_day_night_exclusion_masks_and_tables(
    df_index: pd.DatetimeIndex,
    missing_indicator: pd.Series,
    lights_sub: pd.DataFrame | None,
    night_missing_thresh: float = 0.5,
    day_missing_thresh: float = 0.5,
) -> tuple[pd.Series, pd.Series, pd.DataFrame, pd.DataFrame]:
    raise RuntimeError(
        "Do not call compute_day_night_exclusion_masks_and_tables directly. "
        "Use compute_day_night_exclusion_masks_and_tables_utc(...) in this script."
    )


def compute_day_night_exclusion_masks_and_tables_utc(
    df_index_utc: pd.DatetimeIndex,
    missing_indicator: pd.Series,
    lights_sub_local: pd.DataFrame | None,
    align_lag_min: int,
    night_missing_thresh: float = 0.5,
    day_missing_thresh: float = 0.5,
) -> tuple[pd.Series, pd.Series, pd.DataFrame, pd.DataFrame]:
    """
    Build night/day windows in LOCAL, convert to UTC using utc = local - align_lag,
    and compute missingness on the UTC index.

    Returns:
      night_mask_utc, day_mask_utc, night_table, day_table
    """
    if not isinstance(df_index_utc, pd.DatetimeIndex):
        raise ValueError("df_index_utc must be a DatetimeIndex")
    if not missing_indicator.index.equals(df_index_utc):
        raise ValueError("missing_indicator index must match df_index_utc")

    miss = missing_indicator.astype(np.float32)

    night_mask = pd.Series(False, index=df_index_utc)
    day_mask = pd.Series(False, index=df_index_utc)

    night_rows = []
    day_rows = []

    nights_local, days_local = _build_night_day_windows_from_lights(lights_sub_local, df_index_utc)
    shift = pd.Timedelta(minutes=int(align_lag_min))

    for r in nights_local:
        local_start = pd.to_datetime(r["start"])
        local_end = pd.to_datetime(r["end"])
        utc_start = local_start - shift
        utc_end = local_end - shift

        sel = (df_index_utc >= utc_start) & (df_index_utc < utc_end)
        if not sel.any():
            continue

        frac = float(miss.loc[sel].mean())
        excl = frac > night_missing_thresh
        if excl:
            night_mask.loc[sel] = True

        night_rows.append(
            {
                "subj_day": r.get("subj_day", None),
                "night_start_local": str(local_start),
                "night_end_local": str(local_end),
                "night_start_utc": str(utc_start),
                "night_end_utc": str(utc_end),
                "missing_fraction": frac,
                "excluded": bool(excl),
            }
        )

    for r in days_local:
        local_start = pd.to_datetime(r["start"])
        local_end = pd.to_datetime(r["end"])
        utc_start = local_start - shift
        utc_end = local_end - shift

        sel = (df_index_utc >= utc_start) & (df_index_utc < utc_end)
        if not sel.any():
            continue

        frac = float(miss.loc[sel].mean())
        excl = frac > day_missing_thresh
        if excl:
            day_mask.loc[sel] = True

        day_rows.append(
            {
                "day_start_local": str(local_start),
                "day_end_local": str(local_end),
                "day_start_utc": str(utc_start),
                "day_end_utc": str(utc_end),
                "missing_fraction": frac,
                "excluded": bool(excl),
            }
        )

    return night_mask, day_mask, pd.DataFrame(night_rows), pd.DataFrame(day_rows)


# ----------------------------
# HDF5 writer (whole-file) - UPDATED FORMAT
# ----------------------------

def _safe_attr(v):
    """Convert common non-HDF5-friendly types to safe scalars/strings."""
    if v is None:
        return ""
    if isinstance(v, (np.generic,)):
        return v.item()
    if isinstance(v, (dict, list, tuple)):
        return json.dumps(v, default=str)
    return v


def write_h5_whole(
    outpath: str | Path,
    df: pd.DataFrame,
    meta: dict,
    proc_info: dict,
    mask_dropout: pd.Series,
    mask_nonwear: pd.Series,
    mask_protocol_excl: pd.Series,
    mask_night_excl: pd.Series,
    mask_day_excl: pd.Series,
    night_table: pd.DataFrame,
    day_table: pd.DataFrame,
    chunk_sec: int = 600,
) -> None:
    """
    Writes H5 in the professor/model-required format:

    REQUIRED:
      - root attr: start_time = "YYYY-MM-DD HH:MM:SS" (TRUE UTC)
      - dataset:   data/accelerometry, shape=(3, N), dtype=float32
    Optional extras (kept):
      - masks/*, tables/*, JSON metadata attrs, sampling frequency, etc.
    """
    outpath = str(outpath)
    os.makedirs(os.path.dirname(outpath), exist_ok=True)

    if len(df.index) == 0:
        raise ValueError("Cannot write empty dataframe.")

    # Sampling frequency estimate (diagnostic)
    t_ns = df.index.astype("int64")
    dt = np.diff(t_ns)
    dt = dt[dt > 0]
    fs = float(round(1e9 / np.median(dt), 6)) if dt.size else np.nan

    # Chunk length along time axis (N)
    chunk_len = int(chunk_sec * fs) if np.isfinite(fs) else int(chunk_sec * 30)
    chunk_len = max(1, min(len(df), chunk_len))

    # Required start_time string (UTC-naive representing UTC -> safe)
    start_time_str = pd.Timestamp(df.index[0]).strftime("%Y-%m-%d %H:%M:%S")
    end_time_str = pd.Timestamp(df.index[-1]).strftime("%Y-%m-%d %H:%M:%S")

    # data/accelerometry (3, N)
    x = df["x"].to_numpy(dtype=np.float32)
    y = df["y"].to_numpy(dtype=np.float32)
    z = df["z"].to_numpy(dtype=np.float32)
    acc = np.stack([x, y, z], axis=0).astype(np.float32)  # (3, N)

    with h5py.File(outpath, "w") as f:
        f.create_group("data")
        f.create_group("masks")
        f.create_group("tables")

        # REQUIRED by the model
        f.attrs["start_time"] = start_time_str

        # Nice-to-have diagnostics (won't break model)
        f.attrs["end_time"] = end_time_str
        f.attrs["sample_frequency_hz"] = fs
        f.attrs["num_samples"] = int(len(df))

        # Store proc_info fields as attrs (incl CalibOK)
        for k, v in proc_info.items():
            try:
                f.attrs[k] = _safe_attr(v)
            except Exception:
                f.attrs[k] = str(v)

        # Keep JSON blobs too
        f.attrs["meta_json"] = json.dumps(meta, default=str)
        f.attrs["proc_info_json"] = json.dumps(proc_info, default=str)

        # Required accelerometry dataset
        f["data"].create_dataset(
            "accelerometry",
            data=acc,
            dtype="f4",
            chunks=(3, chunk_len),
        )

        # Masks are length N and aligned to df.index
        def _write_mask(name: str, s: pd.Series):
            s = s.reindex(df.index).fillna(False).astype(bool)
            f["masks"].create_dataset(
                name,
                data=s.to_numpy(dtype=np.uint8),
                dtype="u1",
                chunks=(chunk_len,),
            )

        _write_mask("dropout_zero_vector", mask_dropout)
        _write_mask("detected_nonwear_actipy", mask_nonwear)
        _write_mask("excluded_by_protocol_psg_mwt", mask_protocol_excl)
        _write_mask("excluded_night_missing_gt_50pct", mask_night_excl)
        _write_mask("excluded_day_missing_gt_50pct", mask_day_excl)

        # Tables (embedded CSV bytes)
        f["tables"].create_dataset(
            "night_table_csv",
            data=np.bytes_(night_table.to_csv(index=False).encode("utf-8")) if len(night_table) else np.bytes_(b""),
        )
        f["tables"].create_dataset(
            "day_table_csv",
            data=np.bytes_(day_table.to_csv(index=False).encode("utf-8")) if len(day_table) else np.bytes_(b""),
        )


# ----------------------------
# CLI pipeline
# ----------------------------

def parse_subjid_from_filename(fname: str) -> str:
    return fname.split("_")[0]


def main():
    ap = argparse.ArgumentParser()

    ap.add_argument("--input_dir", default="/oak/stanford/groups/mignot/actigraphy/TAK-994-0001")
    ap.add_argument("--lights_csv", default="/oak/stanford/groups/mignot/actigraphy/TAK-994-0001/lights_0001.csv")
    ap.add_argument("--alignment_csv", default="/oak/stanford/groups/mignot/actigraphy/TAK-994-0001/TAK-994-0001_alignment_tab.csv")
    ap.add_argument("--output_dir", default="results/takeda/preprocessed_h5")
    ap.add_argument("--exclusion_dir", default="results/takeda/excluded_h5")
    ap.add_argument("--only_subjid", default="")

    ap.add_argument("--fs_out", type=int, default=30)

    ap.add_argument("--nonwear_patience_min", type=int, default=90)
    ap.add_argument("--nonwear_window", type=str, default="10s")
    ap.add_argument("--nonwear_stdtol", type=float, default=0.013)

    ap.add_argument("--calib_cube", type=float, default=0.3)

    ap.add_argument("--night_missing_thresh", type=float, default=0.5)
    ap.add_argument("--day_missing_thresh", type=float, default=0.5)
    ap.add_argument("--chunk_sec", type=int, default=600)
    ap.add_argument("--start", type=int, default=0)
    ap.add_argument("--end", type=int, default=-1)

    args = ap.parse_args()

    Path(args.output_dir).mkdir(parents=True, exist_ok=True)
    Path(args.exclusion_dir).mkdir(parents=True, exist_ok=True)

    input_dir = Path(args.input_dir)
    output_dir = Path(args.output_dir)
    exclusion_dir = Path(args.exclusion_dir)

    lights = load_lights(args.lights_csv)
    alignment_tab = load_alignment_tab(args.alignment_csv)

    candidates = [
        p for p in input_dir.iterdir()
        if p.is_file()
        and p.suffix.lower() == ".csv"
        and not p.name.lower().endswith("lights_0001.csv")
    ]
    candidates.sort()

    files: list[Path] = [p for p in candidates if is_actigraphy_csv(p)]
    files = files[args.start:args.end] if args.end != -1 else files[args.start:]

    for p in files:
        fname = p.name
        subjid = parse_subjid_from_filename(fname).strip()
        if args.only_subjid and subjid != args.only_subjid:
            continue

        outname = fname.rsplit(".", 1)[0] + ".h5"
        outpath_ok = output_dir / outname
        outpath_excl = exclusion_dir / outname

        if outpath_ok.exists() or outpath_excl.exists():
            print(f"SKIP exists: {outname}")
            continue

        print(f"\n=== Processing {fname} (subjid={subjid}) ===")

        # --- load raw (timestamps are UTC-naive representing UTC) ---
        df_raw_utc, meta = read_takeda_centerpoint_csv(p, default_sample_rate=32)
        fs_in = int(meta["SampleRate"])

        # --- read align_lag (minutes) ---
        align_lag_min = get_align_shift_minutes(alignment_tab, subjid)
        if align_lag_min != 0:
            print(f"  Alignment: align_lag={align_lag_min} minutes (local = utc + align_lag)")
        else:
            print("  Alignment: align_lag=0 minutes")

        # Dropout mask on RAW UTC timeline (prior to preprocessing)
        raw_force = np.linalg.norm(df_raw_utc[["x", "y", "z"]].values, axis=1)
        mask_dropout_raw_utc = pd.Series((raw_force == 0), index=df_raw_utc.index)

        # Preprocess (keeps UTC index basis)
        df_proc_utc, proc_info, mask_nonwear_added = preprocess_actigraphy_df(
            data_df=df_raw_utc,
            fs_in=fs_in,
            fs_out=args.fs_out,
            calib_cube=args.calib_cube,
            nonwear_patience_min=args.nonwear_patience_min,
            nonwear_window=args.nonwear_window,
            nonwear_stdtol=args.nonwear_stdtol,
            verbose=True,
        )

        # Align dropout mask to processed timeline (nearest)
        mask_dropout_proc_utc = (
            mask_dropout_raw_utc.reindex(
                df_proc_utc.index,
                method="nearest",
                tolerance=pd.Timedelta(milliseconds=1000),
            )
            .fillna(False)
            .astype(bool)
        )

        # Explicit nonwear mask (NaNs introduced by flag_nonwear specifically)
        mask_nonwear_proc_utc = mask_nonwear_added.astype(bool)

        # STEP A (QC): missingness indicator = dropout OR detected_nonwear
        lights_sub_local = lights[lights["subjid"] == subjid].copy()
        missing_indicator_qc = (mask_dropout_proc_utc | mask_nonwear_proc_utc).astype(bool)

        # Compute night/day missing masks/tables correctly by converting local windows -> UTC
        night_excl, day_excl, night_table, day_table = compute_day_night_exclusion_masks_and_tables_utc(
            df_index_utc=df_proc_utc.index,
            missing_indicator=missing_indicator_qc,
            lights_sub_local=lights_sub_local if not lights_sub_local.empty else None,
            align_lag_min=align_lag_min,
            night_missing_thresh=args.night_missing_thresh,
            day_missing_thresh=args.day_missing_thresh,
        )

        # STEP B (Protocol): compute PSG/MWT windows and store as mask ONLY (no NaN-ing)
        if lights_sub_local.empty:
            print(f"  NOTE: subjid {subjid} not found in lights_0001 -> PSG/MWT protocol mask empty.")
            windows_utc = []
            windows_local = []
            mask_protocol = pd.Series(False, index=df_proc_utc.index)
        else:
            windows_utc, windows_local = psg_mwt_exclusion_windows_utc_from_lights_local_days(
                lights=lights,
                subjid=subjid,
                align_lag_min=align_lag_min,
            )
            mask_protocol = build_mask_from_windows_naive(df_proc_utc.index, windows_utc)

        meta_out = {
            **meta,
            "input_file": str(p),
            "subjid": subjid,
            "lights_csv": str(args.lights_csv),
            "alignment_csv": str(args.alignment_csv),
            "align_lag_minutes": int(align_lag_min),
            "psg_mwt_windows_local_days": [
                {"start": str(s), "end": str(e), "label": lab} for (s, e, lab) in windows_local
            ],
            "psg_mwt_windows_utc": [
                {"start": str(s), "end": str(e), "label": lab} for (s, e, lab) in windows_utc
            ],
        }

        # Exclude if calibration failed (match Niels behavior)
        final_outpath = outpath_ok
        if int(proc_info.get("CalibOK", 0)) == 0:
            final_outpath = outpath_excl

        tmp_outpath = str(final_outpath) + ".tmp"
        if os.path.exists(tmp_outpath):
            os.remove(tmp_outpath)

        write_h5_whole(
            outpath=tmp_outpath,
            df=df_proc_utc,
            meta=meta_out,
            proc_info=proc_info,
            mask_dropout=mask_dropout_proc_utc,
            mask_nonwear=mask_nonwear_proc_utc,
            mask_protocol_excl=mask_protocol.astype(bool),
            mask_night_excl=night_excl.astype(bool),
            mask_day_excl=day_excl.astype(bool),
            night_table=night_table,
            day_table=day_table,
            chunk_sec=args.chunk_sec,
        )

        os.replace(tmp_outpath, final_outpath)
        print(f"WROTE: {final_outpath}")


if __name__ == "__main__":
    main()