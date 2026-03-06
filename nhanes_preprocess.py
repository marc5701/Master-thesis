#!/usr/bin/env python3
"""
NHANES 2013–2014 PAX80_H preprocessing 
=======================================================================================

Inputs:
- tar_dir contains per-participant archives: SEQN.tar.bz2
  Each tar contains:
    - many hourly accelerometer files:
        GT3XPLUS-AccelerationCalibrated-<fw>.<sensor_id>.<YYYY>-<MM>-<DD>-<HH>-<MM>-<SS>-000-P0000.sensor.csv[.gz]
      Columns: HEADER_TIMESTAMP, X, Y, Z  (80 Hz; timestamps appear as alternating 12/13ms steps but span the full hour)
    - one QC log file: [SEQN]_Logs.csv
      Columns: DAY_OF_DATA (1–9), START_TIME, END_TIME, DATA_QUALITY_FLAG_CODE, DATA_QUALITY_FLAG_VALUE

Processing per participant:
1) Read all hourly files from the tar; sort by the timestamp in the filename; concatenate -> one raw timeline.
2) Build NHANES QC suspect mask from SEQN_Logs.csv and APPLY as NaN to x/y/z (hard-applied; no deletions).
3) Dropout: x=y=z=0 on raw grid -> NaN.
4) Low-pass filter: cutoff = min(fs_out/2 - 0.1, fs_in/2 - 0.1).
5) Resample to fs_out (default 30 Hz) uniform grid (actipy.processing.resample).
6) Enforce any-axis-NaN => all-axes-NaN.
7) Gravity calibration (Niels params).
8) Nonwear (actipy.processing.flag_nonwear) with Niels/Takeda params.
   Build "detected_nonwear_actipy" mask = new NaNs introduced by nonwear step.
9) Align masks (dropout / qc_removed) to processed timeline (nearest with tolerance).
10) Night/day QC flags ONLY (do not apply):
    missing_indicator = dropout_proc OR nonwear_added OR qc_removed_proc
    fixed windows: night 21:00-09:00, day 09:00-21:00
11) Write one H5:
    - root attr: start_time = first processed timestamp "YYYY-MM-DD HH:MM:SS"
    - data/accelerometry float32 shape (3, N)
    - masks/* uint8 length N:
        dropout_zero_vector
        detected_nonwear_actipy
        qc_removed_nhanes
        excluded_by_protocol_psg_mwt  (always False)
        excluded_night_missing_gt_50pct
        excluded_day_missing_gt_50pct
    - tables/night_table_csv and tables/day_table_csv as CSV bytes
    - proc_info attrs + meta_json + proc_info_json

NEW: run report (CSV + XLSX)
- For every SEQN, the script writes a per-participant JSON row in:
    <report_dir>/rows/<SEQN>.json
  so this is safe even under SLURM arrays (no shared-file append race).
- At the end of a run, the script can consolidate rows into:
    <report_dir>/nhanes_preproc_report.csv
    <report_dir>/nhanes_preproc_report.xlsx
  (Consolidation is automatically skipped inside SLURM array tasks unless you force it.)

Important note about "real UTC":
- NHANES dates are disclosure-shifted into Jan 2000. This script preserves timestamps as-is (tz-naive).
  The produced start_time is consistent with the implicit time axis (start_time + 30 Hz),
  but it is not the participant's real calendar date.
"""

from __future__ import annotations

import os
import re
import json
import gzip
import tarfile
import argparse
import warnings
from pathlib import Path
from datetime import datetime
from typing import Any

import numpy as np
import pandas as pd
import h5py
import actipy


# -------------------------
# Small utilities
# -------------------------

def _safe_json(obj) -> str:
    return json.dumps(obj, default=str)

def _safe_attr(v):
    if v is None:
        return ""
    if isinstance(v, (np.generic,)):
        return v.item()
    if isinstance(v, (dict, list, tuple)):
        return json.dumps(v, default=str)
    return v

def _ensure_naive_index(idx: pd.DatetimeIndex) -> pd.DatetimeIndex:
    """If tz-aware, convert to UTC then drop tz. If tz-naive, keep as-is."""
    if not isinstance(idx, pd.DatetimeIndex):
        raise ValueError("Index must be a DatetimeIndex")
    if getattr(idx, "tz", None) is not None:
        return idx.tz_convert("UTC").tz_localize(None)
    return idx

def maybe_rebuild_timeline(ts: pd.DatetimeIndex, n: int, fs_hz: float) -> tuple[pd.DatetimeIndex, dict]:
    """
    If non-monotonic or duplicate fraction >1%, rebuild to uniform grid anchored at first timestamp.
    """
    ts = pd.to_datetime(ts, errors="coerce")
    if ts.isna().any():
        raise ValueError("Timestamp parsing produced NaNs; cannot proceed.")

    ns = ts.astype("int64").to_numpy()
    is_monotonic = bool(np.all(np.diff(ns) >= 0))
    dup_frac = float(1.0 - pd.Index(ns).is_unique)

    rebuild = (not is_monotonic) or (dup_frac > 0.01)
    info = {
        "rebuild_timeline": bool(rebuild),
        "is_monotonic_original": bool(is_monotonic),
        "duplicate_fraction_original": float(dup_frac),
    }
    if not rebuild:
        return pd.DatetimeIndex(ts), info

    anchor = ts[0]
    dt_ns = int(round(1e9 / float(fs_hz)))
    new_ns = anchor.value + np.arange(n, dtype=np.int64) * dt_ns
    info["anchor_timestamp"] = str(anchor)
    info["rebuild_dt_ns"] = int(dt_ns)
    return pd.to_datetime(new_ns), info


# -------------------------
# Reporting (safe under parallel execution)
# -------------------------

def _is_slurm_array_task() -> bool:
    return ("SLURM_ARRAY_TASK_ID" in os.environ) or ("SLURM_ARRAY_JOB_ID" in os.environ)

def _write_report_row_json(report_dir: str | Path, seqn: str, row: dict[str, Any]) -> None:
    report_dir = Path(report_dir)
    rows_dir = report_dir / "rows"
    rows_dir.mkdir(parents=True, exist_ok=True)
    tmp = rows_dir / f"{seqn}.json.tmp"
    out = rows_dir / f"{seqn}.json"
    tmp.write_text(json.dumps(row, indent=2, default=str))
    tmp.replace(out)

def _consolidate_report(report_dir: str | Path, out_csv: str | Path | None = None, out_xlsx: str | Path | None = None) -> pd.DataFrame:
    report_dir = Path(report_dir)
    rows_dir = report_dir / "rows"
    rows = []
    if rows_dir.exists():
        for p in sorted(rows_dir.glob("*.json")):
            try:
                rows.append(json.loads(p.read_text()))
            except Exception:
                continue
    df = pd.DataFrame(rows)

    if out_csv is None:
        out_csv = report_dir / "nhanes_preproc_report.csv"
    if out_xlsx is None:
        out_xlsx = report_dir / "nhanes_preproc_report.xlsx"

    # Write CSV always
    df.to_csv(out_csv, index=False)

    # Write XLSX if openpyxl is present (it is in your environment)
    try:
        df.to_excel(out_xlsx, index=False, engine="openpyxl")
    except Exception as e:
        warnings.warn(f"Could not write XLSX report ({out_xlsx}): {e}")

    return df


# -------------------------
# NHANES tar member discovery + sorting
# -------------------------

# Matches: ...<sensor_id>.<YYYY>-<MM>-<DD>-<HH>-<MM>-<SS>-000-P0000.sensor.csv[.gz]
ACC_NAME_RE = re.compile(
    r"^GT3XPLUS-AccelerationCalibrated-.*\.(\d{4}-\d{2}-\d{2}-\d{2}-\d{2}-\d{2})-000-P0000\.sensor\.csv(\.gz)?$",
    re.IGNORECASE,
)
LOG_NAME_RE = re.compile(r"^\d+_Logs\.csv$", re.IGNORECASE)

def _parse_acc_start_from_name(basename: str) -> datetime | None:
    m = ACC_NAME_RE.match(basename)
    if not m:
        return None
    return datetime.strptime(m.group(1), "%Y-%m-%d-%H-%M-%S")

def list_tar_members_sorted(tar_path: str | Path) -> tuple[list[tarfile.TarInfo], tarfile.TarInfo | None, dict]:
    """
    Return:
      acc_members_sorted (by timestamp from filename),
      log_member (or None),
      info dict about parsing.
    """
    tar_path = str(tar_path)

    acc_members: list[tuple[datetime, tarfile.TarInfo]] = []
    log_member: tarfile.TarInfo | None = None
    bad_csv_like: list[str] = []

    with tarfile.open(tar_path, mode="r:bz2") as tf:
        for m in tf.getmembers():
            if not m.isfile():
                continue
            b = os.path.basename(m.name)

            if LOG_NAME_RE.match(b):
                log_member = m
                continue

            t0 = _parse_acc_start_from_name(b)
            if t0 is not None:
                acc_members.append((t0, m))
            else:
                # Keep preview of other csv-like files that aren't sensor files
                if b.lower().endswith((".csv", ".csv.gz")) and ("log" not in b.lower()):
                    bad_csv_like.append(b)

    acc_members.sort(key=lambda x: x[0])
    acc_sorted = [m for _, m in acc_members]

    info = {
        "n_acc_members": int(len(acc_sorted)),
        "n_bad_csv_like_names": int(len(bad_csv_like)),
        "bad_csv_like_names_preview": bad_csv_like[:20],
        "log_member_name": os.path.basename(log_member.name) if log_member else "",
    }
    return acc_sorted, log_member, info


# -------------------------
# Read members
# -------------------------

def _open_member_stream(tf: tarfile.TarFile, member: tarfile.TarInfo):
    fobj = tf.extractfile(member)
    if fobj is None:
        raise IOError(f"Failed to extract {member.name}")
    name = os.path.basename(member.name).lower()
    if name.endswith(".gz"):
        return gzip.GzipFile(fileobj=fobj)  # bytes stream
    return fobj  # bytes stream

def read_acc_member(tf: tarfile.TarFile, member: tarfile.TarInfo) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Reads one hourly accelerometer file.

    Returns:
      t_ns (int64), x (float32), y (float32), z (float32)
    """
    stream = _open_member_stream(tf, member)

    # Robust delimiter handling: allow CSV or TSV via sep=None sniffing
    df = pd.read_csv(
        stream,
        engine="python",
        sep=None,
        usecols=["HEADER_TIMESTAMP", "X", "Y", "Z"],
        dtype={"X": "float32", "Y": "float32", "Z": "float32"},
    )

    t = pd.to_datetime(df["HEADER_TIMESTAMP"], errors="coerce", format="%Y-%m-%d %H:%M:%S.%f")
    if t.isna().any():
        t = pd.to_datetime(df["HEADER_TIMESTAMP"], errors="coerce")
    if t.isna().any():
        raise ValueError(f"Timestamp parsing failed in {member.name} (NaT present).")

    t_ns = t.astype("int64").to_numpy(dtype=np.int64)
    x = df["X"].to_numpy(dtype=np.float32)
    y = df["Y"].to_numpy(dtype=np.float32)
    z = df["Z"].to_numpy(dtype=np.float32)
    return t_ns, x, y, z

def read_logs_member(tf: tarfile.TarFile, member: tarfile.TarInfo) -> pd.DataFrame:
    stream = _open_member_stream(tf, member)
    df = pd.read_csv(stream, engine="python")
    df.columns = [c.strip() for c in df.columns]
    return df


# -------------------------
# NHANES QC mask construction
# -------------------------

def build_nhanes_qc_mask_raw(
    t_index: pd.DatetimeIndex,
    logs: pd.DataFrame | None,
) -> tuple[pd.Series, dict]:
    """
    Build per-sample boolean mask at RAW timeline for NHANES QC suspect intervals.

    Mapping strategy:
    - base_date = normalize(min timestamp)
    - for each logs row with DATA_QUALITY_FLAG_CODE present:
        date = base_date + (DAY_OF_DATA-1) days
        start_dt = date + START_TIME
        end_dt   = date + END_TIME
        if end_dt <= start_dt => end_dt += 1 day
    """
    mask = pd.Series(False, index=t_index)
    qc_meta = {
        "qc_rows_total": 0,
        "qc_rows_used": 0,
        "qc_total_flagged_samples": 0,
        "qc_unique_codes": [],
        "qc_parse_errors": 0,
    }

    if logs is None or logs.empty:
        return mask, qc_meta

    cols = {c.upper(): c for c in logs.columns}
    needed = ["DAY_OF_DATA", "START_TIME", "END_TIME", "DATA_QUALITY_FLAG_CODE"]
    if not all(k in cols for k in needed):
        qc_meta["qc_error"] = f"Logs missing required columns. Found: {list(logs.columns)}"
        return mask, qc_meta

    day_col = cols["DAY_OF_DATA"]
    st_col = cols["START_TIME"]
    en_col = cols["END_TIME"]
    code_col = cols["DATA_QUALITY_FLAG_CODE"]

    qc_meta["qc_rows_total"] = int(len(logs))
    base_date = t_index.min().normalize()

    sub = logs.copy()
    sub[code_col] = sub[code_col].astype(str).str.strip()
    sub = sub[(sub[code_col] != "") & (sub[code_col].str.lower() != "nan")]

    sub[day_col] = pd.to_numeric(sub[day_col], errors="coerce")
    sub = sub[sub[day_col].notna()]

    if sub.empty:
        return mask, qc_meta

    qc_meta["qc_rows_used"] = int(len(sub))
    qc_meta["qc_unique_codes"] = sorted(sub[code_col].unique().tolist())[:2000]

    if not t_index.is_monotonic_increasing:
        raise ValueError("t_index must be sorted before building QC mask.")

    t_ns = t_index.astype("int64").to_numpy()

    for _, r in sub.iterrows():
        try:
            day = int(r[day_col])
            if day < 1:
                continue

            st = str(r[st_col]).strip()
            en = str(r[en_col]).strip()
            if not st or st.lower() == "nan" or not en or en.lower() == "nan":
                continue

            d = base_date + pd.Timedelta(days=day - 1)
            start_dt = pd.to_datetime(f"{d.date()} {st}", errors="coerce")
            end_dt = pd.to_datetime(f"{d.date()} {en}", errors="coerce")
            if pd.isna(start_dt) or pd.isna(end_dt):
                qc_meta["qc_parse_errors"] += 1
                continue

            if end_dt <= start_dt:
                end_dt = end_dt + pd.Timedelta(days=1)

            i0 = int(np.searchsorted(t_ns, start_dt.value, side="left"))
            i1 = int(np.searchsorted(t_ns, end_dt.value, side="left"))

            if i1 > i0:
                mask.iloc[i0:i1] = True
        except Exception:
            qc_meta["qc_parse_errors"] += 1
            continue

    qc_meta["qc_total_flagged_samples"] = int(mask.sum())
    return mask.astype(bool), qc_meta


# -------------------------
# Core preprocessing (same as Montpellier/Bologna)
# -------------------------

def preprocess_actigraphy_df(
    data_df: pd.DataFrame,
    fs_in: int,
    fs_out: int = 30,
    calib_cube: float = 0.3,
    nonwear_patience_min: int = 90,
    nonwear_window: str = "10s",
    nonwear_stdtol: float = 0.013,
    verbose: bool = True,
) -> tuple[pd.DataFrame, dict, pd.Series, pd.Series]:
    """
    Returns:
      df_out: processed with NaNs (dropout + nonwear + whatever NaNs were already present)
      proc_info: diagnostics (incl CalibOK)
      mask_dropout_proc: aligned to processed index
      mask_nonwear_added: new NaNs introduced by nonwear step
    """
    if not isinstance(data_df.index, pd.DatetimeIndex):
        raise ValueError("data_df must have DatetimeIndex")

    proc_info: dict = {
        "input_fs": int(fs_in),
        "output_fs": int(fs_out),
        "lowpass_cutoff_hz_requested": float(fs_out) / 2.0 - 0.1,
        "lowpass_cutoff_hz_used": None,
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

    # Dropout on raw grid: exact zero-vector (only where all finite)
    xyz = df[["x", "y", "z"]].to_numpy()
    finite = np.isfinite(xyz).all(axis=1)
    dropout_raw = np.zeros(len(df), dtype=bool)
    dropout_raw[finite] = np.all(xyz[finite] == 0, axis=1)

    proc_info["dropout_points"] = int(dropout_raw.sum())
    if proc_info["dropout_points"] > 0:
        df.loc[dropout_raw, ["x", "y", "z"]] = np.nan

    # Low-pass cutoff safe w.r.t input Nyquist
    cutoff_req = float(fs_out) / 2.0 - 0.1
    cutoff_max = float(fs_in) / 2.0 - 0.1
    cutoff_used = min(cutoff_req, cutoff_max)
    proc_info["lowpass_cutoff_hz_used"] = float(cutoff_used)

    if verbose:
        print(f"  Low-pass filtering: fs_in={fs_in}, cutoff_used={cutoff_used:.3f} Hz (req={cutoff_req:.3f})")
    try:
        df, filter_info = actipy.processing.lowpass(
            data=df,
            data_sample_rate=fs_in,
            cutoff_rate=cutoff_used,
        )
        proc_info.update({f"filter_{k}": v for k, v in filter_info.items()})
        proc_info["filter_ok"] = True
    except Exception as e:
        warnings.warn(f"Lowpass failed: {e}. Proceeding without filter.")
        proc_info["filter_ok"] = False
        proc_info["filter_error"] = str(e)

    # Resample -> uniform grid at fs_out
    if verbose:
        print(f"  Resampling to {fs_out} Hz")
    df, resample_info = actipy.processing.resample(data=df, sample_rate=fs_out)
    proc_info.update({f"resample_{k}": v for k, v in resample_info.items()})
    proc_info["resample_ok"] = True

    # Enforce any-axis-NaN => all-axes-NaN
    any_nan = df[["x", "y", "z"]].isna().any(axis=1)
    if bool(any_nan.any()):
        df.loc[any_nan, ["x", "y", "z"]] = np.nan

    # Align dropout mask to processed timeline (nearest)
    dropout_raw_ser = pd.Series(dropout_raw.astype(bool), index=data_df.index)
    mask_dropout_proc = (
        dropout_raw_ser.reindex(df.index, method="nearest", tolerance=pd.Timedelta(seconds=1))
        .fillna(False)
        .astype(bool)
    )

    # Gravity calibration
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
        proc_info.update(calib_diag)
        proc_info["CalibOK"] = int(calib_diag.get("CalibOK", 0))
        if proc_info["CalibOK"] == 1:
            df = df_cal
    except Exception as e:
        warnings.warn(f"Calibration failed: {e}. Marking CalibOK=0 and proceeding.")
        proc_info["CalibOK"] = 0
        proc_info["calib_error"] = str(e)

    # Snapshot BEFORE nonwear
    nan_before = df["x"].isna()

    # Nonwear -> NaN
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
        proc_info.update(nonwear_info)
        proc_info["nonwear_ok"] = True
    except Exception as e:
        warnings.warn(f"Non-wear flagging failed: {e}. Proceeding without nonwear.")
        proc_info["nonwear_ok"] = False
        proc_info["nonwear_error"] = str(e)

    # Enforce any-axis-NaN => all-axes-NaN
    any_nan2 = df[["x", "y", "z"]].isna().any(axis=1)
    if bool(any_nan2.any()):
        df.loc[any_nan2, ["x", "y", "z"]] = np.nan

    nan_after = df["x"].isna()
    mask_nonwear_added = (nan_after & (~nan_before)).astype(bool)

    return df, proc_info, mask_dropout_proc, mask_nonwear_added


# -------------------------
# Night/Day QC flags ONLY (fixed windows)
# -------------------------

def compute_fixed_night_day_qc_tables_and_masks(
    idx: pd.DatetimeIndex,
    missing_indicator: pd.Series,
    night_start_hour: int = 21,
    night_end_hour: int = 9,
    missing_thresh: float = 0.5,
) -> tuple[pd.Series, pd.Series, pd.DataFrame, pd.DataFrame]:
    if not missing_indicator.index.equals(idx):
        raise ValueError("missing_indicator must share the same index as idx")

    miss = missing_indicator.astype(np.float32)

    night_mask = pd.Series(False, index=idx)
    day_mask = pd.Series(False, index=idx)
    night_rows = []
    day_rows = []

    start_day = idx.min().normalize()
    end_day = idx.max().normalize()

    cur = start_day
    while cur <= end_day:
        w0 = cur + pd.Timedelta(hours=night_start_hour)
        w1 = cur + pd.Timedelta(days=1, hours=night_end_hour)
        sel = (idx >= w0) & (idx < w1)
        if sel.any():
            frac = float(miss.loc[sel].mean())
            bad = frac > missing_thresh
            if bad:
                night_mask.loc[sel] = True
            night_rows.append(
                {"night_start": str(w0), "night_end": str(w1), "missing_fraction": frac, "excluded": bool(bad)}
            )
        cur += pd.Timedelta(days=1)

    cur = start_day
    while cur <= end_day:
        w0 = cur + pd.Timedelta(hours=night_end_hour)
        w1 = cur + pd.Timedelta(hours=night_start_hour)
        sel = (idx >= w0) & (idx < w1)
        if sel.any():
            frac = float(miss.loc[sel].mean())
            bad = frac > missing_thresh
            if bad:
                day_mask.loc[sel] = True
            day_rows.append(
                {"day_start": str(w0), "day_end": str(w1), "missing_fraction": frac, "excluded": bool(bad)}
            )
        cur += pd.Timedelta(days=1)

    return night_mask, day_mask, pd.DataFrame(night_rows), pd.DataFrame(day_rows)


# -------------------------
# HDF5 writer — REQUIRED FORMAT
# -------------------------

def write_h5_whole(
    outpath: str,
    df: pd.DataFrame,
    meta: dict,
    proc_info: dict,
    mask_dropout: pd.Series,
    mask_nonwear: pd.Series,
    mask_qc_removed: pd.Series,
    mask_protocol_excl: pd.Series,
    mask_night_excl: pd.Series,
    mask_day_excl: pd.Series,
    night_table: pd.DataFrame,
    day_table: pd.DataFrame,
    chunk_sec: int = 600,
) -> None:
    os.makedirs(os.path.dirname(outpath), exist_ok=True)

    if not isinstance(df.index, pd.DatetimeIndex):
        raise ValueError("df must have a DatetimeIndex")

    idx = _ensure_naive_index(df.index)
    acc = df[["x", "y", "z"]].to_numpy(dtype=np.float32).T  # (3, N)
    n = acc.shape[1]

    fs = float(proc_info.get("output_fs", 30))
    if not np.isfinite(fs) or fs <= 0:
        fs = 30.0
    chunk_len = int(round(chunk_sec * fs))
    chunk_len = max(1, min(n, chunk_len))

    def _check_mask(name: str, m: pd.Series):
        if not isinstance(m, pd.Series):
            raise TypeError(f"Mask '{name}' must be a pandas Series")
        if not m.index.equals(df.index):
            raise ValueError(f"Mask '{name}' index does not match df.index")
        if len(m) != n:
            raise ValueError(f"Mask '{name}' length {len(m)} != N {n}")

    _check_mask("dropout_zero_vector", mask_dropout)
    _check_mask("detected_nonwear_actipy", mask_nonwear)
    _check_mask("qc_removed_nhanes", mask_qc_removed)
    _check_mask("excluded_by_protocol_psg_mwt", mask_protocol_excl)
    _check_mask("excluded_night_missing_gt_50pct", mask_night_excl)
    _check_mask("excluded_day_missing_gt_50pct", mask_day_excl)

    with h5py.File(outpath, "w") as f:
        f.create_group("annotations")
        f.create_group("data")
        f.create_group("masks")
        f.create_group("tables")

        f.attrs["start_time"] = idx[0].strftime("%Y-%m-%d %H:%M:%S") if len(idx) else ""

        for k, v in proc_info.items():
            try:
                f.attrs[k] = _safe_attr(v)
            except Exception:
                f.attrs[k] = str(v)

        f.attrs["meta_json"] = _safe_json(meta)
        f.attrs["proc_info_json"] = _safe_json(proc_info)

        f["data"].create_dataset("accelerometry", data=acc, dtype="f4", chunks=(3, chunk_len))

        def _write_mask(name: str, s: pd.Series):
            f["masks"].create_dataset(
                name,
                data=s.to_numpy(dtype=np.uint8),
                dtype="u1",
                chunks=(chunk_len,),
            )

        _write_mask("dropout_zero_vector", mask_dropout.astype(bool))
        _write_mask("detected_nonwear_actipy", mask_nonwear.astype(bool))
        _write_mask("qc_removed_nhanes", mask_qc_removed.astype(bool))
        _write_mask("excluded_by_protocol_psg_mwt", mask_protocol_excl.astype(bool))
        _write_mask("excluded_night_missing_gt_50pct", mask_night_excl.astype(bool))
        _write_mask("excluded_day_missing_gt_50pct", mask_day_excl.astype(bool))

        f["tables"].create_dataset(
            "night_table_csv",
            data=np.bytes_(night_table.to_csv(index=False).encode("utf-8")) if len(night_table) else np.bytes_(b""),
        )
        f["tables"].create_dataset(
            "day_table_csv",
            data=np.bytes_(day_table.to_csv(index=False).encode("utf-8")) if len(day_table) else np.bytes_(b""),
        )


# -------------------------
# Per-participant pipeline
# -------------------------

def process_one_seqn(
    tar_path: str,
    out_dir: str,
    exclusion_dir: str,
    fs_in: int,
    fs_out: int,
    nonwear_patience_min: int,
    nonwear_window: str,
    nonwear_stdtol: float,
    night_missing_thresh: float,
    day_missing_thresh: float,
    night_start_hour: int,
    night_end_hour: int,
    chunk_sec: int,
    verbose: bool,
    report_dir: str,
    overwrite: bool = False,
) -> dict[str, Any]:
    """
    Processes one SEQN tar.

    Returns a report row dict (always), and also writes <report_dir>/rows/<SEQN>.json.
    """
    tar_path = str(tar_path)
    seqn = Path(tar_path).name.split(".", 1)[0]

    outname = f"{seqn}.h5"
    out_ok = os.path.join(out_dir, outname)
    out_excl = os.path.join(exclusion_dir, outname)

    row: dict[str, Any] = {
        "seqn": str(seqn),
        "input_tar": tar_path,
        "status": "",
        "reason": "",
        "output_path": "",
        "calib_ok": "",
        "error": "",
        "n_hourly_files": "",
        "logs_present": "",
        "qc_rows_total": "",
        "qc_rows_used": "",
        "qc_parse_errors": "",
        "qc_unique_codes": "",
        "qc_flagged_samples_raw": "",
        "dropout_points_raw": "",
        "nonwear_ok": "",
        "filter_ok": "",
        "resample_ok": "",
        "timeline_rebuilt": "",
        "start_time": "",
        "n_samples_30hz": "",
        "duration_days_30hz": "",
        "run_timestamp_utc": datetime.utcnow().isoformat(timespec="seconds") + "Z",
    }

    try:
        if (os.path.exists(out_ok) or os.path.exists(out_excl)) and not overwrite:
            row["status"] = "SKIPPED_EXISTS"
            row["reason"] = "output_exists"
            row["output_path"] = out_ok if os.path.exists(out_ok) else out_excl
            _write_report_row_json(report_dir, seqn, row)
            if verbose:
                print(f"SKIP exists: {outname}")
            return row

        acc_members, log_member, member_info = list_tar_members_sorted(tar_path)
        row["n_hourly_files"] = int(len(acc_members))
        row["logs_present"] = bool(log_member is not None)

        if not acc_members:
            row["status"] = "FAILED"
            row["reason"] = "no_acc_files_in_tar"
            row["error"] = "No accelerometer files matched the expected NHANES sensor naming."
            _write_report_row_json(report_dir, seqn, row)
            if verbose:
                print(f"[WARN] SEQN={seqn}: no accelerometer files found in tar.")
                if member_info.get("n_bad_csv_like_names", 0) > 0:
                    print("  CSV-like files that did not match sensor naming (preview):")
                    for b in member_info["bad_csv_like_names_preview"]:
                        print("   ", b)
            return row

        if verbose:
            print(f"\n=== SEQN={seqn} ===")
            print(f"  tar: {tar_path}")
            print(f"  acc files: {len(acc_members)}")
            print(f"  logs: {os.path.basename(log_member.name) if log_member else 'NONE'}")

        t_list: list[np.ndarray] = []
        x_list: list[np.ndarray] = []
        y_list: list[np.ndarray] = []
        z_list: list[np.ndarray] = []
        logs_df: pd.DataFrame | None = None

        with tarfile.open(tar_path, mode="r:bz2") as tf:
            if log_member is not None:
                try:
                    logs_df = read_logs_member(tf, log_member)
                except Exception as e:
                    warnings.warn(f"{seqn}: failed to read logs file: {e}")
                    logs_df = None

            for m in acc_members:
                t_ns, x, y, z = read_acc_member(tf, m)
                t_list.append(t_ns)
                x_list.append(x)
                y_list.append(y)
                z_list.append(z)

        t_ns_all = np.concatenate(t_list)
        x_all = np.concatenate(x_list)
        y_all = np.concatenate(y_list)
        z_all = np.concatenate(z_list)

        # Sort by timestamp (defensive)
        order = np.argsort(t_ns_all, kind="mergesort")
        t_ns_all = t_ns_all[order]
        x_all = x_all[order]
        y_all = y_all[order]
        z_all = z_all[order]

        t_idx = pd.to_datetime(t_ns_all)
        t_idx2, rebuild_info = maybe_rebuild_timeline(pd.DatetimeIndex(t_idx), n=len(t_idx), fs_hz=float(fs_in))
        t_idx = pd.DatetimeIndex(t_idx2)
        row["timeline_rebuilt"] = bool(rebuild_info.get("rebuild_timeline", False))

        df_raw = pd.DataFrame(
            {
                "x": x_all.astype(np.float64, copy=False),
                "y": y_all.astype(np.float64, copy=False),
                "z": z_all.astype(np.float64, copy=False),
            },
            index=t_idx,
        )
        df_raw.index.name = "time"
        if not df_raw.index.is_monotonic_increasing:
            df_raw = df_raw.sort_index()

        # Build NHANES QC suspect mask on raw timeline and APPLY -> NaN
        mask_qc_raw, qc_meta = build_nhanes_qc_mask_raw(df_raw.index, logs_df)
        n_qc = int(mask_qc_raw.sum())
        row["qc_flagged_samples_raw"] = int(n_qc)
        row["qc_rows_total"] = int(qc_meta.get("qc_rows_total", 0))
        row["qc_rows_used"] = int(qc_meta.get("qc_rows_used", 0))
        row["qc_parse_errors"] = int(qc_meta.get("qc_parse_errors", 0))
        row["qc_unique_codes"] = "|".join(qc_meta.get("qc_unique_codes", [])[:50])

        if n_qc > 0:
            df_raw.loc[mask_qc_raw, ["x", "y", "z"]] = np.nan

        # Core preprocess
        df_proc, proc_info, mask_dropout_proc, mask_nonwear_added = preprocess_actigraphy_df(
            data_df=df_raw,
            fs_in=int(fs_in),
            fs_out=int(fs_out),
            calib_cube=0.3,
            nonwear_patience_min=int(nonwear_patience_min),
            nonwear_window=str(nonwear_window),
            nonwear_stdtol=float(nonwear_stdtol),
            verbose=verbose,
        )

        # Align QC mask to processed timeline (nearest)
        mask_qc_proc = (
            mask_qc_raw.reindex(df_proc.index, method="nearest", tolerance=pd.Timedelta(seconds=1))
            .fillna(False)
            .astype(bool)
        )

        # Missingness indicator for QC flags (flags only; NOT applied)
        missing_indicator_qc = (mask_dropout_proc | mask_nonwear_added | mask_qc_proc).astype(bool)

        night_excl, day_excl, night_table, day_table = compute_fixed_night_day_qc_tables_and_masks(
            idx=df_proc.index,
            missing_indicator=missing_indicator_qc,
            night_start_hour=int(night_start_hour),
            night_end_hour=int(night_end_hour),
            missing_thresh=float(night_missing_thresh),
        )

        if float(day_missing_thresh) != float(night_missing_thresh):
            _, day_excl2, _, day_table2 = compute_fixed_night_day_qc_tables_and_masks(
                idx=df_proc.index,
                missing_indicator=missing_indicator_qc,
                night_start_hour=int(night_start_hour),
                night_end_hour=int(night_end_hour),
                missing_thresh=float(day_missing_thresh),
            )
            day_excl = day_excl2
            day_table = day_table2

        # NHANES: no protocol masking
        mask_protocol = pd.Series(False, index=df_proc.index)

        calib_ok = int(proc_info.get("CalibOK", 0))
        row["calib_ok"] = calib_ok
        row["dropout_points_raw"] = int(proc_info.get("dropout_points", 0))
        row["nonwear_ok"] = bool(proc_info.get("nonwear_ok", False))
        row["filter_ok"] = bool(proc_info.get("filter_ok", False))
        row["resample_ok"] = bool(proc_info.get("resample_ok", False))

        final_out = out_ok if calib_ok == 1 else out_excl
        tmp_out = final_out + ".tmp"
        if os.path.exists(tmp_out):
            os.remove(tmp_out)

        meta_out = {
            "dataset": "NHANES_2013_2014_PAX80_H",
            "seqn": str(seqn),
            "input_tar": tar_path,
            "fs_in_hz": int(fs_in),
            "fs_out_hz": int(fs_out),
            "n_hourly_files": int(len(acc_members)),
            "member_discovery_info": member_info,
            "qc_meta_from_logs": qc_meta,
            "qc_flagged_samples_raw": int(n_qc),
            "timeline_rebuild_info": rebuild_info,
            "notes": (
                "NHANES timestamps are disclosure-shifted to Jan 2000; start_time is consistent within file, "
                "but not the participant's real calendar date."
            ),
        }

        write_h5_whole(
            outpath=tmp_out,
            df=df_proc,
            meta=meta_out,
            proc_info=proc_info,
            mask_dropout=mask_dropout_proc.astype(bool),
            mask_nonwear=mask_nonwear_added.astype(bool),
            mask_qc_removed=mask_qc_proc.astype(bool),
            mask_protocol_excl=mask_protocol.astype(bool),
            mask_night_excl=night_excl.astype(bool),
            mask_day_excl=day_excl.astype(bool),
            night_table=night_table,
            day_table=day_table,
            chunk_sec=int(chunk_sec),
        )

        os.replace(tmp_out, final_out)

        # Fill row fields after successful write
        row["output_path"] = final_out
        row["start_time"] = df_proc.index[0].strftime("%Y-%m-%d %H:%M:%S") if len(df_proc.index) else ""
        N = int(len(df_proc.index))
        row["n_samples_30hz"] = N
        row["duration_days_30hz"] = float(N / (float(fs_out) * 3600.0 * 24.0)) if N > 0 else 0.0

        if calib_ok == 1:
            row["status"] = "OK"
            row["reason"] = ""
            print(f"[OK] SEQN={seqn} -> {final_out}")
        else:
            row["status"] = "EXCLUDED"
            row["reason"] = "CalibOK==0"
            print(f"[EXCLUDED CalibOK=0] SEQN={seqn} -> {final_out}")

        _write_report_row_json(report_dir, seqn, row)
        return row

    except Exception as e:
        row["status"] = "FAILED"
        row["reason"] = "exception"
        row["error"] = str(e)
        _write_report_row_json(report_dir, seqn, row)
        print(f"[ERROR] SEQN={seqn}: {e}")
        return row


# -------------------------
# CLI
# -------------------------

def main():
    ap = argparse.ArgumentParser("NHANES PAX80_H preprocessing (one H5 per SEQN)")
    ap.add_argument("--tar_dir", required=True, help="Directory with SEQN.tar.bz2 files")
    ap.add_argument("--output_dir", required=True, help="Where to write .h5 (CalibOK==1)")
    ap.add_argument("--exclusion_dir", required=True, help="Where to write .h5 (CalibOK==0)")

    ap.add_argument("--report_dir", type=str, default="", help="Where to write run reports (default: <output_dir>/reports)")
    ap.add_argument("--overwrite", action="store_true", help="Overwrite existing outputs (dangerous).")
    ap.add_argument("--finalize_report", action="store_true", help="Force consolidation of rows -> CSV/XLSX at end.")

    ap.add_argument("--fs_in", type=int, default=80, help="Input sampling rate (NHANES raw is 80 Hz)")
    ap.add_argument("--fs_out", type=int, default=30, help="Output sampling rate (default 30 Hz)")

    ap.add_argument("--nonwear_patience_min", type=int, default=90)
    ap.add_argument("--nonwear_window", type=str, default="10s")
    ap.add_argument("--nonwear_stdtol", type=float, default=0.013)

    ap.add_argument("--night_missing_thresh", type=float, default=0.5)
    ap.add_argument("--day_missing_thresh", type=float, default=0.5)
    ap.add_argument("--night_start_hour", type=int, default=21)
    ap.add_argument("--night_end_hour", type=int, default=9)

    ap.add_argument("--chunk_sec", type=int, default=600)

    ap.add_argument("--start_idx", type=int, default=0, help="Start index into sorted tar list")
    ap.add_argument("--end_idx", type=int, default=-1, help="End index (exclusive). -1 means all.")
    ap.add_argument("--only_seqn", type=str, default="", help="If set, only process this SEQN (e.g. 73557)")

    ap.add_argument("--verbose", action="store_true")
    args = ap.parse_args()

    tar_dir = Path(args.tar_dir)
    if not tar_dir.exists():
        raise FileNotFoundError(f"tar_dir does not exist: {tar_dir}")

    os.makedirs(args.output_dir, exist_ok=True)
    os.makedirs(args.exclusion_dir, exist_ok=True)

    report_dir = args.report_dir.strip() or str(Path(args.output_dir) / "reports")
    Path(report_dir).mkdir(parents=True, exist_ok=True)

    if args.only_seqn:
        seqn = args.only_seqn.strip()
        tp = tar_dir / f"{seqn}.tar.bz2"
        if not tp.exists():
            raise FileNotFoundError(f"Requested SEQN tar not found: {tp}")
        tars = [str(tp)]
    else:
        tars = sorted([str(p) for p in tar_dir.glob("*.tar.bz2")])
        if args.end_idx != -1:
            tars = tars[args.start_idx:args.end_idx]
        else:
            tars = tars[args.start_idx:]

    if not tars:
        raise RuntimeError("No tar.bz2 files selected for processing.")

    for tp in tars:
        process_one_seqn(
            tar_path=tp,
            out_dir=args.output_dir,
            exclusion_dir=args.exclusion_dir,
            fs_in=args.fs_in,
            fs_out=args.fs_out,
            nonwear_patience_min=args.nonwear_patience_min,
            nonwear_window=args.nonwear_window,
            nonwear_stdtol=args.nonwear_stdtol,
            night_missing_thresh=args.night_missing_thresh,
            day_missing_thresh=args.day_missing_thresh,
            night_start_hour=args.night_start_hour,
            night_end_hour=args.night_end_hour,
            chunk_sec=args.chunk_sec,
            verbose=args.verbose,
            report_dir=report_dir,
            overwrite=args.overwrite,
        )

    # Consolidate report (safe default behavior):
    # - If you run in a SLURM array, multiple tasks will write rows concurrently.
    #   We therefore skip consolidation unless you pass --finalize_report.
    if args.finalize_report or (not _is_slurm_array_task()):
        df = _consolidate_report(report_dir)
        print(f"\n[REPORT] Wrote consolidated report to:")
        print(f"  {Path(report_dir) / 'nhanes_preproc_report.csv'}")
        print(f"  {Path(report_dir) / 'nhanes_preproc_report.xlsx'}")
        print(f"[REPORT] Rows: {len(df)}")
    else:
        print(f"\n[REPORT] Wrote per-SEQN rows to: {Path(report_dir) / 'rows'}")
        print("[REPORT] Skipped consolidation because this looks like a SLURM array task.")
        print("         Run one consolidation step later (or rerun with --finalize_report).")


if __name__ == "__main__":
    main()