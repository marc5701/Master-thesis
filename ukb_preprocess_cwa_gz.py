#!/usr/bin/env python3
"""
UK Biobank AX3 preprocessing (Bologna-style output H5; input .cwa.gz)
====================================================================

Adds deterministic file-list support for SLURM array jobs.

Input:
- UKB Axivity CWA recordings, commonly stored as .cwa.gz

Output:
- One .h5 per file, matching the standard format used for Takeda/Montpellier/Bologna:
  groups: annotations/, data/, masks/, tables/
  data/accelerometry: float32, shape (3, N) with chunks (3, chunk_len)
  masks: 5 required masks, uint8 0/1, length N
  tables: night_table_csv, day_table_csv as bytes
  attrs: start_time, proc_info attrs (+ meta_json, proc_info_json)

Pipeline (same as Bologna):
1) dropout (x=y=z=0) -> NaN on raw grid
2) lowpass with safe cutoff <= Nyquist:
     desired = fs_out/2 - 0.1
     cutoff  = min(desired, fs_in/2 - 0.1)
3) resample to fs_out (uniform)
4) calibrate_gravity
5) flag_nonwear -> NaN
6) night/day missingness masks from (dropout OR nonwear_added)

Notes:
- Prefer using --file_list for huge directories to avoid repeated filesystem crawling.
- --start_idx/--end_idx are applied AFTER loading the file list (whether from --file_list or scanning).
- Adds lightweight, multiprocessing-safe failure logging via --fail_log.
"""

from __future__ import annotations

import os
import json
import gzip
import shutil
import argparse
import warnings
import tempfile
import multiprocessing as mp
from pathlib import Path
from datetime import datetime

import numpy as np
import pandas as pd
import h5py
import actipy


# -------------------------
# Globals for MP-safe logging
# -------------------------

_FAIL_LOG_PATH: str | None = None
_FAIL_LOG_LOCK: mp.Lock | None = None


def _init_worker(fail_log_path: str | None, lock: mp.Lock | None):
    """Initializer for multiprocessing workers."""
    global _FAIL_LOG_PATH, _FAIL_LOG_LOCK
    _FAIL_LOG_PATH = fail_log_path
    _FAIL_LOG_LOCK = lock


def _log_failure(eid: str, run: str, path: str, exc: BaseException):
    """Append a single failure line to TSV log (MP-safe)."""
    global _FAIL_LOG_PATH, _FAIL_LOG_LOCK
    if not _FAIL_LOG_PATH:
        return

    ts = datetime.utcnow().strftime("%Y-%m-%dT%H:%M:%SZ")
    exc_type = type(exc).__name__
    msg = str(exc).replace("\t", " ").replace("\n", " ").replace("\r", " ")
    line = f"{ts}\t{eid}\t{run}\t{path}\t{exc_type}\t{msg}\n"

    # ensure parent exists (best-effort)
    try:
        Path(_FAIL_LOG_PATH).parent.mkdir(parents=True, exist_ok=True)
    except Exception:
        pass

    if _FAIL_LOG_LOCK is None:
        # single-process fallback
        with open(_FAIL_LOG_PATH, "a", encoding="utf-8") as f:
            f.write(line)
        return

    with _FAIL_LOG_LOCK:
        with open(_FAIL_LOG_PATH, "a", encoding="utf-8") as f:
            f.write(line)


# -------------------------
# JSON helpers (same pattern as Bologna)
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

def _ensure_utc_naive_index(idx: pd.DatetimeIndex) -> pd.DatetimeIndex:
    if not isinstance(idx, pd.DatetimeIndex):
        raise ValueError("Index must be a DatetimeIndex")
    if getattr(idx, "tz", None) is not None:
        return idx.tz_convert("UTC").tz_localize(None)
    return idx


# -------------------------
# UKB filename parsing
# -------------------------

def parse_ukb_filename(path: str) -> dict:
    """
    UKB filenames are typically like:
      <eid>_<something>_<run>_<something>.cwa.gz

    Niels used:
      sub_id = split('_')[0]
      run    = split('_')[2]

    We'll follow that but be robust.
    """
    name = os.path.basename(path)

    base = name
    if base.lower().endswith(".cwa.gz"):
        base = base[:-7]
    elif base.lower().endswith(".cwa"):
        base = base[:-4]

    parts = base.split("_")
    eid = parts[0] if len(parts) >= 1 else base
    run = parts[2] if len(parts) >= 3 else "0"
    return {"eid": eid, "run": run, "base": base, "filename": name}


# -------------------------
# Reading .cwa.gz safely
# -------------------------

def read_cwa_or_cwa_gz(path: str):
    """
    Try actipy.read_device(path) directly.
    If that fails for .gz, decompress to a temp .cwa and read.
    """
    try:
        data, meta = actipy.read_device(
            path,
            lowpass_hz=None,
            calibrate_gravity=False,
            detect_nonwear=False,
            resample_hz="uniform",  # keep actipy timestamp grid; we resample later ourselves
            verbose=False,
        )
        return data, meta
    except Exception as e1:
        if not path.lower().endswith(".gz"):
            raise

        tmp_path = None
        try:
            tmp = tempfile.NamedTemporaryFile(prefix="ukb_", suffix=".cwa", delete=False)
            tmp_path = tmp.name
            tmp.close()

            with gzip.open(path, "rb") as fin, open(tmp_path, "wb") as fout:
                shutil.copyfileobj(fin, fout)

            data, meta = actipy.read_device(
                tmp_path,
                lowpass_hz=None,
                calibrate_gravity=False,
                detect_nonwear=False,
                resample_hz="uniform",
                verbose=False,
            )
            return data, meta
        except Exception as e2:
            raise RuntimeError(f"Failed reading {path}. Direct error={e1}; gunzip+read error={e2}") from e2
        finally:
            if tmp_path is not None:
                try:
                    os.remove(tmp_path)
                except Exception:
                    pass


# -------------------------
# Core preprocessing (Bologna logic, generalized)
# -------------------------

def preprocess_actigraphy_df(
    data_df: pd.DataFrame,
    fs_in: float,
    fs_out: int = 30,
    calib_cube: float = 0.3,
    nonwear_patience_min: int = 90,
    nonwear_window: str = "10s",
    nonwear_stdtol: float = 0.013,
    verbose: bool = True,
) -> tuple[pd.DataFrame, dict, pd.Series, pd.Series]:
    if not isinstance(data_df.index, pd.DatetimeIndex):
        raise ValueError("data_df must have DatetimeIndex")

    fs_in = float(fs_in)

    desired = float(fs_out) / 2.0 - 0.1
    max_ok = float(fs_in) / 2.0 - 0.1
    cutoff = float(min(desired, max_ok))

    proc_info: dict = {
        "input_fs": float(fs_in),
        "output_fs": int(fs_out),

        # standard cross-cohort key
        "lowpass_cutoff_hz": float(cutoff),

        # audit fields (optional, but useful)
        "lowpass_cutoff_hz_requested": float(desired),
        "lowpass_cutoff_hz_used": float(cutoff),

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

    # Dropout (x=y=z=0) -> NaN on raw grid
    xyz = df[["x", "y", "z"]].to_numpy()
    dropout_raw = np.all(xyz == 0, axis=1)
    proc_info["dropout_points"] = int(dropout_raw.sum())
    if proc_info["dropout_points"] > 0:
        df.loc[dropout_raw, ["x", "y", "z"]] = np.nan

    # Low-pass
    if verbose:
        print(f"  Low-pass: fs_in={fs_in:.3f} Hz, cutoff_req={desired:.3f} Hz, cutoff_used={cutoff:.3f} Hz")
    try:
        df, filter_info = actipy.processing.lowpass(
            data=df,
            data_sample_rate=fs_in,
            cutoff_rate=cutoff,
        )
        proc_info.update({f"filter_{k}": v for k, v in filter_info.items()})
        proc_info["filter_ok"] = True
    except Exception as e:
        warnings.warn(f"Lowpass failed: {e}. Proceeding without filter.")
        proc_info["filter_ok"] = False
        proc_info["filter_error"] = str(e)

    # Resample to fs_out (uniform grid)
    if verbose:
        print(f"  Resampling to {fs_out} Hz")
    df, resample_info = actipy.processing.resample(data=df, sample_rate=fs_out)
    proc_info.update({f"resample_{k}": v for k, v in resample_info.items()})
    proc_info["resample_ok"] = True

    # any-axis-NaN => all-axes-NaN
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
            "  Non-wear (actipy.flag_nonwear): "
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
        warnings.warn(f"Non-wear failed: {e}. Proceeding without nonwear.")
        proc_info["nonwear_ok"] = False
        proc_info["nonwear_error"] = str(e)

    # any-axis-NaN => all-axes-NaN
    any_nan2 = df[["x", "y", "z"]].isna().any(axis=1)
    if bool(any_nan2.any()):
        df.loc[any_nan2, ["x", "y", "z"]] = np.nan

    nan_after = df["x"].isna()
    mask_nonwear_added = (nan_after & (~nan_before)).astype(bool)

    return df, proc_info, mask_dropout_proc, mask_nonwear_added


# -------------------------
# Night/Day QC tables + masks (same as Bologna)
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
            night_rows.append({
                "night_start": str(w0),
                "night_end": str(w1),
                "missing_fraction": frac,
                "excluded": bool(bad),
            })
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
            day_rows.append({
                "day_start": str(w0),
                "day_end": str(w1),
                "missing_fraction": frac,
                "excluded": bool(bad),
            })
        cur += pd.Timedelta(days=1)

    return night_mask, day_mask, pd.DataFrame(night_rows), pd.DataFrame(day_rows)


# -------------------------
# HDF5 writer (standard format)
# -------------------------

def write_h5_whole(
    outpath: str,
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
    os.makedirs(os.path.dirname(outpath), exist_ok=True)

    idx = _ensure_utc_naive_index(df.index)
    acc = df[["x", "y", "z"]].to_numpy(dtype=np.float32).T
    n = acc.shape[1]

    fs = float(proc_info.get("output_fs", 30))
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

        f["data"].create_dataset(
            "accelerometry",
            data=acc,
            dtype="f4",
            chunks=(3, chunk_len),
        )

        def _write_mask(name: str, s: pd.Series):
            f["masks"].create_dataset(
                name,
                data=s.to_numpy(dtype=np.uint8),
                dtype="u1",
                chunks=(chunk_len,),
            )

        _write_mask("dropout_zero_vector", mask_dropout.astype(bool))
        _write_mask("detected_nonwear_actipy", mask_nonwear.astype(bool))
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
# Per-file pipeline (UKB)
# -------------------------

def process_one_ukb(
    cwa_gz_path: str,
    out_dir: str,
    exclusion_dir: str,
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
) -> None:
    info_name = parse_ukb_filename(cwa_gz_path)
    eid = info_name["eid"]
    run = info_name["run"]

    outname = f"{eid}_{run}.h5"
    outpath_ok = os.path.join(out_dir, outname)
    outpath_excl = os.path.join(exclusion_dir, outname)

    # IMPORTANT: Skip if already processed
    if os.path.exists(outpath_ok) or os.path.exists(outpath_excl):
        return

    if verbose:
        print(f"\n== UKB {eid} (run={run}) ==")

    data, meta = read_cwa_or_cwa_gz(cwa_gz_path)

    if not all(c in data.columns for c in ["x", "y", "z"]):
        raise ValueError(f"Missing x/y/z in {cwa_gz_path}. Columns={list(data.columns)}")

    fs_in = meta.get("SampleRate", None)
    if fs_in is None:
        dt = data.index.to_series().diff().dropna().dt.total_seconds()
        fs_in = float(1.0 / dt.median())
    fs_in = float(fs_in)

    df_proc, proc_info, mask_dropout_proc, mask_nonwear_added = preprocess_actigraphy_df(
        data_df=data[["x", "y", "z"]],
        fs_in=fs_in,
        fs_out=fs_out,
        calib_cube=0.3,
        nonwear_patience_min=nonwear_patience_min,
        nonwear_window=nonwear_window,
        nonwear_stdtol=nonwear_stdtol,
        verbose=verbose,
    )

    missing_indicator_qc = (mask_dropout_proc | mask_nonwear_added).astype(bool)

    night_excl, day_excl, night_table, day_table = compute_fixed_night_day_qc_tables_and_masks(
        idx=df_proc.index,
        missing_indicator=missing_indicator_qc,
        night_start_hour=night_start_hour,
        night_end_hour=night_end_hour,
        missing_thresh=float(night_missing_thresh),
    )

    if float(day_missing_thresh) != float(night_missing_thresh):
        _, day_excl2, _, day_table2 = compute_fixed_night_day_qc_tables_and_masks(
            idx=df_proc.index,
            missing_indicator=missing_indicator_qc,
            night_start_hour=night_start_hour,
            night_end_hour=night_end_hour,
            missing_thresh=float(day_missing_thresh),
        )
        day_excl = day_excl2
        day_table = day_table2

    mask_protocol = pd.Series(False, index=df_proc.index)  # UKB: none by default

    final_outpath = outpath_ok if int(proc_info.get("CalibOK", 0)) == 1 else outpath_excl
    tmp_outpath = final_outpath + ".tmp"
    if os.path.exists(tmp_outpath):
        os.remove(tmp_outpath)

    meta_out = {
        "dataset": "UKBiobank_AX3",
        "input_file": cwa_gz_path,
        "eid": eid,
        "run": run,
        "fs_in_hz": float(fs_in),
        "fs_out_hz": int(fs_out),
        "actipy_meta": {k: _safe_attr(v) for k, v in dict(meta).items()},
        "time_basis_note": (
            "actipy timestamps used for processing; H5 'start_time' is written as UTC naive string. "
            "Assumes actipy index represents true UTC clock time."
        ),
    }

    write_h5_whole(
        outpath=tmp_outpath,
        df=df_proc,
        meta=meta_out,
        proc_info=proc_info,
        mask_dropout=mask_dropout_proc.astype(bool),
        mask_nonwear=mask_nonwear_added.astype(bool),
        mask_protocol_excl=mask_protocol.astype(bool),
        mask_night_excl=night_excl.astype(bool),
        mask_day_excl=day_excl.astype(bool),
        night_table=night_table,
        day_table=day_table,
        chunk_sec=chunk_sec,
    )

    os.replace(tmp_outpath, final_outpath)

    if final_outpath == outpath_excl:
        print(f"[EXCLUDED CalibOK=0] {eid} (run={run}) -> {final_outpath}")
    else:
        print(f"[OK] {eid} (run={run}) -> {final_outpath}")


def _worker(args_tuple):
    """
    Worker wrapper with per-file try/except so one bad file doesn't kill the pool.
    Also logs failures to TSV.
    """
    cwa_gz_path = args_tuple[0]
    info = parse_ukb_filename(cwa_gz_path)
    eid, run = info.get("eid", "?"), info.get("run", "?")

    try:
        return process_one_ukb(*args_tuple)
    except (EOFError, OSError, OverflowError, ValueError) as e:
        print(f"[ERROR] {eid} (run={run}) {cwa_gz_path}: {type(e).__name__}: {e}")
        _log_failure(eid, run, cwa_gz_path, e)
        return None
    except Exception as e:
        print(f"[ERROR] {eid} (run={run}) {cwa_gz_path}: {type(e).__name__}: {e}")
        _log_failure(eid, run, cwa_gz_path, e)
        return None


# -------------------------
# File list utilities
# -------------------------

def load_file_list(path: str) -> list[str]:
    p = Path(path)
    if not p.exists():
        raise FileNotFoundError(f"--file_list not found: {path}")
    files = [ln.strip() for ln in p.read_text().splitlines() if ln.strip()]
    return files


def discover_files(data_dir: str) -> list[str]:
    d = Path(data_dir)
    files = sorted([str(p) for p in d.glob("*.cwa.gz")])
    if not files:
        files = sorted([str(p) for p in d.rglob("*.cwa.gz")])
    return files


def apply_slice(files: list[str], start_idx: int, end_idx: int) -> list[str]:
    if start_idx < 0:
        raise ValueError("--start_idx must be >= 0")
    if end_idx != -1 and end_idx < start_idx:
        raise ValueError("--end_idx must be -1 or >= start_idx")
    return files[start_idx: end_idx if end_idx != -1 else None]


def main():
    ap = argparse.ArgumentParser("UK Biobank preprocessing (Bologna-style output)")

    ap.add_argument("--data_dir", required=False, default=None,
                    help="Folder containing *.cwa.gz (ignored if --file_list is provided)")
    ap.add_argument("--file_list", default=None,
                    help="Path to newline-separated list of .cwa.gz files (recommended for UKB)")
    ap.add_argument("--output_dir", required=True,
                    help="Where to write .h5 (CalibOK==1)")
    ap.add_argument("--exclusion_dir", required=True,
                    help="Where to write .h5 (CalibOK==0)")

    ap.add_argument("--fs_out", type=int, default=30)

    ap.add_argument("--nonwear_patience_min", type=int, default=90)
    ap.add_argument("--nonwear_window", type=str, default="10s")
    ap.add_argument("--nonwear_stdtol", type=float, default=0.013)

    ap.add_argument("--night_missing_thresh", type=float, default=0.5)
    ap.add_argument("--day_missing_thresh", type=float, default=0.5)

    ap.add_argument("--night_start_hour", type=int, default=21)
    ap.add_argument("--night_end_hour", type=int, default=9)

    ap.add_argument("--chunk_sec", type=int, default=600)

    ap.add_argument("--start_idx", type=int, default=0)
    ap.add_argument("--end_idx", type=int, default=-1)

    ap.add_argument("--num_workers", type=int, default=1,
                    help="Use >1 for multiprocessing (e.g. 4, 8, 16).")
    ap.add_argument("--verbose", action="store_true")

    # new: failure log
    ap.add_argument("--fail_log", default="results/ukb/failures.tsv",
                    help="TSV log for per-file failures (timestamp, eid, run, path, exc_type, message). "
                         "Set to '' to disable.")

    args = ap.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)
    os.makedirs(args.exclusion_dir, exist_ok=True)

    if args.file_list:
        files_all = load_file_list(args.file_list)
        source = f"file_list={args.file_list}"
    else:
        if not args.data_dir:
            raise ValueError("Must provide --data_dir if --file_list is not provided")
        files_all = discover_files(args.data_dir)
        source = f"data_dir={args.data_dir}"

    if not files_all:
        raise FileNotFoundError(f"No .cwa.gz files found ({source})")

    files = apply_slice(files_all, args.start_idx, args.end_idx)

    print(f"[FILES] Source: {source}")
    print(f"[FILES] Total available: {len(files_all)}")
    print(f"[FILES] This run slice: start_idx={args.start_idx} end_idx={args.end_idx} -> {len(files)} files")

    worker_args = [
        (
            p,
            args.output_dir,
            args.exclusion_dir,
            args.fs_out,
            args.nonwear_patience_min,
            args.nonwear_window,
            args.nonwear_stdtol,
            args.night_missing_thresh,
            args.day_missing_thresh,
            args.night_start_hour,
            args.night_end_hour,
            args.chunk_sec,
            args.verbose,
        )
        for p in files
    ]

    fail_log_path = args.fail_log.strip() if isinstance(args.fail_log, str) else None
    if fail_log_path == "":
        fail_log_path = None

    if args.num_workers and args.num_workers > 1:
        print(f"[MP] Using {args.num_workers} workers")
        lock = mp.Lock() if fail_log_path else None
        with mp.Pool(
            processes=args.num_workers,
            initializer=_init_worker,
            initargs=(fail_log_path, lock),
        ) as pool:
            for _ in pool.imap_unordered(_worker, worker_args, chunksize=1):
                pass
    else:
        # single-process mode
        _init_worker(fail_log_path, None)
        for wa in worker_args:
            _worker(wa)


if __name__ == "__main__":
    main()