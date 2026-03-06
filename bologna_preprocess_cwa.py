#!/usr/bin/env python3
"""
Bologna AX6 preprocessing (Montpellier-style; CWA input; fs_out=30Hz)
====================================================================

Differences vs Montpellier:
- Input is Axivity .cwa (read via actipy.read_device)
- Bologna files are 25 Hz -> resample to 30 Hz
- Lowpass cutoff must respect Nyquist:
    cutoff = min(fs_out/2 - 0.1, fs_in/2 - 0.1)
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


# -------------------------
# JSON helpers 
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
# Bologna-specific helpers
# -------------------------

_CWA_RE = re.compile(r"^(?P<prefix>\d+?)_(?P<sid>\d{10})\.cwa$", re.IGNORECASE)

def parse_bologna_filename(p: str) -> dict:
    """Extract device-ish prefix + 10-digit subject ID from filename."""
    name = os.path.basename(p)
    m = _CWA_RE.match(name)
    if not m:
        raise ValueError(f"Filename does not match '<prefix>_<10digit>.cwa': {name}")
    return {"prefix": m.group("prefix"), "subject_id": m.group("sid")}

def load_population_bologna(xlsx_path: str) -> pd.DataFrame:
    pop = pd.read_excel(xlsx_path)

    required = ["Num_Pol", "HLA_DQ0602", "Diagnosis ICSD3-TR criteria, awaiting hcrt)", "Gender"]
    missing = [c for c in required if c not in pop.columns]
    if missing:
        raise ValueError(f"Missing expected columns in {xlsx_path}: {missing}. Columns={list(pop.columns)}")

    pop = pop.copy()
    pop["Num_Pol"] = pd.to_numeric(pop["Num_Pol"], errors="coerce")
    pop = pop[pop["Num_Pol"].notna()].copy()

    pop["subject_id_10"] = pop["Num_Pol"].astype(int).astype(str).str.zfill(10)
    return pop

    def looks_like_10digit_series(s: pd.Series) -> bool:
        ss = s.astype(str).str.strip()
        # allow numeric IDs too; convert to int safely then zfill
        ss2 = ss.str.replace(r"\.0$", "", regex=True)
        ok = ss2.str.fullmatch(r"\d{1,10}").fillna(False)
        if ok.mean() < 0.7:
            return False
        # if many are <=10 digits numeric, we can zfill
        return True

    # choose best column
    best = None
    for c in candidates:
        if looks_like_10digit_series(pop[c]):
            best = c
            break

    if best is None:
        # fallback: scan all columns
        for c in cols:
            if looks_like_10digit_series(pop[c]):
                best = c
                break

    if best is None:
        raise ValueError(
            f"Could not detect subject-id column in {xlsx_path}. "
            f"Columns={cols}"
        )

    # standardize to 10-digit zero padded string
    s = pop[best].astype(str).str.strip().str.replace(r"\.0$", "", regex=True)
    s = s.where(s.str.fullmatch(r"\d+"), np.nan)
    pop["subject_id_10"] = s.dropna().astype(str).str.zfill(10)
    pop = pop[pop["subject_id_10"].notna()].copy()

    return pop, best


# -------------------------
# Core preprocessing 
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

    # IMPORTANT: cutoff must be <= Nyquist else you get an error
    desired = float(fs_out) / 2.0 - 0.1
    max_ok = float(fs_in) / 2.0 - 0.1
    cutoff = float(min(desired, max_ok))

    proc_info: dict = {
        "input_fs": float(fs_in),
        "output_fs": int(fs_out),

        # Standard key used 
        # store the actual cutoff applied (safe vs Nyquist)
        "lowpass_cutoff_hz": float(cutoff),

        # Keep extra Bologna-specific audit fields 
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
    force = np.linalg.norm(df[["x", "y", "z"]].values, axis=1)
    dropout_raw = (force == 0)
    proc_info["dropout_points"] = int(dropout_raw.sum())
    if proc_info["dropout_points"] > 0:
        df.loc[dropout_raw, ["x", "y", "z"]] = np.nan

    # Low-pass (use safe cutoff)
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

    # Enforce any-axis-NaN => all-axes-NaN
    any_nan2 = df[["x", "y", "z"]].isna().any(axis=1)
    if bool(any_nan2.any()):
        df.loc[any_nan2, ["x", "y", "z"]] = np.nan

    nan_after = df["x"].isna()
    mask_nonwear_added = (nan_after & (~nan_before)).astype(bool)

    return df, proc_info, mask_dropout_proc, mask_nonwear_added


# -------------------------
# Night/Day QC flags ONLY (same as yours)
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
            night_rows.append({"night_start": str(w0), "night_end": str(w1),
                               "missing_fraction": frac, "excluded": bool(bad)})
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
            day_rows.append({"day_start": str(w0), "day_end": str(w1),
                             "missing_fraction": frac, "excluded": bool(bad)})
        cur += pd.Timedelta(days=1)

    return night_mask, day_mask, pd.DataFrame(night_rows), pd.DataFrame(day_rows)


# -------------------------
# HDF5 writer (exactly your required format)
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
# Per-file pipeline (Bologna)
# -------------------------

def process_one_cwa(
    cwa_path: str,
    pop_row: pd.Series | None,
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
    info_name = parse_bologna_filename(cwa_path)
    sid10 = info_name["subject_id"]
    prefix = info_name["prefix"]

    # output naming: keep sid as 10-digit (your preference)
    outname = f"{sid10}.h5"
    outpath_ok = os.path.join(out_dir, outname)
    outpath_excl = os.path.join(exclusion_dir, outname)

    if os.path.exists(outpath_ok) or os.path.exists(outpath_excl):
        return

    if verbose:
        print(f"\n== {sid10}  (prefix={prefix}) ==")

    # Read CWA (raw)
    # NOTE: we disable actipy's internal steps so we follow YOUR pipeline order.
    data, meta = actipy.read_device(
        cwa_path,
        lowpass_hz=None,
        calibrate_gravity=False,
        detect_nonwear=False,
        resample_hz="uniform",
        verbose=False,
    )

    # Ensure x/y/z exist and keep only those
    if not all(c in data.columns for c in ["x", "y", "z"]):
        raise ValueError(f"Missing x/y/z in {cwa_path}. Columns={list(data.columns)}")

    # Determine input sampling rate from meta (fallback to estimate)
    fs_in = meta.get("SampleRate", None)
    if fs_in is None:
        # estimate from median dt
        dt = data.index.to_series().diff().dropna().dt.total_seconds()
        fs_in = float(1.0 / dt.median())
    fs_in = float(fs_in)

    # Preprocess
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

    # QC missingness indicator (raw-quality only)
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

    mask_protocol = pd.Series(False, index=df_proc.index)  # Bologna: none for now

    final_outpath = outpath_ok if int(proc_info.get("CalibOK", 0)) == 1 else outpath_excl
    tmp_outpath = final_outpath + ".tmp"
    if os.path.exists(tmp_outpath):
        os.remove(tmp_outpath)

    meta_out = {
        "dataset": "Bologna_AX6",
        "input_file": cwa_path,
        "subject_id_10": sid10,
        "prefix_deviceish": prefix,
        "fs_in_hz": float(fs_in),
        "fs_out_hz": int(fs_out),
        "actipy_meta": {k: _safe_attr(v) for k, v in dict(meta).items()},
        "population_row": None if pop_row is None else {k: _safe_attr(v) for k, v in pop_row.to_dict().items()},
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
        print(f"[EXCLUDED CalibOK=0] {sid10} -> {final_outpath}")
    else:
        print(f"[OK] {sid10} -> {final_outpath}")


def main():
    ap = argparse.ArgumentParser("Bologna preprocessing (Montpellier-style)")

    ap.add_argument("--data_dir", required=True,
                    help="Folder containing AX6/*.cwa (already unzipped)")
    ap.add_argument("--population_xlsx", required=True,
                    help="POPULATION_CHARACTERISTICS_bologna.xlsx")
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

    ap.add_argument("--start_idx", type=int, default=0,
                    help="Process files from this index in the sorted list")
    ap.add_argument("--end_idx", type=int, default=-1,
                    help="Process files up to this index (python slicing)")

    ap.add_argument("--verbose", action="store_true")

    args = ap.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)
    os.makedirs(args.exclusion_dir, exist_ok=True)

    # Load population + map subject_id_10 -> row
    pop = load_population_bologna(args.population_xlsx)
    pop = pop.drop_duplicates("subject_id_10")
    pop_map = {r["subject_id_10"]: r for _, r in pop.iterrows()}
    print(f"[POP] Loaded {len(pop)} rows from {args.population_xlsx}")

    # Gather CWA files
    data_dir = Path(args.data_dir)
    cwas = sorted([str(p) for p in data_dir.glob("*.cwa")])
    if not cwas:
        # sometimes nested in AX6/
        cwas = sorted([str(p) for p in data_dir.rglob("*.cwa")])
    if not cwas:
        raise FileNotFoundError(f"No .cwa files found under {args.data_dir}")

    cwas = cwas[args.start_idx:args.end_idx if args.end_idx != -1 else None]
    print(f"[FILES] Found {len(cwas)} .cwa files to process")

    for p in cwas:
        sid10 = parse_bologna_filename(p)["subject_id"]
        pop_row = pop_map.get(sid10, None)
        try:
            process_one_cwa(
                cwa_path=p,
                pop_row=pop_row,
                out_dir=args.output_dir,
                exclusion_dir=args.exclusion_dir,
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
            )
        except Exception as e:
            print(f"[ERROR] {p}: {e}")


if __name__ == "__main__":
    main()




