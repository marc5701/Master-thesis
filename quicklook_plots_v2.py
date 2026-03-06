#!/usr/bin/env python3
"""
Quicklook plots with diagnostics to avoid "flat-line" visualization artifacts.

Changes vs v1:
- Plot epoch mean AND epoch std (x/y/z + magnitude).
- Plot ENMO (|a|-1 clipped at 0) epoch mean.
- Create a "strict masked" view where any epoch with ANY mask active is forced to NaN
  so the signal cannot visually "bridge" through masked time.
"""

import os
import numpy as np
import h5py
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

# ---------- configuration ----------
FS = 30
EPOCH_SEC = 30
EPOCH_SAMPLES = FS * EPOCH_SEC  # 900
HOURS = 48
TOTAL_SAMPLES = HOURS * 3600 * FS  # 5,184,000
TOTAL_EPOCHS = TOTAL_SAMPLES // EPOCH_SAMPLES  # 5,760

OUT_DIR = "/workspace/quicklook_plots_v2"

MASK_DEFS = [
    ("detected_nonwear_actipy",        "nonwear"),
    ("dropout_zero_vector",            "dropout"),
    ("qc_removed_nhanes",              "qc_nhanes"),
    ("excluded_by_protocol_psg_mwt",   "psg/mwt"),
    ("excluded_day_missing_gt_50pct",  "day_miss"),
    ("excluded_night_missing_gt_50pct","night_miss"),
]

COHORTS = {
    "ukb": {
        "label": "UK Biobank",
        "file": "/oak_mignot/mdige/results/ukb/preprocessed_h5/1000062_0.h5",
    },
    "bologna": {
        "label": "Bologna",
        "file": "/oak_mignot/mdige/results/bologna/preprocessed_h5/0000049191.h5",
    },
    "montpellier": {
        "label": "Montpellier",
        "file": "/oak_mignot/mdige/results/montpellier/preprocessed_h5/001.h5",
    },
    "takeda": {
        "label": "Takeda",
        "file": "/oak_mignot/mdige/results/takeda/preprocessed_h5/58001-001_CPW1B51190228_2020-06-30_02_01_00_PM_1.h5",
    },
    "nhanes_2011_2012": {
        "label": "NHANES 2011-2012",
        "file": "/oak_mignot/mdige/results/nhanes_2011_2012/ok/62161.h5",
    },
    "nhanes_2013_2014": {
        "label": "NHANES 2013-2014",
        "file": "/oak_mignot/mdige/results/nhanes/ok/73557.h5",
    },
}

def reshape_epochs(x, n_epochs, epoch_samples):
    usable = n_epochs * epoch_samples
    return x[:usable].reshape(n_epochs, epoch_samples)

def epoch_nanmean(x, n_epochs, epoch_samples):
    return np.nanmean(reshape_epochs(x, n_epochs, epoch_samples), axis=1)

def epoch_nanstd(x, n_epochs, epoch_samples):
    return np.nanstd(reshape_epochs(x, n_epochs, epoch_samples), axis=1)

def epoch_any(mask, n_epochs, epoch_samples):
    """mask is uint8 (0/1). Any 1 in epoch => True."""
    return reshape_epochs(mask, n_epochs, epoch_samples).max(axis=1).astype(bool)

def apply_epoch_mask_to_epochs(ep_values, ep_mask):
    """Force epochs with ep_mask True to NaN (for plotting strict masked view)."""
    out = ep_values.astype(np.float32, copy=True)
    out[ep_mask] = np.nan
    return out

def plot_quicklook_v2(
    x_mean, y_mean, z_mean,
    x_std, y_std, z_std,
    mag_mean, mag_std, enmo_mean,
    mask_epochs, mask_labels,
    cohort_label, fname, out_path,
):
    epochs_idx = np.arange(len(x_mean))

    fig, axes = plt.subplots(
        5, 1, figsize=(16, 12), sharex=True,
        gridspec_kw={"height_ratios": [3, 2, 2, 2, 2]},
    )
    ax_tri, ax_mag, ax_std, ax_enmo, ax_msk = axes

    # Panel 1: axis means
    ax_tri.plot(epochs_idx, x_mean)
    ax_tri.plot(epochs_idx, y_mean)
    ax_tri.plot(epochs_idx, z_mean)
    ax_tri.set_title("Axis epoch MEAN (30s) — strict-masked")
    ax_tri.set_ylabel("g")
    ax_tri.set_ylim(-2, 2)
    ax_tri.grid(True, alpha=0.4)

    # Panel 2: magnitude mean
    ax_mag.plot(epochs_idx, mag_mean)
    ax_mag.axhline(1, ls="--", color="gray", lw=0.8)
    ax_mag.set_title("Vector magnitude epoch MEAN (30s) — strict-masked")
    ax_mag.set_ylabel("g")
    ax_mag.set_ylim(0, 3)
    ax_mag.grid(True, alpha=0.4)

    # Panel 3: variability (std)
    ax_std.plot(epochs_idx, x_std)
    ax_std.plot(epochs_idx, y_std)
    ax_std.plot(epochs_idx, z_std)
    ax_std.plot(epochs_idx, mag_std)
    ax_std.set_title("Epoch STD (30s): x/y/z and |a| — tells you if 'flat mean' is hiding motion")
    ax_std.set_ylabel("g")
    ax_std.set_ylim(0, 0.5)  # adjustable
    ax_std.grid(True, alpha=0.4)

    # Panel 4: ENMO (movement proxy)
    ax_enmo.plot(epochs_idx, enmo_mean)
    ax_enmo.set_title("ENMO epoch MEAN (30s) = max(|a|-1, 0)  — movement stands out here")
    ax_enmo.set_ylabel("g")
    ax_enmo.set_ylim(0, 1.5)  # adjustable
    ax_enmo.grid(True, alpha=0.4)

    # Panel 5: masks
    n_masks = len(mask_labels)
    colors = plt.rcParams["axes.prop_cycle"].by_key()["color"]
    for i, (m_ep, m_label) in enumerate(zip(mask_epochs, mask_labels)):
        c = colors[i % len(colors)]
        ax_msk.fill_between(
            epochs_idx, i - 0.4, i + 0.4,
            where=m_ep.astype(bool), color=c, alpha=0.7,
        )
    ax_msk.set_yticks(range(n_masks))
    ax_msk.set_yticklabels(mask_labels, fontsize=8)
    ax_msk.set_ylim(-0.5, n_masks - 0.5)
    ax_msk.set_title("Masks (epoch-any)")
    ax_msk.set_xlabel("Epochs")
    ax_msk.grid(True, alpha=0.4)

    fig.suptitle(f"{cohort_label}  —  {fname}", fontsize=11, y=0.995)
    fig.tight_layout(rect=[0, 0, 1, 0.98])
    fig.savefig(out_path, dpi=200)
    plt.close(fig)
    print(f"Saved {out_path}")

def main():
    os.makedirs(OUT_DIR, exist_ok=True)

    for key, info in COHORTS.items():
        label = info["label"]
        fpath = info["file"]
        print(f"Processing {label} ...")

        with h5py.File(fpath, "r") as f:
            accel = f["data/accelerometry"]  # (3, N)
            n_total = accel.shape[1]
            if n_total < TOTAL_SAMPLES:
                print(f"  SKIP: only {n_total} samples ({n_total/(FS*3600):.1f}h), need {HOURS}h")
                continue

            raw = accel[:, :TOTAL_SAMPLES]  # (3, 5,184,000)

            # Load masks present in this file
            available_masks = set(f["masks"].keys())
            mask_data = []
            mask_labels = []
            for hdf5_key, short_label in MASK_DEFS:
                if hdf5_key in available_masks:
                    mask_data.append(f[f"masks/{hdf5_key}"][:TOTAL_SAMPLES].astype(np.uint8))
                    mask_labels.append(short_label)

        x_raw, y_raw, z_raw = raw[0], raw[1], raw[2]
        mag_raw = np.sqrt(x_raw**2 + y_raw**2 + z_raw**2)
        enmo_raw = np.maximum(mag_raw - 1.0, 0.0)

        # Epoch stats (not masked yet)
        x_mean = epoch_nanmean(x_raw, TOTAL_EPOCHS, EPOCH_SAMPLES)
        y_mean = epoch_nanmean(y_raw, TOTAL_EPOCHS, EPOCH_SAMPLES)
        z_mean = epoch_nanmean(z_raw, TOTAL_EPOCHS, EPOCH_SAMPLES)

        x_std = epoch_nanstd(x_raw, TOTAL_EPOCHS, EPOCH_SAMPLES)
        y_std = epoch_nanstd(y_raw, TOTAL_EPOCHS, EPOCH_SAMPLES)
        z_std = epoch_nanstd(z_raw, TOTAL_EPOCHS, EPOCH_SAMPLES)

        mag_mean = epoch_nanmean(mag_raw, TOTAL_EPOCHS, EPOCH_SAMPLES)
        mag_std  = epoch_nanstd(mag_raw, TOTAL_EPOCHS, EPOCH_SAMPLES)
        enmo_mean = epoch_nanmean(enmo_raw, TOTAL_EPOCHS, EPOCH_SAMPLES)

        # Epoch-any masks
        mask_epochs = [epoch_any(m, TOTAL_EPOCHS, EPOCH_SAMPLES) for m in mask_data]

        # Build a combined epoch mask: if ANY mask is active in that epoch, consider it "masked"
        if mask_epochs:
            combined_epoch_mask = np.zeros(TOTAL_EPOCHS, dtype=bool)
            for me in mask_epochs:
                combined_epoch_mask |= me
        else:
            combined_epoch_mask = np.zeros(TOTAL_EPOCHS, dtype=bool)

        # Strict-masked view for mean plots (prevents nanmean bridging visuals)
        x_mean = apply_epoch_mask_to_epochs(x_mean, combined_epoch_mask)
        y_mean = apply_epoch_mask_to_epochs(y_mean, combined_epoch_mask)
        z_mean = apply_epoch_mask_to_epochs(z_mean, combined_epoch_mask)
        mag_mean = apply_epoch_mask_to_epochs(mag_mean, combined_epoch_mask)

        fname = os.path.splitext(os.path.basename(fpath))[0]
        out_path = os.path.join(OUT_DIR, f"{key}_{fname}_48h_v2.png")

        plot_quicklook_v2(
            x_mean, y_mean, z_mean,
            x_std, y_std, z_std,
            mag_mean, mag_std, enmo_mean,
            mask_epochs, mask_labels,
            label, fname, out_path
        )

    print("Done.")

if __name__ == "__main__":
    main()
