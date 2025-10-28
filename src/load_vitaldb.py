import os
import random
from pathlib import Path
from collections import defaultdict

import h5py
import numpy as np


# ---------- Helpers ----------
def _read_mat_str(f, ref):
    if ref is None:
        return ""
    try:
        data = f[ref][()]
    except Exception:
        data = ref

    if isinstance(data, (bytes, bytearray)):
        return data.decode("utf-8", errors="ignore").rstrip("\x00").strip()

    if isinstance(data, np.ndarray):
        if data.dtype.kind in ("u", "S"):
            try:
                return data.tobytes().decode("utf-8", errors="ignore").rstrip("\x00").strip()
            except Exception:
                pass
        if data.dtype == object or data.dtype.kind == "O":
            try:
                return _read_mat_str(f, data[0])
            except Exception:
                return ""

    try:
        return str(data).strip()
    except Exception:
        return ""


def load_vitaldb_mat(
    mat_path: str,
    channel_index: int = 2,
    samples_per_seg: int = 625,
    max_subjects: int = 100,
    segs_per_subject: int = 10,
    seed: int = 42
):
    assert os.path.isfile(mat_path), f"File not found: {mat_path}"
    random.seed(seed)

    all_abp_segments, all_sbp_segments, all_dbp_segments = [], [], []
    all_demographics, subject_ids = [], []

    with h5py.File(mat_path, "r") as f:
        subset = f["Subset"]
        signals = subset["Signals"]
        num_segments_raw = signals.shape[0]

        # ABP channel
        abp_all = signals[:, channel_index, :]

        # Per-segment labels
        sbp_per_seg = np.squeeze(subset["SBP"][:])
        dbp_per_seg = np.squeeze(subset["DBP"][:])

        # Align lengths
        num_labelled = min(len(sbp_per_seg), len(dbp_per_seg))
        num_segments = min(num_segments_raw, num_labelled)
        abp_all = abp_all[:num_segments]
        sbp_per_seg = sbp_per_seg[:num_segments]
        dbp_per_seg = dbp_per_seg[:num_segments]

        # Demographics arrays
        def _num_or_fill(name):
            arr = subset[name][:]
            if arr.shape[0] == 1:
                return None, float(np.mean(arr))
            return np.squeeze(arr)[:num_segments], None

        age_arr, age_fill = _num_or_fill("Age")
        bmi_arr, bmi_fill = _num_or_fill("BMI")
        hgt_arr, hgt_fill = _num_or_fill("Height")
        wgt_arr, wgt_fill = _num_or_fill("Weight")
        gdr_raw = subset["Gender"][:]

        def fill_or_take(a_arr, a_fill):
            return np.full(num_segments, a_fill) if a_arr is None else a_arr

        age = fill_or_take(age_arr, age_fill)
        bmi = fill_or_take(bmi_arr, bmi_fill)
        hgt = fill_or_take(hgt_arr, hgt_fill)
        wgt = fill_or_take(wgt_arr, wgt_fill)

        # Normalize gender
        if gdr_raw.shape[0] == 1:
            first_ref = gdr_raw[0, 0] if gdr_raw.ndim == 2 else gdr_raw[0]
            g = _read_mat_str(f, first_ref)
            g_num = 1.0 if g.upper().startswith("M") else 0.0
            gender = np.full(num_segments, g_num)
        else:
            if gdr_raw.dtype == object or gdr_raw.dtype.kind == "O":
                gender = np.zeros(num_segments, dtype=float)
                for i in range(num_segments):
                    ref = gdr_raw[i] if gdr_raw.ndim == 1 else gdr_raw[i, 0]
                    g = _read_mat_str(f, ref)
                    gender[i] = 1.0 if g.upper().startswith("M") else 0.0
            else:
                gender = (np.squeeze(gdr_raw[:num_segments]) == b"M").astype(float)

        demographics = np.column_stack([age, bmi, gender, hgt, wgt])

        # Group segments by a "subject proxy"
        groups = defaultdict(list)
        for i in range(num_segments):
            demog_key = tuple(np.round(demographics[i], 3))
            groups[demog_key].append(i)

        unique_subjects = list(groups.keys())
        if len(unique_subjects) > max_subjects:
            unique_subjects = random.sample(unique_subjects, max_subjects)

        # Subject name prefix
        subject_prefix = "subject"
        if "Subject" in subset:
            try:
                refs = np.squeeze(subset["Subject"][:])
                first = refs[0].item() if isinstance(refs[0], np.ndarray) else refs[0]
                subject_prefix = _read_mat_str(f, first) or "subject"
            except Exception:
                pass

        # Build outputs per selected subject
        for subj_idx, key in enumerate(unique_subjects, start=1):
            seg_indices = groups[key]
            chosen = seg_indices if len(seg_indices) <= segs_per_subject else random.sample(seg_indices, segs_per_subject)

            abp = np.empty((len(chosen), samples_per_seg), dtype=abp_all.dtype)
            sbp = np.empty((len(chosen),), dtype=sbp_per_seg.dtype)
            dbp = np.empty((len(chosen),), dtype=dbp_per_seg.dtype)

            for j, i in enumerate(chosen):
                seg = abp_all[i]
                if seg.shape[-1] >= samples_per_seg:
                    abp[j] = seg[:samples_per_seg]
                else:
                    pad = np.zeros(samples_per_seg, dtype=seg.dtype)
                    pad[: seg.shape[-1]] = seg
                    abp[j] = pad
                sbp[j] = sbp_per_seg[i]
                dbp[j] = dbp_per_seg[i]

            all_abp_segments.append(abp)
            all_sbp_segments.append(sbp)
            all_dbp_segments.append(dbp)
            all_demographics.append(np.array(key))
            subject_ids.append(f"{subject_prefix}_{subj_idx}")

    X = all_abp_segments               # list of arrays, each (k, samples_per_seg)
    Y_sbp = all_sbp_segments           # list of arrays, each (k,)
    Y_dbp = all_dbp_segments           # list of arrays, each (k,)
    DEMO = np.vstack(all_demographics) # (num_subjects, 5)  Age,BMI,Gender(1=M),Height,Weight
    IDS = np.array(subject_ids)

    return X, Y_sbp, Y_dbp, DEMO, IDS


# ---------- Entry point (hardcoded to C:\PulseTemp) ----------
if __name__ == "__main__":
    mat_file = r"C:\PulseTemp\VitalDB_CalBased_Test_Subset.mat"   # <-- your local file
    X, Y_sbp, Y_dbp, DEMO, IDS = load_vitaldb_mat(
        mat_file,
        channel_index=2,
        samples_per_seg=625,
        max_subjects=100,
        segs_per_subject=10,
        seed=42
    )
    total_subjects = len(X)
    total_segments = sum(len(x) for x in X)

    print("\n--- Summary ---")
    print(f"Subjects: {total_subjects}")
    print(f"Total segments: {total_segments}")
    print(f"First subject ABP shape: {X[0].shape if total_subjects else 'N/A'}")
    print(f"First subject SBP/DBP lens: "
          f"{len(Y_sbp[0]) if total_subjects else 0} / {len(Y_dbp[0]) if total_subjects else 0}")
    print(f"Demographics shape: {DEMO.shape}  (Age, BMI, Gender(1=M), Height, Weight)")
    print(f"Sample IDs: {IDS[:min(5, len(IDS))]}")
