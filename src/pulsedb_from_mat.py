# pulsedb_from_mat.py
import os, random, h5py, numpy as np
from collections import defaultdict

def _read_str(f, ref):
    """Decode a MATLAB HDF5 string/char/ref robustly."""
    try:
        data = f[ref][()]
    except Exception:
        data = ref
    if isinstance(data, (bytes, bytearray)):
        return data.decode("utf-8", errors="ignore").rstrip("\x00").strip()
    if isinstance(data, np.ndarray):
        try:
            return data.tobytes().decode("utf-8", errors="ignore").rstrip("\x00").strip()
        except Exception:
            if data.dtype.kind == "O":
                try:
                    return _read_str(f, data[0])
                except Exception:
                    return ""
    return (str(data) if data is not None else "").strip()

def load_mat(mat_path,
             channel_index=2,
             samples_per_seg=625,
             max_subjects=100,
             segs_per_subject=10,
             max_total_segments=1000,
             seed=42,
             **_):

    assert os.path.isfile(mat_path), f"File not found: {mat_path}"
    rng = random.Random(seed)

    all_X, all_SBP, all_DBP = [], [], []
    demos, ids = [], []

    with h5py.File(mat_path, "r") as f:
        S = f["Subset"]
        sig = S["Signals"]                                   # (N, C, T)
        N_raw = sig.shape[0]
        abp = sig[:, channel_index, :]

        sbp = np.squeeze(S["SBP"][:])
        dbp = np.squeeze(S["DBP"][:])
        N = min(N_raw, len(sbp), len(dbp))
        abp, sbp, dbp = abp[:N], sbp[:N], dbp[:N]

        # demographics: per-dataset or per-segment
        def _num_or_fill(name):
            arr = S[name][:]
            if arr.shape[0] == 1:
                return None, float(np.mean(arr))
            return np.squeeze(arr)[:N], None

        def _fill_or_take(arr, fill):
            return np.full(N, fill) if arr is None else arr

        age = _fill_or_take(*_num_or_fill("Age"))
        bmi = _fill_or_take(*_num_or_fill("BMI"))
        hgt = _fill_or_take(*_num_or_fill("Height"))
        wgt = _fill_or_take(*_num_or_fill("Weight"))

        graw = S["Gender"][:]
        if graw.shape[0] == 1:
            ref = graw[0, 0] if graw.ndim == 2 else graw[0]
            g = _read_str(f, ref)
            gnum = 1.0 if g.upper().startswith("M") else 0.0
            gen = np.full(N, gnum)
        else:
            if graw.dtype.kind == "O":
                gen = np.zeros(N)
                for i in range(N):
                    ref = graw[i] if graw.ndim == 1 else graw[i, 0]
                    g = _read_str(f, ref)
                    gen[i] = 1.0 if str(g).upper().startswith("M") else 0.0
            else:
                gen = (np.squeeze(graw[:N]) == b"M").astype(float)

        demo = np.column_stack([age, bmi, gen, hgt, wgt])

        # group segments by "subject proxy" = rounded demographics
        groups = defaultdict(list)
        for i in range(N):
            key = tuple(np.round(demo[i], 3))
            groups[key].append(i)

        subjects = list(groups.keys())
        if len(subjects) > max_subjects:
            subjects = rng.sample(subjects, max_subjects)

        # subject id prefix (optional)
        subj_prefix = "subject"
        if "Subject" in S:
            try:
                refs = np.squeeze(S["Subject"][:])
                first = refs[0].item() if isinstance(refs[0], np.ndarray) else refs[0]
                name = _read_str(f, first)
                subj_prefix = "".join(ch for ch in name if 32 <= ord(ch) <= 126) or "subject"
            except Exception:
                pass

        added = 0
        for si, key in enumerate(subjects, start=1):
            idxs = groups[key]
            k = min(segs_per_subject, len(idxs))
            idxs = idxs if len(idxs) <= k else rng.sample(idxs, k)

            # cap by global max_total_segments
            if added + len(idxs) > max_total_segments:
                idxs = idxs[: max_total_segments - added]
            if not idxs:
                break

            X = np.empty((len(idxs), samples_per_seg), dtype=abp.dtype)
            SBP = np.empty(len(idxs)); DBP = np.empty(len(idxs))
            for j, i in enumerate(idxs):
                s = abp[i]
                if s.shape[-1] >= samples_per_seg:
                    X[j] = s[:samples_per_seg]
                else:
                    pad = np.zeros(samples_per_seg, dtype=s.dtype)
                    pad[: s.shape[-1]] = s
                    X[j] = pad
                SBP[j] = sbp[i]; DBP[j] = dbp[i]

            all_X.append(X); all_SBP.append(SBP); all_DBP.append(DBP)
            demos.append(np.array(key)); ids.append(f"{subj_prefix}_{si}")
            added += len(idxs)
            if added >= max_total_segments:
                break

    DEMO = np.vstack(demos) if demos else np.zeros((0,5))
    IDS = np.array(ids)
    return all_X, all_SBP, all_DBP, DEMO, IDS
