# kadane.py
import numpy as np

def kadane_algorithm(arr, use_diff=True):

    x = np.diff(np.asarray(arr, dtype=float)) if use_diff else np.asarray(arr, dtype=float)

    best = -1e18
    cur = 0.0
    s = 0
    best_s = 0
    best_e = -1  # no segment yet

    for i, val in enumerate(x):
        if cur <= 0:
            cur = val
            s = i
        else:
            cur += val
        if cur > best:
            best = cur
            best_s = s
            best_e = i

    if best_e < best_s:
        # fallback: empty improvement; pick a degenerate 0-length interval
        best_s = best_e = 0

    # map indices back to original signal coordinates if we used diff
    if use_diff:
        # diff index i corresponds to transition between arr[i] and arr[i+1]
        # choose original segment [best_s, best_e+1] inclusive
        return float(best), int(best_s), int(best_e + 1)
    else:
        return float(best), int(best_s), int(best_e)
