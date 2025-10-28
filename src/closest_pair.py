# closest_pair.py
import numpy as np

# inline similarity helpers
def euclidean(a, b):
    a = np.asarray(a); b = np.asarray(b)
    return float(np.linalg.norm(a - b))

def corr_distance(a, b):
    a = np.asarray(a); b = np.asarray(b)
    na, nb = a - a.mean(), b - b.mean()
    r = float(np.dot(na, nb) / ((np.linalg.norm(na) * np.linalg.norm(nb)) + 1e-8))
    return 1.0 - r

def dtw(a, b, w=None):
    a = np.asarray(a); b = np.asarray(b)
    n, m = len(a), len(b)
    w = max(w or max(n, m), abs(n - m))
    INF = 1e18
    D = np.full((n + 1, m + 1), INF); D[0, 0] = 0.0
    for i in range(1, n + 1):
        j0 = max(1, i - w); j1 = min(m, i + w)
        for j in range(j0, j1 + 1):
            cost = (a[i - 1] - b[j - 1]) ** 2
            D[i, j] = cost + min(D[i - 1, j], D[i, j - 1], D[i - 1, j - 1])
    return float(np.sqrt(D[n, m]))

def find_closest_pair(segments, metric="dtw", window=50):

    METR = {"dtw": dtw, "euclidean": euclidean, "corr": corr_distance}[metric]
    use_w = (metric == "dtw")

    n = len(segments)
    best = (None, None, float("inf"))
    for i in range(n):
        for j in range(i + 1, n):
            d = METR(segments[i], segments[j], w=window) if use_w else METR(segments[i], segments[j])
            if d < best[2]:
                best = (i, j, d)
    return {"i": best[0], "j": best[1], "dist": best[2]}
