# clustering.py
import numpy as np

# keep your inline similarity helpers here (zscore, euclidean, corr_distance, dtw)
def zscore(x, eps=1e-8):
    x = np.asarray(x, dtype=float)
    return (x - x.mean()) / (x.std() + eps)

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

def recursive_clustering(
    segments,
    metric: str = "dtw",
    window: int = 50,
    max_depth: int = 6,
    min_cluster: int = 20,
    tol: float = 0.15,
    pivot_sample: int = 12,
):

    METR = {"dtw": dtw, "euclidean": euclidean, "corr": corr_distance}[metric]
    use_w = (metric == "dtw")

    X = [zscore(s) for s in segments]
    N = len(X)
    tree, clusters = [], []

    # distance cache to avoid recomputing the same DTWs
    dist_cache = {}
    def dist(a_idx, b_idx):
        i, j = (a_idx, b_idx) if a_idx <= b_idx else (b_idx, a_idx)
        key = (i, j)
        if key in dist_cache:
            return dist_cache[key]
        d = METR(X[i], X[j], w=window) if use_w else METR(X[i], X[j])
        dist_cache[key] = d
        return d

    def avg_dist_to_all(idxs, pivot_idx):
        s = 0.0; c = 0
        for k in idxs:
            if k == pivot_idx: continue
            s += dist(k, pivot_idx); c += 1
        return s / max(c, 1)

    def spread(idxs):
        # approximate spread via medoid: pick a small sample to estimate medoid
        if len(idxs) < 2: return 0.0
        samp = idxs if len(idxs) <= pivot_sample else np.random.choice(idxs, pivot_sample, replace=False)
        # pick candidate with minimal average distance to sample
        best_med = None; best_score = float("inf")
        for i in samp:
            s = 0.0
            for j in samp:
                if i == j: continue
                s += dist(i, j)
            sc = s / max(len(samp) - 1, 1)
            if sc < best_score:
                best_score, best_med = sc, i
        # compute avg distance to that medoid (cheap)
        return avg_dist_to_all(idxs, best_med)

    def split_once(idxs):
        n = len(idxs)
        if n <= 2:
            return idxs, []
        # choose pivot among a small random subset
        cand = idxs if n <= pivot_sample else np.random.choice(idxs, pivot_sample, replace=False)
        best_pivot, best_avg = cand[0], -1.0
        for p in cand:
            ad = avg_dist_to_all(idxs, p)
            if ad > best_avg:
                best_avg = ad; best_pivot = p
        # split by distance-to-pivot median
        dists = [dist(k, best_pivot) for k in idxs]
        thr = float(np.median(dists))
        L = [k for k, d in zip(idxs, dists) if d <= thr]
        R = [k for k, d in zip(idxs, dists) if d >  thr]
        return L, R

    stack = [(0, list(range(N)), 0)]
    while stack:
        node, idxs, depth = stack.pop()
        sp = spread(idxs)
        leaf = (depth >= max_depth) or (len(idxs) <= min_cluster) or (sp <= tol)
        tree.append({"node": node, "depth": depth, "size": len(idxs), "spread": sp,
                     "leaf": leaf, "indices": idxs})
        if leaf:
            clusters.append(idxs); continue
        L, R = split_once(idxs)
        if not L or not R or L == idxs or R == idxs:
            clusters.append(idxs); continue
        stack.append((node*2+2, R, depth+1))
        stack.append((node*2+1, L, depth+1))

    labels = np.empty(N, dtype=int)
    for ci, idxs in enumerate(clusters):
        for i in idxs: labels[i] = ci
    return {"clusters": clusters, "labels": labels, "tree": tree}
