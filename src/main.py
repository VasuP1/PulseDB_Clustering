import os, argparse, numpy as np, matplotlib.pyplot as plt
from clustering import recursive_clustering
from closest_pair import find_closest_pair
from kadane import kadane_algorithm
from pulsedb_from_mat import load_mat

def plot_example(seg, outpath, kadane=True):
    import numpy as np
    import matplotlib.pyplot as plt
    plt.figure()
    plt.plot(seg)
    if kadane:
        ssum, s, e = kadane_algorithm(seg)  # now returns (sum, start, end)
        s = max(0, min(len(seg)-1, s))
        e = max(s, min(len(seg)-1, e))
        xs = np.arange(s, e+1)
        plt.plot(xs, seg[s:e+1])
        plt.title(f"Kadane sum={ssum:.2f} [{s},{e}]")
    plt.xlabel("sample"); plt.ylabel("amp")
    plt.savefig(outpath); plt.close()

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--mat", required=False, default=r"C:\PulseTemp\VitalDB_CalBased_Test_Subset.mat")
    ap.add_argument("--out", default="results")
    ap.add_argument("--max_segments", type=int, default=1000)
    ap.add_argument("--metric", choices=["dtw","euclidean","corr"], default="dtw")
    ap.add_argument("--window", type=int, default=50)
    ap.add_argument("--max_depth", type=int, default=6)
    ap.add_argument("--min_cluster", type=int, default=20)
    ap.add_argument("--tol", type=float, default=0.15)
    ap.add_argument("--segs_per_subject", type=int, default=50)
    args = ap.parse_args()

    os.makedirs(args.out, exist_ok=True)

    X_list, SBP_list, DBP_list, DEMO, IDS = load_mat(
        args.mat,
        channel_index=2,
        samples_per_seg=625,
        max_subjects=100,
        segs_per_subject=args.segs_per_subject,  # <— here
        max_total_segments=args.max_segments,
        seed=42
    )

    X = [seg for subj in X_list for seg in subj]
    print(f"Loaded segments: {len(X)}")

    res = recursive_clustering(X, metric=args.metric, window=args.window,
                               max_depth=args.max_depth, min_cluster=args.min_cluster, tol=args.tol)
    clusters = res["clusters"]
    print(f"Clusters: {len(clusters)}")

    # Save cluster sizes
    with open(os.path.join(args.out, "clusters.txt"), "w") as f:
        for ci, idxs in enumerate(clusters):
            f.write(f"Cluster {ci}: size={len(idxs)}\n")

    # Examples + closest-pair + Kadane overlays for first few clusters
    for ci, idxs in enumerate(clusters[:3]):
        cdir = os.path.join(args.out, f"c{ci}"); os.makedirs(cdir, exist_ok=True)
        for k in idxs[:3]:
            plot_example(X[k], os.path.join(cdir, f"ex_{k}.png"))
        cp = find_closest_pair([X[k] for k in idxs], metric=args.metric, window=args.window)
        for tag, local_idx in [("a", cp["i"]), ("b", cp["j"])]:
            plot_example(X[idxs[local_idx]], os.path.join(cdir, f"closest_{tag}.png"))

    np.save(os.path.join(args.out, "X_flat.npy"), np.array(X, dtype=object))
    print("Done.")

if __name__ == "__main__":
    main()
