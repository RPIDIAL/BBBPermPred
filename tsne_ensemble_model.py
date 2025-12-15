from sklearn.preprocessing import StandardScaler
from sklearn.manifold import TSNE, trustworthiness
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import os

base_dir    = "Outputs_Sept2025/embeddings/trained_on_B3DB"
types       = ["Descriptors"] #, "Embeddings", "Combined"
datasets    = ["BBBP", "B3DB", "NeuTox"]

# final color palette (impermeable=0, permeable=1)
colors  = {0: "#1f77b4", 1: "#ff7f0e"}

rng = np.random.default_rng(0)   # reproducibility for any random sampling you do

for t in types:
    for ds in datasets:
        path   = os.path.join(base_dir, ds, "test", "0", f"{t}.csv")
        y_path = os.path.join(base_dir, ds, "test", "0", "labels.csv")

        df = pd.read_csv(path)

        # numeric features only
        X_df = df.select_dtypes(include=np.number)
        X    = X_df.values
        y    = pd.read_csv(y_path).values.flatten()

        # basic shape sanity check
        if len(X) != len(y):
            raise ValueError(f"Length mismatch for {t}/{ds}: X={len(X)} vs y={len(y)}")

        # scale
        Xs = StandardScaler().fit_transform(X)

        n_samples = len(Xs)
        # t-SNE requires 5 < perplexity < n_samples-1
        base_perp = min(50, max(5, n_samples // 30))  # ~3–5% of n
        if n_samples <= 6:
            # fall back to a tiny perplexity if sample size is very small
            base_perp = max(2, n_samples // 2)

        tsne = TSNE(
            n_components=2,
            perplexity=min(base_perp, max(2, n_samples - 1) - 1),
            metric="cosine",
            n_iter=2_000,
            init="pca",
            random_state=0,
            learning_rate="auto"
        )
        coords = tsne.fit_transform(Xs)  # (n_samples, 2)

        # Build a DataFrame to compute outliers on coords only (exclude y!)
        coords_df = pd.DataFrame(coords, columns=["t-SNE 1", "t-SNE 2"])

        # Remove outliers via z-score on the two t-SNE axes
        z = (coords_df - coords_df.mean()) / coords_df.std(ddof=0)
        z = z.abs()
        mask = (z < 3).all(axis=1)  # within 3 std dev on both axes

        # Apply the same mask to everything (this was the main bug before)
        coords_f = coords[mask]
        Xs_f     = Xs[mask]
        y_f      = y[mask]

        # Make sure trustworthiness args line up and neighbors are valid
        n_keep = len(Xs_f)
        if n_keep < 3:
            print(f"{t}/{ds}: too few points after outlier removal (n={n_keep}); skipping plot.")
            continue

        n_neighbors = max(2, min(base_perp, n_keep - 2))
        trust = trustworthiness(Xs_f, coords_f, n_neighbors=n_neighbors, metric="cosine")

        print(f"{t}/{ds}: trust={trust:.3f}, perp={min(base_perp, n_samples-2)} (neighbors={n_neighbors}, kept {n_keep}/{n_samples})")

        plt.figure(figsize=(6, 5))
        for lab in (0, 1):
            m = (y_f == lab)
            if m.any():
                plt.scatter(
                    coords_f[m, 0], coords_f[m, 1],
                    label="Permeable" if lab == 1 else "Impermeable",
                    marker='o',
                    c=colors[lab],
                    alpha=0.6,
                    edgecolors="none"
                )
        plt.title(f"{t}/{ds} • trust={trust:.3f}")
        plt.xlabel("t-SNE 1")
        plt.ylabel("t-SNE 2")
        plt.legend(title="BBB Permeability")
        plt.tight_layout()

        output_dir = "Outputs_Sept2025/tsnes"
        os.makedirs(output_dir, exist_ok=True)
        out_path = os.path.join(output_dir, f"tsne_{t}_{ds}.png")
        plt.savefig(out_path, dpi=660)
        plt.close()
