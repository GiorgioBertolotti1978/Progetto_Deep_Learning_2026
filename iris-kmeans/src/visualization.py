"""Grafici statici in italiano, salvati senza finestre interattive."""

from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

COLORS = ["#0072B2", "#D55E00", "#009E73"]


def create_plots(X, y, species_names, model, scaler, contingency, elbow, plots_dir):
    """Le figure in 2D proiettano un clustering calcolato in quattro dimensioni."""
    plots_dir = Path(plots_dir)
    plots_dir.mkdir(parents=True, exist_ok=True)
    centers_cm = scaler.inverse_transform(model.cluster_centers_)
    plt.rcParams.update({"font.size": 10, "figure.dpi": 110})

    def save(fig, filename):
        fig.tight_layout()
        fig.savefig(plots_dir / filename, dpi=160, bbox_inches="tight")
        plt.close(fig)

    def petal_axes(ax):
        ax.set_xlabel("Lunghezza petalo (cm)")
        ax.set_ylabel("Larghezza petalo (cm)")
        ax.grid(alpha=0.2)

    fig, ax = plt.subplots(figsize=(8, 5.5))
    for cluster, color in enumerate(COLORS):
        members = model.labels_ == cluster
        ax.scatter(X[members, 2], X[members, 3], color=color, alpha=0.7,
                   edgecolors="white", linewidths=0.4, label=f"Cluster {cluster}")
        ax.scatter(centers_cm[cluster, 2], centers_cm[cluster, 3], color=color,
                   marker="X", s=210, edgecolors="black", zorder=5)
    ax.scatter([], [], marker="X", s=100, color="gray", edgecolors="black", label="Centroidi")
    ax.set_title("K-means: cluster in 4D, proiezione sui petali")
    petal_axes(ax)
    ax.legend()
    save(fig, "01_kmeans_clusters.png")

    fig, ax = plt.subplots(figsize=(8, 5.5))
    for species, color in enumerate(COLORS):
        ax.scatter(X[y == species, 2], X[y == species, 3], color=color,
                   alpha=0.7, edgecolors="white", linewidths=0.4,
                   label=f"Iris {species_names[species]}")
    ax.set_title("Specie reali: riferimento per la valutazione")
    petal_axes(ax)
    ax.legend()
    save(fig, "02_real_species.png")

    fig, ax = plt.subplots(figsize=(8, 5))
    ax.plot(elbow["k"], elbow["inertia"], "o-", color=COLORS[0])
    ax.set(xticks=elbow["k"], xlabel="Numero di cluster K",
           ylabel="Inerzia (SSE nello spazio standardizzato)",
           title="Metodo del gomito: migliore di 20 inizializzazioni per K")
    ax.grid(alpha=0.25)
    save(fig, "03_elbow_method.png")

    fig, ax = plt.subplots(figsize=(8, 5))
    ax.plot(range(len(model.inertia_history_)), model.inertia_history_, "o-", color=COLORS[1])
    ax.set(xlabel="Aggiornamenti dei centroidi (0 = inizializzazione)",
           ylabel="Inerzia (SSE nello spazio standardizzato)",
           title="Convergenza del tentativo selezionato, K = 3")
    ax.grid(alpha=0.25)
    save(fig, "04_convergence.png")

    fig, ax = plt.subplots(figsize=(7, 5))
    sizes = np.bincount(model.labels_, minlength=3)
    bars = ax.bar([f"Cluster {i}" for i in range(3)], sizes, color=COLORS)
    ax.bar_label(bars, padding=3)
    ax.set(ylabel="Numero di campioni", title="Dimensioni dei cluster", ylim=(0, max(sizes) * 1.18))
    save(fig, "05_cluster_sizes.png")

    fig, ax = plt.subplots(figsize=(7, 5))
    heatmap = ax.imshow(contingency.to_numpy(), cmap="Blues")
    ax.set(xticks=range(3), xticklabels=[f"Cluster {i}" for i in range(3)],
           yticks=range(3), yticklabels=[f"Iris {s}" for s in species_names],
           xlabel="Cluster individuati", ylabel="Specie reali",
           title="Tabella di contingenza specie–cluster")
    for row in range(3):
        for col in range(3):
            count = int(contingency.iloc[row, col])
            ax.text(col, row, str(count), ha="center", va="center",
                    color="white" if count > contingency.to_numpy().max() / 2 else "black")
    fig.colorbar(heatmap, ax=ax, label="Numero di campioni")
    save(fig, "06_contingency_matrix.png")

    fig, ax = plt.subplots(figsize=(8, 5.5))
    ax.scatter(X[:, 2], X[:, 3], color="lightgray", s=25, alpha=0.6)
    # Ogni stato viene riportato in centimetri con lo stesso scaler del fit.
    history_cm = np.array([scaler.inverse_transform(c) for c in model.centroids_history_])
    for cluster, color in enumerate(COLORS):
        path = history_cm[:, cluster, 2:4]
        ax.plot(path[:, 0], path[:, 1], "o-", color=color, markersize=3,
                label=f"Centroide {cluster}", zorder=3)
        for start, end in zip(path[:-1], path[1:]):
            ax.annotate("", xy=end, xytext=start,
                        arrowprops={"arrowstyle": "->", "color": color})
        ax.scatter(*path[0], marker="s", s=80, color=color, edgecolors="black", zorder=4)
        ax.scatter(*path[-1], marker="X", s=140, color=color, edgecolors="black", zorder=5)
    ax.scatter([], [], marker="s", color="gray", label="Inizio")
    ax.scatter([], [], marker="X", color="gray", label="Fine")
    petal_axes(ax)
    ax.set_title("Movimento dei centroidi: proiezione 2D di traiettorie in 4D")
    ax.legend(fontsize=9)
    save(fig, "07_centroid_movement.png")
