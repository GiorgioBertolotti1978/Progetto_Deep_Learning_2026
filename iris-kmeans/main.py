"""Esegue tutti gli esperimenti Iris e salva risultati riproducibili."""

import json
from importlib.metadata import version
from pathlib import Path

import numpy as np
import pandas as pd

from src.data_loader import FEATURE_COLUMNS, load_and_export_iris
from src.evaluation import evaluate_clustering
from src.kmeans import fit_best_kmeans
from src.visualization import create_plots

ROOT = Path(__file__).resolve().parent


def main():
    """Pipeline senza input interattivi; percorsi relativi al file main.py."""
    results_dir = ROOT / "results"
    results_dir.mkdir(parents=True, exist_ok=True)
    X, X_scaled, y, species_names, scaler = load_and_export_iris(ROOT / "data")

    # L'algoritmo e la scelta della migliore inizializzazione vedono solo X.
    model = fit_best_kmeans(X_scaled, n_clusters=3, n_init=20, random_state=42)
    scores, contingency = evaluate_clustering(X_scaled, y, model.labels_, species_names, 3)

    elbow_rows = []
    for k in range(1, 11):
        candidate = model if k == 3 else fit_best_kmeans(X_scaled, n_clusters=k, n_init=20, random_state=42)
        elbow_rows.append({"k": k, "inertia": candidate.inertia_, "n_iter": candidate.n_iter_})
    elbow = pd.DataFrame(elbow_rows)
    elbow.to_csv(results_dir / "elbow_results.csv", index=False)
    contingency.to_csv(results_dir / "contingency_matrix.csv")

    assignments = pd.DataFrame(X, columns=FEATURE_COLUMNS)
    assignments.insert(0, "campione_id", np.arange(len(X)))
    assignments["cluster"] = model.labels_
    assignments["specie_id"] = y
    assignments["specie_reale"] = species_names[y]
    assignments.to_csv(results_dir / "cluster_assignments.csv", index=False)

    centers_cm = scaler.inverse_transform(model.cluster_centers_)
    centers_table = pd.DataFrame(centers_cm, columns=FEATURE_COLUMNS)
    centers_table.insert(0, "cluster", np.arange(3))
    for index, name in enumerate(FEATURE_COLUMNS):
        centers_table[name.removesuffix("_cm") + "_standardizzata"] = model.cluster_centers_[:, index]
    centers_table.to_csv(results_dir / "centroids.csv", index=False)
    pd.DataFrame({"iterazione": range(len(model.inertia_history_)), "inertia": model.inertia_history_}).to_csv(
        results_dir / "convergence.csv", index=False)
    history_rows = []
    for iteration, centers in enumerate(model.centroids_history_):
        for cluster, center in enumerate(centers):
            history_rows.append({"iterazione": iteration, "cluster": cluster,
                                 **dict(zip(FEATURE_COLUMNS, scaler.inverse_transform(center.reshape(1, -1))[0]))})
    pd.DataFrame(history_rows).to_csv(results_dir / "centroids_history.csv", index=False)

    metrics = {
        "n_samples": len(X), "n_features": X.shape[1], "n_clusters": 3,
        "n_init": 20, "random_state": 42, "best_initialization_seed": model.random_state,
        "max_iter": model.max_iter, "tol": model.tol,
        "n_iter": model.n_iter_, "converged": model.converged_, "inertia": model.inertia_,
        **scores,
        "cluster_sizes": np.bincount(model.labels_, minlength=3).tolist(),
        "centroids_standardized": model.cluster_centers_.tolist(),
        "centroids_cm": centers_cm.tolist(),
        "inertia_history": model.inertia_history_,
        "initialization_inertias": model.initialization_inertias_,
        "initialization_seeds": model.initialization_seeds_,
        "empty_cluster_relocations": model.empty_cluster_relocations_,
        "scaler_mean": scaler.mean_.tolist(), "scaler_scale": scaler.scale_.tolist(),
        "versions": {package: version(package) for package in ["numpy", "pandas", "scikit-learn", "matplotlib"]},
    }
    (results_dir / "metrics.json").write_text(
        json.dumps(metrics, ensure_ascii=False, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    create_plots(X, y, species_names, model, scaler, contingency, elbow, results_dir / "plots")

    print("CLUSTERING IRIS — K-means manuale, 4 feature standardizzate")
    print(f"Campioni: {len(X)} | K: 3 | Inizializzazioni: 20 | Seed generale: 42")
    print(f"Iterazioni del tentativo migliore: {model.n_iter_} | Convergenza: {model.converged_}")
    print(f"Inerzia: {model.inertia_:.6f}")
    print(f"Silhouette: {scores['silhouette_score']:.6f}")
    print(f"Purezza: {scores['purity']:.6f} | ARI: {scores['adjusted_rand_index']:.6f}")
    print("\nCentroidi in unità originali (cm) e standardizzate:")
    print(centers_table.to_string(index=False, float_format=lambda number: f"{number:.4f}"))
    print("\nCampioni per cluster:", metrics["cluster_sizes"])
    print("\nTabella di contingenza (etichette usate solo per valutare):")
    print(contingency.to_string())
    print(f"\nRisultati: {results_dir}\nGrafici: {results_dir / 'plots'}")


if __name__ == "__main__":
    main()
