"""Metriche interne ed esterne, separate dall'addestramento."""

import numpy as np
import pandas as pd
from sklearn.metrics import adjusted_rand_score, silhouette_score


def evaluate_clustering(X_scaled, y, labels, species_names, n_clusters):
    """Calcola purezza per colonna, ARI e silhouette euclidea in 4D."""
    table = pd.crosstab(
        pd.Series(np.asarray(species_names)[y], name="specie"),
        pd.Series(labels, name="cluster"),
    ).reindex(index=species_names, columns=range(n_clusters), fill_value=0)
    purity = float(table.max(axis=0).sum() / len(y))
    n_occupied = len(np.unique(labels))
    silhouette = (
        float(silhouette_score(X_scaled, labels))
        if 1 < n_occupied < len(labels) else None
    )
    metrics = {
        "silhouette_score": silhouette,
        "purity": purity,
        "adjusted_rand_index": float(adjusted_rand_score(y, labels)),
    }
    return metrics, table
