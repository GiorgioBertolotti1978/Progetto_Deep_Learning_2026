"""Caricamento, esportazione e standardizzazione del dataset Iris."""

from pathlib import Path

import pandas as pd
from sklearn.datasets import load_iris
from sklearn.preprocessing import StandardScaler

FEATURE_COLUMNS = [
    "lunghezza_sepalo_cm", "larghezza_sepalo_cm",
    "lunghezza_petalo_cm", "larghezza_petalo_cm",
]
FEATURE_LABELS = [
    "Lunghezza sepalo (cm)", "Larghezza sepalo (cm)",
    "Lunghezza petalo (cm)", "Larghezza petalo (cm)",
]


def load_and_export_iris(data_dir):
    """Esporta feature ed etichette in file separati, uniti da campione_id."""
    data_dir = Path(data_dir)
    data_dir.mkdir(parents=True, exist_ok=True)
    iris = load_iris()
    X = iris.data.copy()
    y = iris.target.copy()
    features = pd.DataFrame(X, columns=FEATURE_COLUMNS)
    features.index.name = "campione_id"
    features.to_csv(data_dir / "iris_features.csv")
    pd.DataFrame({
        "campione_id": features.index,
        "specie_id": y,
        "specie": iris.target_names[y],
    }).to_csv(data_dir / "iris_labels.csv", index=False)
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    return X, X_scaled, y, iris.target_names, scaler
