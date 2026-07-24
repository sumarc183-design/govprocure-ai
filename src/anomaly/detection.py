"""
Détection d'anomalies sur les marchés publics.

Deux méthodes comparées en profondeur (plutôt que plusieurs traitées
superficiellement) :
- Isolation Forest : isole les points en les séparant récursivement par
  des coupures aléatoires ; un point isolé rapidement (peu de coupures
  nécessaires) est considéré comme anormal. Rapide, scalable, bon pour
  les gros volumes.
- Local Outlier Factor (LOF) : compare la densité locale d'un point à
  celle de ses voisins. Détecte des anomalies "locales" (un marché
  normal dans l'absolu mais atypique dans son contexte) que Isolation
  Forest peut manquer.

Important : une anomalie détectée ici n'est PAS présentée comme une
fraude. Il s'agit uniquement d'une priorisation des dossiers à examiner.
"""

from __future__ import annotations

import pandas as pd
from sklearn.ensemble import IsolationForest
from sklearn.neighbors import LocalOutlierFactor
from sklearn.preprocessing import StandardScaler

from src.anomaly.features import build_feature_matrix

RANDOM_STATE = 42


def _scale_features(df_features: pd.DataFrame) -> pd.DataFrame:
    """Standardise les features (moyenne 0, écart-type 1).

    Nécessaire car montant et durée ont des échelles très différentes ;
    sans ça, les modèles seraient dominés par le montant.
    """
    scaler = StandardScaler()
    scaled = scaler.fit_transform(df_features)
    return pd.DataFrame(scaled, columns=df_features.columns, index=df_features.index)


def detect_isolation_forest(
    df: pd.DataFrame, contamination: float = 0.05
) -> pd.DataFrame:
    """Détecte les anomalies avec Isolation Forest.

    contamination : proportion attendue d'anomalies dans les données
    (0.05 = 5%, à ajuster selon le contexte métier).

    Ajoute deux colonnes :
    - anomaly_score_iforest : score brut (plus négatif = plus anormal)
    - is_anomaly_iforest : booléen selon le seuil de contamination
    """
    df, feature_cols = build_feature_matrix(df)
    X = _scale_features(df[feature_cols])

    model = IsolationForest(
        contamination=contamination, random_state=RANDOM_STATE, n_jobs=-1
    )
    predictions = model.fit_predict(X)
    scores = model.score_samples(X)

    df = df.copy()
    df["anomaly_score_iforest"] = scores
    df["is_anomaly_iforest"] = predictions == -1
    return df


def detect_lof(
    df: pd.DataFrame, contamination: float = 0.05, n_neighbors: int = 20
) -> pd.DataFrame:
    """Détecte les anomalies avec Local Outlier Factor.

    n_neighbors : nombre de voisins utilisés pour estimer la densité locale.
    Attention : LOF est plus coûteux en calcul que Isolation Forest
    (complexité proche de O(n log n) à O(n²) selon l'implémentation) —
    à utiliser sur un échantillon si le dataset est très volumineux.

    Ajoute deux colonnes :
    - anomaly_score_lof : score brut (plus négatif = plus anormal)
    - is_anomaly_lof : booléen selon le seuil de contamination
    """
    df, feature_cols = build_feature_matrix(df)
    X = _scale_features(df[feature_cols])

    model = LocalOutlierFactor(
        n_neighbors=n_neighbors, contamination=contamination, n_jobs=-1
    )
    predictions = model.fit_predict(X)
    scores = model.negative_outlier_factor_

    df = df.copy()
    df["anomaly_score_lof"] = scores
    df["is_anomaly_lof"] = predictions == -1
    return df


def compare_methods(df: pd.DataFrame, contamination: float = 0.05) -> dict:
    """Compare Isolation Forest et LOF sur le même dataset.

    Retourne un dict avec :
    - n_total : nombre de marchés analysés
    - n_anomalies_iforest / n_anomalies_lof : nombre détecté par chaque méthode
    - n_accord : nombre de marchés détectés par les deux méthodes à la fois
    - taux_accord : proportion de marchés en accord entre les deux méthodes
      (parmi ceux détectés par au moins une méthode)

    Un faible taux d'accord ne signifie pas qu'une méthode a tort : les
    deux méthodes détectent des types d'anomalies différents par nature
    (Isolation Forest = anomalies globales, LOF = anomalies contextuelles
    locales). C'est une information à interpréter, pas un simple défaut.
    """
    df_if = detect_isolation_forest(df, contamination=contamination)
    df_lof = detect_lof(df, contamination=contamination)

    is_anomaly_if = df_if["is_anomaly_iforest"]
    is_anomaly_lof = df_lof["is_anomaly_lof"]

    n_union = (is_anomaly_if | is_anomaly_lof).sum()
    n_accord = (is_anomaly_if & is_anomaly_lof).sum()

    return {
        "n_total": len(df),
        "n_anomalies_iforest": int(is_anomaly_if.sum()),
        "n_anomalies_lof": int(is_anomaly_lof.sum()),
        "n_accord": int(n_accord),
        "taux_accord": round(n_accord / n_union, 3) if n_union > 0 else 0.0,
    }
