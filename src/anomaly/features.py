"""
Préparation des variables pour la détection d'anomalies.

Principe : transformer les colonnes brutes DECP en variables numériques
exploitables par Isolation Forest / LOF, et calculer un écart par rapport
à des marchés "comparables" (même division CPV).

Choix de granularité : on regroupe par division CPV (2 premiers chiffres
du codeCPV, ex: '45' = travaux de construction), pas par code CPV complet.
Le code complet (8 chiffres) donne ~8 900 catégories, beaucoup trop fines
pour calculer des médianes fiables (certaines n'ont qu'1 ou 2 marchés).
La division CPV donne ~174 groupes, tous bien peuplés.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

MIN_MARCHES_PAR_GROUPE = 30  # en dessous, la médiane du groupe n'est pas fiable


def add_cpv_division(df: pd.DataFrame) -> pd.DataFrame:
    """Extrait la division CPV (2 premiers chiffres) depuis codeCPV."""
    df = df.copy()
    df["cpv_division"] = df["codeCPV"].astype(str).str[:2]
    df.loc[df["codeCPV"].isna(), "cpv_division"] = np.nan
    return df


def add_comparable_deviation(df: pd.DataFrame) -> pd.DataFrame:
    """Calcule l'écart du montant et de la durée par rapport à la médiane
    des marchés comparables (même division CPV).

    Ajoute :
    - montant_ratio_mediane_cpv : montant / médiane du groupe (1.0 = dans la norme)
    - duree_ratio_mediane_cpv : idem pour dureeMois
    - cpv_groupe_taille : nombre de marchés dans le groupe (pour juger la fiabilité)
    """
    df = add_cpv_division(df)

    group = df.groupby("cpv_division")
    montant_median = group["montant"].transform("median")
    duree_median = group["dureeMois"].transform("median")
    group_size = group["uid"].transform("size")

    # Éviter division par zéro / médiane nulle
    df["montant_ratio_mediane_cpv"] = np.where(
        montant_median > 0, df["montant"] / montant_median, np.nan
    )
    df["duree_ratio_mediane_cpv"] = np.where(
        duree_median > 0, df["dureeMois"] / duree_median, np.nan
    )
    df["cpv_groupe_taille"] = group_size
    df["cpv_groupe_fiable"] = group_size >= MIN_MARCHES_PAR_GROUPE

    return df


def build_feature_matrix(df: pd.DataFrame) -> tuple[pd.DataFrame, list[str]]:
    """Construit la matrice de variables numériques pour les modèles d'anomalie.

    Retourne (df_enrichi, liste_des_colonnes_features).
    Les valeurs manquantes dans les features sont imputées par la médiane
    de la colonne (choix simple et documenté ; à raffiner si besoin).
    """
    df = add_comparable_deviation(df)

    feature_cols = [
        "montant",
        "dureeMois",
        "montant_ratio_mediane_cpv",
        "duree_ratio_mediane_cpv",
    ]

    df_features = df[feature_cols].copy()
    for col in feature_cols:
        median = df_features[col].median()
        df_features[col] = df_features[col].fillna(median)

    df = df.copy()
    df[feature_cols] = df_features

    return df, feature_cols
