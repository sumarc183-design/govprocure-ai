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


def deduplicate_marches(df: pd.DataFrame) -> pd.DataFrame:
    """Déduplique par `uid` (garde une seule ligne par marché).

    Correction d'un bug trouvé lors d'une revue externe du projet : le
    dataset a une granularité (marché × titulaire) — un même marché en
    cotraitance peut apparaître jusqu'à une dizaine de fois (une ligne par
    titulaire associé). Sans cette déduplication, ces marchés sont
    surreprésentés dans l'entraînement des modèles d'anomalie, ce qui peut
    fausser les voisinages de LOF, le taux d'accord entre les méthodes, et
    faire apparaître plusieurs fois le même marché dans les alertes.

    Le même principe avait déjà été identifié et corrigé pour le moteur de
    recherche (bloc 3, `apply_strict_filters`) mais n'avait pas été
    réappliqué ici avant la détection d'anomalies — erreur reconnue,
    voir docs/decisions.md.
    """
    if "uid" not in df.columns:
        return df
    return df.drop_duplicates(subset="uid", keep="first").copy()


def _log_signe(x: pd.Series) -> pd.Series:
    """Transformation log signée : sign(x) * log1p(|x|).

    Utilisée plutôt qu'un log1p classique car `montant` contient des
    valeurs négatives (~2% du dataset, jusqu'à -7,1M€, voir audit qualité
    du bloc 1) — un log1p standard échoue (indéfini) pour x <= -1. Le log
    signé compresse l'échelle des deux côtés du zéro de façon symétrique,
    en gardant le signe d'origine, plutôt que de forcer un filtrage des
    valeurs négatives (qu'on veut justement garder comme candidates à la
    détection d'anomalies, pas exclure comme au bloc prédiction).

    Suggestion reçue en revue externe (log1p sur montant, avant LOF) —
    adaptée ici pour gérer ce cas réel non anticipé dans la suggestion
    d'origine.
    """
    return np.sign(x) * np.log1p(np.abs(x))


def build_feature_matrix(
    df: pd.DataFrame, transformation_log: bool = False
) -> tuple[pd.DataFrame, list[str]]:
    """Construit la matrice de variables numériques pour les modèles d'anomalie.

    Retourne (df_enrichi, liste_des_colonnes_features).
    Les valeurs manquantes dans les features sont imputées par la médiane
    de la colonne (choix simple et documenté ; à raffiner si besoin).

    Déduplique par marché (voir deduplicate_marches) avant tout calcul,
    pour que les médianes de groupe et l'entraînement des modèles portent
    sur des marchés uniques, pas sur des lignes (marché × titulaire).

    transformation_log : si True, applique une transformation log signée
    (voir _log_signe) sur `montant` et `montant_ratio_mediane_cpv` avant
    la standardisation. Motivation : ces deux variables sont extrêmement
    asymétriques (montants jusqu'à plusieurs milliards), ce qui peut faire
    dominer les distances calculées par LOF par une poignée de valeurs
    extrêmes. Désactivé par défaut pour ne pas changer le comportement
    déjà validé dans les blocs précédents — à activer explicitement pour
    comparer (voir docs/limitations.md pour le résultat de cette
    comparaison).
    """
    df = deduplicate_marches(df)
    df = add_comparable_deviation(df)

    feature_cols = [
        "montant",
        "dureeMois",
        "montant_ratio_mediane_cpv",
        "duree_ratio_mediane_cpv",
    ]

    df_features = df[feature_cols].copy()

    if transformation_log:
        df_features["montant"] = _log_signe(df_features["montant"])
        df_features["montant_ratio_mediane_cpv"] = _log_signe(
            df_features["montant_ratio_mediane_cpv"]
        )

    for col in feature_cols:
        median = df_features[col].median()
        df_features[col] = df_features[col].fillna(median)

    df = df.copy()
    df[feature_cols] = df_features

    return df, feature_cols
