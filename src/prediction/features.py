"""
Préparation des données pour prédire `offresRecues` (nombre d'offres
reçues par un marché) à partir de ses caractéristiques.

Choix de cible : offresRecues est une vraie variable observée (pas une
étiquette construite artificiellement), utile métier (mesure de la
compétitivité d'un marché), qui comble un trou de compétences identifié
dans la synthèse finale du projet (aucune régression/classification
supervisée n'existait avant ce module).

Deux problèmes de qualité de données découverts en marge de ce module,
cohérents avec ceux déjà rencontrés sur `montant` au bloc 1 :
- ~60% de valeurs manquantes, et le taux de manquant dépend fortement
  du type de procédure (de 31% à quasiment 100% selon la procédure) —
  biais de sélection réel, pas un hasard. Documenté dans
  docs/limitations.md : le modèle ne peut apprendre que sur les
  procédures où offresRecues est majoritairement renseigné, et
  généralisera mal aux autres.
- valeurs négatives (impossible pour un compte d'offres, ex: -8) et
  distribution très asymétrique (médiane 3, max 20 300) — mêmes
  symptômes que `montant`, mêmes corrections : filtrage des valeurs
  invalides, transformation log pour l'entraînement.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from src.anomaly.features import add_cpv_division
from src.common.text_normalization import normaliser_texte_categorie

COLONNES_REQUISES = [
    "uid", "offresRecues", "montant", "dureeMois", "codeCPV",
    "procedure", "nature", "acheteur_region_nom",
]


DUREE_MAX_PLAUSIBLE_MOIS = 240  # 20 ans — voir docstring filtrer_cible_valide


def filtrer_cible_valide(df: pd.DataFrame) -> pd.DataFrame:
    """Garde uniquement les lignes exploitables pour l'entraînement :
    - offresRecues renseigné et valide (positif ou nul — un compte
      d'offres négatif est une valeur invalide, même logique que le
      filtrage des montants négatifs au bloc 1) ;
    - montant strictement positif (nécessaire pour la transformation
      log1p ; un montant négatif ou nul ferait planter log1p ou
      produirait une valeur non exploitable — mêmes 64 408 lignes
      identifiées comme suspectes dans l'audit qualité du bloc 1) ;
    - dureeMois dans une plage plausible (0, 240] mois (20 ans).

    Le filtre sur dureeMois a été ajouté après un bug découvert en
    validation croisée : une ligne avec dureeMois=21886 (>1800 ans,
    donnée manifestement erronée) a fait exploser numériquement une
    prédiction Ridge (coefficient raisonnable × valeur aberrante = valeur
    log extrême, qui donne un nombre quasi infini une fois reconverti via
    expm1 — R² de l'ordre de -10^71 observé). Vérifié sur l'ensemble du
    dataset : 1042 valeurs négatives/nulles et 1864 valeurs > 240 mois
    (jusqu'à 31410 mois, ~2618 ans) — 0,09% des lignes au total, un
    filtrage qui n'amputerait quasiment pas le volume d'entraînement mais
    protège contre ce type d'instabilité numérique.
    """
    df = df.dropna(subset=["offresRecues"])
    df = df[df["offresRecues"] >= 0]
    df = df[df["montant"] > 0]
    df = df[(df["dureeMois"] > 0) & (df["dureeMois"] <= DUREE_MAX_PLAUSIBLE_MOIS)]
    return df


def preparer_donnees_completes(df: pd.DataFrame) -> pd.DataFrame:
    """Pipeline complet de préparation, dans l'ordre qui compte :

    1. Filtrer la cible valide (avant tout le reste, pas la peine de
       traiter des lignes qu'on va de toute façon exclure).
    2. Dédupliquer par marché (leçon du bloc 1, réappliquée ici pour la
       4e fois — voir docs/decisions.md sur la centralisation à
       envisager).
    3. Normaliser les catégories textuelles (procedure, nature) via le
       module commun — corrige au passage un bug jamais traité sur
       `procedure` (16 valeurs distinctes, 11 réelles une fois
       normalisées).
    4. Ajouter la division CPV (réutilise la fonction du bloc anomalies,
       même principe : le code CPV complet est trop fin, la division à
       2 chiffres donne des groupes fiables).
    """
    df = filtrer_cible_valide(df)
    df = df.drop_duplicates(subset="uid", keep="first").copy()

    df["procedure_normalisee"] = df["procedure"].apply(normaliser_texte_categorie)
    df["nature_normalisee"] = df["nature"].apply(normaliser_texte_categorie)

    df = add_cpv_division(df)

    return df


def construire_matrice_features(
    df: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.Series, list[str]]:
    """Construit la matrice de features (X) et la cible transformée (y)
    pour l'entraînement.

    Transformation log1p sur montant et sur la cible (offresRecues) :
    justifiée par leur asymétrie extrême, même raisonnement que pour les
    montants au bloc 2 (recommandation reçue en revue externe, testée
    ici). log1p plutôt que log simple pour gérer les zéros sans erreur
    (log(0) est indéfini, log1p(0) = 0).

    Retourne (X, y, feature_cols) où X contient les colonnes numériques
    déjà transformées et les catégorielles encodées en one-hot.
    """
    df = preparer_donnees_completes(df)

    df["log_montant"] = np.log1p(df["montant"].fillna(df["montant"].median()))
    df["log_offresRecues"] = np.log1p(df["offresRecues"])

    features_numeriques = ["log_montant", "dureeMois"]
    df["dureeMois"] = df["dureeMois"].fillna(df["dureeMois"].median())

    features_categorielles = ["procedure_normalisee", "nature_normalisee", "cpv_division", "acheteur_region_nom"]
    for col in features_categorielles:
        df[col] = df[col].fillna("inconnu")

    X_categoriel = pd.get_dummies(df[features_categorielles], prefix=features_categorielles)
    X = pd.concat([df[features_numeriques].reset_index(drop=True), X_categoriel.reset_index(drop=True)], axis=1)
    y = df["log_offresRecues"].reset_index(drop=True)

    return X, y, list(X.columns)
