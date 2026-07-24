"""
Chargement des données DECP (Données Essentielles de la Commande Publique).

Point d'attention découvert lors de l'audit initial :
- La granularité du fichier est (marché x titulaire), pas (marché).
  Un même `uid` peut apparaître plusieurs fois si plusieurs titulaires
  sont associés au même marché (cotraitance / groupement). Ce n'est PAS
  un doublon à supprimer.
- Le champ `donneesActuelles` distingue la version actuelle d'un marché
  de ses versions historiques (modifications). Pour une vue "état actuel",
  filtrer sur donneesActuelles == True.
"""

from pathlib import Path

import pandas as pd

# Colonnes jugées essentielles pour l'audit qualité et les blocs suivants.
# On évite de charger les 58 colonnes d'un coup pour rester raisonnable en mémoire.
CORE_COLUMNS = [
    "uid",
    "id",
    "nature",
    "acheteur_id",
    "acheteur_nom",
    "titulaire_id",
    "titulaire_nom",
    "objet",
    "montant",
    "type",
    "codeCPV",
    "procedure",
    "dureeMois",
    "offresRecues",
    "dateNotification",
    "datePublicationDonnees",
    "acheteur_region_nom",
    "acheteur_departement_nom",
    "titulaire_region_nom",
    "modification_id",
    "donneesActuelles",
    "sourceDataset",
]


def load_raw(path: str | Path, columns: list[str] | None = None) -> pd.DataFrame:
    """Charge le fichier Parquet DECP tel quel, sans aucun nettoyage.

    Utile pour l'audit qualité, qui a justement besoin de voir les données brutes.
    """
    columns = columns or CORE_COLUMNS
    df = pd.read_parquet(path, columns=columns)

    # Les dates sont parfois mal typées quand elles sont manquantes ou aberrantes
    # (ex: 0001-01-01). On les convertit proprement, les valeurs invalides
    # deviennent NaT plutôt que de faire planter le reste du pipeline.
    for date_col in ("dateNotification", "datePublicationDonnees"):
        if date_col in df.columns:
            df[date_col] = pd.to_datetime(df[date_col], errors="coerce")

    return df


def load_current_only(path: str | Path, columns: list[str] | None = None) -> pd.DataFrame:
    """Charge uniquement la version actuelle de chaque marché.

    Filtre sur donneesActuelles == True. Utile pour toute analyse qui veut
    une vue "état actuel" plutôt que l'historique complet des modifications.
    """
    df = load_raw(path, columns=columns)
    if "donneesActuelles" in df.columns:
        df = df[df["donneesActuelles"] == True]  # noqa: E712
    return df


def group_by_marche(df: pd.DataFrame) -> pd.DataFrame:
    """Regroupe les lignes (marché x titulaire) au niveau du marché (uid).

    Utile quand on veut analyser un marché une seule fois, indépendamment
    du nombre de titulaires associés (cotraitance/groupement).
    Le montant, la date, etc. sont supposés identiques entre titulaires
    d'un même marché ; on garde la première occurrence pour ces champs,
    et on agrège le nombre de titulaires.
    """
    if "uid" not in df.columns:
        raise ValueError("La colonne 'uid' est nécessaire pour grouper par marché.")

    agg = df.groupby("uid").agg(
        nb_titulaires=("titulaire_id", "nunique"),
    )
    first_values = df.drop_duplicates(subset="uid", keep="first").set_index("uid")
    result = first_values.join(agg, how="left")
    return result.reset_index()
