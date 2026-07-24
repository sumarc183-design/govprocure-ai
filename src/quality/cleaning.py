"""
Nettoyage des données DECP, basé sur les règles identifiées lors de l'audit
qualité (voir docs/data_sources.md et docs/limitations.md).

Ce module ne supprime pas silencieusement les données problématiques :
il les *marque* (colonnes de flags), pour que les blocs suivants
(anomalies, recherche) puissent décider eux-mêmes quoi en faire plutôt
que de perdre l'information.
"""

from __future__ import annotations

import unicodedata

import pandas as pd

from src.quality.audit import DATE_MAX_PLAUSIBLE, DATE_MIN_PLAUSIBLE, MONTANT_MAX_PLAUSIBLE


def _normalize_text(value: str) -> str:
    """Normalise une valeur catégorielle : minuscules, sans accents, sans espaces superflus."""
    if not isinstance(value, str):
        return value
    value = value.strip().lower()
    value = unicodedata.normalize("NFKD", value)
    value = value.encode("ascii", errors="ignore").decode("ascii")
    return value


# Table de correspondance vers un libellé canonique lisible, après normalisation.
# À enrichir si de nouvelles variantes apparaissent dans de futures extractions.
NATURE_CANONICAL = {
    "marche": "Marché",
    "marche subsequent": "Marché subséquent",
    "accord-cadre": "Accord-cadre",
    "accord cadre": "Accord-cadre",
    "marche de partenariat": "Marché de partenariat",
}


def normalize_nature(df: pd.DataFrame) -> pd.DataFrame:
    """Normalise la colonne `nature` vers des libellés canoniques.

    Remplace les variantes de casse/accents ('MARCHE', 'marché', 'Marché')
    par un seul libellé cohérent, sans perdre l'information d'origine
    (conservée dans `nature_original`).
    """
    df = df.copy()
    if "nature" not in df.columns:
        return df

    df["nature_original"] = df["nature"]
    normalized = df["nature"].apply(_normalize_text)
    df["nature"] = normalized.map(NATURE_CANONICAL).fillna(df["nature"])
    return df


def flag_montant_issues(df: pd.DataFrame) -> pd.DataFrame:
    """Ajoute des colonnes de flags sur les montants suspects, sans les supprimer.

    - montant_negatif_ou_nul : True si montant <= 0
    - montant_extreme : True si montant > seuil plausible (1 Md€ par défaut)

    Ces marchés restent dans le dataset : à charge du bloc anomalies de
    décider s'ils doivent être exclus, corrigés, ou traités comme des cas
    à examiner en priorité.
    """
    df = df.copy()
    if "montant" not in df.columns:
        return df

    df["montant_negatif_ou_nul"] = df["montant"] <= 0
    df["montant_extreme"] = df["montant"] > MONTANT_MAX_PLAUSIBLE
    return df


def flag_date_issues(df: pd.DataFrame) -> pd.DataFrame:
    """Ajoute un flag sur les dates hors plage plausible, colonne par colonne."""
    df = df.copy()
    for col in ("dateNotification", "datePublicationDonnees"):
        if col not in df.columns:
            continue
        flag_col = f"{col}_hors_plage"
        df[flag_col] = (df[col] < DATE_MIN_PLAUSIBLE) | (df[col] > DATE_MAX_PLAUSIBLE)
        # Une date manquante (NaT) n'est pas "hors plage", juste absente.
        df.loc[df[col].isna(), flag_col] = False
    return df


def clean_pipeline(df: pd.DataFrame) -> pd.DataFrame:
    """Applique l'ensemble des règles de nettoyage, dans l'ordre.

    Retourne un dataframe enrichi de colonnes de flags, sans perte
    d'information par rapport à l'original.
    """
    df = normalize_nature(df)
    df = flag_montant_issues(df)
    df = flag_date_issues(df)
    return df
