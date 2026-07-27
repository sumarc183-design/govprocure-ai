"""
Outil d'annotation humaine, pour valider (ou invalider) la vérité
terrain par mots-clés utilisée jusqu'ici dans evaluation.py.

Toutes les métriques du bloc 5 reposent sur une vérité terrain
approximative (un marché est "pertinent" s'il contient certains
mots-clés). Cette approximation n'a jamais été comparée à un vrai
jugement humain. Ce module permet de :
1. générer un fichier CSV avec les résultats de recherche à annoter
   manuellement (colonne `pertinent` vide, à remplir avec 0 ou 1) ;
2. recharger ce fichier une fois annoté, et comparer le jugement humain
   à la vérité terrain par mots-clés (taux d'accord, désaccords précis).

Ne lance pas la recherche elle-même : prend en entrée des résultats déjà
calculés (voir src/search/run_annotation.py pour le script qui appelle
réellement engine.search() et génère le fichier).
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd

from src.search.evaluation import RequeteTest, est_pertinent


def generer_fichier_annotation(
    resultats_par_requete: dict[str, pd.DataFrame],
    chemin_sortie: str | Path,
    top_k: int = 8,
    colonne_objet: str = "objet",
) -> pd.DataFrame:
    """Construit un CSV à annoter manuellement à partir des résultats de
    recherche déjà obtenus pour plusieurs requêtes.

    resultats_par_requete : dict {nom_requete: dataframe de résultats
    déjà classés par le moteur}. Prend les top_k premiers résultats de
    chaque requête.

    La colonne `pertinent` est laissée vide — à remplir manuellement avec
    0 (non pertinent) ou 1 (pertinent) après avoir lu chaque `objet`.
    """
    lignes = []
    for nom_requete, df_resultats in resultats_par_requete.items():
        top = df_resultats.head(top_k)
        for rang, (_, row) in enumerate(top.iterrows(), start=1):
            lignes.append({
                "requete": nom_requete,
                "rang": rang,
                "objet": row.get(colonne_objet, ""),
                "montant": row.get("montant", ""),
                "region": row.get("acheteur_region_nom", ""),
                "pertinent": "",  # à remplir : 0 ou 1
            })

    df_annotation = pd.DataFrame(lignes)
    df_annotation.to_csv(chemin_sortie, index=False, encoding="utf-8-sig")
    return df_annotation


def charger_annotations(chemin: str | Path) -> pd.DataFrame:
    """Recharge le fichier annoté (CSV ou Excel .xlsx, détecté automatiquement
    par l'extension). Pour le CSV, détecte aussi automatiquement le
    séparateur (`,` ou `;`) — Excel en français exporte par défaut avec
    `;`, ce qui casserait un simple pd.read_csv(sep=',').

    Lève une erreur claire si des lignes n'ont pas été annotées (colonne
    `pertinent` vide), pour éviter de calculer des métriques sur un
    fichier partiellement rempli sans s'en rendre compte.
    """
    chemin = Path(chemin)
    if chemin.suffix.lower() in (".xlsx", ".xls"):
        df = pd.read_excel(chemin)
    else:
        # sep=None + engine="python" : pandas détecte automatiquement le
        # séparateur (virgule ou point-virgule) plutôt que d'en imposer un.
        df = pd.read_csv(chemin, encoding="utf-8-sig", sep=None, engine="python")

    n_vides = df["pertinent"].isna().sum() + (df["pertinent"] == "").sum()
    if n_vides > 0:
        raise ValueError(
            f"{n_vides} ligne(s) non annotée(s) (colonne 'pertinent' vide). "
            "Remplir toutes les lignes avant de comparer."
        )
    df["pertinent"] = df["pertinent"].astype(int)
    return df


def comparer_annotation_vs_motscles(
    df_annote: pd.DataFrame,
    requetes_test: list[RequeteTest],
) -> pd.DataFrame:
    """Compare le jugement humain à la vérité terrain par mots-clés,
    requête par requête.

    Retourne un dataframe avec, pour chaque requête : le taux d'accord
    entre les deux jugements, et le nombre de désaccords dans chaque
    sens (mots-clés dit pertinent mais humain dit non, et inversement).
    """
    mots_cles_par_nom = {rt.nom: rt.mots_cles_pertinence for rt in requetes_test}

    resultats = []
    for nom_requete in df_annote["requete"].unique():
        sous_df = df_annote[df_annote["requete"] == nom_requete]
        mots_cles = mots_cles_par_nom.get(nom_requete)
        if mots_cles is None:
            continue

        jugement_motscles = sous_df["objet"].astype(str).apply(
            lambda o: int(est_pertinent(o, mots_cles))
        )
        jugement_humain = sous_df["pertinent"]

        accord = (jugement_motscles == jugement_humain).mean()
        # Mots-clés dit pertinent, humain dit non pertinent (faux positif du mot-clé)
        faux_positifs_motscles = ((jugement_motscles == 1) & (jugement_humain == 0)).sum()
        # Mots-clés dit non pertinent, humain dit pertinent (faux négatif du mot-clé)
        faux_negatifs_motscles = ((jugement_motscles == 0) & (jugement_humain == 1)).sum()

        resultats.append({
            "requete": nom_requete,
            "n_lignes": len(sous_df),
            "taux_accord": round(accord, 3),
            "faux_positifs_motscles": int(faux_positifs_motscles),
            "faux_negatifs_motscles": int(faux_negatifs_motscles),
        })

    return pd.DataFrame(resultats)
