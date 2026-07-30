"""
Génère le fichier CSV à annoter manuellement (voir src/search/annotation.py).

Lancement : python -m src.search.run_annotation

Produit annotation_a_remplir.csv à la racine du projet, avec les 8
premiers résultats du pipeline complet pour chacune des 4 requêtes de
test "faciles" (32 lignes au total — proche des 20-30 lignes suggérées).

Une fois le fichier rempli (colonne `pertinent` = 0 ou 1 pour chaque
ligne), relancer avec --comparer pour voir le taux d'accord avec la
vérité terrain par mots-clés :

    python -m src.search.run_annotation --comparer
"""

import sys

from src.quality.loader import load_current_only
from src.search.annotation import (
    charger_annotations,
    comparer_annotation_vs_motscles,
    generer_fichier_annotation,
)
from src.search.engine import search
from src.search.evaluation import REQUETES_TEST

CHEMIN_FICHIER = "annotation_a_remplir.csv"
TAILLE_ECHANTILLON = 50_000
TOP_K = 8


def generer():
    df = load_current_only(
        "data/raw/decp.parquet",
        columns=["uid", "objet", "montant", "acheteur_region_nom"],
    )
    df = df.dropna(subset=["objet"])
    sample = df.sample(n=TAILLE_ECHANTILLON, random_state=42)

    resultats_par_requete = {}
    for rt in REQUETES_TEST:
        result, _ = search(sample, rt.requete, top_k=TOP_K)
        resultats_par_requete[rt.nom] = result
        print(f"  {rt.nom}: {len(result)} résultats")

    generer_fichier_annotation(resultats_par_requete, CHEMIN_FICHIER, top_k=TOP_K)
    print(f"\nFichier généré : {CHEMIN_FICHIER}")
    print("Ouvre-le (Excel, LibreOffice, ou un éditeur de texte) et remplis la")
    print("colonne 'pertinent' avec 0 ou 1 pour chaque ligne, en lisant l'objet.")
    print("Puis relance : python -m src.search.run_annotation --comparer")


def comparer():
    try:
        df_annote = charger_annotations(CHEMIN_FICHIER)
    except ValueError as e:
        print(f"Erreur : {e}")
        return
    except FileNotFoundError:
        print(f"Fichier {CHEMIN_FICHIER} introuvable. Lance d'abord sans argument.")
        return

    resultats = comparer_annotation_vs_motscles(df_annote, REQUETES_TEST)
    print(resultats.to_string(index=False))


if __name__ == "__main__":
    if "--comparer" in sys.argv:
        comparer()
    else:
        generer()
