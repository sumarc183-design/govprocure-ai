"""
Génère le fichier CSV à annoter manuellement (voir src/search/annotation.py).

Lancement : python -m src.search.run_annotation

Produit annotation_a_remplir.csv à la racine du projet, avec les 20
premiers résultats du pipeline complet pour chacune des 4 requêtes de
test "faciles" (80 lignes au total).

Extension au-delà de l'annotation initiale (32 lignes, TOP_K=8) — voir
docs/limitations.md, "Non fait, pour aller plus loin" : TOP_K passé à
20 pour couvrir plus de profondeur par thème (jusqu'au rang 20 plutôt
que 8 seulement), sur la même méthodologie déjà validée (lecture de
chaque `objet`, jugement proposé puis relu). Reste volontairement sur
les 4 requêtes "faciles" (pas les difficiles) pour garder un volume de
lecture manuelle raisonnable — voir docs/limitations.md pour la
discussion de ce compromis de portée.

Une fois le fichier rempli (colonne `pertinent` = 0 ou 1 pour chaque
ligne), relancer avec --comparer pour voir le taux d'accord avec la
vérité terrain par mots-clés :

    python -m src.search.run_annotation --comparer

Taille d'échantillon différenciée par thème (voir
`src.search.evaluation.TAILLE_ECHANTILLON_PAR_THEME`) : ce script tirait
initialement un échantillon unique de 50 000 pour les 4 requêtes, ce qui
reproduisait le même problème de lenteur déjà rencontré et corrigé dans
run_evaluation.py (seule "cybersécurité" a un filtre qui réduit le
volume ; les 3 autres thèmes forcent l'encodage de l'échantillon
entier). Corrigé en réutilisant la même fonction centralisée que
run_evaluation.py, plutôt que de recopier une nouvelle fois la logique.
"""

import sys

from src.quality.loader import load_current_only
from src.search.annotation import (
    charger_annotations,
    comparer_annotation_vs_motscles,
    generer_fichier_annotation,
)
from src.search.engine import search
from src.search.evaluation import REQUETES_TEST, taille_echantillon_pour

CHEMIN_FICHIER = "annotation_a_remplir.csv"
TOP_K = 20


def generer():
    df = load_current_only(
        "data/raw/decp.parquet",
        columns=["uid", "objet", "montant", "acheteur_region_nom"],
    )
    df = df.dropna(subset=["objet"])

    resultats_par_requete = {}
    for rt in REQUETES_TEST:
        sample = df.sample(n=min(taille_echantillon_pour(rt.nom), len(df)), random_state=42)
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
