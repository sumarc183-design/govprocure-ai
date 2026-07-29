"""
Compare la fusion RRF à poids égal (par défaut) et une fusion pondérée
en faveur de BM25, sur le cas où la fusion à poids égal avait dégradé un
résultat BM25 correct ("travaux_voirie_difficile", voir docs/decisions.md).

Lancement : python -m src.search.compare_weighted_rrf

Nécessite le modèle d'embeddings (accès internet la première fois).
"""

from src.quality.loader import load_current_only
from src.search.engine import search
from src.search.evaluation import REQUETES_TEST_DIFFICILES, evaluer_requete

TAILLE_ECHANTILLON = 15_000  # cohérent avec run_evaluation.py pour ce thème


def main():
    df = load_current_only(
        "data/raw/decp.parquet",
        columns=["uid", "objet", "montant", "acheteur_region_nom"],
    )
    df = df.dropna(subset=["objet"])
    sample = df.sample(n=TAILLE_ECHANTILLON, random_state=42)

    requete_test = next(
        rt for rt in REQUETES_TEST_DIFFICILES if rt.nom == "travaux_voirie_difficile"
    )

    print(f"Requête : {requete_test.requete}")
    print()

    print("=" * 60)
    print("RRF à poids égal (comportement par défaut)")
    print("=" * 60)
    result_egal, _ = search(sample, requete_test.requete, top_k=20, poids_fusion=None)
    metriques_egal = evaluer_requete(result_egal, requete_test, k_values=[5, 10], df_corpus=sample)
    print(metriques_egal)

    print()
    print("=" * 60)
    print("RRF pondéré (BM25 x3, embeddings x1)")
    print("=" * 60)
    result_pondere, _ = search(
        sample, requete_test.requete, top_k=20, poids_fusion=(3.0, 1.0)
    )
    metriques_pondere = evaluer_requete(result_pondere, requete_test, k_values=[5, 10], df_corpus=sample)
    print(metriques_pondere)

    print()
    print("=" * 60)
    print("RRF pondéré (BM25 x1, embeddings x3) — pour comparaison inverse")
    print("=" * 60)
    result_inverse, _ = search(
        sample, requete_test.requete, top_k=20, poids_fusion=(1.0, 3.0)
    )
    metriques_inverse = evaluer_requete(result_inverse, requete_test, k_values=[5, 10], df_corpus=sample)
    print(metriques_inverse)


if __name__ == "__main__":
    main()
