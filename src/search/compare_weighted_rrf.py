"""
Compare la fusion RRF à poids égal (par défaut) et des fusions pondérées
(en faveur de BM25, puis des embeddings), sur les 4 thèmes difficiles
(voir docs/limitations.md, section "Pondération RRF").

Généralisation : la première version de ce script ne testait la
pondération que sur "travaux_voirie_difficile" (le seul thème où le
poids égal dégradait un résultat BM25 correct par coïncidence). Non
tranché à l'époque : est-ce qu'un poids donné (BM25 x3 ou embeddings x3)
généralise aux 3 autres thèmes, ou est-ce spécifique à ce cas précis ?
Ce script répond à cette question en testant les 3 configurations sur
les 4 thèmes difficiles plutôt qu'un seul.

Lancement : python -m src.search.compare_weighted_rrf

Nécessite le modèle d'embeddings (accès internet la première fois).
"""

import pandas as pd

from src.quality.loader import load_current_only
from src.search.engine import search
from src.search.evaluation import (
    REQUETES_TEST_DIFFICILES,
    evaluer_requete,
    taille_echantillon_pour,
)

CONFIGURATIONS = [
    ("egal", None),
    ("bm25_x3", (3.0, 1.0)),
    ("embeddings_x3", (1.0, 3.0)),
]


def evaluer_theme(df: pd.DataFrame, requete_test) -> list[dict]:
    """Évalue les 3 configurations de pondération pour un thème donné.

    Un échantillon distinct par thème (taille centralisée dans
    evaluation.py) : même principe que run_evaluation.py, pas un
    échantillon unique partagé entre thèmes.
    """
    sample = df.sample(n=taille_echantillon_pour(requete_test.nom), random_state=42)

    lignes = []
    for nom_config, poids in CONFIGURATIONS:
        result, _ = search(sample, requete_test.requete, top_k=20, poids_fusion=poids)
        metriques = evaluer_requete(result, requete_test, k_values=[5, 10], df_corpus=sample)
        lignes.append({"theme": requete_test.nom, "configuration": nom_config, **metriques})
    return lignes


def main():
    df = load_current_only(
        "data/raw/decp.parquet",
        columns=["uid", "objet", "montant", "acheteur_region_nom"],
    )
    df = df.dropna(subset=["objet"])

    toutes_lignes = []
    for requete_test in REQUETES_TEST_DIFFICILES:
        print(f"Thème : {requete_test.nom} — requête : {requete_test.requete}")
        lignes = evaluer_theme(df, requete_test)
        toutes_lignes.extend(lignes)
        for ligne in lignes:
            print(f"  {ligne['configuration']:>15}  P@5={ligne['precision@5']}  P@10={ligne['precision@10']}")
        print()

    resultats = pd.DataFrame(toutes_lignes)
    print("=" * 70)
    print("Résumé complet")
    print("=" * 70)
    print(resultats.to_string(index=False))

    resultats.to_csv("data/processed/comparaison_rrf_pondere.csv", index=False)
    print("\nRésultats sauvegardés dans data/processed/comparaison_rrf_pondere.csv")


if __name__ == "__main__":
    main()
