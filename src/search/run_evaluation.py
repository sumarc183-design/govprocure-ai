"""
Script d'évaluation du moteur de recherche sur le jeu de requêtes de test.

Lancement : python -m src.search.run_evaluation

Affiche Precision@K, Recall@K et NDCG@K pour BM25 seul et pour le
pipeline hybride complet (BM25 + embeddings + RRF), sur :
- REQUETES_TEST : requêtes "faciles" (vocabulaire proche des mots-clés
  de vérité terrain) ;
- REQUETES_TEST_DIFFICILES : mêmes thèmes reformulés sans recouvrement
  lexical avec la vérité terrain, pour évaluer spécifiquement l'apport
  des embeddings (voir docs/decisions.md, section Bloc 5, sur le biais
  de fuite reconnu et sa correction).

Nécessite un accès internet la première fois (téléchargement du modèle
d'embeddings) pour la partie pipeline complet.

Taille d'échantillon : 100 000 par défaut. Ne pas réduire sans vérifier
le nombre de candidats obtenus après filtrage strict pour chaque requête
(le thème cybersécurité, plus rare, tombe à seulement 1 candidat
pertinent sur un échantillon de 50 000 après filtre Île-de-France +
montant élevé — sensibilité découverte en pratique, voir
docs/limitations.md).
"""

from src.quality.loader import load_current_only
from src.search.bm25_search import BM25Index
from src.search.engine import apply_strict_filters, search
from src.search.filters import parse_query
from src.search.evaluation import REQUETES_TEST, REQUETES_TEST_DIFFICILES, evaluer_requete

TAILLE_ECHANTILLON = 100_000


def _afficher_tableau(titre, lignes):
    print()
    print("=" * 78)
    print(titre)
    print("=" * 78)
    print(f"{'Requête':<30} {'n_pert':>7} {'P@5':>6} {'R@5':>6} {'NDCG@5':>8} {'P@10':>6} {'R@10':>6}")
    for r in lignes:
        print(
            f"{r['nom']:<30} {r.get('n_pertinents_corpus', '-'):>7} "
            f"{r['precision@5']:>6} {r.get('recall@5', '-'):>6} {r['ndcg@5']:>8} "
            f"{r['precision@10']:>6} {r.get('recall@10', '-'):>6}"
        )


def evaluer_bm25_seul(sample, requetes):
    lignes = []
    bm25 = BM25Index(sample)
    for rt in requetes:
        result = bm25.search(rt.requete, top_k=20)
        lignes.append(evaluer_requete(result, rt, k_values=[5, 10], df_corpus=sample))
    return lignes


def evaluer_pipeline_complet(sample, requetes):
    lignes = []
    for rt in requetes:
        # Vérité terrain calculée sur le sous-ensemble réellement filtré
        # par la requête (région/montant), pas sur tout l'échantillon —
        # cohérent avec ce que le pipeline peut effectivement retrouver.
        parsed = parse_query(rt.requete)
        df_filtre = apply_strict_filters(sample, parsed)
        result, explication = search(sample, rt.requete, top_k=20)
        lignes.append(evaluer_requete(result, rt, k_values=[5, 10], df_corpus=df_filtre))
    return lignes


def main():
    df = load_current_only(
        "data/raw/decp.parquet",
        columns=["uid", "objet", "montant", "acheteur_region_nom"],
    )
    df = df.dropna(subset=["objet"])
    sample = df.sample(n=TAILLE_ECHANTILLON, random_state=42)
    print(f"Échantillon : {len(sample):,} marchés")

    _afficher_tableau("BM25 seul — requêtes faciles", evaluer_bm25_seul(sample, REQUETES_TEST))
    _afficher_tableau("BM25 seul — requêtes difficiles (sans recouvrement lexical)", evaluer_bm25_seul(sample, REQUETES_TEST_DIFFICILES))
    _afficher_tableau("Pipeline complet (BM25+embeddings+RRF) — requêtes faciles", evaluer_pipeline_complet(sample, REQUETES_TEST))
    _afficher_tableau("Pipeline complet (BM25+embeddings+RRF) — requêtes difficiles", evaluer_pipeline_complet(sample, REQUETES_TEST_DIFFICILES))


if __name__ == "__main__":
    main()
