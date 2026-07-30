"""
Script d'évaluation du moteur de recherche sur le jeu de requêtes de test.

Lancement : python -m src.search.run_evaluation

Affiche Precision@K, Recall@K et NDCG@K pour BM25 seul et pour le
pipeline hybride complet (BM25 + embeddings + RRF), sur :
- REQUETES_TEST : requêtes "faciles" (vocabulaire proche des mots-clés
  de vérité terrain) ;
- REQUETES_TEST_DIFFICILES : mêmes thèmes reformulés sans recouvrement
  lexical avec la vérité terrain, pour évaluer spécifiquement l'apport
  des embeddings (voir docs/decisions.md, section Bloc 5).

Nécessite un accès internet la première fois (téléchargement du modèle
d'embeddings) pour la partie pipeline complet.

Taille d'échantillon PAR REQUÊTE (pas une taille unique pour toutes) :
voir `src.search.evaluation.TAILLE_ECHANTILLON_PAR_THEME` pour le détail
et le raisonnement (centralisé là, réutilisé aussi par run_annotation.py
et compare_weighted_rrf.py, pour éviter que ce problème de lenteur déjà
rencontré une fois ici ne se reproduise dans un script similaire — ce
qui est arrivé une première fois avant cette centralisation).
"""

from src.quality.loader import load_current_only
from src.search.bm25_search import BM25Index
from src.search.engine import apply_strict_filters, search
from src.search.evaluation import (
    REQUETES_TEST,
    REQUETES_TEST_DIFFICILES,
    evaluer_requete,
    taille_echantillon_pour,
)
from src.search.filters import parse_query


def _afficher_tableau(titre, lignes):
    print()
    print("=" * 78)
    print(titre)
    print("=" * 78)
    print(f"{'Requête':<30} {'n_ech':>7} {'n_pert':>7} {'P@5':>6} {'R@5':>6} {'NDCG@5':>8} {'P@10':>6} {'R@10':>6}")
    for r in lignes:
        print(
            f"{r['nom']:<30} {r.get('n_echantillon', '-'):>7} {r.get('n_pertinents_corpus', '-'):>7} "
            f"{r['precision@5']:>6} {r.get('recall@5', '-'):>6} {r['ndcg@5']:>8} "
            f"{r['precision@10']:>6} {r.get('recall@10', '-'):>6}"
        )


def evaluer_bm25_seul(df, requetes):
    lignes = []
    for rt in requetes:
        taille = taille_echantillon_pour(rt.nom)
        sample = df.sample(n=min(taille, len(df)), random_state=42)
        bm25 = BM25Index(sample)
        result = bm25.search(rt.requete, top_k=20)
        r = evaluer_requete(result, rt, k_values=[5, 10], df_corpus=sample)
        r["n_echantillon"] = len(sample)
        lignes.append(r)
    return lignes


def evaluer_pipeline_complet(df, requetes):
    lignes = []
    for rt in requetes:
        taille = taille_echantillon_pour(rt.nom)
        sample = df.sample(n=min(taille, len(df)), random_state=42)

        parsed = parse_query(rt.requete)
        df_filtre = apply_strict_filters(sample, parsed)
        result, explication = search(sample, rt.requete, top_k=20)
        r = evaluer_requete(result, rt, k_values=[5, 10], df_corpus=df_filtre)
        r["n_echantillon"] = len(sample)
        lignes.append(r)
    return lignes


def main():
    df = load_current_only(
        "data/raw/decp.parquet",
        columns=["uid", "objet", "montant", "acheteur_region_nom"],
    )
    df = df.dropna(subset=["objet"])
    print(f"Corpus disponible : {len(df):,} marchés (taille d'échantillon adaptée par thème, voir en-tête du script)")

    _afficher_tableau("BM25 seul — requêtes faciles", evaluer_bm25_seul(df, REQUETES_TEST))
    _afficher_tableau("BM25 seul — requêtes difficiles (sans recouvrement lexical)", evaluer_bm25_seul(df, REQUETES_TEST_DIFFICILES))
    _afficher_tableau("Pipeline complet (BM25+embeddings+RRF) — requêtes faciles", evaluer_pipeline_complet(df, REQUETES_TEST))
    _afficher_tableau("Pipeline complet (BM25+embeddings+RRF) — requêtes difficiles", evaluer_pipeline_complet(df, REQUETES_TEST_DIFFICILES))


if __name__ == "__main__":
    main()
