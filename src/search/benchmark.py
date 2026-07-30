"""
Mesure du temps de réponse du pipeline de recherche.

Lancement : python -m src.search.benchmark

Mesure séparément :
- construction de l'index BM25 (dépend de la taille du sous-ensemble filtré) ;
- recherche BM25 elle-même (rapide, structure déjà construite) ;
- construction de l'index embeddings (le plus coûteux : encode tous les
  textes du sous-ensemble filtré) ;
- recherche embeddings elle-même ;
- pipeline complet de bout en bout (engine.search).

Pourquoi mesurer séparément plutôt que juste le temps total : pour savoir
QUELLE étape domine le temps de réponse. Si la construction d'index domine
(c'est le cas attendu pour les embeddings), la vraie optimisation à faire
en production est de précalculer/mettre en cache les embeddings du corpus
une fois pour toutes, pas d'optimiser la recherche elle-même — décision
déjà notée comme piste dans docs/limitations.md, ce benchmark permet de
la chiffrer concrètement plutôt que de rester une intuition.
"""

import os
import time

from src.quality.loader import load_current_only
from src.search.bm25_search import BM25Index
from src.search.embeddings_search import EmbeddingIndex
from src.search.engine import apply_strict_filters, search
from src.search.filters import parse_query

REQUETE_BENCHMARK = "marchés de cybersécurité en Île-de-France de montant élevé"


def mesurer(label: str, fonction) -> float:
    t0 = time.perf_counter()
    fonction()
    duree = time.perf_counter() - t0
    print(f"  {label:<40} {duree:>8.3f} s")
    return duree


def main():
    df = load_current_only(
        "data/raw/decp.parquet",
        columns=["uid", "objet", "montant", "acheteur_region_nom"],
    )
    df = df.dropna(subset=["objet"])

    for taille in [10_000, 50_000, 100_000]:
        print(f"\n{'=' * 60}")
        print(f"Échantillon : {taille:,} marchés")
        print("=" * 60)

        sample = df.sample(n=min(taille, len(df)), random_state=42)
        parsed = parse_query(REQUETE_BENCHMARK)
        df_filtre = apply_strict_filters(sample, parsed)
        print(f"  Candidats après filtre strict : {len(df_filtre)}")
        print()

        # BM25 : construction + recherche séparément
        bm25_index_holder = {}
        mesurer("Construction index BM25", lambda: bm25_index_holder.update(
            idx=BM25Index(df_filtre)
        ))
        mesurer("Recherche BM25", lambda: bm25_index_holder["idx"].search(
            parsed.texte_libre, top_k=20
        ))

        # Embeddings : construction + recherche séparément
        embedding_index_holder = {}
        mesurer("Construction index embeddings", lambda: embedding_index_holder.update(
            idx=EmbeddingIndex(df_filtre)
        ))
        mesurer("Recherche embeddings", lambda: embedding_index_holder["idx"].search(
            parsed.texte_libre, top_k=20
        ))

        # Pipeline complet de bout en bout, SANS cache (cache_embeddings_path=None)
        # — préserve la mesure de référence "sans optimisation" déjà
        # documentée dans docs/limitations.md. La démonstration du cache
        # est faite séparément plus bas.
        mesurer("Pipeline complet (search(), sans cache)", lambda: search(
            sample, REQUETE_BENCHMARK, top_k=20, cache_embeddings_path=None
        ))

    # Démonstration du cache d'embeddings : le premier appel est "à froid"
    # (rien en cache), le second "à chaud" (tout déjà calculé) — sur le
    # même échantillon de 100 000, pour mesurer le gain réel du mécanisme
    # de cache ajouté au bloc 5 (voir docs/decisions.md).
    print(f"\n{'=' * 60}")
    print("Démonstration du cache d'embeddings (échantillon 100 000)")
    print("=" * 60)
    sample = df.sample(n=min(100_000, len(df)), random_state=42)
    chemin_cache = "data/processed/embeddings_cache_demo.npz"
    if os.path.exists(chemin_cache):
        os.remove(chemin_cache)  # partir d'un cache vide pour la démo

    def recherche_avec_cache():
        return search(sample, REQUETE_BENCHMARK, top_k=20, cache_embeddings_path=chemin_cache)

    duree_froide = mesurer("Recherche à froid (cache vide)", recherche_avec_cache)
    duree_chaude = mesurer("Recherche à chaud (cache rempli)", recherche_avec_cache)
    if duree_chaude > 0:
        print(f"\n  Accélération : {duree_froide / duree_chaude:.1f}x plus rapide à chaud")


if __name__ == "__main__":
    main()
