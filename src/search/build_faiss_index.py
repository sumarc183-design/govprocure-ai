"""
Construit un vrai index FAISS sur un échantillon réaliste de marchés
uniques, et le compare au bruteforce numpy actuellement utilisé par
`EmbeddingIndex` (embeddings_search.py) — réponse concrète à la piste
notée dans docs/limitations.md ("construire un vrai index FAISS/Qdrant",
jusqu'ici seulement un cache disque .npz, voir docs/decisions.md pour le
raisonnement qui avait motivé ce choix intermédiaire).

Lancement : python -m src.search.build_faiss_index

Encode un nouvel échantillon de TAILLE_ECHANTILLON marchés (au-delà des
15 402 déjà présents dans embeddings_cache.npz — trop petit pour un test
représentatif). Premier lancement lent (CPU, plusieurs minutes) ; les
lancements suivants réutilisent data/processed/embeddings_cache_faiss_demo.npz.

Ce qui n'est PAS fait ici (limite assumée, cohérente avec
docs/decisions.md) : précalculer les 1,73M marchés uniques du corpus
complet, estimé à plusieurs heures sur cette machine. Ce script
démontre l'index à une échelle comparable à celle déjà testée ailleurs
dans le projet (échantillons "sans filtre" de 15 000 à 100 000 lignes),
pas sur le corpus entier.
"""

from __future__ import annotations

import time

from src.anomaly.features import deduplicate_marches
from src.quality.loader import load_current_only
from src.search.embeddings_search import EmbeddingIndex, _get_model
from src.search.faiss_index import FaissEmbeddingIndex, recherche_bruteforce

TAILLE_ECHANTILLON = 50_000
CHEMIN_CACHE = "data/processed/embeddings_cache_faiss_demo.npz"
CHEMIN_INDEX_FAISS = "data/processed/faiss_index_demo.bin"
REQUETE_BENCHMARK = "entretien des espaces verts et tonte"
N_REPETITIONS = 20


def mesurer(label: str, fonction, n_repeats: int = N_REPETITIONS):
    t0 = time.perf_counter()
    resultat = None
    for _ in range(n_repeats):
        resultat = fonction()
    duree = (time.perf_counter() - t0) / n_repeats
    print(f"  {label:<45} {duree * 1000:>9.4f} ms")
    return duree, resultat


def main():
    df = load_current_only(
        "data/raw/decp.parquet",
        columns=["uid", "objet", "montant", "acheteur_region_nom"],
    )
    df = df.dropna(subset=["objet"])
    df = deduplicate_marches(df)
    sample = df.sample(n=min(TAILLE_ECHANTILLON, len(df)), random_state=7)
    print(f"Corpus : {len(sample):,} marchés uniques")

    t0 = time.perf_counter()
    embedding_index = EmbeddingIndex(sample, cache_path=CHEMIN_CACHE)
    print(f"Encodage/chargement du cache : {time.perf_counter() - t0:.1f}s\n")

    embeddings = embedding_index.embeddings.astype("float32")
    sample_ordonne = embedding_index.df

    model = _get_model()
    query_embedding = model.encode([REQUETE_BENCHMARK], normalize_embeddings=True)[0].astype("float32")

    print(f'Requête benchmark : "{REQUETE_BENCHMARK}"')
    print(f"Recherche répétée {N_REPETITIONS}x sur {len(sample):,} vecteurs (temps moyen) :\n")

    duree_bf, (idx_bf, _scores_bf) = mesurer(
        "Bruteforce numpy (actuel)",
        lambda: recherche_bruteforce(embeddings, query_embedding, top_k=20),
    )

    index_flat = FaissEmbeddingIndex(sample_ordonne, embeddings, use_ivf=False)
    duree_flat, result_flat = mesurer(
        "FAISS IndexFlatIP (exact)",
        lambda: index_flat.search(query_embedding, top_k=20),
    )

    index_ivf = FaissEmbeddingIndex(sample_ordonne, embeddings, use_ivf=True, nlist=100)
    duree_ivf, result_ivf = mesurer(
        "FAISS IndexIVFFlat (approx, nlist=100)",
        lambda: index_ivf.search(query_embedding, top_k=20),
    )

    # Vérification de correction : IndexFlatIP doit retourner les mêmes
    # marchés que le bruteforce (même méthode mathématique, seule la
    # vitesse change) — pas juste "plus rapide", aussi correct. IVF est
    # approximatif : un recouvrement partiel est attendu, pas un défaut.
    uids_bruteforce = set(sample_ordonne.iloc[idx_bf]["uid"])
    uids_flat = set(result_flat["uid"])
    uids_ivf = set(result_ivf["uid"])
    accord_flat = len(uids_bruteforce & uids_flat) / len(uids_bruteforce)
    accord_ivf = len(uids_bruteforce & uids_ivf) / len(uids_bruteforce)

    print("\nAccord avec le bruteforce (recouvrement du top 20) :")
    print(f"  FAISS Flat (exact)                     {accord_flat:.0%}")
    print(f"  FAISS IVF (approx, nlist=100)           {accord_ivf:.0%}")

    print(f"\nAccélération FAISS Flat vs bruteforce : {duree_bf / duree_flat:.2f}x")
    print(f"Accélération FAISS IVF vs bruteforce  : {duree_bf / duree_ivf:.2f}x")

    index_flat.save(CHEMIN_INDEX_FAISS)
    print(f"\nIndex FAISS sauvegardé : {CHEMIN_INDEX_FAISS} ({len(sample):,} vecteurs)")


if __name__ == "__main__":
    main()
