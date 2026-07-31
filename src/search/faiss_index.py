"""
Vrai index FAISS de similarité sémantique, précalculé une fois pour un
sous-ensemble de marchés, par opposition au cache disque de
`embeddings_search.py` (qui recalcule un produit scalaire en pur numpy
sur les vecteurs du sous-ensemble filtré à chaque recherche).

Pourquoi un module séparé plutôt qu'une modification de `EmbeddingIndex` :
`EmbeddingIndex` reste la voie utilisée par `engine.search()` en
production (rapide et suffisante pour des sous-ensembles filtrés de
quelques centaines de milliers de lignes, voir docs/decisions.md pour le
choix motivé du cache plutôt que FAISS par défaut). Ce module est le
prototype demandé dans docs/limitations.md ("construire un vrai index
FAISS/Qdrant") pour mesurer concrètement l'apport d'un index dédié,
proportionné à ce qui est réellement calculable sur cette machine
(voir build_faiss_index.py — précalculer les 1,73M marchés uniques du
corpus complet prendrait encore l'ordre de plusieurs heures, non refait
ici, cf. estimation déjà documentée).

IndexFlatIP : recherche exacte par produit scalaire (équivalent à la
similarité cosinus ici, les vecteurs étant normalisés) — donne
exactement les mêmes résultats que le bruteforce numpy actuel, seule la
vitesse change. IndexIVFFlat : recherche approximative (clustering),
plus rapide à grande échelle mais peut manquer quelques voisins proches
— utile seulement au-delà d'un nombre de vecteurs suffisant pour que le
clustering ait un sens (voir `nlist`).
"""

from __future__ import annotations

from pathlib import Path

import faiss
import numpy as np
import pandas as pd


class FaissEmbeddingIndex:
    """Encapsule un index FAISS construit sur des embeddings déjà calculés.

    Contrairement à `EmbeddingIndex` (embeddings_search.py), ne calcule
    aucun embedding lui-même : prend en entrée une matrice déjà encodée
    (voir build_faiss_index.py pour la construction du corpus encodé) et
    se contente d'indexer/rechercher dedans. Sépare volontairement
    "encoder du texte" (coûteux, modèle transformer) de "indexer des
    vecteurs" (rapide, FAISS) — les deux ne doivent pas être refaits
    ensemble à chaque test d'index.
    """

    def __init__(self, df: pd.DataFrame, embeddings: np.ndarray, use_ivf: bool = False, nlist: int = 100):
        self.df = df.reset_index(drop=True)
        self.embeddings = np.ascontiguousarray(embeddings, dtype="float32")
        dim = self.embeddings.shape[1]

        if use_ivf and len(self.embeddings) >= nlist * 40:
            quantizer = faiss.IndexFlatIP(dim)
            self.index = faiss.IndexIVFFlat(quantizer, dim, nlist, faiss.METRIC_INNER_PRODUCT)
            self.index.train(self.embeddings)
            self.index.nprobe = max(1, nlist // 10)
        else:
            self.index = faiss.IndexFlatIP(dim)
        self.index.add(self.embeddings)

    def search(self, query_embedding: np.ndarray, top_k: int = 50) -> pd.DataFrame:
        """Retourne les top_k lignes les plus proches sémantiquement de la
        requête déjà encodée (même convention que EmbeddingIndex.search,
        mais l'encodage de la requête reste à la charge de l'appelant ici
        pour ne pas recharger le modèle dans ce module dédié à l'index).
        """
        query = np.ascontiguousarray(query_embedding, dtype="float32").reshape(1, -1)
        scores, indices = self.index.search(query, top_k)

        result = self.df.iloc[indices[0]].copy()
        result["embedding_score"] = scores[0]
        return result

    def save(self, chemin: str | Path) -> None:
        chemin = Path(chemin)
        chemin.parent.mkdir(parents=True, exist_ok=True)
        faiss.write_index(self.index, str(chemin))


def recherche_bruteforce(embeddings: np.ndarray, query_embedding: np.ndarray, top_k: int = 50) -> np.ndarray:
    """Recherche par produit scalaire en pur numpy — même logique que
    `EmbeddingIndex.search` (embeddings_search.py), extraite ici pour
    comparer sa vitesse à FAISS sur exactement les mêmes vecteurs.
    """
    scores = embeddings @ query_embedding
    top_indices = np.argsort(-scores)[:top_k]
    return top_indices, scores[top_indices]
