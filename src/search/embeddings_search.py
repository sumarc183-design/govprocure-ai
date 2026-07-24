"""
Recherche par similarité sémantique (embeddings) sur le champ `objet`.

Modèle utilisé : paraphrase-multilingual-MiniLM-L12-v2 (sentence-transformers).
Choisi pour deux raisons :
- multilingue, donc adapté au français sans fine-tuning ;
- petit modèle (~118M paramètres) : rapide à exécuter en CPU, important
  vu qu'on n'a pas de GPU garanti dans l'environnement de déploiement.

Limite connue : contrairement à BM25, l'encodage de tout le corpus est
coûteux (doit être fait une fois par texte, puis réutilisable). Sur un
sous-ensemble déjà filtré (voir engine.py) ça reste rapide ; sur les 3M
lignes complètes, il faudrait un encodage préalable et un stockage des
vecteurs (ex: FAISS/Qdrant), pas un recalcul à chaque requête — décision
à prendre selon le volume réel en production (voir docs/limitations.md).
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from sentence_transformers import SentenceTransformer

_MODEL_NAME = "paraphrase-multilingual-MiniLM-L12-v2"
_model_cache: SentenceTransformer | None = None


def _get_model() -> SentenceTransformer:
    """Charge le modèle une seule fois par process (coûteux à recharger)."""
    global _model_cache
    if _model_cache is None:
        _model_cache = SentenceTransformer(_MODEL_NAME)
    return _model_cache


class EmbeddingIndex:
    """Encapsule un index de similarité sémantique sur la colonne `objet`."""

    def __init__(self, df: pd.DataFrame, text_column: str = "objet"):
        self.df = df.reset_index(drop=True)
        self.text_column = text_column
        model = _get_model()
        texts = self.df[text_column].fillna("").astype(str).tolist()
        self.embeddings = model.encode(
            texts, show_progress_bar=False, normalize_embeddings=True
        )

    def search(self, query: str, top_k: int = 50) -> pd.DataFrame:
        """Retourne les top_k lignes les plus proches sémantiquement de la requête.

        Similarité cosinus : comme les embeddings sont normalisés
        (normalize_embeddings=True), un simple produit scalaire suffit
        (équivalent à la similarité cosinus sans division supplémentaire).
        """
        if not query.strip():
            return self.df.iloc[0:0].copy()

        model = _get_model()
        query_embedding = model.encode([query], normalize_embeddings=True)[0]

        scores = self.embeddings @ query_embedding  # produit scalaire = cosinus ici

        result = self.df.copy()
        result["embedding_score"] = scores
        result = result.sort_values("embedding_score", ascending=False).head(top_k)
        return result
