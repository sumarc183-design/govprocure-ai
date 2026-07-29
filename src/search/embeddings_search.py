"""
Recherche par similarité sémantique (embeddings) sur le champ `objet`.

Modèle utilisé : paraphrase-multilingual-MiniLM-L12-v2 (sentence-transformers).
Choisi pour deux raisons :
- multilingue, donc adapté au français sans fine-tuning ;
- petit modèle (~118M paramètres) : rapide à exécuter en CPU, important
  vu qu'on n'a pas de GPU garanti dans l'environnement de déploiement.

Cache disque des embeddings (voir EmbeddingIndex, paramètre cache_path) :
mesuré au bloc 5 (docs/limitations.md), le recalcul des embeddings à
chaque recherche domine presque intégralement le temps de réponse
(~28 secondes sur un cas d'usage typique). Le cache résout ce problème
pour les marchés déjà rencontrés lors d'une recherche précédente : leur
vecteur est stocké sur disque et réutilisé, seuls les marchés jamais vus
sont réellement encodés.

Limite assumée de cette solution : un cache local (fichier .npz) reste
une solution intermédiaire, adaptée à un sous-ensemble raisonnable
(quelques centaines de milliers de marchés). Une vraie mise à l'échelle
sur les 1,73M marchés uniques du corpus complet nécessiterait un calcul
batch (potentiellement des heures sur une machine sans GPU, mesuré
proportionnellement à partir du benchmark existant) et un index dédié
(FAISS/Qdrant), pas juste un fichier cache — voir docs/decisions.md.
"""

from __future__ import annotations

from pathlib import Path

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


def charger_cache_embeddings(chemin: str | Path) -> dict[str, np.ndarray]:
    """Charge un cache d'embeddings depuis un fichier .npz.

    Retourne un dict vide si le fichier n'existe pas encore (premier
    lancement) — pas une erreur, un cache vide est un état normal.
    """
    chemin = Path(chemin)
    if not chemin.exists():
        return {}
    data = np.load(chemin, allow_pickle=False)
    return {uid: data[uid] for uid in data.files}


def sauver_cache_embeddings(cache: dict[str, np.ndarray], chemin: str | Path) -> None:
    """Sauvegarde le cache sur disque, un tableau nommé par uid dans un .npz.

    Écrase le fichier existant avec la version mise à jour (cache
    cumulatif : les marchés déjà présents restent, les nouveaux s'ajoutent).
    """
    chemin = Path(chemin)
    chemin.parent.mkdir(parents=True, exist_ok=True)
    np.savez(chemin, **cache)


def identifier_uids_manquants(uids: list[str], cache: dict[str, np.ndarray]) -> list[str]:
    """Retourne les uids présents dans `uids` mais absents du cache.

    Logique pure, testable sans modèle ni réseau — sépare la décision
    "quoi encoder" du calcul d'encodage lui-même.
    """
    return [uid for uid in uids if uid not in cache]


class EmbeddingIndex:
    """Encapsule un index de similarité sémantique sur la colonne `objet`.

    cache_path (optionnel) : si fourni, réutilise les embeddings déjà
    calculés pour les marchés déjà rencontrés (identifiés par `uid`), et
    n'encode que les nouveaux. Le cache est mis à jour et sauvegardé à
    chaque appel — il grossit au fil des recherches successives.
    """

    def __init__(
        self,
        df: pd.DataFrame,
        text_column: str = "objet",
        cache_path: str | Path | None = None,
        id_column: str = "uid",
    ):
        self.df = df.reset_index(drop=True)
        self.text_column = text_column
        self.cache_path = cache_path
        model = _get_model()

        if cache_path is not None and id_column in self.df.columns:
            cache = charger_cache_embeddings(cache_path)
            uids = self.df[id_column].astype(str).tolist()
            uids_manquants = identifier_uids_manquants(uids, cache)

            if uids_manquants:
                textes_manquants = (
                    self.df[self.df[id_column].astype(str).isin(uids_manquants)]
                    [text_column].fillna("").astype(str).tolist()
                )
                nouveaux_vecteurs = model.encode(
                    textes_manquants, show_progress_bar=False, normalize_embeddings=True
                )
                for uid, vecteur in zip(uids_manquants, nouveaux_vecteurs):
                    cache[uid] = vecteur
                sauver_cache_embeddings(cache, cache_path)

            self.embeddings = np.array([cache[uid] for uid in uids])
        else:
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
