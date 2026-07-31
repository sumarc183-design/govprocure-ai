"""
Moteur de recherche hybride : orchestre l'ensemble du pipeline.

1. Parser la requête en langage naturel (filtres stricts + texte libre)
2. Appliquer les filtres stricts sur le dataset (réduction avant recherche)
3. Recherche BM25 sur le sous-ensemble filtré
4. Recherche par embeddings sur le même sous-ensemble filtré
5. Fusion des deux classements par RRF
6. Retour du top K avec une explication des critères utilisés

Voir docs/decisions.md pour le raisonnement détaillé derrière cette
architecture.
"""

from __future__ import annotations

import pandas as pd

from src.anomaly.features import deduplicate_marches
from src.search.bm25_search import BM25Index
from src.search.embeddings_search import EmbeddingIndex
from src.search.filters import ParsedQuery, parse_query
from src.search.fusion import reciprocal_rank_fusion


def apply_strict_filters(df: pd.DataFrame, parsed: ParsedQuery) -> pd.DataFrame:
    """Applique les filtres stricts extraits de la requête, puis déduplique
    au niveau marché.

    Le montant qualitatif ("montant élevé") est résolu ici en percentile
    du dataset *déjà réduit par les autres filtres* (ex: le 90e percentile
    des marchés en Île-de-France, pas du dataset entier) — plus pertinent
    qu'un seuil absolu qui ignorerait le contexte régional/sectoriel.

    Déduplication par `uid` (via `deduplicate_marches`, module anomalies) :
    cohérent avec la découverte du bloc 1 (un même marché peut avoir
    plusieurs lignes, une par titulaire, en cas de cotraitance/accord-cadre
    multi-attributaire — voir docs/decisions.md). Sans cette déduplication,
    un utilisateur verrait le même marché apparaître plusieurs fois à
    l'identique dans les résultats de recherche, ce qui dégrade l'expérience
    alors que la pertinence de recherche (objet, montant, région) est
    identique pour toutes ces lignes.
    """
    result = df.copy()

    if parsed.region is not None and "acheteur_region_nom" in result.columns:
        result = result[result["acheteur_region_nom"] == parsed.region]

    if parsed.montant_min is not None:
        result = result[result["montant"] >= parsed.montant_min]
    elif parsed.montant_eleve_qualitatif and len(result) > 0:
        seuil = result["montant"].quantile(0.90)
        result = result[result["montant"] >= seuil]

    return deduplicate_marches(result)


def search(
    df: pd.DataFrame,
    query: str,
    top_k: int = 20,
    bm25_candidates: int = 200,
    embedding_candidates: int = 200,
    cache_embeddings_path: str | None = "data/processed/embeddings_cache.npz",
    poids_fusion: tuple[float, float] | None = None,
    embedding_model_name: str | None = None,
) -> tuple[pd.DataFrame, dict]:
    """Exécute le pipeline de recherche hybride complet.

    Retourne (résultats, explication) où explication détaille les
    critères utilisés (filtres appliqués, taille du sous-ensemble
    résultant), pour répondre à l'exigence de transparence de la mission.

    bm25_candidates / embedding_candidates : nombre de candidats retenus
    par chaque méthode avant fusion. Plus grand que top_k pour laisser
    la fusion RRF un vrai choix parmi plusieurs candidats de chaque
    méthode, pas seulement le top_k déjà tronqué par chacune.

    cache_embeddings_path : fichier de cache des embeddings (voir
    embeddings_search.py). Activé par défaut — chaque marché déjà
    rencontré lors d'une recherche précédente réutilise son vecteur
    plutôt que de le recalculer. Passer None pour désactiver (utile en
    test, pour ne pas laisser de fichier de cache derrière soi).

    poids_fusion : (poids_bm25, poids_embeddings) pour la fusion RRF.
    None (par défaut) = poids égaux, comportement d'origine inchangé.
    Ajouté suite à la découverte du bloc 5 (cas "travaux de voirie") où
    la fusion à poids égal a dégradé un résultat BM25 correct par
    coïncidence — voir docs/decisions.md pour le résultat empirique de
    la comparaison.

    embedding_model_name : None (par défaut) = MiniLM (voir
    embeddings_search._MODEL_NAME), comportement d'origine inchangé.
    Permet de tester un modèle d'embeddings plus grand (voir
    compare_embedding_models.py) sans dupliquer le pipeline complet.
    """
    parsed = parse_query(query)
    df_filtre = apply_strict_filters(df, parsed)

    explication = {
        "region_filtree": parsed.region,
        "montant_min_applique": parsed.montant_min,
        "montant_eleve_qualitatif": parsed.montant_eleve_qualitatif,
        "texte_recherche": parsed.texte_libre,
        "n_marches_avant_filtre": len(df),
        "n_marches_apres_filtre_strict": len(df_filtre),
    }

    if len(df_filtre) == 0 or not parsed.texte_libre.strip():
        return df_filtre.head(top_k), explication

    bm25_index = BM25Index(df_filtre)
    ranking_bm25 = bm25_index.search(parsed.texte_libre, top_k=bm25_candidates)

    if embedding_model_name is not None:
        embedding_index = EmbeddingIndex(
            df_filtre, cache_path=cache_embeddings_path, model_name=embedding_model_name
        )
    else:
        embedding_index = EmbeddingIndex(df_filtre, cache_path=cache_embeddings_path)
    ranking_embeddings = embedding_index.search(
        parsed.texte_libre, top_k=embedding_candidates
    )

    weights = list(poids_fusion) if poids_fusion is not None else None
    fused = reciprocal_rank_fusion([ranking_bm25, ranking_embeddings], weights=weights)
    fused_top = fused.head(top_k)

    result = df_filtre[df_filtre["uid"].isin(fused_top["uid"])].merge(
        fused_top, on="uid"
    )
    result = result.sort_values("rrf_score", ascending=False).reset_index(drop=True)

    explication["n_candidats_bm25"] = len(ranking_bm25)
    explication["n_candidats_embeddings"] = len(ranking_embeddings)

    return result, explication
