"""
Fusion de plusieurs classements via Reciprocal Rank Fusion (RRF).

Pourquoi RRF plutôt qu'une moyenne pondérée des scores bruts : voir
docs/decisions.md, section "Bloc 3 — Recherche hybride". En résumé,
BM25 et la similarité cosinus des embeddings n'ont pas la même échelle ;
RRF ne compare que les rangs (positions dans chaque classement), ce qui
évite le problème de normalisation et de pondération arbitraire.

Formule : pour un document apparaissant au rang r dans un classement,
son score RRF pour ce classement est 1 / (k + r). k est une constante
(60 par défaut, valeur standard dans la littérature) qui amortit l'écart
entre les tout premiers rangs — sans k, la différence entre le rang 1 et
le rang 2 serait disproportionnée par rapport à la différence entre le
rang 50 et le rang 51.
"""

from __future__ import annotations

import pandas as pd

RRF_K = 60  # constante standard, voir docstring du module


def reciprocal_rank_fusion(
    rankings: list[pd.DataFrame],
    id_column: str = "uid",
    k: int = RRF_K,
) -> pd.DataFrame:
    """Fusionne plusieurs classements (dataframes triés par pertinence) en un seul.

    Chaque dataframe dans `rankings` doit être trié du plus pertinent au
    moins pertinent (l'ordre des lignes fait foi, pas une colonne de score
    particulière) et contenir la colonne `id_column` pour identifier les
    marchés de façon unique.

    Un marché absent d'un classement (ex: filtré en amont, ou score nul)
    ne contribue simplement pas au score RRF pour ce classement — il n'est
    pas pénalisé au-delà de son absence.

    Retourne un dataframe avec les colonnes [id_column, rrf_score], trié
    par rrf_score décroissant.
    """
    scores: dict = {}

    for ranking in rankings:
        for rang, uid in enumerate(ranking[id_column].tolist(), start=1):
            scores[uid] = scores.get(uid, 0.0) + 1.0 / (k + rang)

    result = pd.DataFrame(
        [{"uid": uid, "rrf_score": score} for uid, score in scores.items()]
    )
    result = result.rename(columns={"uid": id_column})
    result = result.sort_values("rrf_score", ascending=False).reset_index(drop=True)
    return result
