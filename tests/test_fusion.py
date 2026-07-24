import pandas as pd
import pytest

from src.search.fusion import reciprocal_rank_fusion


def test_rrf_favors_document_ranked_high_in_both():
    """Un document 1er dans les deux classements doit dominer le résultat."""
    ranking_bm25 = pd.DataFrame({"uid": ["a", "b", "c"]})
    ranking_embeddings = pd.DataFrame({"uid": ["a", "c", "b"]})

    result = reciprocal_rank_fusion([ranking_bm25, ranking_embeddings])
    assert result.iloc[0]["uid"] == "a"


def test_rrf_handles_disjoint_rankings():
    """Un document présent dans un seul classement doit quand même apparaître,
    avec un score plus faible qu'un document présent dans les deux.
    """
    ranking_bm25 = pd.DataFrame({"uid": ["a", "b"]})
    ranking_embeddings = pd.DataFrame({"uid": ["c", "d"]})

    result = reciprocal_rank_fusion([ranking_bm25, ranking_embeddings])
    assert set(result["uid"]) == {"a", "b", "c", "d"}
    # "a" et "c" sont 1er dans leur classement respectif -> même score, ex-aequo
    top_uids = set(result.iloc[0:2]["uid"])
    assert top_uids == {"a", "c"}


def test_rrf_document_in_both_beats_document_in_one():
    ranking_bm25 = pd.DataFrame({"uid": ["a", "b"]})
    ranking_embeddings = pd.DataFrame({"uid": ["b", "c"]})

    result = reciprocal_rank_fusion([ranking_bm25, ranking_embeddings])
    # "b" apparaît dans les deux classements -> doit être 1er
    assert result.iloc[0]["uid"] == "b"


def test_rrf_score_column_present_and_sorted():
    ranking = pd.DataFrame({"uid": ["a", "b", "c"]})
    result = reciprocal_rank_fusion([ranking])
    assert "rrf_score" in result.columns
    scores = result["rrf_score"].tolist()
    assert scores == sorted(scores, reverse=True)
