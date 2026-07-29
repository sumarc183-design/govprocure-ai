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


def test_rrf_poids_egaux_par_defaut_identique_a_avant():
    """Non-régression : sans weights, le comportement doit être strictement
    identique à la version non pondérée d'origine.
    """
    ranking_bm25 = pd.DataFrame({"uid": ["a", "b", "c"]})
    ranking_embeddings = pd.DataFrame({"uid": ["a", "c", "b"]})

    sans_weights = reciprocal_rank_fusion([ranking_bm25, ranking_embeddings])
    avec_poids_egaux = reciprocal_rank_fusion(
        [ranking_bm25, ranking_embeddings], weights=[1.0, 1.0]
    )
    pd.testing.assert_frame_equal(sans_weights, avec_poids_egaux)


def test_rrf_poids_plus_eleve_favorise_le_classement_correspondant():
    """Un poids plus élevé sur BM25 doit faire remonter les documents bien
    classés par BM25, même s'ils sont mal classés par les embeddings.
    """
    ranking_bm25 = pd.DataFrame({"uid": ["a", "b", "c"]})  # "a" en tête pour BM25
    ranking_embeddings = pd.DataFrame({"uid": ["c", "b", "a"]})  # "a" en dernier pour embeddings

    resultat_poids_egaux = reciprocal_rank_fusion([ranking_bm25, ranking_embeddings])
    resultat_bm25_favorise = reciprocal_rank_fusion(
        [ranking_bm25, ranking_embeddings], weights=[5.0, 1.0]
    )

    rang_a_egal = resultat_poids_egaux[resultat_poids_egaux["uid"] == "a"].index[0]
    rang_a_favorise = resultat_bm25_favorise[resultat_bm25_favorise["uid"] == "a"].index[0]
    assert rang_a_favorise <= rang_a_egal  # "a" remonte (ou reste) quand BM25 est favorisé


def test_rrf_weights_longueur_incorrecte_leve_erreur():
    ranking = pd.DataFrame({"uid": ["a", "b"]})
    with pytest.raises(ValueError, match="même longueur"):
        reciprocal_rank_fusion([ranking, ranking], weights=[1.0])
