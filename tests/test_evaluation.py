import pandas as pd
import pytest

from src.search.evaluation import (
    REQUETES_TEST,
    RequeteTest,
    est_pertinent,
    evaluer_requete,
    ndcg_at_k,
    precision_at_k,
)


def test_est_pertinent_insensible_accents_casse():
    assert est_pertinent("SECURITE DES SYSTEMES D INFORMATION", ["sécurité"])
    assert est_pertinent("Sécurité des systèmes", ["securite"])
    assert not est_pertinent("Travaux de voirie", ["cybersécurité"])


def test_precision_at_k_cas_parfait():
    objets = ["sécurité informatique A", "sécurité informatique B", "travaux voirie"]
    # 2 pertinents sur les 2 premiers -> precision@2 = 1.0
    assert precision_at_k(objets, ["sécurité"], k=2) == 1.0


def test_precision_at_k_cas_partiel():
    objets = ["sécurité informatique", "travaux voirie", "travaux voirie 2"]
    # 1 pertinent sur les 3 premiers -> precision@3 = 1/3
    assert precision_at_k(objets, ["sécurité"], k=3) == pytest.approx(1 / 3)


def test_precision_at_k_moins_de_k_resultats_divise_par_k():
    """Correction suite à revue externe : Precision@K doit diviser par k
    (définition standard), pas par le nombre réel de résultats retournés.
    Un seul résultat pertinent pour k=10 doit donner 0.1, pas 1.0 —
    sinon un moteur qui retourne trop peu de résultats est faussement
    récompensé.
    """
    objets = ["sécurité informatique"]
    assert precision_at_k(objets, ["sécurité"], k=10) == pytest.approx(0.1)


def test_precision_at_k_liste_vide():
    assert precision_at_k([], ["sécurité"], k=10) == 0.0


def test_ndcg_at_k_classement_ideal_avec_corpus_complet():
    """NDCG = 1.0 seulement si le classement est idéal ET que tous les
    documents pertinents du corpus sont présents dans le top k.
    """
    objets = ["sécurité A", "sécurité B", "travaux voirie"]
    # 2 pertinents trouvés, et le corpus n'en contient que 2 au total -> idéal
    assert ndcg_at_k(objets, ["sécurité"], k=3, n_pertinents_corpus=2) == pytest.approx(1.0)


def test_ndcg_at_k_penalise_documents_pertinents_manques():
    """Correction suite à revue externe : si le corpus contient plus de
    documents pertinents que ceux trouvés dans le top k, NDCG doit être
    strictement inférieur à 1.0 — reproduit le cas observé en pratique
    (Precision@10=0.7 avec NDCG@10=1.0, incohérent avec l'ancien calcul).
    """
    objets = ["sécurité A", "sécurité B", "travaux voirie"]
    # 2 pertinents trouvés, mais le corpus en contient 10 au total -> pénalisé
    ndcg = ndcg_at_k(objets, ["sécurité"], k=3, n_pertinents_corpus=10)
    assert ndcg < 1.0


def test_ndcg_at_k_classement_inverse_penalise():
    """Le même nombre de résultats pertinents, mais classés en dernier,
    doit donner un NDCG strictement inférieur au classement idéal.
    """
    ideal = ["sécurité A", "sécurité B", "travaux voirie", "travaux X"]
    inverse = ["travaux voirie", "travaux X", "sécurité A", "sécurité B"]
    ndcg_ideal = ndcg_at_k(ideal, ["sécurité"], k=4, n_pertinents_corpus=2)
    ndcg_inverse = ndcg_at_k(inverse, ["sécurité"], k=4, n_pertinents_corpus=2)
    assert ndcg_ideal == pytest.approx(1.0)
    assert ndcg_inverse < ndcg_ideal


def test_ndcg_at_k_aucun_pertinent_dans_corpus_retourne_zero():
    objets = ["travaux voirie", "travaux voirie 2"]
    assert ndcg_at_k(objets, ["sécurité"], k=2, n_pertinents_corpus=0) == 0.0


def test_evaluer_requete_structure_resultat():
    requete = RequeteTest(
        nom="test", requete="sécurité", mots_cles_pertinence=["sécurité"]
    )
    df = pd.DataFrame({"objet": ["sécurité A", "travaux voirie"]})
    df_corpus = pd.DataFrame({"objet": ["sécurité A", "travaux voirie"]})
    resultat = evaluer_requete(df, requete, k_values=[1, 2], df_corpus=df_corpus)

    assert resultat["nom"] == "test"
    assert resultat["n_resultats"] == 2
    assert "precision@1" in resultat
    assert "ndcg@1" in resultat
    assert "precision@2" in resultat
    assert "ndcg@2" in resultat


def test_evaluer_requete_ndcg_none_sans_corpus():
    """Sans df_corpus, NDCG ne peut pas être calculé correctement (il
    nécessite n_pertinents_corpus) — doit valoir None plutôt qu'un chiffre
    silencieusement biaisé.
    """
    requete = RequeteTest(
        nom="test", requete="sécurité", mots_cles_pertinence=["sécurité"]
    )
    df = pd.DataFrame({"objet": ["sécurité A", "travaux voirie"]})
    resultat = evaluer_requete(df, requete, k_values=[2])
    assert resultat["ndcg@2"] is None
    assert resultat["precision@2"] is not None  # precision reste calculable sans corpus


def test_requetes_test_predefinies_non_vides():
    """Vérifie que le jeu de requêtes de test prédéfini est bien formé
    (non-régression si quelqu'un modifie REQUETES_TEST par erreur).
    """
    assert len(REQUETES_TEST) >= 3
    for rt in REQUETES_TEST:
        assert rt.requete.strip() != ""
        assert len(rt.mots_cles_pertinence) > 0


def test_requetes_difficiles_meme_verite_terrain_que_faciles():
    """Les requêtes difficiles doivent partager la vérité terrain (mots-clés)
    de leur équivalent facile, mais avec un texte de requête totalement
    différent (aucun mot significatif en commun), pour tester spécifiquement
    l'apport des embeddings sans recouvrement lexical.
    """
    from src.search.evaluation import REQUETES_TEST_DIFFICILES

    mots_vides = {"le", "la", "les", "un", "une", "des", "de", "du", "et", "en"}

    assert len(REQUETES_TEST_DIFFICILES) == len(REQUETES_TEST)
    for facile, difficile in zip(REQUETES_TEST, REQUETES_TEST_DIFFICILES):
        assert facile.mots_cles_pertinence == difficile.mots_cles_pertinence
        mots_facile = set(facile.requete.lower().split()) - mots_vides
        mots_difficile = set(difficile.requete.lower().split()) - mots_vides
        assert mots_facile.isdisjoint(mots_difficile), (
            f"Recouvrement lexical inattendu entre '{facile.requete}' "
            f"et '{difficile.requete}'"
        )


def test_evaluer_requete_calcule_recall_si_corpus_fourni():
    requete = RequeteTest(
        nom="test", requete="sécurité", mots_cles_pertinence=["sécurité"]
    )
    df_resultats = pd.DataFrame({"objet": ["sécurité A", "travaux voirie"]})
    df_corpus = pd.DataFrame({
        "objet": ["sécurité A", "sécurité B", "sécurité C", "travaux voirie"]
    })
    resultat = evaluer_requete(df_resultats, requete, k_values=[2], df_corpus=df_corpus)

    assert resultat["n_pertinents_corpus"] == 3  # A, B, C dans le corpus
    # 1 seul pertinent trouvé sur 3 dans le corpus -> recall@2 = 1/3 (arrondi à 3 décimales)
    assert resultat["recall@2"] == round(1 / 3, 3)


def test_evaluer_requete_recall_none_si_pas_de_pertinent_dans_corpus():
    requete = RequeteTest(
        nom="test", requete="sécurité", mots_cles_pertinence=["inexistant_xyz"]
    )
    df_resultats = pd.DataFrame({"objet": ["travaux voirie"]})
    df_corpus = pd.DataFrame({"objet": ["travaux voirie"]})
    resultat = evaluer_requete(df_resultats, requete, k_values=[5], df_corpus=df_corpus)
    assert resultat["n_pertinents_corpus"] == 0
    assert resultat["recall@5"] is None
