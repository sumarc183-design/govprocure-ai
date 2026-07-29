import numpy as np
import pytest

from src.search.embeddings_search import (
    charger_cache_embeddings,
    identifier_uids_manquants,
    sauver_cache_embeddings,
)


def test_charger_cache_fichier_absent_retourne_dict_vide(tmp_path):
    """Premier lancement : le fichier de cache n'existe pas encore.
    Ce n'est pas une erreur, juste un cache vide.
    """
    chemin = tmp_path / "cache_inexistant.npz"
    cache = charger_cache_embeddings(chemin)
    assert cache == {}


def test_sauver_puis_charger_cache_conserve_les_vecteurs(tmp_path):
    chemin = tmp_path / "cache.npz"
    cache_original = {
        "marche_1": np.array([0.1, 0.2, 0.3]),
        "marche_2": np.array([0.4, 0.5, 0.6]),
    }
    sauver_cache_embeddings(cache_original, chemin)

    cache_charge = charger_cache_embeddings(chemin)
    assert set(cache_charge.keys()) == {"marche_1", "marche_2"}
    np.testing.assert_array_almost_equal(cache_charge["marche_1"], [0.1, 0.2, 0.3])


def test_sauver_cache_cree_le_dossier_parent(tmp_path):
    chemin = tmp_path / "sous_dossier" / "cache.npz"
    sauver_cache_embeddings({"m1": np.array([1.0])}, chemin)
    assert chemin.exists()


def test_identifier_uids_manquants_tous_absents():
    uids = ["a", "b", "c"]
    cache = {}
    assert identifier_uids_manquants(uids, cache) == ["a", "b", "c"]


def test_identifier_uids_manquants_certains_presents():
    uids = ["a", "b", "c"]
    cache = {"b": np.array([1.0])}
    assert identifier_uids_manquants(uids, cache) == ["a", "c"]


def test_identifier_uids_manquants_tous_presents():
    uids = ["a", "b"]
    cache = {"a": np.array([1.0]), "b": np.array([2.0])}
    assert identifier_uids_manquants(uids, cache) == []


def test_cache_grandit_de_facon_cumulative(tmp_path):
    """Un cache existant, complété avec de nouveaux uids, doit conserver
    les anciens en plus des nouveaux (pas de perte lors de la mise à jour).
    """
    chemin = tmp_path / "cache.npz"
    sauver_cache_embeddings({"ancien": np.array([1.0, 2.0])}, chemin)

    cache = charger_cache_embeddings(chemin)
    cache["nouveau"] = np.array([3.0, 4.0])
    sauver_cache_embeddings(cache, chemin)

    cache_final = charger_cache_embeddings(chemin)
    assert set(cache_final.keys()) == {"ancien", "nouveau"}
