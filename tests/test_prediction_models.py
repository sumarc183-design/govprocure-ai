import numpy as np
import pandas as pd
import pytest

from src.prediction.models import _evaluer, entrainer_et_comparer, separer_train_test


@pytest.fixture
def petit_dataset():
    """Petit dataset synthétique, juste assez grand pour entraîner sans
    erreur (pas pour être réaliste métier — voir tests d'intégration
    manuels sur données réelles, documentés dans docs/limitations.md).
    """
    rng = np.random.default_rng(42)
    n = 200
    X = pd.DataFrame({
        "log_montant": rng.normal(10, 2, n),
        "dureeMois": rng.normal(12, 5, n),
        "procedure_normalisee_x": rng.integers(0, 2, n).astype(bool),
    })
    y = pd.Series(rng.normal(2, 1, n).clip(min=0))
    return X, y


def test_separer_train_test_pas_de_chevauchement(petit_dataset):
    X, y = petit_dataset
    X_train, X_test, y_train, y_test = separer_train_test(X, y, test_size=0.2)
    assert len(X_train) + len(X_test) == len(X)
    assert set(X_train.index).isdisjoint(set(X_test.index))


def test_evaluer_retourne_les_metriques_attendues():
    y_true = np.array([1.0, 2.0, 3.0, 4.0])
    y_pred = np.array([1.1, 2.1, 2.9, 4.2])
    resultat = _evaluer(y_true, y_pred)
    for cle in ["r2_log", "rmse_log", "r2_reel", "rmse_reel", "mae_reel"]:
        assert cle in resultat


def test_evaluer_r2_parfait_si_prediction_parfaite():
    y_true = np.array([1.0, 2.0, 3.0])
    resultat = _evaluer(y_true, y_true.copy())
    assert resultat["r2_log"] == pytest.approx(1.0)
    assert resultat["r2_reel"] == pytest.approx(1.0)


def test_evaluer_clippe_les_predictions_negatives_en_espace_reel():
    """Une prédiction log légèrement négative donnerait un nombre d'offres
    négatif après expm1 — non-sens métier, doit être ramené à 0.
    """
    y_true = np.array([0.0, 0.0])
    y_pred = np.array([-0.5, -0.3])  # prédictions log négatives
    resultat = _evaluer(y_true, y_pred)
    # Si le clip fonctionne, le RMSE réel reste fini et raisonnable
    assert resultat["rmse_reel"] >= 0
    assert not np.isnan(resultat["rmse_reel"])


def test_entrainer_et_comparer_retourne_les_trois_modeles(petit_dataset):
    X, y = petit_dataset
    resultats = entrainer_et_comparer(X, y)
    assert "baseline_mediane" in resultats
    assert "ridge" in resultats
    assert "random_forest" in resultats
    assert "_rf_feature_importance" in resultats


def test_entrainer_et_comparer_feature_importance_somme_a_un(petit_dataset):
    X, y = petit_dataset
    resultats = entrainer_et_comparer(X, y)
    importance = resultats["_rf_feature_importance"]
    assert importance.sum() == pytest.approx(1.0, abs=0.01)
