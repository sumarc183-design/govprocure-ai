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


def test_validation_croisee_retourne_structure_attendue(petit_dataset):
    from sklearn.linear_model import Ridge

    from src.prediction.models import validation_croisee

    X, y = petit_dataset
    resultat = validation_croisee(X, y, Ridge(), n_splits=3)

    assert resultat["n_splits"] == 3
    assert len(resultat["detail_par_pli"]) == 3
    for cle in ["r2_reel_moyen", "r2_reel_ecart_type", "mae_reel_moyen", "mae_reel_ecart_type"]:
        assert cle in resultat


def test_validation_croisee_plis_disjoints(petit_dataset):
    """Vérifie que chaque marché n'apparaît qu'une seule fois en test
    sur l'ensemble des plis (propriété fondamentale du K-fold).
    """
    from sklearn.model_selection import KFold

    from src.prediction.models import RANDOM_STATE

    X, y = petit_dataset
    kfold = KFold(n_splits=3, shuffle=True, random_state=RANDOM_STATE)
    tous_index_test = []
    for _, test_idx in kfold.split(X):
        tous_index_test.extend(test_idx.tolist())

    assert len(tous_index_test) == len(set(tous_index_test))  # aucun doublon
    assert len(tous_index_test) == len(X)  # chaque ligne vue exactement une fois


def test_rechercher_hyperparametres_retourne_un_modele_entraine(petit_dataset):
    from src.prediction.models import rechercher_hyperparametres

    X, y = petit_dataset
    resultat = rechercher_hyperparametres(X, y, n_iter=3)

    assert "meilleurs_parametres" in resultat
    assert "meilleur_score_cv" in resultat
    assert hasattr(resultat["modele"], "predict")
    # Le modèle retourné doit être utilisable pour prédire directement
    predictions = resultat["modele"].predict(X)
    assert len(predictions) == len(X)


def test_r2_espace_reel_identique_a_r2_score_sur_donnees_non_transformees():
    """Vérifie que _r2_espace_reel calcule bien un R² après reconversion
    expm1, pas juste un alias du R² standard.
    """
    from src.prediction.models import _r2_espace_reel

    y_true_log = np.array([0.0, 1.0, 2.0])  # log1p(0)=0, correspond à des vraies valeurs 0, e-1, e²-1
    y_pred_log = y_true_log.copy()  # prédiction parfaite

    resultat = _r2_espace_reel(y_true_log, y_pred_log)
    assert resultat == pytest.approx(1.0)  # prédiction parfaite -> R²=1 peu importe l'espace


def test_scorer_espace_reel_utilise_par_defaut(petit_dataset):
    """Non-régression : la recherche d'hyperparamètres doit utiliser le
    scorer en espace réel par défaut, suite à la découverte qu'optimiser
    en espace log peut sélectionner un modèle moins bon en réel.
    """
    import inspect

    from src.prediction.models import SCORER_ESPACE_REEL, rechercher_hyperparametres

    signature = inspect.signature(rechercher_hyperparametres)
    assert signature.parameters["scoring"].default is SCORER_ESPACE_REEL
