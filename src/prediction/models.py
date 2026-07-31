"""
Entraînement et comparaison de modèles pour prédire `offresRecues`.

Trois modèles comparés, du plus simple au plus complexe (même principe
que la comparaison Isolation Forest / LOF au bloc 2 : comparer plutôt
que choisir un seul modèle à l'aveugle) :
- Baseline naïve : prédit toujours la médiane. Sert à vérifier que les
  modèles suivants apportent réellement quelque chose — un modèle qui
  ne bat pas cette baseline ne vaut rien.
- Ridge (régression linéaire régularisée) : simple, rapide, interprétable
  (coefficients directement lisibles).
- Random Forest : capture les relations non linéaires, donne une
  importance des features.

Métriques : R² (variance expliquée), RMSE et MAE, calculées en espace
log (celui de l'entraînement) ET reconverties en espace réel (nombre
d'offres) via expm1, pour rester interprétables métier ("le modèle se
trompe en moyenne de X offres", pas juste "de X en log").
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.dummy import DummyRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import Ridge
from sklearn.metrics import make_scorer, mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import KFold, RandomizedSearchCV, train_test_split

RANDOM_STATE = 42


def separer_train_test(X: pd.DataFrame, y: pd.Series, test_size: float = 0.2):
    """Découpe train/test.

    Pas de risque de fuite par cotraitance ici : X et y proviennent de
    `construire_matrice_features`, qui déduplique déjà par marché en
    amont (un seul `uid` par ligne) — un même marché ne peut donc pas se
    retrouver à la fois dans le train et dans le test.
    """
    return train_test_split(X, y, test_size=test_size, random_state=RANDOM_STATE)


def evaluer_predictions(y_true_log: np.ndarray, y_pred_log: np.ndarray) -> dict:
    """Calcule les métriques en espace log (entraînement) et en espace
    réel (expm1, nombre d'offres) pour rester interprétable métier.

    Clip à 0 après expm1 : le modèle peut prédire une valeur log légèrement
    négative (bruit), ce qui donnerait un nombre d'offres < 0 après
    reconversion — non-sens métier, corrigé par un plancher à 0.

    Rendue publique (préfixe `_` retiré) quand run_hyperparameter_search.py
    en a eu besoin pour évaluer un modèle sur un jeu de test tenu à part —
    même principe que `retirer_accents()` au bloc 3/prédiction : une
    fonction utilisée dans un seul module au départ, formalisée en API
    partagée dès qu'un deuxième module en a besoin, plutôt que dupliquée.
    """
    y_true_reel = np.expm1(y_true_log)
    y_pred_reel = np.clip(np.expm1(y_pred_log), 0, None)

    return {
        "r2_log": round(r2_score(y_true_log, y_pred_log), 3),
        "rmse_log": round(np.sqrt(mean_squared_error(y_true_log, y_pred_log)), 3),
        "r2_reel": round(r2_score(y_true_reel, y_pred_reel), 3),
        "rmse_reel": round(np.sqrt(mean_squared_error(y_true_reel, y_pred_reel)), 2),
        "mae_reel": round(mean_absolute_error(y_true_reel, y_pred_reel), 2),
    }


def entrainer_et_comparer(X: pd.DataFrame, y: pd.Series) -> dict:
    """Entraîne baseline, Ridge et Random Forest, et retourne leurs
    métriques comparées sur le même jeu de test.
    """
    X_train, X_test, y_train, y_test = separer_train_test(X, y)

    resultats = {}

    baseline = DummyRegressor(strategy="median")
    baseline.fit(X_train, y_train)
    resultats["baseline_mediane"] = evaluer_predictions(y_test.values, baseline.predict(X_test))

    ridge = Ridge(random_state=RANDOM_STATE)
    ridge.fit(X_train, y_train)
    resultats["ridge"] = evaluer_predictions(y_test.values, ridge.predict(X_test))

    rf = RandomForestRegressor(
        n_estimators=100, max_depth=15, random_state=RANDOM_STATE, n_jobs=-1
    )
    rf.fit(X_train, y_train)
    resultats["random_forest"] = evaluer_predictions(y_test.values, rf.predict(X_test))

    resultats["_rf_feature_importance"] = pd.Series(
        rf.feature_importances_, index=X.columns
    ).sort_values(ascending=False)

    return resultats


def validation_croisee(X: pd.DataFrame, y: pd.Series, modele, n_splits: int = 5) -> dict:
    """Validation croisée K-fold pour un modèle donné (pas un seul
    découpage train/test).

    Pourquoi : le résultat d'un seul découpage train/test (voir
    entrainer_et_comparer) pourrait être un coup de chance ou de
    malchance selon les marchés qui atterrissent dans le test — même
    principe que la vérification de stabilité déjà faite pour la
    détection d'anomalies au bloc 2 (plusieurs tirages plutôt qu'un seul).

    Retourne les métriques moyennes et l'écart-type sur les n_splits
    plis, en espace réel (interprétable métier), plus le détail par pli.
    """
    kfold = KFold(n_splits=n_splits, shuffle=True, random_state=RANDOM_STATE)

    resultats_par_pli = []
    for train_idx, test_idx in kfold.split(X):
        X_train, X_test = X.iloc[train_idx], X.iloc[test_idx]
        y_train, y_test = y.iloc[train_idx], y.iloc[test_idx]

        modele.fit(X_train, y_train)
        pred = modele.predict(X_test)
        resultats_par_pli.append(evaluer_predictions(y_test.values, pred))

    r2_reel_list = [r["r2_reel"] for r in resultats_par_pli]
    mae_reel_list = [r["mae_reel"] for r in resultats_par_pli]

    return {
        "n_splits": n_splits,
        "r2_reel_moyen": round(float(np.mean(r2_reel_list)), 3),
        "r2_reel_ecart_type": round(float(np.std(r2_reel_list)), 3),
        "mae_reel_moyen": round(float(np.mean(mae_reel_list)), 2),
        "mae_reel_ecart_type": round(float(np.std(mae_reel_list)), 2),
        "detail_par_pli": resultats_par_pli,
    }


def _r2_espace_reel(y_true_log: np.ndarray, y_pred_log: np.ndarray) -> float:
    """R² calculé en espace réel (nombre d'offres) plutôt qu'en espace log.

    À utiliser comme fonction de score pour la recherche d'hyperparamètres
    (via make_scorer), suite à la découverte qu'optimiser le R² en espace
    log peut sélectionner un modèle moins bon en espace réel — voir
    docs/decisions.md, section "Recherche d'hyperparamètres : optimiser en
    espace log peut dégrader le résultat réel".
    """
    y_true_reel = np.expm1(y_true_log)
    y_pred_reel = np.clip(np.expm1(y_pred_log), 0, None)
    return r2_score(y_true_reel, y_pred_reel)


SCORER_ESPACE_REEL = make_scorer(_r2_espace_reel, greater_is_better=True)


def rechercher_hyperparametres(
    X: pd.DataFrame, y: pd.Series, n_iter: int = 15, scoring=SCORER_ESPACE_REEL
) -> dict:
    """Recherche d'hyperparamètres pour Random Forest via RandomizedSearchCV.

    RandomizedSearchCV plutôt que GridSearchCV exhaustif : avec 4
    hyperparamètres à explorer, une grille complète serait coûteuse en
    temps ; un tirage aléatoire de n_iter combinaisons donne un bon
    compromis exploration/temps de calcul, cohérent avec la contrainte
    de temps déjà rencontrée ailleurs dans ce projet (ex: la taille
    d'échantillon du bloc recherche).

    scoring : SCORER_ESPACE_REEL par défaut (pas le R² scikit-learn
    standard, qui note en espace log). Changement suite à la découverte
    qu'optimiser le R² en espace log peut sélectionner un modèle moins
    bon en espace réel (voir docs/decisions.md) — corrigé ici en scorant
    directement dans l'espace qui compte pour l'usage métier.
    """
    parametres = {
        "n_estimators": [50, 100, 200, 300],
        "max_depth": [8, 12, 15, 20, None],
        "min_samples_leaf": [1, 2, 5, 10],
        "max_features": ["sqrt", "log2", 0.5],
    }

    recherche = RandomizedSearchCV(
        RandomForestRegressor(random_state=RANDOM_STATE, n_jobs=-1),
        param_distributions=parametres,
        n_iter=n_iter,
        cv=3,
        scoring=scoring,
        random_state=RANDOM_STATE,
        n_jobs=-1,
    )
    recherche.fit(X, y)

    return {
        "meilleurs_parametres": recherche.best_params_,
        "meilleur_score_cv": round(recherche.best_score_, 3),
        "modele": recherche.best_estimator_,
    }
