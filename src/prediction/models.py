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
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import train_test_split

RANDOM_STATE = 42


def separer_train_test(X: pd.DataFrame, y: pd.Series, test_size: float = 0.2):
    """Découpe train/test.

    Pas de risque de fuite par cotraitance ici : X et y proviennent de
    `construire_matrice_features`, qui déduplique déjà par marché en
    amont (un seul `uid` par ligne) — un même marché ne peut donc pas se
    retrouver à la fois dans le train et dans le test.
    """
    return train_test_split(X, y, test_size=test_size, random_state=RANDOM_STATE)


def _evaluer(y_true_log: np.ndarray, y_pred_log: np.ndarray) -> dict:
    """Calcule les métriques en espace log (entraînement) et en espace
    réel (expm1, nombre d'offres) pour rester interprétable métier.

    Clip à 0 après expm1 : le modèle peut prédire une valeur log légèrement
    négative (bruit), ce qui donnerait un nombre d'offres < 0 après
    reconversion — non-sens métier, corrigé par un plancher à 0.
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
    resultats["baseline_mediane"] = _evaluer(y_test.values, baseline.predict(X_test))

    ridge = Ridge(random_state=RANDOM_STATE)
    ridge.fit(X_train, y_train)
    resultats["ridge"] = _evaluer(y_test.values, ridge.predict(X_test))

    rf = RandomForestRegressor(
        n_estimators=100, max_depth=15, random_state=RANDOM_STATE, n_jobs=-1
    )
    rf.fit(X_train, y_train)
    resultats["random_forest"] = _evaluer(y_test.values, rf.predict(X_test))

    resultats["_rf_feature_importance"] = pd.Series(
        rf.feature_importances_, index=X.columns
    ).sort_values(ascending=False)

    return resultats
