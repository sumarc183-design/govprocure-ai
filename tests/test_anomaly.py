import numpy as np
import pandas as pd
import pytest

from src.anomaly.detection import compare_methods, detect_isolation_forest, detect_lof
from src.anomaly.features import add_comparable_deviation, add_cpv_division


@pytest.fixture
def synthetic_df():
    """Dataset synthétique : 100 marchés "normaux" + 5 anomalies évidentes.

    Les anomalies ont un montant extrême (100x la médiane) pour être
    détectables sans ambiguïté par les deux méthodes.
    """
    rng = np.random.default_rng(42)
    n_normal = 100

    normal = pd.DataFrame({
        "uid": [f"normal_{i}" for i in range(n_normal)],
        "montant": rng.normal(100_000, 15_000, n_normal).clip(min=1000),
        "dureeMois": rng.normal(12, 3, n_normal).clip(min=1),
        "codeCPV": ["45000000"] * n_normal,
    })

    anomalies = pd.DataFrame({
        "uid": [f"anomalie_{i}" for i in range(5)],
        "montant": [10_000_000.0] * 5,  # 100x la médiane
        "dureeMois": [12.0] * 5,
        "codeCPV": ["45000000"] * 5,
    })

    return pd.concat([normal, anomalies], ignore_index=True)


def test_cpv_division_extraction():
    df = pd.DataFrame({"codeCPV": ["45452100", "71200000", None]})
    result = add_cpv_division(df)
    assert result.loc[0, "cpv_division"] == "45"
    assert result.loc[1, "cpv_division"] == "71"
    assert pd.isna(result.loc[2, "cpv_division"])


def test_comparable_deviation_flags_group_reliability():
    df = pd.DataFrame({
        "uid": ["a", "b", "c"],
        "montant": [100.0, 200.0, 300.0],
        "dureeMois": [1.0, 2.0, 3.0],
        "codeCPV": ["45000000", "45000000", "71000000"],
    })
    result = add_comparable_deviation(df)
    # Groupe 45 a 2 marchés, groupe 71 en a 1 — les deux sont sous le seuil de fiabilité (30)
    assert not result["cpv_groupe_fiable"].any()
    assert result.loc[0, "cpv_groupe_taille"] == 2
    assert result.loc[2, "cpv_groupe_taille"] == 1


def test_isolation_forest_detects_synthetic_anomalies(synthetic_df):
    result = detect_isolation_forest(synthetic_df, contamination=0.05)
    anomalies_detectees = result[result["is_anomaly_iforest"]]["uid"].tolist()
    # Les 5 anomalies synthétiques (montant extrême) doivent être majoritairement détectées
    n_vraies_anomalies_detectees = sum(
        1 for uid in anomalies_detectees if uid.startswith("anomalie_")
    )
    assert n_vraies_anomalies_detectees >= 4  # au moins 4 sur 5


def test_lof_detects_synthetic_anomalies(synthetic_df):
    result = detect_lof(synthetic_df, contamination=0.05, n_neighbors=10)
    anomalies_detectees = result[result["is_anomaly_lof"]]["uid"].tolist()
    n_vraies_anomalies_detectees = sum(
        1 for uid in anomalies_detectees if uid.startswith("anomalie_")
    )
    assert n_vraies_anomalies_detectees >= 4


def test_compare_methods_returns_expected_keys(synthetic_df):
    result = compare_methods(synthetic_df, contamination=0.05)
    expected_keys = {
        "n_total", "n_anomalies_iforest", "n_anomalies_lof",
        "n_accord", "taux_accord",
    }
    assert expected_keys == set(result.keys())
    assert result["n_total"] == len(synthetic_df)
    assert 0.0 <= result["taux_accord"] <= 1.0


def test_build_feature_matrix_imputes_missing_values():
    """Non-régression : build_feature_matrix doit retourner un dataframe
    dont les features sont bien imputées (bug corrigé : l'imputation était
    calculée mais jamais réinjectée dans le dataframe retourné).
    """
    from src.anomaly.features import build_feature_matrix

    df = pd.DataFrame({
        "uid": ["a", "b", "c", "d"],
        "montant": [100.0, 200.0, None, 400.0],
        "dureeMois": [1.0, None, 3.0, 4.0],
        "codeCPV": ["45000000", "45000000", "45000000", "45000000"],
    })
    df_result, feature_cols = build_feature_matrix(df)
    assert df_result[feature_cols].isna().sum().sum() == 0
