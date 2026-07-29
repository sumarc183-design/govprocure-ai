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


def test_build_feature_matrix_deduplique_par_marche():
    """Non-régression : bug trouvé lors d'une revue externe — un marché en
    cotraitance (plusieurs titulaires, même uid) était compté plusieurs
    fois dans l'entraînement des modèles d'anomalie, faussant les médianes
    de groupe et les voisinages LOF. Même principe que la déduplication
    déjà appliquée au moteur de recherche (bloc 3).
    """
    from src.anomaly.features import build_feature_matrix

    df = pd.DataFrame({
        "uid": ["m1", "m1", "m1", "m2", "m3"],  # m1 en cotraitance, 3 titulaires
        "titulaire_id": ["t1", "t2", "t3", "t4", "t5"],
        "montant": [100000.0, 100000.0, 100000.0, 50000.0, 75000.0],
        "dureeMois": [12.0, 12.0, 12.0, 6.0, 24.0],
        "codeCPV": ["45000000"] * 5,
    })
    df_result, _ = build_feature_matrix(df)
    assert len(df_result) == 3  # un par marché (m1, m2, m3), pas 5
    assert set(df_result["uid"]) == {"m1", "m2", "m3"}


def test_log_signe_gere_les_valeurs_negatives():
    """La transformation log signée doit rester définie pour les montants
    négatifs (~2% du dataset réel, jusqu'à -7,1M€) — un log1p classique
    échouerait (indéfini pour x <= -1).
    """
    from src.anomaly.features import _log_signe

    valeurs = pd.Series([-7_000_000.0, -1.0, 0.0, 1.0, 7_000_000.0])
    resultat = _log_signe(valeurs)
    assert not resultat.isna().any()
    # Le signe d'origine doit être préservé
    assert resultat.iloc[0] < 0
    assert resultat.iloc[4] > 0
    assert resultat.iloc[2] == 0.0


def test_log_signe_compresse_les_valeurs_extremes():
    """La transformation doit réduire l'écart relatif entre une valeur
    normale et une valeur extrême (c'est tout l'intérêt de la transformation).
    """
    from src.anomaly.features import _log_signe

    valeurs = pd.Series([100_000.0, 100_000_000_000.0])  # ratio brut : 1 000 000x
    resultat = _log_signe(valeurs)
    ratio_brut = valeurs.iloc[1] / valeurs.iloc[0]
    ratio_apres_log = resultat.iloc[1] / resultat.iloc[0]
    assert ratio_apres_log < ratio_brut


def test_build_feature_matrix_transformation_log_desactivee_par_defaut():
    """Non-régression : le comportement par défaut (sans transformation_log)
    ne doit pas changer par rapport aux blocs précédents déjà validés.
    """
    from src.anomaly.features import build_feature_matrix

    df = pd.DataFrame({
        "uid": ["m1", "m2"],
        "montant": [100_000.0, 200_000.0],
        "dureeMois": [12.0, 24.0],
        "codeCPV": ["45000000", "45000000"],
    })
    df_defaut, _ = build_feature_matrix(df)
    df_explicite, _ = build_feature_matrix(df, transformation_log=False)
    pd.testing.assert_frame_equal(df_defaut, df_explicite)


def test_build_feature_matrix_transformation_log_change_les_valeurs():
    from src.anomaly.features import build_feature_matrix

    df = pd.DataFrame({
        "uid": ["m1", "m2"],
        "montant": [100_000.0, 200_000_000.0],
        "dureeMois": [12.0, 24.0],
        "codeCPV": ["45000000", "45000000"],
    })
    df_sans_log, _ = build_feature_matrix(df, transformation_log=False)
    df_avec_log, _ = build_feature_matrix(df, transformation_log=True)
    assert not df_sans_log["montant"].equals(df_avec_log["montant"])


def test_deduplicate_marches_garde_une_ligne_par_uid():
    from src.anomaly.features import deduplicate_marches

    df = pd.DataFrame({
        "uid": ["m1", "m1", "m2"],
        "montant": [100.0, 100.0, 200.0],
    })
    result = deduplicate_marches(df)
    assert len(result) == 2
    assert set(result["uid"]) == {"m1", "m2"}


def test_deduplicate_marches_sans_colonne_uid_ne_plante_pas():
    from src.anomaly.features import deduplicate_marches

    df = pd.DataFrame({"montant": [100.0, 200.0]})
    result = deduplicate_marches(df)
    assert len(result) == 2  # inchangé, pas d'erreur
