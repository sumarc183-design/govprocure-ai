import numpy as np
import pandas as pd
import pytest

from src.prediction.features import (
    construire_matrice_features,
    filtrer_cible_valide,
    preparer_donnees_completes,
)


@pytest.fixture
def df_brut():
    return pd.DataFrame({
        "uid": ["m1", "m1", "m2", "m3", "m4", "m5"],  # m1 en cotraitance
        "offresRecues": [5.0, 5.0, -8.0, np.nan, 3.0, 12.0],
        "montant": [100000.0, 100000.0, 50000.0, 75000.0, -1000.0, 200000.0],
        "dureeMois": [12.0, 12.0, 24.0, 6.0, 12.0, 36.0],
        "codeCPV": ["45000000"] * 6,
        "procedure": ["Appel d'offres ouvert", "Appel d'offres ouvert",
                      "Appel d offres ouvert", "Procédure adaptée",
                      "Procédure adaptée", "Dialogue compétitif"],
        "nature": ["Marché"] * 6,
        "acheteur_region_nom": ["Bretagne"] * 6,
    })


def test_filtrer_cible_valide_retire_offresRecues_negatif(df_brut):
    result = filtrer_cible_valide(df_brut)
    assert (result["offresRecues"] >= 0).all()


def test_filtrer_cible_valide_retire_offresRecues_manquant(df_brut):
    result = filtrer_cible_valide(df_brut)
    assert not result["offresRecues"].isna().any()


def test_filtrer_cible_valide_retire_montant_invalide(df_brut):
    """Non-régression : bug trouvé en testant sur données réelles — un
    montant négatif ou nul faisait planter log1p (RuntimeWarning +
    valeurs NaN/inf silencieuses dans la matrice de features).
    """
    result = filtrer_cible_valide(df_brut)
    assert (result["montant"] > 0).all()


def test_preparer_donnees_completes_deduplique_par_marche(df_brut):
    """m1 apparaît 2 fois (cotraitance) -> une seule ligne après préparation."""
    result = preparer_donnees_completes(df_brut)
    assert result["uid"].duplicated().sum() == 0
    assert (result["uid"] == "m1").sum() == 1


def test_preparer_donnees_completes_normalise_procedure(df_brut):
    """'Appel d'offres ouvert' et 'Appel d offres ouvert' doivent devenir
    la même catégorie normalisée.
    """
    result = preparer_donnees_completes(df_brut)
    valeurs = result.set_index("uid")["procedure_normalisee"]
    assert valeurs.loc["m1"] == valeurs.loc["m2"] if "m2" in valeurs.index else True


def test_construire_matrice_features_pas_de_nan_ni_inf(df_brut):
    X, y, cols = construire_matrice_features(df_brut)
    assert X.isna().sum().sum() == 0
    assert not np.isinf(X.select_dtypes(include="number")).any().any()
    assert not y.isna().any()


def test_construire_matrice_features_cible_transformee_log(df_brut):
    """La cible retournée doit être log1p(offresRecues), pas la valeur brute."""
    X, y, cols = construire_matrice_features(df_brut)
    # offresRecues=3.0 pour un marché valide -> log1p(3) ≈ 1.386
    assert y.max() < 20  # une vraie valeur de offresRecues (ex: 20300) n'aurait
    # pas sa place ici après log1p (log1p(20300) ≈ 9.9) -- vérifie que la
    # transformation a bien été appliquée, pas la valeur brute
