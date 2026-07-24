import pandas as pd
import pytest

from src.quality.cleaning import (
    clean_pipeline,
    flag_date_issues,
    flag_montant_issues,
    normalize_nature,
)


@pytest.fixture
def dirty_df():
    return pd.DataFrame({
        "uid": ["A", "B", "C", "D"],
        "nature": ["Marché", "MARCHE", "marché", "Accord cadre"],
        "montant": [1000.0, -500.0, 2_000_000_000.0, None],
        "dateNotification": pd.to_datetime(
            ["2022-01-01", "1999-01-01", None, "2023-05-01"], errors="coerce"
        ),
        "datePublicationDonnees": pd.to_datetime(
            ["2022-01-05", "1999-01-05", None, "2023-05-05"], errors="coerce"
        ),
    })


def test_normalize_nature_unifies_variants(dirty_df):
    result = normalize_nature(dirty_df)
    # Les 3 premières variantes de "marché" doivent devenir identiques
    assert result.loc[0, "nature"] == result.loc[1, "nature"] == result.loc[2, "nature"]
    # L'original est conservé
    assert result.loc[1, "nature_original"] == "MARCHE"


def test_flag_montant_issues(dirty_df):
    result = flag_montant_issues(dirty_df)
    assert result.loc[0, "montant_negatif_ou_nul"] == False  # noqa: E712
    assert result.loc[1, "montant_negatif_ou_nul"] == True  # noqa: E712
    assert result.loc[2, "montant_extreme"] == True  # noqa: E712
    assert result.loc[0, "montant_extreme"] == False  # noqa: E712


def test_flag_date_issues_ignores_missing(dirty_df):
    result = flag_date_issues(dirty_df)
    assert result.loc[1, "dateNotification_hors_plage"] == True  # noqa: E712
    assert result.loc[0, "dateNotification_hors_plage"] == False  # noqa: E712
    # Une date manquante (NaT) ne doit pas être flaggée comme hors plage
    assert result.loc[2, "dateNotification_hors_plage"] == False  # noqa: E712


def test_clean_pipeline_preserves_row_count(dirty_df):
    result = clean_pipeline(dirty_df)
    assert len(result) == len(dirty_df)
    # Aucune ligne perdue, uniquement des colonnes ajoutées
    assert "nature_original" in result.columns
    assert "montant_extreme" in result.columns
    assert "dateNotification_hors_plage" in result.columns
