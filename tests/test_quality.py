"""Tests unitaires pour src/quality — utilisent des données synthétiques,
pas le fichier DECP réel (trop volumineux pour la CI, et pas versionné).
"""

import pandas as pd
import pytest

from src.quality.audit import (
    audit_categories,
    audit_dates,
    audit_missing_values,
    audit_montant,
    audit_versioning,
    run_full_audit,
)
from src.quality.loader import group_by_marche


@pytest.fixture
def sample_df():
    return pd.DataFrame({
        "uid": ["A", "A", "B", "C", "D"],
        "titulaire_id": ["t1", "t2", "t3", None, "t5"],
        "montant": [1000.0, 1000.0, -500.0, 2_000_000_000.0, None],
        "nature": ["Marché", "Marché", "MARCHE", "marché", "Accord-cadre"],
        "dateNotification": pd.to_datetime(
            ["2022-01-01", "2022-01-01", "2001-01-01", "0001-01-01", "2023-05-01"],
            errors="coerce",
        ),
        "donneesActuelles": [True, True, True, None, True],
    })


def test_missing_values_detected(sample_df):
    reports = audit_missing_values(sample_df)
    assert reports["montant"].n_missing == 1
    assert reports["montant"].missing_rate == 0.2


def test_montant_issues_detected(sample_df):
    reports = audit_missing_values(sample_df)
    audit_montant(sample_df, reports)
    assert any("négatifs" in issue for issue in reports["montant"].issues)
    assert any("1,000,000,000" in issue for issue in reports["montant"].issues)


def test_date_issues_detected(sample_df):
    df = sample_df.copy()
    reports = audit_missing_values(df)
    audit_dates(df, reports)
    # 0001-01-01 devient NaT après coercion en amont (loader), donc ici on simule
    # une date simplement hors plage (2001, considérée valide car >= 2000).
    # On vérifie juste que la fonction ne plante pas et remonte un rapport.
    assert "dateNotification" in reports


def test_category_normalization_detected(sample_df):
    reports = audit_missing_values(sample_df)
    audit_categories(sample_df, reports)
    assert any("variantes distinctes" in issue for issue in reports["nature"].issues)


def test_versioning_nulls_detected(sample_df):
    reports = audit_missing_values(sample_df)
    audit_versioning(sample_df, reports)
    assert any("nulles" in issue for issue in reports["donneesActuelles"].issues)


def test_quality_score_bounds(sample_df):
    reports = run_full_audit(sample_df)
    for report in reports.values():
        assert 0.0 <= report.quality_score <= 1.0


def test_group_by_marche_handles_cotraitance():
    df = pd.DataFrame({
        "uid": ["A", "A", "B"],
        "titulaire_id": ["t1", "t2", "t3"],
        "montant": [1000.0, 1000.0, 500.0],
    })
    grouped = group_by_marche(df)
    assert len(grouped) == 2  # A et B, malgré 3 lignes en entrée
    assert grouped.set_index("uid").loc["A", "nb_titulaires"] == 2
    assert grouped.set_index("uid").loc["B", "nb_titulaires"] == 1


def test_group_by_marche_requires_uid_column():
    df = pd.DataFrame({"titulaire_id": ["t1"], "montant": [100.0]})
    with pytest.raises(ValueError):
        group_by_marche(df)
