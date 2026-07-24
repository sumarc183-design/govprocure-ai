import numpy as np
import pandas as pd
import pytest

from src.anomaly.robustness import check_agreement_stability


@pytest.fixture
def synthetic_df():
    rng = np.random.default_rng(0)
    n = 500
    return pd.DataFrame({
        "uid": [f"m_{i}" for i in range(n)],
        "montant": rng.normal(100_000, 20_000, n).clip(min=1000),
        "dureeMois": rng.normal(12, 3, n).clip(min=1),
        "codeCPV": ["45000000"] * n,
    })


def test_check_agreement_stability_returns_expected_structure(synthetic_df):
    result = check_agreement_stability(
        synthetic_df, sample_size=200, n_repeats=3, seeds=[1, 2, 3]
    )
    assert result["n_repeats"] == 3
    assert "taux_accord_moyen" in result
    assert "taux_accord_ecart_type" in result
    assert len(result["detail_par_tirage"]) == 3


def test_check_agreement_stability_handles_small_dataset(synthetic_df):
    """Si sample_size dépasse la taille du dataset, on ne doit pas planter :
    on prend simplement tout le dataset disponible.
    """
    result = check_agreement_stability(
        synthetic_df, sample_size=10_000, n_repeats=2, seeds=[1, 2]
    )
    assert result["n_repeats"] == 2
