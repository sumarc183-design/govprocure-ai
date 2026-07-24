"""
Vérification de la stabilité (reproductibilité) des résultats de détection
d'anomalies.

Un résultat obtenu sur un seul tirage d'échantillon peut être un coup de
chance (ou de malchance). Ce module répète l'analyse sur plusieurs
échantillons indépendants pour vérifier que le résultat est stable, pas
un artefact du tirage précis utilisé.
"""

from __future__ import annotations

import statistics

import pandas as pd

from src.anomaly.detection import compare_methods


def check_agreement_stability(
    df: pd.DataFrame,
    sample_size: int = 20_000,
    n_repeats: int = 5,
    contamination: float = 0.05,
    seeds: list[int] | None = None,
) -> dict:
    """Répète compare_methods() sur plusieurs échantillons indépendants.

    Retourne un dict avec la moyenne et l'écart-type du taux d'accord entre
    Isolation Forest et LOF, plus le détail par tirage.

    Un écart-type faible (par rapport à la moyenne) indique un résultat
    stable et reproductible. Un écart-type élevé indiquerait au contraire
    que le résultat dépend fortement du tirage — signe qu'il faudrait
    revoir la méthode ou augmenter la taille d'échantillon.
    """
    seeds = seeds or list(range(n_repeats))
    resultats = []

    for seed in seeds:
        sample = df.sample(n=min(sample_size, len(df)), random_state=seed)
        result = compare_methods(sample, contamination=contamination)
        resultats.append({"seed": seed, **result})

    taux_accord_list = [r["taux_accord"] for r in resultats]

    return {
        "n_repeats": len(seeds),
        "sample_size": sample_size,
        "taux_accord_moyen": round(statistics.mean(taux_accord_list), 4),
        "taux_accord_ecart_type": round(statistics.stdev(taux_accord_list), 4)
        if len(taux_accord_list) > 1
        else 0.0,
        "detail_par_tirage": resultats,
    }
