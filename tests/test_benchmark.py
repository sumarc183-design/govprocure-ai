"""Test de fumée pour le module de benchmark — vérifie juste l'import,
le benchmark lui-même est un script de mesure manuelle (voir
docs/limitations.md pour les résultats), pas une fonction à tester
unitairement.
"""


def test_benchmark_importe_sans_erreur():
    import src.search.benchmark  # noqa: F401
