# Changelog

Toutes les évolutions significatives du projet sont consignées dans ce document.

Le format s'inspire de [Keep a Changelog](https://keepachangelog.com/fr/1.1.0/) et le versioning suit les principes de [Semantic Versioning](https://semver.org/lang/fr/).

## [1.0.0] — 2026-09-16

### Ajouté

- Documentation de statut de livraison et roadmap post-v1.0 priorisée.
- README restructuré : positionnement, démarrage rapide, architecture, validation et limites.

### Modifié

- Mise à niveau et résolution des dépendances : PyArrow 25.0.1, Streamlit 1.64.0, scikit-learn 1.9.0, Requests 2.34.2, Playwright 1.61.0 et Ruff 0.16.0.
- Actions GitHub `checkout` et `setup-python` mises à niveau vers v7.

### Validé

- Résolution complète des dépendances.
- CI GitHub Actions exécutée avec succès sur la branche `main`.

## Historique antérieur

Les décisions, résultats expérimentaux et corrections antérieures à v1.0 sont documentés dans [docs/decisions.md](docs/decisions.md) et [docs/limitations.md](docs/limitations.md).
