# GovProcure AI

Plateforme d'analyse des marchés publics — qualité des données, détection d'anomalies, recherche hybride, tableau de bord d'aide à la décision.

Le projet s'appuie sur les données ouvertes de la commande publique (open data) pour :

- **contrôler la qualité des données** (valeurs manquantes, doublons, incohérences, dérive) ;
- **détecter des opérations atypiques** à examiner (sans jamais qualifier de fraude) ;
- **rechercher des marchés en langage naturel** via un moteur hybride (SQL + BM25 + embeddings) ;
- **comparer plusieurs modèles** et documenter leurs limites ;
- **produire un tableau de bord** d'aide à la décision.

## Statut du projet

🚧 Projet en cours de développement — voir [docs/roadmap.md](docs/roadmap.md) pour l'avancement par bloc.

## Structure du repo

```
govprocure-ai/
├── data/
│   ├── raw/            # données brutes (non versionnées, voir .gitignore)
│   └── processed/      # données nettoyées / transformées
├── src/
│   ├── quality/        # Bloc 1 — contrôle qualité des données
│   ├── anomaly/        # Bloc 2 — détection d'anomalies
│   ├── search/         # Bloc 3 — moteur de recherche hybride
│   └── dashboard/       # Bloc 4 — interface Streamlit
├── tests/              # tests unitaires et de robustesse (pytest)
├── notebooks/          # exploration et prototypage
├── docs/               # documentation, roadmap, limites du système
└── .github/workflows/  # CI (GitHub Actions)
```

## Installation

```bash
python -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

## Lancer les tests

```bash
pytest tests/ -v
```

## Documentation

- [Roadmap et avancement](docs/roadmap.md)
- [Décisions techniques et pourquoi (ADR)](docs/decisions.md)
- [Limites et robustesse du système](docs/limitations.md)
- [Sources de données](docs/data_sources.md)

## Stack technique

Python, pandas, Polars, SQL, scikit-learn, PyTorch, sentence-transformers, BM25, FAISS/Qdrant, SHAP, Evidently, FastAPI, Streamlit, pytest, GitHub Actions, Docker.
