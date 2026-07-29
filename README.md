# GovProcure AI

[![Tests](https://github.com/sumarc183-design/govprocure-ai/actions/workflows/tests.yml/badge.svg)](https://github.com/sumarc183-design/govprocure-ai/actions/workflows/tests.yml)

> Une plateforme d'analyse des marchés publics français — qualité des données, détection d'anomalies, recherche hybride en langage naturel — construite sur les 3,14 millions de contrats publiés en open data, avec chaque décision technique et chaque bug documentés dans le code de leur découverte à leur correction.

**[📄 Lire la synthèse finale](docs/synthese_finale.md)** — ce que le projet sait faire, ce qu'il ne sait pas faire, ce qui serait fait différemment en production.

## En bref

| | |
|---|---|
| **Données** | 3,14M marchés publics (data.gouv.fr / decp.info), 1,73M marchés uniques après regroupement |
| **Qualité** | Score de fiabilité par colonne — 64 408 montants incohérents détectés, catégories normalisées |
| **Anomalies** | Isolation Forest + LOF comparés, taux d'accord de 4,5% (13,0% avec transformation log testée) — résultat stable, cohérent avec des notions différentes de l'anomalie (pas une preuve de complémentarité en soi, une validation métier serait nécessaire) |
| **Recherche** | Filtres + BM25 + embeddings + RRF — passé de 0/10 à 8/10 résultats pertinents après diagnostic et correction de 2 bugs réels, cache disque des embeddings (gain mesuré : 47x) |
| **Prédiction** | Régression du nombre d'offres reçues (Random Forest, R²=0,676, MAE=5,45 offres), biais de sélection des données identifié et documenté |
| **Robustesse** | 2 bugs méthodologiques trouvés via revue externe et corrigés (déduplication, calcul NDCG), documentés avec preuve avant/après |
| **Tests** | 112 tests collectés — 107 s'exécutent intégralement en CI ; les 5 tests fonctionnels du dashboard nécessitent les données locales (sautés en CI, dataset non versionné) et, une fois exécutés en local, 3 passent et 2 sont `xfail` (limite documentée de l'automatisation headless) |

## Le projet en images

![Qualité des données](docs/images/dashboard-qualite-donnees.png)
![Détection d'anomalies](docs/images/dashboard-anomalies-resultats.png)
![Recherche hybride](docs/images/dashboard-recherche-resultats.png)

## Pourquoi ce projet est différent d'un portfolio classique

Ce n'est pas un projet où "tout marche parfaitement". Trois exemples concrets :

- **Un bug trouvé et corrigé en direct** : la fonction de détection d'anomalies ne dédupliquait pas les marchés en cotraitance, faussant le taux d'accord entre modèles et faisant apparaître les mêmes marchés plusieurs fois dans les alertes. Trouvé via une revue externe du code, corrigé, chiffres recalculés avant/après (voir [decisions.md](docs/decisions.md)).
- **Un calcul de métrique corrigé** : le NDCG@K initial ne pénalisait pas les résultats pertinents manqués (`P@10=0.7` donnait pourtant `NDCG@10=1.0`, incohérent). Corrigé, effet mesuré : `0.431 → 0.146` sur un cas réel.
- **Un résultat nuancé plutôt qu'idéalisé** : sur 4 requêtes de recherche reformulées sans aucun mot en commun avec la vérité terrain, les embeddings sauvent la mise sur 2 thèmes, dégradent le résultat sur 1, et échouent totalement sur le dernier. Documenté tel quel plutôt que présenté comme une victoire uniforme.
- **Un biais de sélection assumé, pas caché** : le modèle de prédiction (nombre d'offres reçues) n'apprend que sur les ~40% de marchés où cette donnée est renseignée — et ce taux de renseignement varie de 31% à 100% selon le type de procédure. Documenté comme une vraie limite du modèle, pas contourné par une imputation artificielle.

## Structure du repo

```
govprocure-ai/
├── data/
│   ├── raw/            # données brutes (non versionnées, voir .gitignore)
│   └── processed/      # données nettoyées / transformées (cache embeddings, etc.)
├── src/
│   ├── common/          # normalisation de texte partagée
│   ├── quality/         # Bloc 1 — contrôle qualité des données
│   ├── anomaly/         # Bloc 2 — détection d'anomalies
│   ├── search/          # Bloc 3 — moteur de recherche hybride + évaluation
│   ├── dashboard/        # Bloc 4 — interface Streamlit
│   └── prediction/       # Bloc bonus — régression supervisée (offresRecues)
├── tests/              # tests unitaires et de robustesse (pytest)
├── docs/               # documentation, roadmap, décisions, limites
└── .github/workflows/  # CI (GitHub Actions)
```

## Installation

```bash
python -m venv venv
source venv/bin/activate  # Windows : venv\Scripts\activate
pip install -r requirements.txt
```

## Récupérer les données

Le dataset (~200 Mo) n'est pas versionné dans ce repo (voir `.gitignore`) —
il faut le télécharger séparément avant de lancer le dashboard ou les scripts.

1. **Télécharger le fichier Parquet** directement ici :
   [`https://www.data.gouv.fr/api/1/datasets/r/11cea8e8-df3e-4ed1-932b-781e2635e432`](https://www.data.gouv.fr/api/1/datasets/r/11cea8e8-df3e-4ed1-932b-781e2635e432)
   (page du jeu de données : [data.gouv.fr — DECP consolidées, format tabulaire](https://www.data.gouv.fr/datasets/donnees-essentielles-de-la-commande-publique-consolidees-format-tabulaire))

2. **Renommer** le fichier téléchargé en `decp.parquet`

3. **Placer** le fichier dans `data/raw/decp.parquet` (créer le dossier `data/raw/` s'il n'existe pas)

```bash
mkdir -p data/raw
# après téléchargement, déplacer/renommer le fichier :
mv ~/Downloads/*.parquet data/raw/decp.parquet
```

⚠️ Ce dataset est mis à jour quasi quotidiennement par sa source
(decp.info) — les chiffres exacts documentés dans `docs/limitations.md`
peuvent légèrement varier d'un téléchargement à l'autre (nombre de
lignes, valeurs précises), sans remettre en cause les conclusions.

## Lancer le dashboard

```bash
streamlit run src/dashboard/app.py
```

## Lancer les tests

```bash
pytest tests/ -v
```

Pour lancer aussi les tests fonctionnels du dashboard (Playwright, nécessite le dataset local — voir "Récupérer les données" ci-dessus) :

```bash
pip install -r requirements-dev.txt
playwright install chromium
pytest tests/test_dashboard_functional.py -v
```

## Documentation complète

- [**Synthèse finale**](docs/synthese_finale.md) — vue d'ensemble : ce qui fonctionne, les limites, les recommandations production
- [Décisions techniques (ADR)](docs/decisions.md) — chaque choix technique, pourquoi, et les alternatives écartées
- [Limites et résultats détaillés](docs/limitations.md) — tous les chiffres, bugs, et découvertes, bloc par bloc
- [Roadmap](docs/roadmap.md) — avancement du projet
- [Sources de données](docs/data_sources.md) — origine et structure du dataset

## Stack technique

Python, pandas, scikit-learn, sentence-transformers, rank_bm25, Streamlit, pytest, GitHub Actions.

## Licence

Projet personnel réalisé dans le cadre d'une candidature. Tous droits
réservés — pas de licence open source explicite à ce stade. Les données
utilisées (DECP, data.gouv.fr) sont sous [Licence Ouverte / Open Licence 2.0](https://www.etalab.gouv.fr/licence-ouverte-open-licence).
