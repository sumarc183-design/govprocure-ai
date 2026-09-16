# Roadmap produit et technique

## Statut actuel

**Version : v1.0 — finalisée**
**Positionnement : démonstrateur data/IA reproductible pour l'analyse de la commande publique française.**

La version v1.0 couvre le périmètre annoncé. Les éléments ci-dessous sont livrés, documentés et couverts par la chaîne de qualité du dépôt. Les pistes post-v1.0 sont des améliorations identifiées ; elles ne sont pas présentées comme des fonctionnalités déjà disponibles.

## Livré en v1.0

| Domaine | État | Livrable | Preuve / documentation |
|---|---|---|---|
| Fondations | Terminé | Repo Python, dépendances épinglées, CI, données externes documentées | `README.md`, `.github/workflows/tests.yml` |
| Qualité des données | Terminé | Audit par colonne, normalisation et règles de cohérence | `src/quality/`, `docs/data_sources.md` |
| Anomalies | Terminé | Isolation Forest + LOF, déduplication par marché, tests de stabilité | `src/anomaly/`, `docs/limitations.md` |
| Recherche hybride | Terminé | Filtres, BM25, embeddings, RRF et métriques d'évaluation | `src/search/`, `docs/decisions.md` |
| Interface | Terminé | Dashboard Streamlit et export CSV | `src/dashboard/` |
| Prédiction | Terminé | Baseline, Ridge et Random Forest sur `offresRecues` | `src/prediction/` |
| Consolidation | Terminé | Documentation des biais, correctifs méthodologiques, tests et validation CI | `docs/synthese_finale.md`, `tests/` |

## Améliorations post-v1.0

| Priorité | Sujet | Objectif | Critère de complétion |
|---|---|---|---|
| Haute | Index vectoriel persistant | Pré-calculer les embeddings de tout le corpus dans FAISS ou Qdrant | Recherche sémantique sur le corpus complet sans recalcul à la requête |
| Haute | Jeu d'évaluation humain | Étendre et séparer l'annotation de la personne qui construit les requêtes | Jeu de test versionné, protocole d'annotation et accord inter-annotateur mesurés |
| Moyenne | Modèle d'embeddings | Comparer un modèle multilingue plus performant au modèle CPU léger | Gain de qualité mesuré sur les requêtes difficiles, avec coût documenté |
| Moyenne | Fusion des rangs | Évaluer les poids RRF sur un corpus de requêtes plus large | Poids sélectionnés sur validation, sans dégrader les requêtes de référence |
| Moyenne | Observabilité des données | Suivre volume, schéma, taux de manquants et dérive entre deux téléchargements | Rapport de drift automatisé et seuils d'alerte documentés |
| Basse | Déploiement | Ajouter API, authentification, stockage et supervision | Architecture, sécurité et SLA définis avant toute mise à disposition externe |

## Hors périmètre v1.0

Les sujets suivants nécessitent une décision produit, des données supplémentaires ou une infrastructure dédiée. Ils ne font pas partie du livrable actuel :

- qualification juridique ou détection automatique de fraude ;
- décision automatisée sur un marché public ;
- mise à jour temps réel des données ;
- index vectoriel complet et hébergé ;
- gestion des utilisateurs, droits d'accès, API publique et SLA.

## Principes d'évolution

Toute évolution significative doit :

1. partir d'un besoin métier ou d'une limite documentée ;
2. inclure des critères de succès quantifiés ;
3. préserver la traçabilité des données et des décisions ;
4. être testée dans la CI et documentée dans le changelog ;
5. ne jamais transformer une anomalie statistique en accusation de fraude.
