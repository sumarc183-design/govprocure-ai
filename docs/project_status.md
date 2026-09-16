# Statut de livraison

## Résumé exécutif

**GovProcure AI v1.0 est finalisé comme démonstrateur professionnel de data science appliquée à la commande publique.** Il met à disposition une chaîne cohérente allant de l'audit des données au dashboard, avec une documentation explicite des compromis et limites.

| Élément | Statut |
|---|---|
| Périmètre fonctionnel v1.0 | Terminé |
| Documentation produit et technique | Disponible |
| Tests, lint et typage | Intégrés à la CI GitHub Actions |
| Jeu de données | Externe et non versionné ; procédure de récupération documentée |
| Déploiement public / production | Hors périmètre |

## Ce qui est livré

- Audit de qualité et nettoyage non destructif des données DECP.
- Détection d'anomalies par Isolation Forest et LOF, interprétée comme aide à la priorisation humaine.
- Recherche hybride associant filtres, BM25, embeddings multilingues et Reciprocal Rank Fusion.
- Évaluation de la recherche avec Precision@K, Recall@K et NDCG@K.
- Interface Streamlit et export CSV.
- Modèle de régression pour `offresRecues`, comparé à une baseline et à Ridge.
- Tests automatisés, lint Ruff, vérification de types Mypy et CI.

## Conditions de reproduction

1. Installer les dépendances avec `pip install -r requirements.txt`.
2. Télécharger le Parquet DECP et le placer dans `data/raw/decp.parquet`.
3. Lancer `streamlit run src/dashboard/app.py` ou les scripts de chaque module.
4. Exécuter `pytest tests/ -v`, `ruff check src/ tests/` et `mypy --ignore-missing-imports src/`.

Les tests fonctionnels du dashboard dépendent volontairement du jeu de données local et de Playwright. Ils sont donc séparés des contrôles exécutés en CI sans dataset.

## Validation de livraison

La livraison v1.0 est considérée terminée lorsque :

- les dépendances sont résolubles avec des versions épinglées ;
- la CI est verte sur la branche principale ;
- le README explique l'installation, les données, l'utilisation et les limites ;
- les résultats chiffrés et les choix méthodologiques restent traçables dans `docs/` ;
- aucune fonctionnalité annoncée ne transforme une anomalie statistique en preuve de fraude.

## Limites importantes

- Les résultats dépendent de la version téléchargée d'un dataset mis à jour fréquemment.
- Le modèle supervisé ne couvre que les marchés dont `offresRecues` est renseigné ; ce sous-ensemble est biaisé.
- Les embeddings sont calculés sur CPU et le premier appel sur un nouveau sous-ensemble reste coûteux.
- Une mise en production demanderait notamment un index vectoriel, un mécanisme d'actualisation, une validation métier, une authentification et une supervision.

Pour les résultats détaillés, consulter la [synthèse finale](synthese_finale.md), les [limites](limitations.md) et la [roadmap](roadmap.md).
