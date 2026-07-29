# Roadmap

Planning indicatif sur 8 semaines. Chaque bloc intègre ses propres tests de robustesse au fur et à mesure plutôt qu'en fin de projet.

| Semaine | Bloc | Statut | Livrable attendu |
|---|---|---|---|
| 0.5 | Setup + audit des données | ✅ Terminé | Repo initialisé, CI en place, rapport d'audit du dataset |
| 1 | Bloc 1 — Qualité des données | ✅ Terminé | Score de qualité par variable/période |
| 2–3 | Bloc 2 — Détection d'anomalies | ✅ Terminé | Isolation Forest + LOF comparés, tests de stabilité |
| 4–5.5 | Bloc 3 — Recherche hybride | ✅ Terminé | Moteur SQL + BM25 + embeddings, métriques Precision@K/NDCG |
| 6 | Bloc 4 — Dashboard | ✅ Terminé | Interface Streamlit fonctionnelle |
| 7–8 | Bloc 5 — Consolidation robustesse | ✅ Terminé | Rapport de limites, drift, reproductibilité, biais |
| Bonus | Bloc prédiction — Régression supervisée | ✅ Terminé | Prédiction du nombre d'offres reçues, comparaison de modèles |

Légende statut : 🔲 à faire · 🟡 en cours · ✅ terminé

## Bloc 5 — détail des points traités

1. ✅ Déduplication par marché avant détection d'anomalies (bug trouvé via revue externe, corrigé)
2. ✅ Correction NDCG@K et Precision@K (bugs trouvés via revue externe, corrigés)
3. ✅ Évaluation du pipeline hybride sur requêtes difficiles (résultat nuancé documenté)
4. ✅ Annotation humaine (100% d'accord sur 3 thèmes/4)
5. ✅ Mesure du temps de réponse (~28s, goulot = embeddings, identifié et chiffré)
6. ✅ Vérification manuelle finale du dashboard (aucune régression)
7. ✅ Synthèse finale (docs/synthese_finale.md)
8. ✅ README recruteur (accroche, chiffres clés, captures, données, badge CI, licence)

## Itérations post-bloc 5 (recommandations de la synthèse finale, réalisées)

1. ✅ Cache disque des embeddings (recommandation n°1) — gain mesuré : 47x plus rapide (23,7s → 0,5s)
2. ✅ Centralisation partielle de la normalisation de texte (recommandation n°2) — module `src/common/text_normalization.py`, utilisé pour le nouveau code, modules existants non retouchés (risque de régression jugé trop élevé)
3. ✅ Transformation log des montants avant LOF (suggestion différée depuis la 1ère revue externe) — taux d'accord Isolation Forest/LOF triplé (4,5% → 13,0%)

## Bloc prédiction — détail

- Cible : `offresRecues` (nombre d'offres reçues par marché), vraie variable observée, pas une étiquette fabriquée
- Modèles comparés : baseline (médiane), Ridge, Random Forest
- Résultat : Random Forest R²(réel)=0,676, MAE=5,45 offres — bat nettement la baseline et Ridge
- Biais de sélection identifié et documenté : le taux de valeurs manquantes dépend fortement du type de procédure (31% à 100%)
- Bug bonus trouvé et corrigé : la colonne `procedure` n'avait jamais été normalisée (16 → 11 vraies catégories)

## Décisions de périmètre

- Anomalies : Isolation Forest + Local Outlier Factor traités en profondeur. Autoencodeur et XGBoost en extension possible si le temps le permet.
- Recherche : priorité donnée à ce bloc (le plus différenciant techniquement).
- Dashboard : volontairement simple, assemblage des blocs existants plutôt que recherche UX poussée.
