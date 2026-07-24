# Roadmap

Planning indicatif sur 8 semaines. Chaque bloc intègre ses propres tests de robustesse au fur et à mesure plutôt qu'en fin de projet.

| Semaine | Bloc | Statut | Livrable attendu |
|---|---|---|---|
| 0.5 | Setup + audit des données | ✅ À faire | Repo initialisé, CI en place, rapport d'audit du dataset |
| 1 | Bloc 1 — Qualité des données | ✅ À faire | Score de qualité par variable/période |
| 2–3 | Bloc 2 — Détection d'anomalies | 🟡est À faire | Isolation Forest + LOF comparés, tests de stabilité |
| 4–5.5 | Bloc 3 — Recherche hybride | 🔲 À faire | Moteur SQL + BM25 + embeddings, métriques Precision@K/NDCG |
| 6 | Bloc 4 — Dashboard | 🔲 À faire | Interface Streamlit fonctionnelle |
| 7–8 | Bloc 5 — Consolidation robustesse | 🔲 À faire | Rapport de limites, drift, reproductibilité, biais |

Légende statut : 🔲 à faire · 🟡 en cours · ✅ terminé

## Décisions de périmètre

- Anomalies : Isolation Forest + Local Outlier Factor traités en profondeur. Autoencodeur et XGBoost en extension possible si le temps le permet.
- Recherche : priorité donnée à ce bloc (le plus différenciant techniquement).
- Dashboard : volontairement simple, assemblage des blocs existants plutôt que recherche UX poussée.
