# Synthèse finale

Ce document résume, en une lecture, ce que le projet sait faire, ce
qu'il ne sait pas faire, et ce qui serait fait différemment avant une
mise en production réelle. Le détail complet de chaque résultat, bug,
et décision se trouve dans `docs/decisions.md` (le pourquoi) et
`docs/limitations.md` (les résultats et limites chiffrés).

## Ce que le projet sait faire

**Qualité des données** — Score de fiabilité par colonne sur 3,14M
lignes, détectant automatiquement montants incohérents (négatifs, ou
supérieurs à 1 Md€), dates hors plage, catégories mal normalisées
(casse/accents), sans jamais supprimer de données (flags plutôt que
suppression, pour garder la traçabilité).

**Détection d'anomalies** — Isolation Forest et Local Outlier Factor
comparés en profondeur sur les marchés (dédupliqués par cotraitance).
Les deux méthodes ne sont d'accord que sur ~4,5% des cas — résultat
stable et reproductible (écart-type 0,85% sur 5 tirages), qui illustre
concrètement leur complémentarité (anomalies globales vs contextuelles)
plutôt qu'une faiblesse. Isolation Forest passe à l'échelle sur les 3M
lignes ; LOF non (limite mémoire au-delà de 2-3M lignes).

**Recherche hybride** — Filtres stricts (région, montant) + BM25 +
embeddings sémantiques, fusionnés par Reciprocal Rank Fusion. Sur des
requêtes reformulées sans aucun recouvrement lexical avec les mots
attendus, le pipeline complet transforme un échec total de BM25 seul
(score nul) en résultat exploitable sur 2 thèmes testés sur 4
(cybersécurité, restauration scolaire) — preuve mesurée, pas supposée,
de la valeur ajoutée des embeddings.

**Évaluation rigoureuse** — Precision@K, Recall@K et NDCG@K implémentés
selon leur définition standard (après correction de deux bugs trouvés
lors d'une revue externe), vérité terrain par mots-clés elle-même
validée par un échantillon d'annotation humaine (100% d'accord sur 3
thèmes sur 4).

**Dashboard fonctionnel** — Interface Streamlit à 3 onglets, validée de
bout en bout après chaque correction majeure, export CSV des résultats.

**Documentation du raisonnement** — Chaque décision technique
(pourquoi cette méthode plutôt qu'une autre) et chaque bug trouvé (y
compris via une revue externe du code) sont documentés avec leur
correction, pas seulement listés.

**Régression supervisée** — Prédiction du nombre d'offres reçues par un
marché (Random Forest, R² réel = 0,676, MAE = 5,45 offres), comparée à
une baseline naïve et à une régression Ridge, avec un biais de sélection
des données (taux de manquant dépendant du type de procédure) découvert
et documenté plutôt qu'ignoré.

## Ce que le projet ne sait pas faire (limites assumées)

- **Ne détecte pas la fraude** — uniquement des anomalies statistiques à
  faire vérifier par un humain. Aucune vérité terrain de fraude confirmée
  n'existe pour entraîner un modèle supervisé, et ce choix méthodologique
  est assumé plutôt que contourné artificiellement.
- **La prédiction supervisée reste limitée à un cas d'usage** — la
  régression sur `offresRecues` comble le manque de compétences en
  apprentissage supervisé, mais reste soumise à un biais de sélection
  fort (le sous-ensemble annoté n'est pas représentatif de toutes les
  procédures) et n'a pas été optimisée (pas de validation croisée, pas
  de recherche d'hyperparamètres).
- **Recherche lente sans cache, largement corrigée avec** — ~28 secondes
  de temps de réponse sur un cas d'usage typique sans optimisation, parce
  que les embeddings étaient recalculés à chaque requête. **Mise à jour** :
  un cache disque des embeddings a été ajouté après cette synthèse
  initiale — mesuré, il donne un gain de **47x** sur une recherche
  répétée (23,7 s → 0,5 s). Limite résiduelle : le tout premier appel sur
  un sous-ensemble de marchés jamais rencontré reste aussi lent qu'avant;
  le gain ne profite qu'aux recherches répétées sur des filtres communs
  (voir docs/limitations.md, "Itération post-bloc 5").
- **Échec persistant sur certains thèmes de recherche** — même avec les
  embeddings, la requête "maintenance des parcs municipaux" (reformulation
  de "espaces verts") reste à 0% de précision. Le modèle d'embeddings
  choisi (léger, pour tourner sur CPU) ne capture pas cette proximité
  sémantique précise.
- **Vérité terrain d'évaluation imparfaite** — construite par mots-clés,
  partiellement biaisée pour le thème cybersécurité (mots-clés choisis
  après avoir observé des résultats de recherche), validée seulement sur
  32 exemples annotés, pas sur un jeu de test à grande échelle.
- **RRF peut dégrader un résultat correct** — observé concrètement sur le
  thème "travaux de voirie" : la fusion à poids égal entre BM25 et
  embeddings a fait chuter la précision par rapport à BM25 seul, quand
  BM25 avait par coïncidence un bon score et les embeddings un moins bon.

## Ce qui serait fait différemment en production

1. ✅ **Fait (partiellement) — Précalculer les embeddings.** Un cache
   disque a été ajouté (gain mesuré : 47x, 23,7s → 0,5s sur une
   recherche répétée). Reste non fait : un vrai index FAISS/Qdrant
   couvrant l'intégralité du corpus (1,73M marchés), qui nécessiterait un
   calcul batch d'environ 14h sur cette machine sans GPU — hors
   périmètre réalisé, mais chemin clair pour la suite.
2. ✅ **Fait (partiellement) — Centraliser la normalisation.** Un module
   commun (`src/common/text_normalization.py`) a été créé après avoir
   trouvé un 4e cas du même problème (`procedure`, jamais normalisée).
   Utilisé pour le nouveau code (bloc prédiction). Reste non fait : migrer
   les modules existants (`cleaning.py`, `bm25_search.py`) pour qu'ils
   réutilisent ce module plutôt que leur propre logique dupliquée — repoussé
   par prudence (code déjà testé et validé, risque de régression jugé
   supérieur au bénéfice à ce stade avancé du projet).
3. **Tester un modèle d'embeddings plus grand** pour les cas où le modèle
   léger actuel échoue (ex: "espaces verts"), en évaluant le compromis
   avec le temps de calcul supplémentaire.
4. **Étendre l'annotation humaine** au-delà de 32 exemples et de 4 thèmes,
   avec un vrai annotateur indépendant du projet, pour une validation
   plus robuste de la méthodologie d'évaluation.
5. **Reconsidérer la stratégie de fusion RRF** à la lumière du cas
   "travaux de voirie" — envisager une pondération non uniforme ou un
   score composite, en testant empiriquement si ça corrige la régression
   observée sans introduire de nouveaux problèmes.
6. **Tester une transformation logarithmique des montants** avant LOF
   (suggestion reçue en revue externe, pas encore testée) — les montants
   très asymétriques (jusqu'à plusieurs milliards) pourraient dominer les
   distances de LOF sans cette transformation.
7. **Construire un vrai jeu de test de non-régression pour le dashboard**
   (au-delà du smoke test d'import actuel), pour attraper automatiquement
   les régressions visuelles/fonctionnelles plutôt que de les vérifier
   manuellement à chaque changement.

## En une phrase

Le projet démontre une chaîne complète et fonctionnelle (qualité →
anomalies → recherche → interface), avec des résultats mesurés
honnêtement plutôt qu'idéalisés, plusieurs bugs réels trouvés et
corrigés en cours de route (dont certains via une revue externe), et une
compréhension claire de ses propres limites — ce qui compte, pour ce
projet, autant que le code lui-même.
