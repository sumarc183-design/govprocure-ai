# Décisions techniques (ADR)

Ce document trace les choix techniques du projet et **pourquoi** ils ont été
pris — y compris les alternatives envisagées et écartées. Objectif : pouvoir
justifier chaque décision plus tard (soutenance, revue de code, ou juste
pour soi-même dans 2 mois), plutôt que de devoir deviner a posteriori
pourquoi le code est écrit ainsi.

Mis à jour au fil du projet, dans l'ordre chronologique.

---

## Bloc 1 — Qualité des données

### Filtrer sur `donneesActuelles == True`

**Décision** : pour toute analyse "état actuel", ne garder que les lignes
où `donneesActuelles == True`.

**Pourquoi** : un marché public peut être modifié après signature (avenant,
prolongation). Le dataset garde une trace de chaque version. Sans filtre,
un même marché modifié 3 fois est compté 3 fois — ça fausse les comptages,
moyennes et médianes.

**Alternative écartée** : garder tout l'historique pour étudier l'évolution
des marchés dans le temps (analyse des avenants). Intéressant mais hors
périmètre du projet actuel — pourrait être une extension future.

### Flagger les données suspectes plutôt que les supprimer

**Décision** : les montants négatifs/extrêmes et les dates aberrantes sont
marqués via des colonnes booléennes (`montant_extreme`, `dateNotification_hors_plage`,
etc.), jamais supprimés du dataset.

**Pourquoi** : supprimer silencieusement fait perdre l'information sans
distinguer une vraie erreur de saisie d'une donnée légitime mal comprise.
Exemple concret : certains montants > 1 Md€ sont probablement des erreurs
(confusion d'unité), mais d'autres pourraient être de vrais très gros
marchés d'État. Une fois supprimés, impossible de vérifier après coup.
En flaggant, on garde la ligne et on laisse le bloc suivant (anomalies)
décider quoi en faire selon le contexte.

**Alternative écartée** : nettoyage direct (suppression ou correction
automatique). Plus simple à coder, mais irréversible et risqué sans
validation métier au cas par cas.

### Normaliser les catégories tout en gardant l'original

**Décision** : `nature` est normalisée (`"MARCHE"` → `"Marché"`) dans la
colonne `nature`, mais la valeur brute est conservée dans `nature_original`.

**Pourquoi** : la normalisation est nécessaire pour grouper/compter
correctement (sinon `groupby("nature")` sépare artificiellement "Marché"
et "MARCHE"). Mais garder l'original permet de revenir à la donnée brute
si besoin d'auditer une valeur plus tard.

---

## Bloc 2 — Détection d'anomalies

### Regrouper par division CPV (2 chiffres) plutôt que code CPV complet (8 chiffres)

**Décision** : le calcul de "marchés comparables" (médiane de montant/durée)
se fait par division CPV (ex: '45' = travaux de construction, ~174 groupes),
pas par code CPV complet.

**Pourquoi** : vérifié empiriquement avant de coder — le code complet donne
~8 900 catégories, dont beaucoup n'ont qu'1 ou 2 marchés. Une médiane sur
2 valeurs n'a aucun sens statistique. La division CPV donne des groupes
de centaines/milliers de marchés, donc des médianes fiables.

**Compromis assumé** : plus fin (code complet) = plus pertinent métier mais
peu fiable statistiquement. Plus large (division) = fiable mais moins
précis. On a choisi le niveau où les deux s'équilibrent, et ajouté un flag
`cpv_groupe_fiable` pour signaler quand un groupe reste trop petit.

### Standardiser les features avant les modèles

**Décision** : montant et durée sont standardisés (moyenne 0, écart-type 1)
via `StandardScaler` avant d'entraîner Isolation Forest / LOF.

**Pourquoi** : le montant varie de quelques centaines à plusieurs milliards
d'euros, la durée de 1 à ~120 mois. Sans standardisation, les algorithmes
(basés sur des distances) seraient dominés par le montant simplement parce
que ses valeurs numériques sont plus grandes — pas parce qu'il est plus
important métier.

### Comparer Isolation Forest et LOF plutôt que choisir un seul modèle

**Décision** : les deux méthodes sont implémentées et comparées, pas juste
l'une des deux.

**Pourquoi** : elles définissent l'anomalie différemment. Isolation Forest
isole les valeurs extrêmes dans l'absolu (ex: un marché à 12 Md€). LOF
détecte des anomalies contextuelles/locales (un point atypique par rapport
à son voisinage immédiat, même s'il n'est pas la valeur la plus extrême du
dataset). Résultat observé sur les vraies données : seulement 5,3% d'accord
entre les deux — preuve concrète qu'elles ne font pas la même chose.

### Vérifier la scalabilité et la stabilité avant de considérer le bloc terminé

**Décision** : avant de clore le bloc 2, on a testé (1) si chaque méthode
tient la charge sur le dataset complet, et (2) si le taux d'accord entre
les deux méthodes est stable sur plusieurs échantillons ou dépend du
hasard du tirage.

**Pourquoi** : un résultat mesuré une seule fois (sur un seul échantillon)
ne prouve rien en soi — ça pourrait être un coup de chance. Un modèle qui
fonctionne sur 20 000 lignes en test peut très bien planter en production
sur les données réelles. Ces deux vérifications sont exactement ce que
prévoyait la roadmap pour le bloc 5 (stabilité, reproductibilité) — autant
les faire dès maintenant, pendant qu'on code le bloc concerné, plutôt que
tout reporter à la fin où on n'aurait plus le temps de corriger si un
problème apparaît.

**Ce qu'on a trouvé** : Isolation Forest scale sans problème (41s sur
3,09M lignes). LOF plante par manque de mémoire au-delà d'un seuil entre
2M et 3,09M lignes (sur une machine à ~4 Go de RAM). Le taux d'accord
entre les deux méthodes (~5,4%) est stable sur 5 tirages indépendants
(écart-type de seulement 1,4 point).

**Alternative écartée** : ignorer ces vérifications et considérer le bloc
terminé dès que le code tournait sur l'échantillon de test. Aurait laissé
un bug de passage à l'échelle non détecté jusqu'au bloc 5, trop tard pour
le corriger sereinement.

---

## Bloc 3 — Recherche hybride

### Combiner filtres SQL, BM25 et embeddings plutôt qu'une seule technique

**Décision** : le moteur de recherche combine trois techniques différentes
plutôt que d'en choisir une seule.

**Pourquoi** : une requête comme *"marchés informatiques de montant élevé
en Île-de-France, similaires aux prestations de cybersécurité"* mélange
trois besoins de nature différente :
- des filtres exacts (région, montant) → une technique de similarité
  (embeddings) n'a pas de sens ici, on veut une comparaison stricte ;
- une recherche par mots-clés (BM25) → efficace mais rate les synonymes
  ("cybersécurité" vs "sécurité des systèmes d'information", aucun mot
  commun) ;
- une notion de similarité de sujet (embeddings) → capture les synonymes
  et reformulations, mais n'a pas de sens pour un filtre numérique strict
  comme un montant.

**Alternative écartée** : n'utiliser que des embeddings partout (plus
simple à coder) — écarté parce que les filtres stricts (montant, région)
doivent rester exacts, pas approximés par une similarité.

### Reciprocal Rank Fusion (RRF) plutôt qu'une moyenne pondérée des scores

**Décision** : la fusion entre le classement BM25 et le classement
embeddings se fait via RRF (Reciprocal Rank Fusion), pas via une moyenne
pondérée des scores bruts.

**Pourquoi** : BM25 et la similarité cosinus des embeddings n'ont pas la
même échelle (BM25 dépend du corpus et de la rareté des mots, la
similarité cosinus est bornée entre 0 et 1). Une moyenne pondérée
nécessiterait de normaliser les scores (sensible aux valeurs extrêmes) et
de choisir des poids arbitraires (pourquoi 50/50 et pas 70/30 ?). RRF
compare uniquement les **rangs** de chaque méthode, pas les scores bruts —
ça évite le problème d'échelles incompatibles, et c'est une méthode
standard (utilisée par exemple par Elasticsearch) plutôt qu'un choix de
poids inventé.

**Où s'appliquent les filtres stricts** : avant la fusion, pas dedans. Les
filtres (région, montant, dates) réduisent d'abord le dataset ; BM25 et
embeddings tournent ensuite uniquement sur ce sous-ensemble déjà filtré,
puis sont fusionnés par RRF. Un filtre exact n'a pas sa place dans une
fusion floue.

### Déduplication par marché appliquée aussi dans le bloc recherche

**Décision** : `apply_strict_filters` déduplique par `uid` (garde une
seule ligne par marché) après application des filtres région/montant,
avant de construire les index de recherche.

**Pourquoi** : trouvé en testant sur données réelles — un marché en
cotraitance (plusieurs titulaires, même `uid`, jusqu'à une dizaine de
lignes identiques pour un même appel d'offres) apparaissait dupliqué à
l'identique dans les résultats de recherche. C'est le même phénomène
identifié dans le bloc 1 (`group_by_marche`), mais appliqué ici à un
nouveau contexte (recherche plutôt qu'audit qualité).

**Leçon générale** : une règle de nettoyage/structuration des données
établie dans un bloc doit être reconduite dans tous les blocs suivants
qui manipulent les mêmes données brutes — elle ne se propage pas
automatiquement d'un module à l'autre. À vérifier systématiquement
pour le bloc 4 (dashboard) : les données affichées devront aussi être
dédupliquées au niveau marché.

### Itération d'amélioration BM25 : normalisation d'accents + mots composés

**Contexte** : testé en conditions réelles, la requête "cybersécurité"
ne remontait aucun résultat pertinent dans le top 10, alors que des
marchés clairement pertinents existaient (ex: "SÉCURITÉ DES SYSTÈMES
D'INFORMATION", 150 M€, Île-de-France). Plutôt que d'accepter cette
limite ou de basculer directement vers une fusion à poids arbitraires
(voir plus haut, écartée), diagnostic poussé avant de choisir une
correction.

**Diagnostic en deux temps** :
1. D'abord suspecté : "cybersécurité" est un mot composé sans overlap
   de token avec "sécurité" — décomposition de préfixes composés
   ajoutée (`cyber`, `télé`, etc.). Amélioration mesurée mais modeste
   (rang 121 → 88).
2. Cause plus profonde trouvée ensuite : une partie du corpus est écrite
   en majuscules sans accents (`"SECURITE"` au lieu de `"Sécurité"`).
   BM25 fait une correspondance exacte de caractères — sans
   normalisation, ces deux formes ne partagent jamais de token, quel que
   soit le sens.

**Décision** : normaliser les accents dans le tokenizer BM25 (même
technique NFKD que `cleaning.py` au bloc 1), en plus de la décomposition
de préfixes composés.

**Résultat mesuré** : rang du meilleur résultat pertinent passé de
121/931 à 3/931 — la normalisation d'accents s'est révélée être la
correction la plus impactante, plus que la décomposition de mots
composés seule.

**Pourquoi avoir creusé plutôt que d'accepter la limite ou de passer à
une fusion pondérée directement** : la fusion pondérée aurait masqué le
symptôme (en donnant plus de poids aux embeddings) sans corriger la
cause réelle (BM25 aveugle à une simple différence d'accentuation).
Corriger la cause profite à toutes les requêtes futures contenant des
mots accentués, pas seulement à ce cas précis de "cybersécurité".

**Alternative encore possible mais non retenue** : centraliser la
normalisation de texte (accents, casse) dans une fonction unique
partagée entre `cleaning.py` (bloc 1) et `bm25_search.py` (bloc 3), pour
éviter la duplication de logique actuelle. Repoussé pour ne pas
complexifier la structure du projet à ce stade — à reconsidérer si un
troisième module a besoin de la même normalisation.

---

## Bloc 4 — Dashboard

### Une seule page à onglets plutôt que plusieurs pages Streamlit

**Décision** : `src/dashboard/app.py` utilise `st.tabs()` (qualité,
anomalies, recherche) dans une seule page, plutôt que le système
multi-pages natif de Streamlit.

**Pourquoi** : cohérent avec la roadmap (bloc 4 = assemblage simple des
blocs précédents, pas de nouvelle recherche technique). Une seule page
évite de dupliquer le chargement des données entre plusieurs fichiers de
page, et suffit largement pour 3 sections.

### Mise en cache systématique du chargement de données

**Décision** : les fonctions de chargement (`charger_donnees_qualite`,
`charger_donnees_recherche`) sont décorées avec `@st.cache_data`.

**Pourquoi** : Streamlit ré-exécute tout le script à chaque interaction
(chaque clic, chaque changement de champ). Sans cache, un simple clic
rechargerait le fichier Parquet de 3M lignes depuis le disque à chaque
fois — inutilisable en pratique. `@st.cache_data` ne recharge que si les
paramètres d'entrée changent.

### Limite de test assumée pour ce bloc

**Décision** : le dashboard est testé par un smoke test d'import
(`test_dashboard.py`), pas par un test du rendu réel de l'interface.

**Pourquoi** : tester le rendu visuel réel d'une app Streamlit nécessite
un navigateur et un vrai contexte d'exécution, hors périmètre raisonnable
de pytest pour ce projet. Le smoke test attrape déjà la plupart des
erreurs bloquantes (import cassé, erreur de syntaxe, dépendance
manquante) avant même de lancer l'app. Le test complet de l'interface
(rendu, clics, onglet recherche avec le vrai modèle d'embeddings) reste
une vérification manuelle.

---

---

## Bloc 5 — Consolidation robustesse

### Vérité terrain par mots-clés plutôt qu'annotation humaine

**Décision** : Precision@K et NDCG@K sont calculés contre une vérité
terrain construite par mots-clés (un marché est "pertinent" s'il contient
l'un des mots-clés associés à la requête de test), pas contre une
annotation humaine.

**Pourquoi** : une annotation humaine de pertinence (faire lire des
centaines de marchés et juger un par un) serait la référence idéale,
mais hors périmètre raisonnable de temps pour ce projet. La vérité
terrain par mots-clés est une approximation practicable, vérifiable
(on peut relire les mots-clés choisis et juger s'ils sont raisonnables),
et surtout reproductible — elle permet de mesurer objectivement une
amélioration ou une régression du pipeline dans le temps, ce qui est le
principal usage recherché ici.

**Limite assumée et documentée** : cette méthode avantage mécaniquement
BM25 (basé sur la présence de mots) par rapport aux embeddings (qui
capturent des synonymes non listés dans les mots-clés). Les scores
Precision@K/NDCG obtenus ne mesurent donc pas la qualité absolue du
pipeline hybride, seulement sa performance sur un sous-ensemble de cas
où la pertinence est définissable par mots-clés. Alternative non retenue
faute de temps : construire un petit jeu de requêtes avec de vrais
synonymes non couverts par les mots-clés de vérité terrain (ce qui
avantagerait au contraire les embeddings) pour équilibrer le biais.

### Bug trouvé lors d'une revue externe : anomalies calculées sans déduplication par marché

**Contexte** : une revue externe du projet a signalé que
`build_feature_matrix` (utilisé par `detect_isolation_forest`,
`detect_lof`, `compare_methods` et `check_agreement_stability`)
n'appliquait pas la déduplication par marché, alors que ce principe avait
déjà été identifié et corrigé au bloc 3 pour le moteur de recherche.

**Pourquoi c'est important** : sans déduplication, un marché en
cotraitance (jusqu'à une dizaine de titulaires, donc autant de lignes
identiques) est surreprésenté dans l'entraînement des modèles. Ça peut
fausser les médianes de groupe (division CPV), les voisinages de LOF, et
faire apparaître plusieurs fois le même marché dans les alertes — ce
qu'on avait effectivement observé sans le diagnostiquer sur une capture
du dashboard (`uid` répétés dans le tableau "Top 10 anomalies").

**Pourquoi ça nous avait échappé** : la leçon "dédupliquer par marché
avant tout traitement" avait été apprise et documentée au bloc 1
(`group_by_marche`) et réappliquée au bloc 3 (recherche), mais jamais
reportée au bloc 2 (anomalies) — exactement le type d'erreur que la
note méthodologique du bloc 3 avait pourtant anticipée ("à vérifier
systématiquement pour le bloc 4"), sans qu'on l'étende explicitement au
bloc 2 après coup.

**Décision de correction** : centraliser la déduplication dans
`build_feature_matrix` (fonction `deduplicate_marches`, dans
`features.py`) plutôt que dans chaque fonction appelante. Comme toutes
les fonctions de détection passent par `build_feature_matrix`, la
correction s'applique automatiquement partout sans avoir à modifier
`detection.py`, `robustness.py`, ni le dashboard séparément.

**Résultat mesuré après correction** : le taux d'accord moyen entre
Isolation Forest et LOF passe de 5,4% à 4,5% (écart-type 1,4% → 0,85%)
sur les mêmes 5 tirages. Changement réel mais modeste — expliqué par le
fait qu'un tirage aléatoire uniforme sur l'ensemble du dataset a
statistiquement peu de chances d'inclure plusieurs lignes du même
marché en cotraitance. L'impact le plus visible de la correction n'est
donc pas ce taux d'accord global, mais l'élimination des `uid` dupliqués
dans les listes d'anomalies présentées (vérifié : 10/10 `uid` uniques
dans le top 10 après correction, contre des répétitions visibles avant).

**Leçon méthodologique (troisième occurrence du même problème)** : c'est
la troisième fois qu'une règle de nettoyage établie dans un bloc n'est
pas systématiquement reportée aux blocs suivants. Signal clair qu'une
fonction de déduplication centralisée, appliquée une seule fois au
chargement des données plutôt que réappliquée séparément dans chaque
module consommateur, réduirait ce risque structurellement plutôt que de
compter sur la vigilance à chaque nouveau bloc.

### Correction NDCG@K et Precision@K suite à une revue externe

**Bug NDCG** : la version initiale calculait l'IDCG (dénominateur de
normalisation) à partir du nombre de documents pertinents *trouvés* dans
le top k (`n_pertinents = sum(pertinences)`), pas du nombre total de
documents pertinents *existant dans le corpus*. Conséquence : un
classement qui rate une grande partie des documents pertinents pouvait
quand même obtenir NDCG=1.0, du moment que les quelques documents
trouvés étaient bien classés — cas observé concrètement (Precision@10 =
0,7 avec NDCG@10 = 1,0, incohérent).

**Correction** : `ndcg_at_k` prend maintenant `n_pertinents_corpus` en
paramètre obligatoire (pas de valeur par défaut, pour empêcher un appel
qui retomberait silencieusement sur l'ancien calcul biaisé). Le
classement idéal utilisé pour l'IDCG est désormais
`min(n_pertinents_corpus, k)` documents pertinents en tête, pas le
nombre trouvé.

**Effet mesuré** : sur `travaux_voirie_difficile` (corpus contenant 4066
marchés pertinents, mais un seul trouvé dans le top 5), NDCG@5 passe de
0,431 (ancien calcul) à 0,146 (corrigé) — une différence significative,
qui reflète correctement une mauvaise couverture de la recherche plutôt
que de la masquer.

**Bug Precision@K** : la version initiale divisait par le nombre réel de
résultats retournés (`len(top_k)`) plutôt que par k systématiquement. Un
moteur qui ne retourne qu'un seul résultat, même parfaitement pertinent,
obtenait Precision@10 = 1.0 au lieu de 0.1 — masquant le fait qu'il ne
retourne presque rien.

**Correction** : `precision_at_k` divise maintenant toujours par k
(définition standard en recherche d'information), les positions non
remplies comptant comme non pertinentes.

**Pourquoi ces deux bugs allaient dans le même sens** : les deux
anciennes implémentations partageaient le même défaut de fond — elles
évaluaient uniquement "parmi ce qui a été retourné/trouvé, est-ce bien
classé", sans jamais pénaliser ce qui aurait dû être trouvé mais ne l'a
pas été. Un rappel de la distinction précision/rappel qu'on connaissait
en théorie, mais qu'on avait mal transposée dans le code la première
fois.

### Mesurer le temps de réponse par composant plutôt que de bout en bout uniquement

**Décision** : `src/search/benchmark.py` mesure séparément la
construction de l'index BM25, la recherche BM25, la construction de
l'index embeddings, la recherche embeddings, et le pipeline complet —
pas seulement un temps total.

**Pourquoi** : un temps total ne dit pas où optimiser. Si la
construction de l'index embeddings domine largement (hypothèse
attendue, confirmée par les difficultés de lenteur déjà rencontrées),
la vraie solution en production est de précalculer/mettre en cache les
embeddings du corpus une fois pour toutes — pas d'essayer d'accélérer la
recherche elle-même, qui n'est pas le goulot d'étranglement.

### Annotation humaine découplée de l'appel au moteur de recherche

**Décision** : `src/search/annotation.py` prend en entrée des résultats
de recherche déjà calculés (dict de dataframes), pas une fonction qui
appelle `engine.search()` elle-même. Un script séparé
(`run_annotation.py`) fait l'appel réel et génère le fichier CSV.

**Pourquoi** : même principe de découplage que pour `evaluation.py`
(bloc 5) — la logique de génération du CSV et de comparaison
mots-clés/humain est ainsi testable sans dépendre du modèle
d'embeddings, seul le script d'appel final nécessite un accès internet.

**Objectif de cette annotation** : la vérité terrain par mots-clés
utilisée pour Precision@K/NDCG depuis le début du bloc 5 n'a jamais été
comparée à un vrai jugement humain — ce module permet de vérifier si
elle est fiable ou si elle génère des faux positifs/négatifs
significatifs (ex: le cas SSI = Sécurité Incendie vs Système
d'Information, déjà découvert au bloc 3, est exactement le genre
d'erreur qu'une vérité terrain par mots-clés seule ne peut pas éviter).

### Transparence sur la méthode d'annotation réellement utilisée

**Décision** : documenter explicitement que l'annotation des 32 lignes
n'a pas été réalisée par une tierce personne indépendante du projet,
mais par une lecture ligne par ligne, jugements proposés puis validés.

**Pourquoi le dire clairement plutôt que de présenter ça comme une
"vraie" annotation humaine indépendante** : la valeur méthodologique
d'une annotation dépend de qui juge et comment. Prétendre à une
indépendance qui n'existe pas serait plus trompeur que d'assumer la
limite. Ceci dit, la méthode reste plus rigoureuse qu'une simple
confirmation sans lecture : chaque ligne a été jugée individuellement
avec une justification explicite (voir la conversation du projet), pas
une approbation en bloc.

**Résultat obtenu malgré cette limite** : 3 thèmes sur 4 à 100% d'accord
avec la vérité terrain par mots-clés, un seul écart trouvé (faux positif
logique sur "cantine" désignant un bâtiment plutôt qu'un service, même
famille que le cas SSI du bloc 3). Ça apporte un premier niveau de
validation, même partiel, plutôt que de laisser cette question
complètement ouverte comme au début du bloc 5.

---

## Itération post-bloc 5 : cache disque des embeddings

**Contexte** : le benchmark du bloc 5 a chiffré le vrai goulot
d'étranglement du projet (~28 secondes par recherche, dominé par le
recalcul des embeddings à chaque requête). Cette itération s'attaque
directement à ce point, identifié comme la recommandation n°1 de la
synthèse finale.

**Décision** : un cache disque (fichier `.npz`, `uid` → vecteur), plutôt
qu'un index FAISS/Qdrant complet.

**Pourquoi ce choix, et pas directement FAISS/Qdrant** : un vrai index
de similarité (FAISS/Qdrant) demande une infrastructure et un calcul
initial sur l'intégralité du corpus. Précalculer les embeddings des
1,73M marchés uniques prendrait, en extrapolant linéairement le
benchmark existant (27,5 s pour 931 candidats), environ **14 heures**
sur une machine sans GPU — irréaliste dans ce contexte. Le cache disque
est une solution intermédiaire honnête : il ne résout pas le cas d'un
tout premier lancement sur un sous-ensemble jamais vu (toujours aussi
lent la première fois), mais élimine le recalcul pour tout marché déjà
rencontré lors d'une recherche précédente — pertinent dès que le même
sous-ensemble de marchés est interrogé plusieurs fois (cas réaliste
d'un dashboard utilisé en continu sur les mêmes filtres récurrents,
ex: toujours "Île-de-France").

**Design retenu** : le cache est cumulatif (grossit à chaque nouvelle
recherche, ne perd jamais les entrées précédentes) et partiel (seuls les
`uid` manquants sont réellement encodés à chaque appel). La logique de
décision ("quels `uid` sont déjà en cache") est séparée du calcul
d'encodage lui-même, pour rester testable sans modèle ni réseau.

**Limite assumée, à nouveau** : ce n'est toujours pas une solution de
production à l'échelle du corpus complet. Un vrai déploiement
nécessiterait soit un calcul batch unique sur toute la base (plusieurs
heures, une seule fois, sur une machine dédiée ou avec GPU), soit un
index FAISS/Qdrant construit progressivement en tâche de fond. Le cache
disque reste une amélioration réelle et mesurable pour l'usage actuel du
projet (dashboard avec un nombre de filtres limité et réutilisés), pas
une solution générale.

---

## Bloc prédiction : régression supervisée sur `offresRecues`

### Choix de la cible : offresRecues plutôt qu'une cible de fraude fabriquée

**Décision** : combler le manque de compétences en apprentissage
supervisé identifié dans la synthèse finale en prédisant `offresRecues`
(nombre d'offres reçues par un marché), pas une cible de fraude
artificielle.

**Pourquoi** : `offresRecues` est une variable réellement observée dans
les données, pas une étiquette inventée pour l'occasion. Elle a un vrai
sens métier (mesure de compétitivité d'un marché) et évite l'écueil
qu'on s'était fixé dès le bloc 2 : ne jamais fabriquer artificiellement
une vérité terrain de fraude qui n'existe pas.

### Biais de sélection découvert : le taux de manquant dépend de la procédure

**Constat** : `offresRecues` a ~60% de valeurs manquantes globalement,
mais ce taux varie énormément selon le type de procédure — de 31%
("Marché passé sans publicité") à quasiment 100% ("Procédure
concurrentielle avec négociation"). Ce n'est pas un hasard.

**Implication assumée, pas contournée** : le modèle entraîné sur les
~40% de marchés où `offresRecues` est connu ne peut, par construction,
apprendre que sur les types de procédure où cette donnée est
généralement renseignée. Il généralisera mal (voire pas du tout) sur les
procédures où elle est presque toujours absente. Documenté comme limite
du modèle, pas corrigé artificiellement (imputer une valeur inventée
pour ces cas serait pire que de reconnaître l'angle mort).

### Bug bonus découvert : `procedure` n'a jamais été normalisée

**Constat** : en creusant le biais de sélection ci-dessus, découverte
que `procedure` contient 16 valeurs distinctes qui se réduisent à 11
vraies catégories une fois normalisées — variantes d'apostrophe (`'`
droite, `'` typographique, absente : "Appel d'offres" / "Appel
d'offres" / "Appel d offres") et d'accents ("Procedure" / "Procédure"),
jamais traitées jusqu'ici. Contrairement à `nature` (normalisée au bloc
1), `procedure` n'avait jamais été passée dans une fonction de
nettoyage.

**Décision** : créer `src/common/text_normalization.py`, un module de
normalisation partagé — implémente enfin la recommandation n°2 de la
synthèse finale (centraliser une logique dupliquée 3 fois : bloc 1,
bloc 3, et maintenant ce bug sur `procedure`). Utilisé pour ce nouveau
bloc ; **le refactoring des modules existants** (`cleaning.py`,
`bm25_search.py`) pour réutiliser ce module commun n'a **pas** été fait
— jugé trop risqué de toucher du code déjà testé et validé si tard dans
le projet, sans bénéfice fonctionnel immédiat. Centralisation partielle
assumée : le nouveau code l'utilise, l'ancien reste tel quel.

### Exclusion de données invalides plutôt que flag, ici seulement

**Décision** : pour ce bloc spécifiquement, les lignes avec
`offresRecues < 0` ou `montant <= 0` sont **exclues** de l'entraînement,
pas seulement flaggées (contrairement à la philosophie générale du
projet établie au bloc 1).

**Pourquoi cette exception assumée** : une valeur invalide dans la
cible ou une feature d'entraînement ne peut de toute façon pas
contribuer utilement à l'apprentissage (elle ferait planter la
transformation log1p, ou fausserait silencieusement le modèle). La
distinction : flagger a du sens quand on veut *analyser* les données
telles quelles (bloc 1, bloc 2) ; exclure a du sens quand une valeur est
structurellement inutilisable pour la tâche précise (entraînement d'un
modèle nécessitant des valeurs positives pour une transformation log).

### Comparaison de modèles : pourquoi rapporter les métriques en espace log ET en espace réel

**Décision** : chaque modèle est évalué avec R²/RMSE en espace log
(celui de l'entraînement) et après reconversion (`expm1`) en espace réel
(nombre d'offres).

**Pourquoi les deux, pas un seul** : découvert en pratique que les deux
peuvent se contredire. Ridge obtient R²=0,179 en log mais seulement
R²=0,032 en réel ; Random Forest fait l'inverse (0,381 en log, 0,676 en
réel — bien meilleur). Diagnostic : l'espace log compresse l'influence
des valeurs extrêmes (un marché à 2500 offres et un à 10 offres sont
proches une fois transformés en log), ce qui peut masquer qu'un modèle
échoue à capturer la "queue" de la distribution (les cas à très
nombreuses offres). Random Forest, plus flexible, capture mieux ces cas
extrêmes, d'où son bien meilleur score en espace réel. Rapporter les
deux évite de se fier à une seule métrique qui raconterait une histoire
incomplète.

### Résultats obtenus (référence, dataset complet)

656 023 marchés exploitables (après filtrage cible valide + montant
positif + déduplication marché). Comparaison sur le jeu de test (20%) :

| Modèle | R² (log) | R² (réel) | MAE (réel) |
|---|---|---|---|
| Baseline (médiane) | -0,054 | -0,031 | 7,71 offres |
| Ridge | 0,179 | 0,032 | 7,59 offres |
| Random Forest | 0,381 | **0,676** | **5,45 offres** |

Random Forest bat nettement la baseline et Ridge, en particulier en
espace réel. Features les plus importantes (Random Forest) : le montant
(log-transformé) domine largement, suivi de la durée, de la division
CPV (notamment la division 45 - travaux), et du type de procédure.

---

## Test différé du bloc 2 : transformation log des montants avant LOF

**Contexte** : suggestion reçue lors de la toute première revue externe
du projet ("tester une transformation logarithmique des montants avant
LOF"), notée comme travail futur mais jamais testée jusqu'ici — refermée
maintenant, après le bloc prédiction, par souci de ne pas laisser de
promesse non tenue.

**Décision** : `build_feature_matrix` accepte un paramètre
`transformation_log` (désactivé par défaut, pour ne pas changer le
comportement déjà documenté dans les résultats précédents). Utilise un
**log signé** (`sign(x) * log1p(|x|)`), pas un `log1p` classique : le
dataset contient ~2% de montants négatifs (jusqu'à -7,1M€, voir audit
qualité du bloc 1), et un `log1p` standard est indéfini pour ces
valeurs. Le log signé compresse l'échelle des deux côtés du zéro tout en
gardant le signe d'origine — cohérent avec le choix de ne jamais exclure
ces montants négatifs de la détection d'anomalies (contrairement au
bloc prédiction, où ils sont exclus car inutilisables pour
l'entraînement).

**Résultat mesuré** (5 tirages de 20 000 marchés, mêmes seeds que la
mesure de référence) :

| | Sans transformation (référence) | Avec transformation log |
|---|---|---|
| Taux d'accord moyen | 4,5% | **13,0%** |
| Écart-type | 0,85% | 2,51% |

**Interprétation** : le taux d'accord entre Isolation Forest et LOF
**triple** avec la transformation, tout en restant proportionnellement
stable (écart-type relatif similaire dans les deux cas, ~19%). Ça
confirme l'hypothèse de la revue externe : sans transformation, LOF
était probablement dominé par les valeurs extrêmes de montant dans son
calcul de distances, le faisant diverger fortement d'Isolation Forest.
Avec la transformation, LOF détecte des anomalies bien mieux alignées
avec Isolation Forest — signe d'un comportement plus sensé, pas
seulement d'un chiffre différent.

**Décision finale** : garder `transformation_log=False` par défaut
(pour ne pas invalider silencieusement les résultats déjà documentés et
discutés ailleurs dans ce projet), mais **recommander explicitement**
`transformation_log=True` pour tout usage futur ou toute nouvelle
analyse — le gain est net et le raisonnement qui le justifie est solide.

---

## Aller plus loin sur la prédiction : validation croisée et hyperparamètres

### Bug trouvé en validation croisée : dureeMois contient des valeurs aberrantes

**Contexte** : en ajoutant la validation croisée K-fold pour consolider
les résultats du bloc prédiction, un pli a produit un R² catastrophique
(de l'ordre de -10⁷¹) pour Ridge.

**Diagnostic** : une ligne avec `dureeMois = 21886` (plus de 1800 ans,
donnée manifestement erronée) a fait exploser numériquement la
prédiction — un coefficient Ridge raisonnable multiplié par une valeur
aberrante donne une prédiction log extrême, qui devient quasi infinie
une fois reconvertie via `expm1`. Vérification sur l'ensemble du
dataset : 1042 valeurs négatives/nulles et 1864 valeurs supérieures à
240 mois (jusqu'à 31 410 mois, ~2618 ans) — 0,09% des lignes, une
proportion infime mais suffisante pour casser un modèle linéaire non
robuste à ce type d'aberration.

**Correction** : `filtrer_cible_valide` exclut maintenant aussi les
lignes avec `dureeMois` hors de la plage (0, 240] mois (20 ans, seuil
arbitraire mais raisonnable pour un marché public). Troisième variable
(après `montant` et `offresRecues`) où le dataset s'est révélé contenir
des valeurs numériquement aberrantes — motif récurrent de ce dataset
plutôt qu'un cas isolé.

**Leçon méthodologique** : ce bug n'était pas détectable avec un simple
découpage train/test (la ligne aberrante n'était probablement pas tombée
dans le jeu de test lors du split initial) — c'est justement la
validation croisée, en testant plusieurs découpages, qui l'a fait
apparaître. Argument concret en faveur de la validation croisée au-delà
du principe théorique : elle attrape des problèmes qu'un seul découpage
peut manquer par chance.

### Validation croisée : les résultats initiaux ne sont pas un coup de chance

**Résultat** (5-fold, sous-échantillon de 50 000 marchés pour rester
rapide à exécuter ; résultat initial mesuré sur les 656 023 marchés
complets) :

| Modèle | R² réel (single-split, référence) | R² réel (5-fold CV) |
|---|---|---|
| Ridge | 0,032 | 0,039 ± 0,012 |
| Random Forest | 0,676 | 0,633 ± 0,194 |

Les deux résultats restent cohérents avec la mesure initiale — la
validation croisée confirme que ce n'était pas un artefact du découpage
initial. L'écart-type de Random Forest (0,194, soit ~30% de sa moyenne)
est notable : sa performance varie sensiblement selon les plis, une
limite honnête à garder en tête plutôt qu'à ignorer.

### Recherche d'hyperparamètres : optimiser en espace log peut dégrader le résultat réel

**Ce qui a été fait** : `RandomizedSearchCV` sur Random Forest (15
combinaisons, validation croisée à 3 plis interne), optimisant d'abord
le R² en espace log (l'espace d'entraînement).

**Premier résultat, sur un découpage train/test identique pour une
comparaison équitable** :

| Modèle | R² réel | MAE réel |
|---|---|---|
| Random Forest par défaut (100 arbres, profondeur 15) | 0,596 | 5,83 offres |
| Random Forest "optimisé" (scorer log) | 0,515 | 6,23 offres |

**Le modèle "optimisé" est en réalité moins bon** en espace réel, malgré
un meilleur score en validation croisée pendant la recherche (0,345 en
log). Même leçon que celle découverte avec Ridge (bloc prédiction
initial) : optimiser une métrique en espace log ne garantit pas un gain
en espace réel.

**Correction tentée** : un scorer personnalisé (`_r2_espace_reel`,
`SCORER_ESPACE_REEL`) a été ajouté, calculant le R² directement en
espace réel (après `expm1`) pour que `RandomizedSearchCV` optimise la
bonne métrique.

**Résultat après correction, honnête** : le modèle sélectionné avec ce
nouveau scorer (`n_estimators=200, min_samples_leaf=2, max_features=0.5,
max_depth=12`) obtient R²_réel=**0,517** sur le jeu de test — **toujours
moins bon** que le Random Forest par défaut (0,596), malgré un score de
0,566 pendant la recherche elle-même (en validation croisée interne à 3
plis sur le train). La correction de la métrique de score n'a donc pas
suffi à battre la configuration par défaut sur ce jeu de test précis.

**Interprétation honnête** : deux explications possibles, non
tranchées : (1) le budget de recherche (15 combinaisons aléatoires sur
un sous-échantillon de 50 000) est probablement trop limité pour
explorer l'espace des hyperparamètres efficacement ; (2) les paramètres
par défaut choisis initialement (100 arbres, profondeur 15) étaient déjà
une configuration raisonnablement bonne pour ce problème précis, pas
si simple à améliorer avec un budget de recherche limité. Ce résultat
illustre que la recherche d'hyperparamètres n'est pas une solution
magique — elle peut ne pas apporter de gain, et c'est un résultat
légitime à rapporter tel quel plutôt qu'à cacher.

**Non fait, pour aller plus loin** : augmenter le budget de recherche
(plus d'itérations, jeu de données complet plutôt qu'un sous-échantillon
de 50 000), ou tester une grille plus ciblée autour des valeurs par
défaut plutôt qu'un espace de recherche aussi large.

---

## Pondération RRF : résultat plus nuancé que l'hypothèse de départ

**Hypothèse de départ** : sur "travaux de voirie" (requête difficile),
BM25 seul avait un score correct par coïncidence lexicale ; la fusion à
poids égal l'aurait dilué avec un moins bon score embeddings.
L'hypothèse suggérait de favoriser BM25 pour corriger.

**Résultat empirique** (`compare_weighted_rrf.py`, 581 marchés pertinents
dans le corpus) :

| Configuration | P@5 | NDCG@5 | P@10 | NDCG@10 |
|---|---|---|---|---|
| Poids égal (référence) | 0,2 | 0,214 | 0,2 | 0,217 |
| BM25 ×3 | 0,2 | 0,214 | **0,5** | **0,42** |
| Embeddings ×3 | **0,4** | **0,345** | 0,3 | 0,288 |

**L'hypothèse n'était que partiellement confirmée** : favoriser BM25
améliore bien le résultat, mais surtout sur le top 10 (P@10 : 0,2 → 0,5).
Favoriser au contraire les **embeddings** améliore le top 5 (P@5 : 0,2 →
0,4) — l'inverse de ce qu'on attendait initialement. Les deux
pondérations battent le poids égal, mais à des profondeurs de classement
différentes.

**Conclusion honnête** : il n'y a pas de poids universellement meilleur
sur ce cas — le meilleur choix dépend de si on privilégie la précision
en tête de liste (favoriser embeddings) ou une meilleure couverture sur
une liste plus longue (favoriser BM25). Pas de solution tranchée à
adopter comme nouveau défaut ; le paramètre `poids_fusion` reste
optionnel (`None` par défaut, poids égal) plutôt que de figer un choix
qui ne serait optimal que pour un cas précis testé sur un seul thème.

**Non fait, pour aller plus loin** : tester la pondération sur les 3
autres thèmes (pas seulement "travaux de voirie") pour voir si un poids
donné généralise, ou si l'effet est spécifique à ce cas précis.

---

## Tests fonctionnels du dashboard (Playwright)

**Décision** : ajouter `tests/test_dashboard_functional.py`, qui lance
réellement le serveur Streamlit et un navigateur headless (Playwright),
au-delà du simple test d'import déjà existant (`test_dashboard.py`).

**Pourquoi c'était une vraie limite avant** : le test d'import attrape
les erreurs de syntaxe ou d'import cassé, mais pas les régressions
visuelles/fonctionnelles. Seule une vérification manuelle comblait ce
trou jusqu'ici.

**Protections de conception** :
- `pytest.mark.skipif` si `data/raw/decp.parquet` est absent (toujours
  le cas en CI, fichier volontairement non versionné) — skip propre et
  rapide plutôt qu'échec systématique en CI.
- `pytest.importorskip` + `try/except` si Playwright/Chromium n'est pas
  installé — même logique de skip propre.

**Investigation complète sur 2 des 5 tests (widgets des onglets
anomalies/recherche), 9 tentatives indépendantes, chacune avec un
raisonnement technique distinct** :

1. **Portabilité Windows** : `["streamlit", "run", ...]` échouait sur
   Windows → corrigé via `sys.executable -m streamlit`.
2. **Sélecteurs de clic** : `get_by_text` remplacé par `get_by_role("tab", ...)`
   (structure React Aria de Streamlit), avec repli.
3. **Délais progressifs** : de 10s à 120s selon les étapes — les imports
   `sentence-transformers`/`torch` au niveau module peuvent être lents.
4. **Vérifier les éléments plutôt que le texte** : `[data-testid=...]`
   plutôt que du texte, au cas où le texte ne soit pas dans le DOM
   cherchable (tableaux virtualisés sur canvas, confirmé pour l'onglet
   qualité).
5. **Attendre la disparition du squelette** (`stSkeleton`, trouvé en
   inspectant le HTML brut) plutôt que l'apparition du widget final.
6. **Clic souris brut au niveau OS** (`page.mouse.click` aux coordonnées)
   plutôt que `locator.click()`, pour simuler une vraie interaction
   matérielle.
7. **Isolation totale serveur+page par test** (port TCP dynamique,
   fixtures `function`-scoped) — éliminait toute possibilité de
   contamination d'état entre tests.
8. **Attente de fin d'exécution du script complet** avant de cliquer sur
   un autre onglet — découverte que Streamlit exécute tout son script
   Python dans l'ordre en une seule passe (titre → audit qualité →
   onglets suivants), et qu'un clic trop précoce bascule sur un panneau
   dont le contenu n'a simplement pas encore été généré. Cette piste a
   corrigé le symptôme "panneau totalement vide" (plus de caption du
   tout) en "caption visible mais widget en squelette" — un vrai
   progrès partiel, mais pas une résolution complète.
9. **Changement de version de Streamlit** (1.60.0 → 1.32.0, testé
   explicitement pour écarter l'hypothèse d'une refonte trop récente de
   l'interface — identifiants React Aria observés dans le HTML,
   suggérant un changement d'architecture frontend récent) : **aucun
   effet**, symptôme identique bit pour bit sur les deux versions.
   Élimine définitivement l'hypothèse "bug de version".

**Résultat final** : 3 tests sur 5 passent de façon fiable (titre,
présence des 3 onglets, tableau qualité). Les 2 tests sur les widgets
des onglets anomalies/recherche échouent de façon parfaitement
reproductible et documentée, marqués `@pytest.mark.xfail` plutôt que
supprimés ou laissés en échec silencieux.

**Théorie retenue pour expliquer le symptôme** (non vérifiée
empiriquement plus loin, faute de temps) : les widgets interactifs
Streamlit dépendent probablement d'API de détection de visibilité
(dans l'esprit d'`IntersectionObserver`) pour décider quand
s'hydrater réellement — un mécanisme déjà confirmé pour le tableau
qualité (rendu sur canvas virtualisé). En mode headless, ces API se
comportent différemment d'un navigateur avec un vrai affichage.
L'onglet actif dès le chargement initial (qualité) s'hydrate avant que
cette différence n'entre en jeu ; les onglets activés après coup par un
clic automatisé n'y échappent pas. Piste testable identifiée mais non
explorée : lancer le navigateur en mode `headless=False` pour confirmer
cette théorie précisément.

**Pourquoi s'arrêter à 9 tentatives plutôt que continuer** : chacune
reposait sur un raisonnement technique différent et légitime, et
plusieurs ont fait progresser le diagnostic (la piste n°8 a changé la
nature du symptôme, la n°9 a définitivement écarté une cause plausible).
Le rendement marginal de nouvelles tentatives devenait cependant très
faible, et ce point était explicitement le moins prioritaire des 5
recommandations de la synthèse finale. Le dashboard fonctionne
correctement en usage réel (confirmé à de multiples reprises par
captures d'écran manuelles tout au long du projet) — la limite documentée
concerne l'automatisation des tests, pas le produit lui-même.

## Pratiques transverses

### Un commit = un changement logique

**Décision** : séparer les commits par nature (code / tests / documentation),
plutôt que tout regrouper en un seul commit fourre-tout.

**Pourquoi** : permet de retrouver rapidement, via `git log`, quand et
pourquoi un changement précis a été fait — utile pour soi-même en cours de
projet, et pour quiconque relit l'historique plus tard.
