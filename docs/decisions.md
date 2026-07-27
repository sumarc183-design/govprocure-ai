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

## Pratiques transverses

### Un commit = un changement logique

**Décision** : séparer les commits par nature (code / tests / documentation),
plutôt que tout regrouper en un seul commit fourre-tout.

**Pourquoi** : permet de retrouver rapidement, via `git log`, quand et
pourquoi un changement précis a été fait — utile pour soi-même en cours de
projet, et pour quiconque relit l'historique plus tard.
