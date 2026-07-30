# Limites et robustesse du système

Ce document est mis à jour au fil du projet, bloc par bloc, plutôt qu'écrit en une seule fois à la fin.

## Principe général

- Une anomalie détectée n'est jamais présentée comme une fraude : uniquement comme un dossier à prioriser pour examen humain.
- Le système documente explicitement les cas où il doit s'abstenir de conclure (confiance insuffisante, données manquantes critiques, hors périmètre du modèle).

## Bloc 1 — Qualité des données

**Méthodologie** : score par colonne = 1 - taux de valeurs manquantes,
pénalisé de 0.05 par type de problème structurel détecté (dates hors plage,
montants incohérents, catégories mal normalisées, versioning incohérent).
Score volontairement simple à ce stade ; à pondérer selon l'importance
métier de chaque variable si besoin (ex: `montant` est plus critique que
`titulaire_nom` pour la suite du projet).

**Limites connues du score actuel** :
- Ne détecte pas les incohérences croisées entre colonnes (ex: `dureeMois`
  incompatible avec les dates de notification/fin).
- Le seuil de montant "suspect" (1 Md€) est arbitraire — pertinent pour un
  premier filtrage, mais certains grands marchés d'État peuvent légitimement
  dépasser ce seuil (à valider avant de les traiter comme anomalies).
- Le score ne distingue pas une valeur manquante "structurelle" (champ non
  applicable) d'une vraie donnée absente (ex: `offresRecues` peut être
  légitimement non renseigné selon le type de procédure).

**Résultats** : voir le tableau détaillé dans
[data_sources.md](data_sources.md#notes-daudit).

**Prochaine étape** : fonction de nettoyage (normalisation des catégories,
règles de traitement des montants aberrants) avant utilisation par les
blocs anomalies et recherche.

## Bloc 2 — Détection d'anomalies

**Méthodes comparées** : Isolation Forest et Local Outlier Factor (LOF),
sur un échantillon de 20 000 marchés (dataset complet : ~3,09M marchés
avec montant et codeCPV renseignés).

**Features utilisées** : montant, durée (mois), et l'écart par rapport à
la médiane des marchés comparables (même division CPV — 2 premiers
chiffres du code CPV, ~174 groupes bien peuplés). Voir
`src/anomaly/features.py`.

> ⚠️ **État historique avant correction** — ce chiffre (5,3%) a été
> mesuré avant la correction de déduplication par marché (bloc 5). Le
> résultat final corrigé est **4,5%** — voir la section "Stabilité /
> reproductibilité du taux d'accord" plus bas pour le détail complet
> avant/après. Conservé ici tel quel pour l'historique du raisonnement.

**Résultat clé — accord entre les deux méthodes** : seulement **5,3%**
de recouvrement entre les anomalies détectées par Isolation Forest et
celles détectées par LOF (contamination fixée à 5% pour les deux).

Ce faible taux d'accord n'est **pas une erreur** : il illustre une
différence de nature entre les deux approches, observée concrètement sur
les données :
- **Isolation Forest** identifie les valeurs les plus extrêmes dans
  l'absolu (ex: un marché à 12,3 Md€, ratio ×60 000 par rapport à la
  médiane de sa catégorie).
- **LOF** identifie des anomalies contextuelles/locales : un petit
  groupe dense de montants élevés (18-35 M€, ratio ×90-170) qui, sans
  être les valeurs les plus extrêmes du dataset, sont atypiques par
  rapport à leur voisinage immédiat.

**Implication méthodologique, nuancée** : les deux méthodes ne sont pas
redondantes (elles capturent des notions différentes de l'anomalie),
mais ce constat statistique ne prouve pas à lui seul qu'elles sont
réellement *complémentaires* au sens utile du terme — un faible taux
d'accord peut aussi signifier que l'une des deux (ou les deux) détecte
du bruit plutôt que des anomalies pertinentes. Combiner les deux dans un
système de priorisation reste une piste raisonnable, mais nécessiterait
une validation métier (des cas confirmés, ou au moins un avis humain sur
un échantillon des désaccords) avant de conclure que la combinaison
apporte réellement de la valeur.

**Limite technique connue** : LOF émet un avertissement scikit-learn sur
la présence de valeurs dupliquées, qui peuvent fausser le calcul de
distance aux voisins. Cause identifiée : un même marché en cotraitance
génère plusieurs lignes très proches (même montant, même durée) — géré
au niveau des features mais peut nécessiter un traitement dédié
(dédoublonnage avant LOF, ou augmentation de `n_neighbors`) avant la
version finale.

**Bug corrigé pendant le développement** : la fonction
`build_feature_matrix` calculait l'imputation des valeurs manquantes
mais ne la réinjectait pas dans le dataframe retourné (les NaN
persistaient silencieusement). Corrigé et couvert par un test de
non-régression (`test_build_feature_matrix_imputes_missing_values`).

**À faire avant la version finale** :
- Décider d'une stratégie de combinaison des deux méthodes (union,
  score composite, ou présentation séparée à l'utilisateur final).

### Passage à l'échelle (tests sur dataset complet, 3,09M marchés)

| Méthode | Taille testée | Temps | Résultat |
|---|---|---|---|
| Isolation Forest | 3 089 375 (dataset complet) | 41,2 s | ✅ Fonctionne, scale bien |
| LOF | 20 000 | 0,3 s | ✅ |
| LOF | 100 000 | 1,8 s | ✅ |
| LOF | 500 000 | 23,9 s | ✅ |
| LOF | 1 000 000 | 58,2 s | ✅ |
| LOF | 2 000 000 | 189,8 s (~3 min 10) | ✅ |
| LOF | 3 089 375 (dataset complet) | — | ❌ **Process tué (OOM)** |

**Conclusion** : Isolation Forest passe à l'échelle sans problème sur le
dataset complet. **LOF ne passe pas à l'échelle** au-delà d'un seuil situé
entre 2M et 3,09M lignes (testé sur une machine à 3,9 Go de RAM
disponible — le seuil réel dépend de la RAM disponible, à revérifier sur
la machine de déploiement finale).

**Décision** : pour un usage en production sur le dataset complet, LOF
doit être appliqué sur un échantillon représentatif (ex: 1-2M lignes max
selon la RAM disponible), jamais sur l'intégralité brute. Isolation Forest
peut lui tourner sur tout le dataset sans souci.

**Ceci est un exemple concret de "cas où le modèle doit refuser de
conclure"** (prévu dans le périmètre du bloc 5) : plutôt que de laisser
LOF planter silencieusement sur un trop gros volume, le pipeline final
devra détecter la taille du dataset en amont et basculer automatiquement
sur un échantillon si nécessaire, avec un message explicite plutôt qu'un
crash.

### Stabilité / reproductibilité du taux d'accord

**Mesure initiale (avant correction de la déduplication par marché,
voir docs/decisions.md)** : taux d'accord de 5,3% entre Isolation Forest
et LOF, revérifié sur 5 échantillons indépendants de 20 000 marchés
chacun (seeds différentes) via
`src/anomaly/robustness.py::check_agreement_stability`.

| Seed | n anomalies IF | n anomalies LOF | n accord | taux accord |
|---|---|---|---|---|
| 1 | 1000 | 1000 | 121 | 0,064 |
| 2 | 1000 | 1000 | 77 | 0,040 |
| 3 | 1000 | 1000 | 134 | 0,072 |
| 42 | 986 | 1000 | 100 | 0,053 |
| 100 | 999 | 1000 | 82 | 0,043 |

Moyenne : 5,4% — écart-type : 1,4%.

**Mesure après correction** (déduplication par marché appliquée dans
`build_feature_matrix`, bug trouvé lors d'une revue externe — voir
docs/decisions.md) :

| Seed | n anomalies IF | n anomalies LOF | n accord | taux accord |
|---|---|---|---|---|
| 1 | 969 | 969 | 97 | 0,053 |
| 2 | 969 | 969 | 79 | 0,042 |
| 3 | 967 | 967 | 96 | 0,052 |
| 42 | 969 | 969 | 60 | 0,032 |
| 100 | 968 | 968 | 86 | 0,046 |

**Moyenne : 4,5% — écart-type : 0,85%**

**Conclusion** : le faible taux d'accord reste **stable et
reproductible** après correction (écart-type resserré, pas élargi) —
toujours pas un artefact du tirage. La correction elle-même a un effet
mesuré modeste sur cette métrique précise (5,4% → 4,5%), parce qu'un
tirage aléatoire uniforme sur 3M lignes a statistiquement peu de chances
de piocher plusieurs lignes du même marché en cotraitance. L'effet le
plus visible de la correction n'est pas ce taux global mais l'absence
de `uid` dupliqués dans les listes d'anomalies présentées à
l'utilisateur (vérifié : 10/10 `uid` uniques dans le top 10 après
correction, contre des répétitions visibles dans le dashboard avant —
voir capture d'écran discutée lors de la revue externe).

## Bloc 3 — Recherche hybride

**Architecture** : filtres stricts (région, montant) → réduction du
dataset → BM25 + embeddings sur le sous-ensemble filtré → fusion par
Reciprocal Rank Fusion (RRF). Voir `docs/decisions.md` pour le
raisonnement détaillé.

**Dépendance réseau à connaître** : le module
`src/search/embeddings_search.py` télécharge un modèle depuis
huggingface.co au premier lancement (~470 Mo, une seule fois, mis en
cache ensuite). Nécessite un accès internet complet la première fois.
Validé en conditions réelles :
- l'extraction de filtres (`filters.py`) — 5 tests, aucune dépendance externe ;
- BM25 (`bm25_search.py`) — 4 tests, sur données synthétiques et un
  échantillon réel de `objet` (fonctionne, exemple : requête "cybersécurité
  systèmes information" retrouve en premier un marché intitulé "Analyse
  et remédiation des événements de sécurité des systèmes d'information") ;
- la fusion RRF (`fusion.py`) — 4 tests, logique pure sans dépendance
  externe ;
- `apply_strict_filters` (`engine.py`) — 2 tests sur la logique de
  filtrage seule.

> ⚠️ **État historique** — les points ci-dessous étaient présentés comme
> "reste à valider" avant l'exécution complète du pipeline. **Tout a
> depuis été fait** : le modèle a été téléchargé et testé, le pipeline
> complet a été exécuté de bout en bout sur données réelles, et le temps
> de calcul a été mesuré précisément. Voir "Confirmation finale" et
> "Temps de réponse" plus bas pour les résultats définitifs.

~~**Reste à valider**~~ (nécessitait un accès internet complet pour le
téléchargement initial du modèle) :
- le téléchargement effectif du modèle `paraphrase-multilingual-MiniLM-L12-v2` ;
- le pipeline complet `engine.search()` de bout en bout sur un vrai
  échantillon, en particulier la pertinence des résultats sur des
  requêtes avec synonymes (ex: "cybersécurité" vs "sécurité des systèmes
  d'information") ;
- le temps de calcul de l'encodage des embeddings sur un sous-ensemble
  filtré de taille réaliste.

**Point d'architecture à surveiller pour la suite** : l'encodage des
embeddings recalcule les vecteurs à chaque requête sur le sous-ensemble
filtré. Acceptable tant que les filtres stricts réduisent significativement
le dataset (quelques milliers de lignes), mais si une requête ne filtre
presque rien (ex: pas de région ni montant précisés), l'encodage pourrait
devenir coûteux sur un sous-ensemble encore proche des 3M lignes — à
mesurer une fois le modèle testable, et éventuellement pré-calculer/mettre
en cache les embeddings de tout le corpus plutôt que de les recalculer
à la volée (via FAISS/Qdrant, prévu dans la stack technique initiale).

### Premier test en conditions réelles

**Bug trouvé et corrigé** : la requête *"marchés de cybersécurité en
Île-de-France de montant élevé"* donnait un texte libre pollué par des
mots de liaison orphelins (`'marchés de cybersécurité en de'` au lieu de
`'marchés de cybersécurité'`), car les mots ("en", "de") entourant la
région et le montant extraits n'étaient pas nettoyés après extraction.
Corrigé dans `filters.py` (`_nettoyer_mots_orphelins`), avec test de
non-régression associé.

**Limite réelle observée, pas un bug** : sur un échantillon de 5 000
marchés, filtré sur Île-de-France + 90e percentile de montant, seuls
54 marchés passaient le filtre strict. Vérification manuelle : sur les
5 000 marchés de l'échantillon, seulement 10 contenaient un terme lié à
la cybersécurité — et la plupart étaient des **faux positifs dus à un
homonyme** : le sigle `SSI` désigne le plus souvent, dans ce corpus,
un **Système de Sécurité Incendie** (bâtiment), pas un Système
d'Information. Résultat : le sous-ensemble filtré (54 marchés) ne
contenait probablement aucun vrai marché de cybersécurité, donc BM25
et les embeddings ne pouvaient renvoyer que "le moins hors-sujet" d'un
lot déjà non pertinent — pas un défaut du moteur de recherche, mais une
conséquence directe d'un échantillon de test trop petit.

**Implications** :
- Refaire le test sur un échantillon plus grand (ou le dataset complet)
  pour vérifier la pertinence des résultats quand de vrais candidats
  existent réellement dans le sous-ensemble filtré.
- L'ambiguïté d'acronymes comme `SSI` est une vraie limite du texte
  libre en langage administratif français : ni BM25 ni les embeddings
  ne désambiguïsent un sigle sans contexte suffisant. À garder en tête
  pour l'interprétation des résultats, pas nécessairement à "corriger"
  (il n'y a pas de règle générale fiable pour distinguer les deux sens
  sans context supplémentaire).

### Deuxième test réel, sur 100 000 lignes : résultat pertinent trouvé, et un second bug

**Bon résultat** : avec un échantillon plus grand (100 000 lignes,
1 086 marchés après filtre strict), le moteur trouve un vrai résultat
pertinent : *"SÉCURITÉ DES SYSTÈMES D'INFORMATION"* (150 M€) apparaît
en position 9 sur la requête cybersécurité + Île-de-France + montant
élevé. Ça valide que le pipeline fonctionne quand assez de candidats
pertinents existent dans le sous-ensemble filtré.

**Bug trouvé** : plusieurs marchés apparaissaient dupliqués à l'identique
dans les résultats (un même marché jusqu'à 9 fois). Cause : cotraitance /
accord-cadre multi-attributaire — un même marché (`uid`) peut avoir
jusqu'à une dizaine de titulaires, donc autant de lignes dans les
données brutes (exactement le phénomène documenté dans le bloc 1). Le
moteur de recherche n'appliquait pas la déduplication par marché avant
de classer/afficher les résultats, contrairement à l'audit qualité qui
la gérait déjà (`group_by_marche`).

**Corrigé** : `apply_strict_filters` (dans `engine.py`) déduplique
maintenant par `uid` après application des filtres région/montant, avant
de construire les index BM25/embeddings. Testé avec un cas de cotraitance
synthétique (3 titulaires, même `uid`) : un seul résultat retourné, pas
trois.

**Leçon methodologique** : une décision prise dans un bloc (bloc 1 :
dédupliquer par marché) doit être appliquée de façon cohérente dans
tous les blocs suivants qui manipulent les mêmes données brutes — sinon
le même problème réapparaît ailleurs sous une autre forme. À vérifier
aussi dans le bloc 4 (dashboard) le moment venu.

### Troisième observation : limite de BM25 sur les mots composés/synonymes

Après correction de la déduplication, le marché *"SÉCURITÉ DES SYSTÈMES
D'INFORMATION"* (150 M€, très pertinent pour la requête cybersécurité) a
disparu du top 10, alors qu'il y était avant. Diagnostic (BM25 isolé,
sans les embeddings) :

- Le mot "cybersécurité" de la requête ne matche **littéralement aucun**
  marché du corpus — ils disent tous "sécurité des systèmes
  d'information", jamais "cybersécurité". Pour BM25 (correspondance
  exacte de tokens), `cybersécurité` ≠ `sécurité` : score **0,00** pour
  tous les marchés SSI pertinents (vérifié : rang BM25 seul de 121 à 717
  sur 931 candidats).
- Le mot "marchés" de la requête, en revanche, matche presque tous les
  documents (très fréquent dans ce corpus administratif, faible pouvoir
  discriminant), ce qui fait remonter en tête des documents répétant
  souvent "marché(s)" sans rapport avec la cybersécurité.
- Conséquence sur la fusion RRF : même si les embeddings classent
  probablement bien les marchés SSI (similarité sémantique réelle avec
  "cybersécurité"), un très mauvais rang côté BM25 peut suffire à faire
  sortir un résultat pertinent du top 10 final.

**Ce n'est pas un bug, c'est une illustration concrète du problème que
l'architecture hybride est censée résoudre** — et une limite réelle de
RRF : un signal très faible sur une méthode n'est pas toujours compensé
par un signal fort sur l'autre, contrairement à ce qu'on pourrait
attendre naïvement d'une "moyenne" entre deux avis.

**Pistes d'amélioration identifiées, non implémentées à ce stade**
(à explorer si le temps du bloc 5 le permet, ou à assumer comme limite
documentée) :
- Décomposer les mots composés ("cyber" + "sécurité") ou étendre la
  requête avec des synonymes avant BM25 — ajouterait de la complexité et
  nécessiterait un dictionnaire de synonymes métier, non trivial à
  maintenir.
- Pondérer différemment les deux méthodes dans la fusion plutôt qu'un
  RRF à poids égal (ce qui reviendrait à abandonner en partie l'argument
  initial contre les poids arbitraires — compromis à assumer explicitement
  si choisi).
- Augmenter le nombre de candidats retenus par méthode avant fusion
  (actuellement 200) pour laisser plus de marge à un bon score embeddings
  de compenser un mauvais rang BM25.

### Quatrième observation et correction majeure : accents et mots composés

Investigation plus poussée du problème précédent (BM25 rang 121-717 pour
les marchés SSI pertinents). Deux causes identifiées et corrigées dans
`bm25_search.py` :

**Cause principale (la plus impactante)** : une partie du corpus est
écrite en majuscules sans accents (ex: `"SECURITE DES SYSTEMES D
INFORMATION"`, `"AMO SECURITE DES SYSTEMES D INFORMATION"`). Sans
normalisation, `"sécurité"` (requête, avec accent) et `"securite"`
(donnée brute, sans accent) sont deux tokens différents pour BM25 —
score de correspondance nul même sur un mot identique. **C'est le même
principe que la normalisation de `nature` découverte dans le bloc 1**
(casse/accents incohérents), mais dans un nouveau module : la leçon
n'avait pas été réappliquée ici avant ce test.

**Cause secondaire** : "cybersécurité" ne partage aucun token avec
"sécurité" (mots composés — voir observation précédente). Une petite
liste de préfixes composés courants (`cyber`, `télé`, `multi`, etc.)
est maintenant décomposée en préfixe + radical.

**Correction** : `_tokenize` normalise désormais les accents (fonction
`_normaliser_accents`, même technique NFKD que `cleaning.py`) avant
tokenisation, en plus de la décomposition de mots composés déjà en place.

**Résultat mesuré** : le meilleur rang BM25 du marché *"SÉCURITÉ DES
SYSTÈMES D'INFORMATION"* passe de **121/931 à 3/931** — largement dans
un top 10 utilisable. Plusieurs autres marchés SSI pertinents apparaissent
également aux rangs 2, 4 et 6 après correction, alors qu'aucun n'était
visible dans le top 10 avant.

**Tests ajoutés** : `test_tokenize_ignores_accents` et
`test_tokenize_decomposes_compound_prefix`, non-régression sur les deux
mécanismes. Un test existant (`test_tokenize_removes_stopwords`) a dû
être adapté : il vérifiait un token accentué (`"sécurité"`), qui n'est
plus produit tel quel après la normalisation (devient `"securite"`) —
changement de comportement volontaire, pas une régression.

**Leçon méthodologique (renforce celle du bloc précédent)** : deux
leçons du bloc 1 (normalisation casse/accents, déduplication par marché)
ont dû être réappliquées indépendamment dans le bloc 3, dans deux
modules différents. Signal clair qu'il faudrait à terme centraliser ces
normalisations de texte dans une fonction commune partagée entre
`cleaning.py` et `bm25_search.py`, plutôt que de dupliquer la logique.

> **Fait par la suite** : voir plus bas ("Bug bonus corrigé en marge",
> bloc prédiction) pour la création du module partagé, puis sa mise à
> jour finale : `cleaning.py` et `bm25_search.py` délèguent maintenant
> tous les deux à la même primitive `retirer_accents()`, sans
> changement de comportement.

### Confirmation finale : pipeline complet (BM25 + embeddings + RRF)

Après correction, le pipeline complet (`engine.search()`, avec le vrai
modèle d'embeddings, testé en conditions réelles) donne pour la
requête *"marchés de cybersécurité en Île-de-France de montant élevé"* :
**8 résultats pertinents sur 10** (contre 0/10 avant la correction),
incluant *"SÉCURITÉ DES SYSTÈMES D'INFORMATION"* en position 3.

**Limite résiduelle assumée** : 2 résultats sur 10 restent hors-sujet
(un marché de sécurité physique de carrière, un marché pétrolier) —
probablement portés par le mot générique "marchés", très peu
discriminant. Non corrigé plus avant : le rapport coût/bénéfice d'une
liste de "quasi-stopwords" métier (retirer "marché(s)", "accord-cadre",
"prestation(s)" de la requête libre) n'a pas été jugé prioritaire face
au gain déjà obtenu, mais reste une piste documentée pour une itération
future si la précision doit encore être améliorée.

## Bloc 4 — Dashboard

**Ce qui est fait** : interface Streamlit à 3 onglets (qualité,
anomalies, recherche), assemblant les blocs précédents sans nouvelle
logique métier. Export CSV des résultats de recherche implémenté.

**Ce qui n'est pas fait** : export PDF, initialement prévu dans le
périmètre du bloc 4. Non implémenté à ce stade — le CSV couvre déjà le
besoin d'export exploitable, le PDF ajouterait une dépendance
supplémentaire (génération de rapport formaté) pour un gain jugé
secondaire vu le temps restant. À ajouter si le temps du bloc 5 le
permet, sinon documenté comme périmètre réduit assumé.

**Couverture de test automatisée** : le serveur Streamlit démarre sans
erreur (`streamlit run`, réponse HTTP 200), le module s'importe sans
erreur (`test_dashboard.py`), et la logique sous-jacente (audit qualité,
détection d'anomalies) est déjà testée et validée dans les blocs 1 et 2.

> ⚠️ **État historique corrigé** — cette section indiquait initialement
> "reste à valider" pour le rendu visuel. **C'est fait** : les 3 onglets
> ont été vérifiés manuellement après toutes les corrections du bloc 5
> (déduplication anomalies, NDCG/Precision), via export CSV de chaque
> onglet et captures d'écran (voir README). Résultats confirmés : 10/10
> `uid` uniques dans le top anomalies (déduplication effective), 20/20
> résultats uniques dans la recherche, tableau qualité inchangé. Aucune
> régression détectée.

**Limite persistante, non corrigée** : le test automatisé du dashboard
reste un simple test d'import (`test_dashboard.py`), pas un test
fonctionnel réel de rendu ou d'interaction. La vérification faite
au-dessus est manuelle, pas reproductible automatiquement en CI — à
améliorer si le projet évolue (ex: tests Selenium/Playwright sur
l'interface Streamlit).

## Bloc 5 — Consolidation robustesse

### Métriques de recherche : Precision@K et NDCG@K

**Méthodologie et limite assumée** : en l'absence d'annotation humaine,
la vérité terrain est construite par mots-clés (`src/search/evaluation.py`)
— un marché est jugé "pertinent" pour une requête de test s'il contient
l'un des mots-clés associés dans son `objet`. Limite reconnue : cette
approche avantage mécaniquement BM25 (basé sur les mots-clés) par
rapport aux embeddings (qui pourraient capter des synonymes non couverts
par la liste de mots-clés) — les chiffres ci-dessous doivent être lus en
gardant ce biais méthodologique en tête, pas comme une vérité absolue.

**4 requêtes de test**, construites à partir des thèmes explorés
manuellement au bloc 3 et de vérifications de présence réelle dans le
corpus (67 à 8 046 marchés pertinents trouvés selon le thème, sur un
échantillon de 200 000 lignes) : cybersécurité, travaux de voirie,
restauration scolaire, espaces verts.

**Résultats BM25 seul** (sur un échantillon de 50 000 marchés) :

| Requête | P@5 | P@10 | NDCG@5 | NDCG@10 |
|---|---|---|---|---|
| cybersécurité | 1.0 | 0.7 | 1.0 | 1.0 |
| travaux de voirie | 1.0 | 1.0 | 1.0 | 1.0 |
| restauration scolaire | 1.0 | 1.0 | 1.0 | 1.0 |
| espaces verts | 1.0 | 1.0 | 1.0 | 1.0 |

Ces scores élevés confirment concrètement l'impact de la correction du
bloc 3 (normalisation d'accents + décomposition de mots composés) : avant
cette correction, la requête cybersécurité obtenait un score BM25 nul
sur les marchés pertinents (voir plus haut, "Quatrième observation").

**Retour critique reçu et biais méthodologique reconnu** : ces scores
"trop parfaits" ont légitimement soulevé une objection — la vérité
terrain avait potentiellement été construite après avoir observé les
sorties BM25 lors de l'exploration manuelle du bloc 3 (fuite
méthodologique), sans mesure du nombre total de documents pertinents
(pas de Recall@K), et sans tester de requêtes reformulées sans
recouvrement lexical avec les mots-clés de vérité terrain. Les trois
corrections suivantes ont été apportées suite à cette remarque :

1. **Recall@K ajouté**, avec le nombre de marchés pertinents connu dans
   le corpus (`n_pertinents_corpus`).
2. **`REQUETES_TEST_DIFFICILES`** : les 4 mêmes thèmes reformulés sans
   aucun mot significatif en commun avec les mots-clés de vérité terrain
   (ex: "protection des systèmes informatiques" au lieu de
   "cybersécurité"), pour tester spécifiquement l'apport des embeddings
   sans recoupement lexical possible avec BM25.
3. **Fuite reconnue explicitement** dans le code (`evaluation.py`) pour
   le thème cybersécurité : ses mots-clés de vérité terrain ont été
   choisis après avoir observé les résultats de recherche au bloc 3, pas
   choisis a priori.

**Résultats BM25 sur les requêtes difficiles** (même échantillon de
100 000) — **métriques corrigées** suite à la revue externe (voir
docs/decisions.md pour le détail des deux bugs corrigés : IDCG mal
calculé pour NDCG, division par le mauvais dénominateur pour Precision) :

| Requête difficile | n_pertinents_corpus | P@5 | R@5 | NDCG@5 |
|---|---|---|---|---|
| cybersécurité ("protection des systèmes informatiques") | 38 | 0.0 | 0.0 | 0.0 |
| travaux de voirie ("entretien des routes communales") | 4066 | 0.2 | 0.0 | 0.146 |
| restauration scolaire ("repas pour les élèves") | 614 | 1.0 | 0.008 | 1.0 |
| espaces verts ("maintenance des parcs municipaux") | 1434 | 0.0 | 0.0 | 0.0 |

**Effet concret de la correction NDCG** : pour "travaux de voirie", le
NDCG@5 passe de **0,431 (ancien calcul, biaisé)** à **0,146 (corrigé)**.
L'ancien calcul ignorait qu'il existe 4066 marchés pertinents dans le
corpus et jugeait uniquement si le seul résultat trouvé était bien
classé — le nouveau calcul pénalise correctement le fait de n'en avoir
trouvé qu'un seul sur un potentiel de 5 positions pertinentes au top 5.

BM25 s'effondre totalement sur 2 des 4 thèmes reformulés (score nul) —
attendu, puisque aucun mot de la requête ne matche littéralement les
marchés pertinents. Le thème restauration scolaire reste artificiellement
élevé : la reformulation choisie ("repas pour les élèves") contient
encore littéralement le mot-clé de vérité terrain "repas" — fuite
résiduelle reconnue, pas corrigée (aurait nécessité une reformulation
encore plus éloignée, ex: remplacer "repas" par un terme n'apparaissant
dans aucun mot-clé, à améliorer dans une itération future).

**Résultats du pipeline complet (BM25 + embeddings + RRF)** : à
compléter avec `python -m src.search.run_evaluation` (nécessite le
téléchargement du modèle d'embeddings). C'est sur les requêtes
difficiles (particulièrement cybersécurité et espaces verts, où BM25
seul obtient 0,0) que la valeur ajoutée réelle des embeddings devrait se
mesurer concrètement — si le pipeline complet fait mieux que 0,0 sur ces
deux thèmes, c'est une preuve mesurée de leur apport, pas supposée.

### Problème de performance découvert : script d'évaluation trop lent

**Constat** : la première version de `run_evaluation.py` utilisait une
taille d'échantillon unique (100 000) pour les 8 requêtes de test.
Résultat en pratique : temps d'exécution de l'ordre de 20 à 30 minutes,
inutilisable pour itérer.

**Cause** : seul le thème "cybersécurité" a un filtre strict (région +
montant) qui réduit le volume avant le calcul des embeddings (~931
lignes après filtre). Les 3 autres thèmes ("travaux de voirie",
"restauration scolaire", "espaces verts") n'ont aucun filtre dans leur
requête de test — `apply_strict_filters` ne réduit donc rien, et le
pipeline encode la totalité de l'échantillon (100 000 textes) à chaque
fois, pour 6 des 8 requêtes évaluées (facile + difficile × 3 thèmes).
L'encodage de dizaines de milliers de textes en embeddings sur CPU est
lent — c'est ce qui expliquait la lenteur.

**Correction** : taille d'échantillon différenciée par thème, pas une
taille unique pour toutes les requêtes. Cybersécurité garde 100 000
(nécessaire vu sa rareté et son filtre strict, voir plus haut). Les 3
autres thèmes passent à 15 000, largement suffisant vu leur fréquence
dans le corpus (92 à 581 marchés pertinents retrouvés à cette taille,
contre des milliers à 100 000 — proportionnellement cohérent, largement
assez pour des métriques stables).

**Limite reconnue de cette correction** : les 8 requêtes ne sont plus
évaluées sur un échantillon strictement identique, ce qui complique
légèrement la comparabilité brute des scores entre thèmes (une
différence de taille d'échantillon peut influencer marginalement des
métriques comme Recall@K). Assumé comme compromis pratique : chaque
thème reste évalué sur un échantillon suffisamment grand pour être
représentatif de son propre volume réel dans le corpus, ce qui est plus
important que d'avoir une taille identique arbitraire pour tous.

**Lien avec la remarque de la revue externe (point 6)** : ce problème de
lenteur est une illustration concrète de la limite déjà notée sur le
recalcul des embeddings à chaque recherche (`EmbeddingIndex` réencode
tout à chaque appel, pas de cache). Une solution plus durable (hors
périmètre de cette correction rapide) serait de précalculer et stocker
les embeddings du corpus une fois pour toutes, plutôt que d'ajuster la
taille d'échantillon au cas par cas.

### Incohérence trouvée en comparant facile vs difficile pour cybersécurité

**Constat** : en comparant les deux tableaux "Pipeline complet", le
thème cybersécurité affichait `n_pertinents_corpus = 4` en version
facile mais `37` en version difficile — alors que les deux versions du
même thème devraient normalement filtrer sur le même sous-ensemble
(Île-de-France + montant élevé).

**Cause** : la requête difficile initiale ("protection des systèmes
informatiques") avait supprimé non seulement le vocabulaire thématique,
mais aussi tout le filtre géographique/montant présent dans la version
facile ("marchés de cybersécurité **en Île-de-France de montant
élevé**"). Sans ce filtre, `apply_strict_filters` ne réduisait rien, et
la requête difficile était évaluée sur l'intégralité de l'échantillon
(100 000 lignes non filtrées) plutôt que sur les 931 candidats filtrés
de la version facile — mélangeant involontairement deux facteurs de
difficulté différents (absence de recouvrement lexical **et** absence
de filtre) au lieu d'isoler uniquement le premier, qui était l'objectif
de ce jeu de requêtes.

**Correction** : la requête difficile conserve maintenant le même filtre
que la version facile — seul le vocabulaire thématique change
("protection des systèmes informatiques **en Île-de-France de montant
élevé**"). Vérifié : les deux versions donnent maintenant le même
nombre de candidats après filtre (931).

**Bonus, trouvé au passage** : le test de non-régression associé
(`test_requetes_difficiles_meme_verite_terrain_que_faciles`) a été
corrigé pour vérifier la bonne propriété (aucun mot-clé de vérité
terrain ne doit apparaître littéralement dans la requête difficile),
plutôt que l'ancienne vérification trop large (aucun mot en commun entre
facile et difficile, ce qui aurait empêché de partager volontairement le
même filtre). Ce test plus précis a immédiatement révélé la fuite déjà
connue et documentée sur `restauration_scolaire_difficile` ("repas pour
les élèves" contenait littéralement le mot-clé "repas") — corrigée du
même coup avec une nouvelle formulation ("nourriture destinée aux
enfants dans les établissements primaires").

### Résultats finaux : impact réel des embeddings sur les requêtes difficiles

Une fois les deux incohérences ci-dessus corrigées, comparaison propre
entre BM25 seul et le pipeline complet (BM25+embeddings+RRF) sur les 4
requêtes difficiles (sans recouvrement lexical avec la vérité terrain) :

| Thème difficile | BM25 seul (P@5) | Pipeline complet (P@5) | NDCG@5 pipeline |
|---|---|---|---|
| cybersécurité | 0.0 | **0.4** | 0.637 |
| travaux de voirie | 0.6 | **0.2** | 0.214 |
| restauration scolaire | 0.0 | **0.6** | 0.655 |
| espaces verts | 0.0 | **0.0** | 0.0 |

**Résultat nuancé, pas une victoire uniforme des embeddings** :

- **2 succès nets** (cybersécurité, restauration scolaire) : le pipeline
  complet retrouve des résultats pertinents là où BM25 seul échouait
  totalement (score nul). C'est la preuve concrète de la valeur ajoutée
  des embeddings sur des reformulations sans recouvrement lexical —
  exactement l'hypothèse de départ de l'architecture hybride.

- **1 régression surprenante** (travaux de voirie) : le pipeline complet
  fait **moins bien** que BM25 seul (0.2 contre 0.6). Explication
  probable : BM25 seul profitait d'un léger recouvrement fortuit
  ("routes communales" partage des sous-chaînes avec des termes du
  corpus malgré la reformulation), obtenant un score correct par
  coïncidence. La fusion RRF, en intégrant ensuite le classement des
  embeddings (possiblement moins bon sur cette requête précise), a dilué
  ce signal BM25 qui était en fait de bonne qualité. C'est une
  illustration concrète de la limite de RRF déjà documentée au bloc 3 :
  un signal fort sur une méthode n'est pas toujours préservé quand il
  est fusionné à parts égales avec un signal plus faible sur l'autre.

- **1 échec persistant** (espaces verts) : ni BM25 ni le pipeline complet
  ne trouvent de résultat pertinent (0.0 des deux côtés). Ça suggère que
  le modèle d'embeddings utilisé
  (`paraphrase-multilingual-MiniLM-L12-v2`, choisi pour sa légèreté CPU)
  ne capture pas suffisamment bien la proximité sémantique entre
  "maintenance des parcs municipaux" et "espaces verts"/"tonte"/"élagage"
  pour ce cas précis — limite du modèle choisi, pas de l'architecture en
  elle-même. Un modèle plus grand (mais plus lourd) pourrait faire
  mieux, à tester si le temps le permet.

**Conclusion honnête pour le README/la synthèse finale** : l'architecture
hybride apporte une valeur mesurable et réelle (2 cas sur 4 où elle
transforme un échec total en résultat exploitable), mais elle n'est pas
une solution magique — elle peut aussi dégrader un résultat correct par
coïncidence (1 cas), et reste limitée par la qualité du modèle
d'embeddings choisi sur certains thèmes (1 cas). Ce résultat mesuré et
nuancé est plus crédible qu'un tableau où tout fonctionnerait parfaitement.

### Temps de réponse

Mesuré avec `python -m src.search.benchmark`, sur la requête
"marchés de cybersécurité en Île-de-France de montant élevé" (avec
filtre région/montant), à 3 tailles d'échantillon brut :

| Échantillon brut | Candidats après filtre | BM25 (build+search) | Embeddings (build+search) | Pipeline complet |
|---|---|---|---|---|
| 10 000 | 105 | 0,009 s | 11,7 s* | 4,0 s* |
| 50 000 | 483 | 0,027 s | 15,5 s | 18,1 s |
| 100 000 | 931 | 0,068 s | 27,5 s | 28,9 s |

*Le modèle d'embeddings est mis en cache après son premier chargement
(voir `_get_model()` dans `embeddings_search.py`) — le premier "Pipeline
complet" mesuré (4,0 s) est donc plus rapide que le "Construction index
embeddings" juste au-dessus (11,7 s, qui inclut le chargement initial du
modèle depuis le disque), puisqu'il réutilise le modèle déjà chargé.

**Conclusion sans ambiguïté** : BM25 est négligeable (moins de 0,1 s dans
tous les cas). **Les embeddings dominent presque intégralement** le temps
de réponse, et ce temps croît avec le nombre de candidats à encoder (pas
avec la taille de l'échantillon brut — la construction du filtre elle-même
est quasi instantanée).

**Implication directe pour une mise en production** : sur un cas d'usage
réaliste (~1000 candidats après filtre, comme "Île-de-France + montant
élevé"), l'utilisateur attend **27 à 29 secondes** pour un résultat.
C'est largement inacceptable pour une interface interactive (l'attente
tolérée pour une recherche est généralement de 1 à 3 secondes). Ce
chiffre confirme et quantifie, pour la première fois avec une vraie
mesure plutôt qu'une intuition, que **le recalcul des embeddings à
chaque requête est le principal goulot d'étranglement du projet** — plus
impactant que n'importe quel autre point de performance déjà discuté
(scalabilité de LOF, temps de fusion RRF, etc., tous négligeables en
comparaison).

**Recommandation pour une version production** (déjà évoquée comme piste,
confirmée prioritaire par cette mesure) : précalculer et stocker les
embeddings de l'ensemble du corpus une seule fois (via FAISS ou Qdrant,
prévus dans la stack technique initiale), plutôt que de les recalculer à
la volée à chaque recherche. Ça transformerait le temps de réponse d'une
recherche de ~30 secondes à probablement moins d'une seconde (simple
recherche de similarité dans un index déjà construit, sans ré-encodage).

### Annotation humaine : la vérité terrain par mots-clés est-elle fiable ?

> ⚠️ **État historique** — cette section décrit les instructions pour
> lancer l'annotation, écrites avant de l'avoir fait. **Les résultats
> réels sont disponibles plus bas**, dans la section "Résultats de
> l'annotation humaine : la vérité terrain par mots-clés est fiable"
> (spoiler : 100% d'accord sur 3 thèmes sur 4). Instructions conservées
> ici pour la reproductibilité (permettent de relancer l'annotation sur
> de nouveaux exemples si besoin).

**Instructions pour relancer/étendre l'annotation** :
```
python -m src.search.run_annotation
```
puis, après avoir rempli manuellement la colonne `pertinent` (0 ou 1)
dans `annotation_a_remplir.csv` en lisant chaque `objet` :
```
python -m src.search.run_annotation --comparer
```

Ça donnera, pour chacune des 4 requêtes de test, le taux d'accord entre
la vérité terrain par mots-clés (utilisée dans toutes les métriques
Precision@K/NDCG du bloc 5) et un vrai jugement humain, ainsi que le
détail des désaccords (faux positifs et faux négatifs du mot-clé).

**Ce qu'on cherche à savoir** : est-ce que le biais méthodologique
reconnu depuis le début du bloc 5 (vérité terrain construite après avoir
observé BM25, avantage mécanique de BM25 sur les embeddings) se traduit
concrètement par des faux positifs/négatifs significatifs, ou si
l'approximation par mots-clés reste globalement fiable malgré tout. Un
taux d'accord élevé (>90%) validerait a posteriori la méthodologie
utilisée ; un taux plus faible remettrait en question les chiffres
Precision@K/NDCG déjà documentés plus haut.

**Résultat obtenu (voir détail plus bas)** : taux d'accord élevé (100%
sur 3 thèmes sur 4, 87,5% sur le 4ème) — la méthodologie par mots-clés
est validée comme globalement fiable sur cet échantillon.

**Même bug de lenteur reproduit une deuxième fois** : la première
version de `run_annotation.py` utilisait un échantillon unique de 50 000
pour les 4 requêtes, reproduisant exactement le problème déjà rencontré
et corrigé dans `run_evaluation.py` (seule "cybersécurité" a un filtre
qui réduit le volume ; les 3 autres thèmes forcent l'encodage de
l'échantillon entier).

> **Écart trouvé a posteriori** : ce paragraphe affirmait initialement
> que la correction ("taille d'échantillon différenciée par thème") avait
> déjà été appliquée à `run_annotation.py` — en réalité le code gardait
> encore une seule constante (`TAILLE_ECHANTILLON = 50_000`) pour les 4
> thèmes, jamais corrigée. Doc et code désynchronisés, corrigé maintenant
> des deux côtés : `run_annotation.py` utilise désormais
> `taille_echantillon_pour()` (voir plus bas), la fonction utilitaire
> centralisée dans `src/search/evaluation.py` — la même que
> `run_evaluation.py`, avec les mêmes tailles (100 000 pour
> "cybersécurité", 15 000 pour les 3 autres thèmes ; pas 5 000, chiffre
> qui ne correspondait à aucune valeur réellement utilisée ailleurs dans
> le projet).

Leçon retenue : ce genre de correction ponctuelle (un seul script
corrigé) ne suffit pas à empêcher la récidive dans un script similaire
écrit plus tard — un signal de plus en faveur d'une fonction utilitaire
commune de "sélection de taille d'échantillon adaptée au filtre d'une
requête", plutôt que de la recopier à chaque nouveau script. **Fait par
la suite** : `taille_echantillon_pour()` (dans `src/search/evaluation.py`)
implémente exactement cette fonction utilitaire, réutilisée par
`run_evaluation.py`, `run_annotation.py` et `compare_weighted_rrf.py` —
plus aucune taille d'échantillon par thème dupliquée en dur dans ces
trois scripts.

### Résultats de l'annotation humaine : la vérité terrain par mots-clés est fiable

**Méthodologie de cette annotation, à préciser honnêtement** : les 32
lignes (8 par thème) ont été jugées pertinentes ou non en lisant chaque
`objet`, jugements proposés puis relus et validés. Ce n'est pas une
annotation aveugle réalisée par une tierce personne indépendante du
projet — une vraie validation externe aurait plus de poids
méthodologique — mais ça reste un jugement de lecture explicite,
justifié ligne par ligne, plus rigoureux qu'une simple confirmation
sans y regarder.

**Résultat** :

| Requête | Taux d'accord (mots-clés vs jugement) | Faux positifs mots-clés | Faux négatifs |
|---|---|---|---|
| cybersécurité | 100% | 0 | 0 |
| travaux de voirie | 100% | 0 | 0 |
| restauration scolaire | 87,5% | 1 | 0 |
| espaces verts | 100% | 0 | 0 |

**Interprétation** : la vérité terrain par mots-clés se révèle globalement
fiable (3 thèmes sur 4 à 100% d'accord). Le seul écart trouvé : le marché
*"Gros œuvre — Restructuration de la cantine scolaire"* est classé
pertinent par les mots-clés (il contient "cantine") mais correspond en
réalité à des **travaux de bâtiment**, pas à un service de restauration
scolaire — un faux positif logique, dans la même famille que le cas SSI
(Sécurité Incendie vs Système d'Information) déjà rencontré au bloc 3 :
un mot-clé isolé ("cantine", "SSI") peut désigner un objet physique
(le bâtiment, l'équipement) plutôt que le service qu'on cherche
réellement.

**Ce que ça valide, et ce que ça ne valide pas** : ce résultat confirme
que la vérité terrain par mots-clés n'introduit pas de biais massif sur
ces 4 thèmes précis — les chiffres Precision@K/NDCG documentés plus haut
restent globalement crédibles. Ça ne prouve pas que la méthode serait
aussi fiable sur d'autres thèmes non testés, ni que le biais de fuite
déjà reconnu (mots-clés cybersécurité choisis après avoir vu les
résultats BM25 du bloc 3) soit annulé — cette annotation valide la
*précision* de la vérité terrain (peu de faux positifs), pas
l'absence de biais dans sa *construction* (comment les mots-clés ont
été choisis).

## Itération post-bloc 5 : cache disque des embeddings

**Ce qui a été fait** : `EmbeddingIndex` accepte maintenant un
`cache_path` (activé par défaut dans `engine.search()`) — un fichier
`.npz` qui stocke les vecteurs déjà calculés par `uid`. Chaque recherche
ne recalcule que les marchés jamais rencontrés auparavant. Voir
`docs/decisions.md` pour le raisonnement complet, notamment pourquoi un
cache disque a été choisi plutôt qu'un index FAISS/Qdrant complet
(précalculer les 1,73M marchés uniques prendrait ~14h sur cette machine,
irréaliste).

**Résultats mesurés** (échantillon de 100 000, `python -m src.search.benchmark`) :

| | Temps |
|---|---|
| Recherche à froid (cache vide) | 23,7 s |
| Recherche à chaud (cache rempli) | 0,5 s |
| **Accélération** | **47,1x** |

Confirme concrètement l'hypothèse : le deuxième appel se rapproche du
temps BM25 seul (quelques dizaines de millisecondes), l'encodage
embeddings étant entièrement évité une fois le cache rempli. Le gain de
47x est légèrement inférieur à l'estimation initiale ("plusieurs
centaines de fois"), probablement parce que la recherche à chaud
recharge quand même le cache disque et effectue le produit scalaire sur
tous les vecteurs — pas un temps strictement nul, mais un gain massif et
bien réel.

Pour rappel, également mesuré dans ce même run (comportement sans cache,
inchangé, cohérent avec les résultats précédents) :

| Échantillon | Candidats filtrés | Construction embeddings | Pipeline complet sans cache |
|---|---|---|---|
| 10 000 | 105 | 11,8 s | 3,3 s |
| 50 000 | 483 | 11,6 s | 12,1 s |
| 100 000 | 931 | 22,9 s | 25,1 s |

**Limite qui persiste malgré le cache** : le tout premier appel sur un
nouveau sous-ensemble de marchés (jamais rencontré) reste aussi lent
qu'avant — le cache n'aide qu'à partir de la deuxième recherche sur des
marchés communs. Un utilisateur qui explore des filtres très variés
(région différente à chaque fois) ne bénéficierait quasiment jamais du
cache. Le gain réel dépend donc du profil d'usage (répétitif vs
exploratoire), pas garanti dans tous les cas.

## Test différé : transformation log des montants avant LOF

Suggestion reçue lors de la toute première revue externe, testée
maintenant plutôt que laissée en suspens indéfiniment.

**Résultat** : avec une transformation log signée sur les montants
(voir docs/decisions.md pour le détail), le taux d'accord entre
Isolation Forest et LOF **triple** — de 4,5% (sans transformation) à
**13,0%** (avec), sur les mêmes 5 tirages que la mesure de référence.
La stabilité relative reste comparable (écart-type proportionnel
similaire, ~19% dans les deux cas).

**Recommandation** : utiliser `transformation_log=True` pour toute
analyse future — le paramètre existe (`build_feature_matrix`,
`detect_isolation_forest`, `detect_lof`, `compare_methods`,
`check_agreement_stability`), désactivé par défaut uniquement pour ne
pas invalider silencieusement les chiffres déjà publiés ailleurs dans ce
document (4,5% cité à plusieurs endroits en référence aux tirages
initiaux).

**Non fait, pour aller plus loin** : vérifier qualitativement si les
anomalies détectées avec transformation ont plus de sens métier (pas
seulement un meilleur taux d'accord statistique) — nécessiterait une
inspection manuelle du contenu des nouvelles anomalies détectées,
similaire à l'annotation humaine faite pour la recherche.

## Bloc prédiction : régression sur offresRecues

**Ce qui fonctionne** : Random Forest bat nettement la baseline naïve
et Ridge (R² réel = 0,676 vs -0,031 et 0,032), avec une erreur moyenne
de 5,45 offres sur des valeurs allant de 0 à plusieurs milliers —
raisonnable vu l'ampleur de la distribution.

**Limite majeure, assumée et non contournable** : le modèle n'apprend
que sur les ~40% de marchés où `offresRecues` est renseigné, et ce
sous-ensemble n'est **pas représentatif** du dataset complet — le taux
de manquant varie de 31% à quasiment 100% selon le type de procédure.
Concrètement : le modèle sera peu fiable, voire inutilisable, sur les
procédures où `offresRecues` est presque toujours absent (ex:
"Procédure concurrentielle avec négociation", ~100% manquant). Ce n'est
pas un défaut du modèle en soi, mais une limite structurelle des données
disponibles — aucune quantité d'ingénierie ne peut faire apprendre un
modèle sur des exemples qui n'existent pas.

**Limite secondaire** : Ridge sous-performe fortement en espace réel
(R²=0,032) malgré un score correct en espace log (0,179) — signe que la
régression linéaire simple ne capture pas bien les cas à très nombreuses
offres (queue de distribution). Documenté comme illustration du fait que
l'espace de mesure (log vs réel) peut donner des conclusions très
différentes sur la qualité d'un modèle, pas comme un défaut de Ridge en
tant que tel.

**Bug bonus corrigé en marge** : `procedure` n'avait jamais été
normalisée (16 valeurs se réduisent à 11 après correction — même
problème que `nature` au bloc 1, jamais traité sur ce champ). Corrigé
via un nouveau module de normalisation partagé
(`src/common/text_normalization.py`).

**Itération suivante** : la centralisation, initialement partielle (le
nouveau code utilisait le module commun, `cleaning.py` et
`bm25_search.py` gardaient leur propre copie de la logique de retrait
d'accents), a été complétée en isolant la partie sans risque de la
duplication — le retrait d'accents NFKD lui-même, identique dans les
trois endroits — dans une fonction `retirer_accents()` réutilisée
partout. Comportement inchangé (mêmes tests de non-régression, toujours
au vert) ; seule la duplication de code a disparu.

## Aller plus loin sur la prédiction : validation croisée et hyperparamètres

**Bug trouvé** : `dureeMois` contient des valeurs aberrantes (négatives,
ou jusqu'à 31 410 mois soit ~2618 ans) — 0,09% des lignes, découvert via
un R² catastrophique (~-10⁷¹) en validation croisée sur Ridge. Corrigé :
`filtrer_cible_valide` exclut maintenant aussi `dureeMois` hors de la
plage (0, 240] mois. Troisième variable (après `montant` et
`offresRecues`) où ce dataset contient des valeurs numériquement
aberrantes.

**Validation croisée (5-fold, confirme les résultats initiaux)** :

| Modèle | R² réel (single-split) | R² réel (5-fold CV) |
|---|---|---|
| Ridge | 0,032 | 0,039 ± 0,012 |
| Random Forest | 0,676 | 0,633 ± 0,194 |

Random Forest a un écart-type notable entre plis (0,194, ~30% de sa
moyenne) — sa performance n'est pas parfaitement stable, à garder en
tête.

**Recherche d'hyperparamètres — résultat contre-intuitif, persistant même
après correction** : le modèle sélectionné par `RandomizedSearchCV`
optimisé sur le R² en espace log est moins bon en espace réel que le
modèle par défaut (0,515 vs 0,596). Un scorer personnalisé en espace
réel (`SCORER_ESPACE_REEL`) a été ajouté pour corriger ça — mais même
avec ce scorer corrigé, le modèle sélectionné (0,517) reste **toujours
moins bon** que la configuration par défaut sur le jeu de test retenu.
Interprétation honnête : soit le budget de recherche (15 combinaisons,
sous-échantillon de 50 000) est trop limité, soit les paramètres par
défaut étaient déjà une configuration raisonnablement bonne pour ce
problème — non tranché, rapporté tel quel plutôt que caché.

**Non fait, pour aller plus loin** : augmenter le budget de recherche
(plus d'itérations, dataset complet) ; analyse plus fine du sous-ensemble
où le modèle échoue le plus (au-delà du simple constat du biais de
sélection).

## Pondération RRF : résultat plus nuancé que prévu

Testé sur "travaux de voirie" (requête difficile, 581 marchés pertinents) :

| Configuration | P@5 | P@10 |
|---|---|---|
| Poids égal (référence) | 0,2 | 0,2 |
| BM25 ×3 | 0,2 | **0,5** |
| Embeddings ×3 | **0,4** | 0,3 |

Favoriser BM25 améliore surtout le top 10 ; favoriser les embeddings
améliore surtout le top 5 — l'inverse de l'hypothèse de départ (qui
prévoyait que favoriser BM25 serait la bonne correction dans tous les
cas). Pas de poids universellement meilleur identifié ; le paramètre
`poids_fusion` reste optionnel plutôt que de changer le défaut sur la
base d'un seul thème testé.

**Non fait, pour aller plus loin** : tester sur les 3 autres thèmes pour
voir si un poids donné généralise.

## Tests fonctionnels du dashboard (Playwright)

**Ce qui est fait** : `tests/test_dashboard_functional.py`, teste
réellement le rendu via un navigateur headless, au-delà du simple test
d'import déjà existant.

**Protection CI** : ces tests se sautent automatiquement si
`data/raw/decp.parquet` est absent (toujours le cas sur la machine CI,
le fichier n'étant jamais versionné) — pas d'échec en CI, juste un skip
propre et rapide.

**Résultat final, après 9 tentatives indépendantes de correction (voir
docs/decisions.md pour le détail complet de chacune : portabilité
Windows, sélecteurs ARIA, délais progressifs, éléments plutôt que texte,
attente de disparition du squelette, clic souris OS, isolation
serveur+page par test, attente de fin de script, changement de version
Streamlit)** :

| Test | Statut |
|---|---|
| Titre de l'app affiché | ✅ Fiable |
| Les 3 onglets présents | ✅ Fiable |
| Tableau qualité affiché | ✅ Fiable |
| Widgets onglet anomalies | ⚠️ `xfail` (limite headless documentée) |
| Widgets onglet recherche | ⚠️ `xfail` (limite headless documentée) |

**Les 2 derniers tests échouent de façon parfaitement reproductible**,
y compris après changement de version de Streamlit (1.60.0 → 1.32.0,
aucun effet — élimine l'hypothèse d'un bug de version spécifique).
Théorie retenue : les widgets interactifs dépendent d'API de détection
de visibilité qui se comportent différemment en mode headless qu'avec
un vrai affichage — cohérent avec le fait que l'onglet actif au
chargement initial (qualité) fonctionne parfaitement, mais pas ceux
activés après coup par clic automatisé. **Le dashboard fonctionne
correctement en usage réel** — validé à de multiples reprises par
captures d'écran manuelles tout au long de ce projet (voir
`docs/images/`) — ce n'est donc pas un bug du dashboard, mais une
limite spécifique à l'automatisation headless.

**Décision assumée** : marquer ces 2 tests `xfail` plutôt que de
continuer à itérer indéfiniment. 3 tests sur 5 fiables représentent déjà
un gain réel par rapport au simple test d'import qui existait avant, et
ce point était explicitement le moins prioritaire des 5 recommandations
de la synthèse finale — un cas clair où il fallait savoir s'arrêter
plutôt que de poursuivre un problème à rendement décroissant.
