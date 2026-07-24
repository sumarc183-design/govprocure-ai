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

## Pratiques transverses

### Un commit = un changement logique

**Décision** : séparer les commits par nature (code / tests / documentation),
plutôt que tout regrouper en un seul commit fourre-tout.

**Pourquoi** : permet de retrouver rapidement, via `git log`, quand et
pourquoi un changement précis a été fait — utile pour soi-même en cours de
projet, et pour quiconque relit l'historique plus tard.
