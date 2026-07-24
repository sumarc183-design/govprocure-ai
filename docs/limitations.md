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

**Implication méthodologique** : les deux méthodes sont complémentaires,
pas redondantes. Un système de priorisation robuste devrait probablement
combiner les deux plutôt que choisir l'une ou l'autre.

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
- Tester sur le dataset complet (LOF est coûteux — à valider en temps
  de calcul sur 3M lignes, éventuellement échantillonner ou utiliser
  une structure d'index approximatif).
- Ajouter les métriques de robustesse prévues : stabilité selon les
  périodes, taux de faux positifs (nécessite un référentiel de
  validation, à construire).
- Décider d'une stratégie de combinaison des deux méthodes (union,
  score composite, ou présentation séparée à l'utilisateur final).

## Bloc 3 — Recherche hybride

_À compléter : Precision@K, Recall@K, NDCG, résistance aux fautes/variantes de requêtes._

## Biais et dérive connus

_À compléter au fur et à mesure des tests (Evidently, comparaison de périodes)._
