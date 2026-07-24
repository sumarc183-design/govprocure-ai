# Sources de données

## Données de la commande publique (open data)

- Source principale envisagée : data.economie.gouv.fr (données essentielles de la commande publique)
- À vérifier avant de figer l'architecture (semaine 0.5) :
  - granularité réelle (marché individuel vs agrégé) ;
  - couverture temporelle (nombre d'années disponibles, continuité) ;
  - complétude des champs clés (montant, durée, catégorie/CPV, localisation, procédure) ;
  - fréquence de mise à jour ;
  - format(s) disponibles (CSV, JSON, API).

## Notes d'audit

**Source retenue** : fichier consolidé Parquet, décp.info / data.gouv.fr
("données essentielles de la commande publique consolidées, format tabulaire").
Choisi plutôt que le fichier officiel `decp_augmente` du ministère (obsolète)
car plus exhaustif et directement exploitable en Parquet.

**Volumétrie** : 3 141 176 lignes, 58 colonnes (22 colonnes conservées pour l'analyse).

**Granularité réelle** : 1 ligne = (marché × titulaire), pas 1 ligne = 1 marché.
Un même `uid` peut apparaître plusieurs fois si plusieurs titulaires sont
associés à un même marché (cotraitance / groupement). Ce n'est **pas** un
doublon — vérifié en comparant `titulaire_id` sur les lignes partageant un uid.
Après regroupement (`group_by_marche`), on obtient environ 1,73M marchés
uniques (sur la version "actuelle" du dataset).

**Versioning** : le champ `donneesActuelles` (bool) distingue la version
actuelle d'un marché de ses versions historiques (modifications, via
`modification_id`). Filtrer sur `donneesActuelles == True` pour une vue
"état actuel".

**Résultats de l'audit qualité (`src/quality/audit.py`)** — voir aussi
[limitations.md](limitations.md) :

| Colonne | Score | % manquant | Problème principal |
|---|---|---|---|
| `offresRecues` | 0.40 | ~60% | Très forte proportion de valeurs manquantes |
| `montant` | 0.88 | ~2% | Valeurs négatives/nulles + valeurs extrêmes (>1 Md€) |
| `procedure` | 0.90 | ~5% | Variantes de casse à normaliser |
| `datePublicationDonnees` | 0.90 | ~5% | Quelques dates hors plage plausible |
| `dateNotification` | 0.94 | ~1% | Quelques dates hors plage plausible |
| `donneesActuelles` | 0.94 | ~1% | Valeurs nulles sur un champ censé être booléen |
| `nature` | 0.95 | <1% | Variantes de casse à normaliser (ex: "Marché"/"MARCHE"/"marché") |

Les autres colonnes conservées ont un score ≥ 0.96.

**Décisions prises suite à l'audit** :
- Ne jamais dédupliquer naïvement sur `uid` seul (risque de perdre des
  titulaires légitimes en cotraitance).
- Normaliser `nature` et `procedure` (casse, accents) avant tout regroupement.
- Traiter les montants négatifs/nuls et les montants > 1 Md€ comme des
  candidats de premier plan pour le bloc détection d'anomalies (bloc 2),
  plutôt que de simplement les exclure — à trancher selon le cas.
- `offresRecues` : décider si le taux de complétude (~40%) le rend
  exploitable tel quel ou seulement comme signal partiel.
