"""
Normalisation de texte centralisée.

Créé pour implémenter la recommandation n°2 de docs/synthese_finale.md :
la même leçon (les catégories textuelles brutes contiennent des variantes
de casse/accents/ponctuation qui devraient être une seule et même
catégorie) a été apprise indépendamment à 3 reprises dans ce projet :
- bloc 1 : normalisation de `nature` (cleaning.py)
- bloc 3 : normalisation d'accents pour BM25 (bm25_search.py)
- bloc prédiction : découverte que `procedure` a exactement le même
  problème, jamais corrigé (16 valeurs distinctes qui se réduisent à
  9 vraies catégories une fois normalisées — apostrophes différentes
  et accents manquants).

Plutôt que de dupliquer une 4e fois cette logique, elle est centralisée
ici et réutilisée partout où c'est pertinent.
"""

from __future__ import annotations

import unicodedata


def retirer_accents(texte: str) -> str:
    """Retire les accents d'un texte (NFKD : décompose "é" en "e" + accent,
    puis ne garde que les caractères ASCII).

    Primitive partagée par `normaliser_texte_categorie` ci-dessous, ainsi
    que par `src/quality/cleaning.py` et `src/search/bm25_search.py`, qui
    dupliquaient chacun cette même transformation (voir docs/decisions.md,
    "recommandation n°2" — centralisation restée partielle jusqu'ici).
    """
    return unicodedata.normalize("NFKD", texte).encode("ascii", errors="ignore").decode("ascii")


def normaliser_texte_categorie(texte: str) -> str:
    """Normalise une valeur catégorielle textuelle pour la comparaison/le
    regroupement : minuscules, sans accents, apostrophes unifiées, espaces
    superflus réduits.

    Ne change PAS la valeur affichée à l'utilisateur (voir les modules
    appelants pour la logique de conservation de l'original, ex:
    `nature_original` dans cleaning.py) — cette fonction sert uniquement
    à la comparaison/au regroupement.
    """
    if not isinstance(texte, str):
        return texte

    texte = texte.strip().lower()

    # Unifier les variantes d'apostrophe (' droite, ' typographique) et
    # l'absence d'apostrophe avant un espace isolé (ex: "d offres" vs
    # "d'offres") — sans ça, "Appel d'offres" et "Appel d offres" restent
    # deux catégories différentes malgré la normalisation d'accents.
    texte = texte.replace("’", "'")

    # Retirer les accents (voir retirer_accents ci-dessus).
    texte = retirer_accents(texte)

    # Après suppression des accents, l'apostrophe droite est déjà
    # préservée par l'encodage ascii (elle est déjà ASCII). Normaliser le
    # cas particulier "d offres" (espace) -> "d'offres" (apostrophe) :
    # remplace un espace isolé entre "d" et une voyelle par une apostrophe,
    # motif spécifique au français administratif de ce corpus.
    texte = texte.replace(" d ", " d'")

    # Réduire les espaces multiples résiduels
    texte = " ".join(texte.split())

    return texte
