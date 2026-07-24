"""
Recherche par mots-clés (BM25) sur le champ `objet` des marchés.

BM25 : algorithme classique de recherche textuelle (utilisé par défaut
dans Elasticsearch, entre autres). Il score un document selon la
fréquence des mots de la requête qu'il contient, pondérée par la rareté
de ces mots dans l'ensemble du corpus (un mot rare qui matche compte plus
qu'un mot très fréquent comme "de" ou "et").

Limite connue : ne capture aucun synonyme ni reformulation — c'est
précisément la limite que les embeddings (embeddings_search.py) viennent
combler.
"""

from __future__ import annotations

import re
import unicodedata

import pandas as pd
from rank_bm25 import BM25Okapi

def _normaliser_accents(text: str) -> str:
    """Retire les accents (é -> e, à -> a, etc.).

    Nécessaire car une partie du corpus est écrite en majuscules sans
    accents (ex: "SECURITE DES SYSTEMES D INFORMATION"), pratique de
    saisie historique fréquente dans ce type de dataset administratif.
    Sans cette normalisation, "sécurité" (requête) et "securite" (donnée
    brute) sont deux tokens différents pour BM25 — trouvé concrètement
    en testant sur données réelles (voir docs/limitations.md). Même
    principe que la normalisation de `nature` dans le bloc 1
    (cleaning.py), réappliqué ici pour la même raison.
    """
    return unicodedata.normalize("NFKD", text).encode("ascii", errors="ignore").decode("ascii")


# Mots vides français basiques à ignorer (liste courte et pragmatique,
# pas une liste NLP exhaustive — suffisante pour améliorer la pertinence
# sans complexifier inutilement). Normalisés sans accent pour être
# comparés à du texte déjà passé par _normaliser_accents.
STOPWORDS_FR = {
    _normaliser_accents(w) for w in [
        "le", "la", "les", "un", "une", "des", "de", "du", "et", "ou", "à",
        "au", "aux", "en", "pour", "sur", "dans", "par", "avec", "ce", "ces",
        "sa", "son", "ses", "est", "sont",
    ]
}


# Préfixes composés courants en français administratif/technique, pour
# lesquels on sépare le préfixe du reste du mot (ex: "cybersécurité" ->
# "cyber" + "sécurité"). Sans ça, BM25 traite "cybersécurité" comme un
# token complètement différent de "sécurité", alors que sémantiquement
# ils partagent l'essentiel du sens — problème découvert concrètement en
# testant sur données réelles (voir docs/limitations.md, "Troisième
# observation"). Normalisés sans accent, même raison que STOPWORDS_FR.
# Limite assumée : liste manuelle non exhaustive, à enrichir si d'autres
# cas similaires sont découverts. Pas une solution générale de
# décomposition morphologique.
PREFIXES_COMPOSES = [
    _normaliser_accents(p) for p in [
        "cyber", "télé", "multi", "pluri", "auto", "socio", "géo", "bio",
        "éco", "micro", "macro", "inter", "intra", "pré", "post", "sous",
        "sur", "anti", "pro", "co",
    ]
]


def _decomposer_mots_composes(words: list[str]) -> list[str]:
    """Ajoute le préfixe et le radical séparément pour les mots composés
    reconnus, en plus du mot original (le mot original est conservé pour
    ne pas perdre un éventuel score sur le terme exact).
    """
    result = []
    for word in words:
        result.append(word)
        for prefixe in PREFIXES_COMPOSES:
            if word.startswith(prefixe) and len(word) > len(prefixe) + 3:
                radical = word[len(prefixe):]
                result.append(prefixe)
                result.append(radical)
                break  # un seul préfixe reconnu par mot, le premier qui matche
    return result


def _tokenize(text: str) -> list[str]:
    """Découpe un texte en mots simples, en minuscules, sans accents, sans
    mots vides.

    Découpage volontairement simple (regex sur les caractères alphanumériques)
    plutôt qu'un tokenizer linguistique complet — suffisant pour BM25 sur ce
    corpus, et beaucoup plus rapide à exécuter sur des millions de lignes.

    Les mots composés reconnus (ex: "cybersécurité") sont en plus décomposés
    en préfixe + radical, pour partager des tokens avec leurs équivalents
    non composés (ex: "sécurité").
    """
    text = _normaliser_accents(text.lower())
    words = re.findall(r"\b\w+\b", text)
    words = [w for w in words if w not in STOPWORDS_FR and len(w) > 1]
    return _decomposer_mots_composes(words)


class BM25Index:
    """Encapsule un index BM25 construit sur la colonne `objet` d'un dataframe.

    L'index doit être reconstruit si le dataframe change (pas de mise à
    jour incrémentale) — acceptable pour ce projet où la recherche
    s'effectue sur un sous-ensemble déjà filtré (voir engine.py), donc
    l'index est petit et rapide à construire à chaque requête.
    """

    def __init__(self, df: pd.DataFrame, text_column: str = "objet"):
        self.df = df.reset_index(drop=True)
        self.text_column = text_column
        corpus = self.df[text_column].fillna("").astype(str).tolist()
        self.tokenized_corpus = [_tokenize(doc) for doc in corpus]
        self.bm25 = BM25Okapi(self.tokenized_corpus)

    def search(self, query: str, top_k: int = 50) -> pd.DataFrame:
        """Retourne les top_k lignes les mieux scorées par BM25, avec leur score.

        Si la requête est vide (aucun texte libre après extraction des
        filtres), retourne un dataframe vide plutôt que planter ou renvoyer
        un classement arbitraire.
        """
        if not query.strip():
            return self.df.iloc[0:0].copy()

        tokenized_query = _tokenize(query)
        scores = self.bm25.get_scores(tokenized_query)

        result = self.df.copy()
        result["bm25_score"] = scores
        result = result.sort_values("bm25_score", ascending=False).head(top_k)
        return result
