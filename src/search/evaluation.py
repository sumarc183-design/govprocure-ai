"""
Métriques d'évaluation du moteur de recherche : Precision@K et NDCG@K.

Méthodologie assumée et limite honnête : en l'absence d'annotation humaine
(coûteuse et hors périmètre raisonnable pour ce projet), la "vérité
terrain" est construite par mots-clés — un marché est considéré pertinent
pour une requête de test s'il contient l'un des mots-clés associés dans
son `objet`. C'est une approximation, pas une vérité terrain parfaite :
- Elle peut sous-estimer la pertinence (un marché pertinent mais formulé
  différemment des mots-clés choisis sera compté comme non pertinent).
- Elle peut légèrement biaiser en faveur de BM25 (qui matche aussi des
  mots-clés) par rapport aux embeddings (qui pourraient capter des
  synonymes non couverts par la liste de mots-clés).
Malgré cette limite, elle permet de mesurer objectivement si le pipeline
s'améliore ou se dégrade au fil des itérations — plus utile qu'aucune
mesure du tout.
"""

from __future__ import annotations

import math
import unicodedata
from dataclasses import dataclass

import pandas as pd


def _normaliser(text: str) -> str:
    """Même normalisation (minuscules, sans accents) que bm25_search.py,
    pour que la vérité terrain et le moteur de recherche jugent le texte
    de la même façon.
    """
    text = text.lower()
    return unicodedata.normalize("NFKD", text).encode("ascii", errors="ignore").decode("ascii")


def est_pertinent(objet: str, mots_cles: list[str]) -> bool:
    """Un marché est jugé pertinent si son objet contient au moins un des
    mots-clés (normalisés, insensibles à la casse/accents)."""
    objet_norm = _normaliser(str(objet))
    return any(_normaliser(mc) in objet_norm for mc in mots_cles)


def precision_at_k(objets_retournes: list[str], mots_cles: list[str], k: int) -> float:
    """Proportion de résultats pertinents parmi les k premiers retournés.

    Définition standard en recherche d'information : on divise toujours
    par k, pas par le nombre réel de résultats retournés. Si le moteur ne
    retourne que 2 résultats pour k=10, les 8 positions manquantes comptent
    comme non pertinentes (précision pénalisée), plutôt que d'être ignorées.

    Correction suite à une revue externe : la version précédente divisait
    par len(top_k) (le nombre réel de résultats), ce qui masquait
    silencieusement les cas où le moteur retourne trop peu de résultats —
    un moteur qui ne retourne qu'un seul résultat parfaitement pertinent
    obtenait Precision@10 = 1.0 au lieu de 0.1.
    """
    top_k = objets_retournes[:k]
    if k == 0:
        return 0.0
    n_pertinents = sum(1 for o in top_k if est_pertinent(o, mots_cles))
    return n_pertinents / k


def ndcg_at_k(
    objets_retournes: list[str],
    mots_cles: list[str],
    k: int,
    n_pertinents_corpus: int,
) -> float:
    """NDCG@K avec pertinence binaire (1 = pertinent, 0 = non pertinent).

    DCG : somme des pertinences pondérées par la position (1/log2(rang+1)),
    pour valoriser un résultat pertinent trouvé tôt plus qu'un résultat
    pertinent trouvé tard.

    IDCG : DCG du classement idéal, qui doit tenir compte du nombre TOTAL
    de documents pertinents existant dans le corpus (n_pertinents_corpus),
    pas seulement de ceux effectivement trouvés dans le top k. Le
    classement idéal contient min(n_pertinents_corpus, k) résultats
    pertinents en tête.

    n_pertinents_corpus est donc un paramètre obligatoire, volontairement
    sans valeur par défaut : NDCG ne peut pas être calculé correctement
    sans connaître ce nombre.

    Correction suite à une revue externe : la version précédente calculait
    l'IDCG à partir du nombre de pertinents *trouvés* dans le top k
    (n_pertinents = sum(pertinences)), ce qui rendait NDCG=1.0 dès que
    tous les résultats trouvés étaient bien classés — même si le moteur
    en avait raté beaucoup d'autres qui existaient dans le corpus. Ça
    expliquait des cas observés en pratique comme Precision@10=0.7 avec
    NDCG@10=1.0 (incohérent : un classement qui rate 30% des pertinents
    ne devrait jamais avoir un NDCG parfait).

    NDCG = DCG / IDCG, entre 0 et 1. Retourne 0 si IDCG = 0 (aucun
    document pertinent dans le corpus pour cette requête).
    """
    top_k = objets_retournes[:k]
    pertinences = [1 if est_pertinent(o, mots_cles) else 0 for o in top_k]

    dcg = sum(rel / math.log2(i + 2) for i, rel in enumerate(pertinences))

    n_ideal = min(n_pertinents_corpus, k)
    idcg = sum(1 / math.log2(i + 2) for i in range(n_ideal))

    return dcg / idcg if idcg > 0 else 0.0


@dataclass
class RequeteTest:
    nom: str
    requete: str
    mots_cles_pertinence: list[str]


# Jeu de requêtes de test, construites à partir des explorations manuelles
# faites sur données réelles au bloc 3 (thème cybersécurité) et de
# vérifications similaires sur d'autres thèmes du corpus DECP.
#
# ATTENTION - biais de fuite reconnu (data leakage) : les mots-clés de
# vérité terrain du thème "cybersécurité" ont été choisis après avoir vu
# les résultats de recherche BM25 lors de l'exploration manuelle du bloc 3
# (ex: "sécurité des systèmes d'information" a été identifié en observant
# les sorties du moteur, pas choisi a priori). C'est une fuite
# méthodologique : elle avantage artificiellement la méthode qui a servi
# à découvrir ces mots-clés. Les autres thèmes (voirie, restauration,
# espaces verts) sont moins à risque mais partagent le même principe de
# construction. Voir REQUETES_TEST_DIFFICILES ci-dessous, conçu
# spécifiquement pour limiter ce biais.
REQUETES_TEST = [
    RequeteTest(
        nom="cybersecurite",
        requete="marchés de cybersécurité en Île-de-France de montant élevé",
        mots_cles_pertinence=[
            "securite des systemes d information", "cybersecurite",
            "cyber securite", "securite du si", "securite informatique",
        ],
    ),
    RequeteTest(
        nom="travaux_voirie",
        requete="travaux de voirie et de chaussée",
        mots_cles_pertinence=["voirie", "chaussee", "trottoir", "revetement"],
    ),
    RequeteTest(
        nom="restauration_scolaire",
        requete="marchés de restauration scolaire et cantine",
        mots_cles_pertinence=["restauration scolaire", "cantine", "repas"],
    ),
    RequeteTest(
        nom="espaces_verts",
        requete="entretien des espaces verts et tonte",
        mots_cles_pertinence=["espaces verts", "tonte", "elagage", "espaces naturels"],
    ),
]


# Requêtes "difficiles" : mêmes thèmes et même vérité terrain (mots-clés)
# que REQUETES_TEST, mais reformulées avec un vocabulaire qui ne partage
# volontairement AUCUN mot avec les mots-clés de vérité terrain. Objectif :
# tester spécifiquement la capacité des embeddings à retrouver un document
# pertinent par similarité sémantique, sans recoupement lexical avec BM25
# ni avec la vérité terrain elle-même — corrige le biais de fuite reconnu
# ci-dessus. C'est sur ce jeu de requêtes que les embeddings devraient, en
# théorie, faire mieux que BM25 seul ; si ce n'est pas le cas en pratique,
# c'est un résultat à documenter honnêtement, pas à ignorer.
REQUETES_TEST_DIFFICILES = [
    RequeteTest(
        nom="cybersecurite_difficile",
        # Garde le même filtre (région + montant) que la version facile —
        # bug corrigé : la première reformulation ("protection des
        # systèmes informatiques") avait supprimé aussi le filtre
        # géographique/montant, pas seulement le vocabulaire thématique,
        # ce qui mélangeait deux facteurs de difficulté différents
        # (absence de recouvrement lexical ET absence de filtre) au lieu
        # d'isoler uniquement le premier. Trouvé en comparant n_pertinents
        # entre facile (4, après filtre) et difficile (37, sans filtre)
        # pour un même thème censé être filtré de façon identique.
        requete="protection des systèmes informatiques en Île-de-France de montant élevé",
        mots_cles_pertinence=REQUETES_TEST[0].mots_cles_pertinence,
    ),
    RequeteTest(
        nom="travaux_voirie_difficile",
        requete="entretien des routes communales",
        mots_cles_pertinence=REQUETES_TEST[1].mots_cles_pertinence,
    ),
    RequeteTest(
        nom="restauration_scolaire_difficile",
        # Reformulé pour éviter toute fuite lexicale avec les mots-clés de
        # vérité terrain ("repas", "cantine", "restauration scolaire") —
        # la première version ("repas pour les élèves") contenait
        # littéralement "repas", ce qui faussait artificiellement le score
        # de BM25 seul sur cette requête soi-disant "difficile".
        requete="nourriture destinée aux enfants dans les établissements primaires",
        mots_cles_pertinence=REQUETES_TEST[2].mots_cles_pertinence,
    ),
    RequeteTest(
        nom="espaces_verts_difficile",
        requete="maintenance des parcs municipaux",
        mots_cles_pertinence=REQUETES_TEST[3].mots_cles_pertinence,
    ),
]


def evaluer_requete(
    df_resultats: pd.DataFrame,
    requete_test: RequeteTest,
    k_values: list[int] | None = None,
    colonne_objet: str = "objet",
    df_corpus: pd.DataFrame | None = None,
) -> dict:
    """Calcule Precision@K, et (si df_corpus fourni) Recall@K et NDCG@K,
    à partir des résultats déjà retournés par le moteur de recherche.

    Ne lance pas la recherche elle-même (voir engine.search) — prend en
    entrée le dataframe de résultats déjà classé, pour découpler le calcul
    des métriques (testable sans réseau) de l'exécution du pipeline
    complet (nécessite le modèle d'embeddings).

    df_corpus : dataframe du corpus complet soumis à la recherche (avant
    filtrage/classement), utilisé pour compter le nombre total de marchés
    pertinents, calculer Recall@K, et calculer NDCG@K correctement (voir
    ndcg_at_k). Sans df_corpus, NDCG ne peut pas être calculé correctement
    (il nécessite de connaître le nombre total de pertinents dans le
    corpus) — dans ce cas, `ndcg@k` vaut `None` plutôt qu'un chiffre
    silencieusement biaisé.
    """
    k_values = k_values or [5, 10, 20]
    objets = df_resultats[colonne_objet].astype(str).tolist()

    resultat = {"nom": requete_test.nom, "n_resultats": len(objets)}

    n_pertinents_corpus = None
    if df_corpus is not None:
        n_pertinents_corpus = int(
            df_corpus[colonne_objet].astype(str)
            .apply(lambda o: est_pertinent(o, requete_test.mots_cles_pertinence))
            .sum()
        )
        resultat["n_pertinents_corpus"] = n_pertinents_corpus

    for k in k_values:
        resultat[f"precision@{k}"] = round(
            precision_at_k(objets, requete_test.mots_cles_pertinence, k), 3
        )

        if n_pertinents_corpus is not None:
            resultat[f"ndcg@{k}"] = round(
                ndcg_at_k(objets, requete_test.mots_cles_pertinence, k, n_pertinents_corpus), 3
            )
            if n_pertinents_corpus > 0:
                n_trouves = sum(
                    1 for o in objets[:k]
                    if est_pertinent(o, requete_test.mots_cles_pertinence)
                )
                resultat[f"recall@{k}"] = round(n_trouves / n_pertinents_corpus, 3)
            else:
                resultat[f"recall@{k}"] = None
        else:
            resultat[f"ndcg@{k}"] = None

    return resultat
