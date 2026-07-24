"""
Extraction de filtres stricts (région, montant) depuis une requête en
langage naturel.

Limite assumée dès le départ (MVP) : ceci est une extraction par règles
simples (correspondance de région connue + expressions régulières pour
les montants), PAS un système de compréhension du langage naturel
complet. Une vraie extraction robuste nécessiterait un NER (Named Entity
Recognition) entraîné ou un LLM dédié — hors périmètre raisonnable pour
ce projet. Documenté comme limite dans docs/limitations.md.

Le texte restant après extraction des filtres est utilisé comme requête
de recherche libre pour BM25 et les embeddings.
"""

from __future__ import annotations

import re
from dataclasses import dataclass

# Régions réellement présentes dans le dataset DECP (voir acheteur_region_nom).
# Trié par longueur décroissante pour matcher les noms les plus spécifiques
# en premier (éviter qu'un nom court ne matche à tort dans un nom plus long).
REGIONS_CONNUES = sorted([
    "Auvergne-Rhône-Alpes", "Bourgogne-Franche-Comté", "Bretagne",
    "Centre-Val de Loire", "Corse", "Grand Est", "Guadeloupe", "Guyane",
    "Hauts-de-France", "La Réunion", "Martinique", "Mayotte", "Normandie",
    "Nouvelle-Aquitaine", "Nouvelle-Calédonie", "Occitanie",
    "Pays de la Loire", "Polynésie française", "Provence-Alpes-Côte d'Azur",
    "Saint-Barthélemy", "Saint-Martin", "Saint-Pierre-et-Miquelon",
    "Wallis-et-Futuna", "Île-de-France",
], key=len, reverse=True)

# Seuil utilisé pour interpréter une expression qualitative ("montant élevé")
# faute de pouvoir déduire un seuil précis du langage naturel.
# Choix arbitraire assumé : 90e percentile calculé sur le dataset filtré
# au moment de la recherche (voir engine.py), pas une valeur fixe ici.
MONTANT_ELEVE_PERCENTILE = 0.90


@dataclass
class ParsedQuery:
    region: str | None
    montant_min: float | None
    montant_eleve_qualitatif: bool
    texte_libre: str


def extract_region(query: str) -> tuple[str | None, str]:
    """Cherche une région connue dans la requête (insensible à la casse).

    Retourne (région trouvée ou None, requête débarrassée de cette mention).
    """
    query_lower = query.lower()
    for region in REGIONS_CONNUES:
        if region.lower() in query_lower:
            # Retire la mention de la requête (insensible à la casse)
            pattern = re.compile(re.escape(region), re.IGNORECASE)
            remaining = pattern.sub("", query)
            return region, remaining
    return None, query


def extract_montant_filter(query: str) -> tuple[float | None, bool, str]:
    """Cherche une contrainte de montant explicite ou qualitative.

    Retourne (montant_min explicite ou None, montant_eleve_qualitatif,
    requête nettoyée).

    Gère deux cas :
    - explicite : "montant supérieur à 100000 euros", "plus de 50000€"
    - qualitatif : "montant élevé" (sans valeur précise) -> flag à
      interpréter par l'appelant via un percentile du dataset filtré.
    """
    # Cas explicite : nombre suivi ou précédé d'indices de montant
    pattern_explicite = re.compile(
        r"(?:sup[ée]rieur[e]?\s+[àa]|plus\s+de)\s+([\d\s]+(?:[.,]\d+)?)\s*(?:€|euros?)?",
        re.IGNORECASE,
    )
    match = pattern_explicite.search(query)
    if match:
        valeur_str = match.group(1).replace(" ", "").replace(",", ".")
        try:
            valeur = float(valeur_str)
            remaining = pattern_explicite.sub("", query)
            return valeur, False, remaining
        except ValueError:
            pass

    # Cas qualitatif : "montant élevé" / "gros montant" sans valeur précise
    pattern_qualitatif = re.compile(
        r"montants?\s+[ée]lev[ée]s?|gros\s+montants?", re.IGNORECASE
    )
    if pattern_qualitatif.search(query):
        remaining = pattern_qualitatif.sub("", query)
        return None, True, remaining

    return None, False, query


# Mots de liaison qui peuvent rester "orphelins" après extraction d'une
# région ou d'un montant (ex: "en Île-de-France" -> "en" reste seul une
# fois "Île-de-France" retiré). Nettoyés en fin de parsing pour ne pas
# polluer la requête de recherche libre transmise à BM25/embeddings.
MOTS_LIAISON_ORPHELINS = {"en", "de", "du", "des", "et", "à", "au", "aux"}


def _nettoyer_mots_orphelins(text: str) -> str:
    """Retire les mots de liaison isolés en début/fin de texte ou en doublon,
    laissés par l'extraction de région/montant. Ne touche pas aux mots de
    liaison au milieu d'une expression significative (ex: "systèmes
    d'information" reste intact).
    """
    mots = text.split()
    # Ne retire que les mots de liaison en tête ou en fin de chaîne,
    # de façon répétée (ex: "en de cybersécurité de" -> "cybersécurité")
    while mots and mots[0].lower() in MOTS_LIAISON_ORPHELINS:
        mots.pop(0)
    while mots and mots[-1].lower() in MOTS_LIAISON_ORPHELINS:
        mots.pop()
    return " ".join(mots)


def parse_query(query: str) -> ParsedQuery:
    """Parse une requête en langage naturel en filtres stricts + texte libre.

    Le montant qualitatif ("montant élevé") est résolu en montant_min
    précis par l'appelant (engine.py), qui a accès au dataset pour calculer
    le percentile approprié — ce module reste indépendant des données.
    """
    region, remaining = extract_region(query)
    montant_min, montant_eleve, remaining = extract_montant_filter(remaining)

    # Nettoyage : espaces multiples, ponctuation résiduelle, puis mots de
    # liaison orphelins laissés par l'extraction de région/montant.
    texte_libre = re.sub(r"\s+", " ", remaining).strip(" ,.")
    texte_libre = _nettoyer_mots_orphelins(texte_libre)

    return ParsedQuery(
        region=region,
        montant_min=montant_min,
        montant_eleve_qualitatif=montant_eleve,
        texte_libre=texte_libre,
    )
