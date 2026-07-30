import pandas as pd
import pytest

from src.search.bm25_search import BM25Index, _tokenize
from src.search.filters import extract_montant_filter, extract_region, parse_query


@pytest.fixture
def sample_df():
    return pd.DataFrame({
        "uid": ["a", "b", "c"],
        "objet": [
            "Analyse et remédiation des événements de sécurité des systèmes d'information",
            "Fourniture de matériel de bureau et papeterie",
            "Travaux de rénovation de la toiture du bâtiment municipal",
        ],
        "montant": [50000.0, 5000.0, 200000.0],
    })


# --- Tests filters.py ---

def test_extract_region_found():
    region, remaining = extract_region("marchés en Bretagne pour travaux")
    assert region == "Bretagne"
    assert "Bretagne" not in remaining


def test_extract_region_not_found():
    region, remaining = extract_region("marchés pour travaux")
    assert region is None
    assert remaining == "marchés pour travaux"


def test_extract_montant_explicite():
    montant, qualitatif, remaining = extract_montant_filter(
        "marchés supérieur à 100000 euros"
    )
    assert montant == 100000.0
    assert qualitatif is False


def test_extract_montant_qualitatif():
    montant, qualitatif, remaining = extract_montant_filter(
        "marchés de montant élevé en informatique"
    )
    assert montant is None
    assert qualitatif is True


def test_parse_query_full_example():
    """Reproduit l'exemple exact du besoin métier initial."""
    result = parse_query(
        "Affiche les marchés informatiques de montant élevé conclus en "
        "Île-de-France et similaires aux prestations de cybersécurité"
    )
    assert result.region == "Île-de-France"
    assert result.montant_eleve_qualitatif is True
    assert result.montant_min is None
    assert "cybersécurité" in result.texte_libre.lower()


def test_parse_query_no_orphan_connector_words():
    """Non-régression : bug trouvé en testant sur données réelles — les mots
    de liaison ('en', 'de') qui entouraient la région/montant extraits
    restaient orphelins dans le texte libre (ex: 'marchés de cybersécurité
    en de'), polluant la requête envoyée à BM25/embeddings.
    """
    result = parse_query(
        "marchés de cybersécurité en Île-de-France de montant élevé"
    )
    assert not result.texte_libre.lower().startswith(("en ", "de ", "du "))
    assert not result.texte_libre.lower().endswith((" en", " de", " du"))
    assert "cybersécurité" in result.texte_libre.lower()


# --- Tests bm25_search.py ---

def test_tokenize_removes_stopwords():
    tokens = _tokenize("Le marché de la sécurité et des systèmes")
    assert "le" not in tokens
    assert "de" not in tokens
    assert "securite" in tokens  # accents normalisés : "sécurité" -> "securite"
    assert "systemes" in tokens


def test_tokenize_ignores_accents():
    """Non-régression : bug trouvé en testant sur données réelles — une
    partie du corpus est écrite en majuscules sans accents (ex: 'SECURITE
    DES SYSTEMES D INFORMATION'). Sans normalisation, cette écriture ne
    matchait jamais une requête en français normal ('sécurité').
    """
    tokens_accentues = _tokenize("Sécurité")
    tokens_sans_accent = _tokenize("SECURITE")
    assert tokens_accentues == tokens_sans_accent


def test_tokenize_decomposes_compound_prefix():
    """Non-régression : 'cybersécurité' doit partager un token avec
    'sécurité' pour que BM25 puisse les rapprocher.
    """
    tokens = _tokenize("cybersécurité")
    assert "securite" in tokens
    assert "cyber" in tokens


def test_bm25_ranks_relevant_document_first(sample_df):
    index = BM25Index(sample_df)
    result = index.search("sécurité systèmes information", top_k=3)
    assert result.iloc[0]["uid"] == "a"


def test_bm25_empty_query_returns_empty(sample_df):
    index = BM25Index(sample_df)
    result = index.search("", top_k=5)
    assert len(result) == 0


def test_bm25_score_column_present(sample_df):
    index = BM25Index(sample_df)
    result = index.search("travaux rénovation", top_k=3)
    assert "bm25_score" in result.columns


# --- Tests engine.py (apply_strict_filters uniquement : le reste du
# pipeline dépend du téléchargement du modèle d'embeddings, non testable
# dans cet environnement sandboxé sans accès à huggingface.co — à valider
# en local, voir docs/limitations.md) ---

def test_apply_strict_filters_by_region():
    from src.search.engine import apply_strict_filters
    from src.search.filters import ParsedQuery

    df = pd.DataFrame({
        "uid": ["a", "b", "c"],
        "montant": [1000.0, 2000.0, 3000.0],
        "acheteur_region_nom": ["Bretagne", "Île-de-France", "Bretagne"],
    })
    parsed = ParsedQuery(
        region="Bretagne", montant_min=None,
        montant_eleve_qualitatif=False, texte_libre="",
    )
    result = apply_strict_filters(df, parsed)
    assert set(result["uid"]) == {"a", "c"}


def test_apply_strict_filters_montant_qualitatif_uses_percentile():
    from src.search.engine import apply_strict_filters
    from src.search.filters import ParsedQuery

    df = pd.DataFrame({
        "uid": [f"m{i}" for i in range(10)],
        "montant": [float(i) * 1000 for i in range(10)],  # 0 à 9000
        "acheteur_region_nom": ["Bretagne"] * 10,
    })
    parsed = ParsedQuery(
        region=None, montant_min=None,
        montant_eleve_qualitatif=True, texte_libre="",
    )
    result = apply_strict_filters(df, parsed)
    # 90e percentile de [0..9000] -> ne garde que les plus gros montants
    assert result["montant"].min() >= df["montant"].quantile(0.90)


def test_apply_strict_filters_deduplicates_cotraitance():
    """Non-régression : bug trouvé en testant sur données réelles — un marché
    en cotraitance (plusieurs titulaires, même uid) apparaissait plusieurs
    fois à l'identique dans les résultats de recherche. Cohérent avec la
    découverte du bloc 1 (group_by_marche) : la déduplication par uid doit
    aussi s'appliquer avant la recherche, pas seulement dans l'audit qualité.
    """
    from src.search.engine import apply_strict_filters
    from src.search.filters import ParsedQuery

    df = pd.DataFrame({
        "uid": ["m1", "m1", "m1", "m2"],  # m1 en cotraitance, 3 titulaires
        "titulaire_id": ["t1", "t2", "t3", "t4"],
        "montant": [100000.0, 100000.0, 100000.0, 50000.0],
        "acheteur_region_nom": ["Bretagne"] * 4,
    })
    parsed = ParsedQuery(
        region=None, montant_min=None,
        montant_eleve_qualitatif=False, texte_libre="",
    )
    result = apply_strict_filters(df, parsed)
    assert len(result) == 2  # un par uid, pas un par titulaire
    assert set(result["uid"]) == {"m1", "m2"}
