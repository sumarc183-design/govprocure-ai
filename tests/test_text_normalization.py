from src.common.text_normalization import normaliser_texte_categorie


def test_normalise_casse():
    assert normaliser_texte_categorie("MARCHE") == normaliser_texte_categorie("marché")


def test_normalise_accents():
    assert normaliser_texte_categorie("Procedure") == normaliser_texte_categorie("Procédure")


def test_normalise_apostrophes_variantes():
    """Bug trouvé lors de la construction du bloc prédiction : 'procedure'
    avait 3 variantes d'apostrophe jamais normalisées (' droite, '
    typographique, absente) — ce que nature/bm25_search normalisaient déjà
    pour les accents, mais pas pour ce cas précis.
    """
    v1 = normaliser_texte_categorie("Appel d'offres ouvert")
    v2 = normaliser_texte_categorie("Appel d’offres ouvert")
    v3 = normaliser_texte_categorie("Appel d offres ouvert")
    assert v1 == v2 == v3


def test_normalise_espaces_superflus():
    assert normaliser_texte_categorie("  Marché   Public  ") == normaliser_texte_categorie("Marché Public")


def test_normalise_gere_valeur_non_string():
    """Ne doit pas planter sur une valeur manquante (NaN) ou non textuelle."""
    assert normaliser_texte_categorie(None) is None
    import math
    resultat = normaliser_texte_categorie(float("nan"))
    assert math.isnan(resultat)
