import pandas as pd
import pytest

from src.search.annotation import (
    charger_annotations,
    comparer_annotation_vs_motscles,
    generer_fichier_annotation,
)
from src.search.evaluation import RequeteTest


@pytest.fixture
def resultats_par_requete():
    return {
        "cybersecurite": pd.DataFrame({
            "objet": ["Sécurité des systèmes A", "Travaux voirie X"],
            "montant": [100000.0, 50000.0],
            "acheteur_region_nom": ["Île-de-France", "Bretagne"],
        }),
    }


def test_generer_fichier_annotation_structure(tmp_path, resultats_par_requete):
    chemin = tmp_path / "annotation.csv"
    df = generer_fichier_annotation(resultats_par_requete, chemin, top_k=2)

    assert len(df) == 2
    assert list(df.columns) == ["requete", "rang", "objet", "montant", "region", "pertinent"]
    assert (df["pertinent"] == "").all()  # colonne vide, à remplir manuellement
    assert chemin.exists()


def test_generer_fichier_annotation_respecte_top_k(tmp_path):
    resultats = {
        "test": pd.DataFrame({"objet": [f"marché {i}" for i in range(10)]}),
    }
    chemin = tmp_path / "annotation.csv"
    df = generer_fichier_annotation(resultats, chemin, top_k=3)
    assert len(df) == 3


def test_charger_annotations_leve_erreur_si_lignes_vides(tmp_path):
    chemin = tmp_path / "annotation.csv"
    pd.DataFrame({
        "requete": ["a", "a"], "rang": [1, 2], "objet": ["x", "y"],
        "montant": [1, 2], "region": ["r", "r"], "pertinent": ["1", ""],
    }).to_csv(chemin, index=False)

    with pytest.raises(ValueError, match="non annotée"):
        charger_annotations(chemin)


def test_charger_annotations_ok_si_tout_rempli(tmp_path):
    chemin = tmp_path / "annotation.csv"
    pd.DataFrame({
        "requete": ["a", "a"], "rang": [1, 2], "objet": ["x", "y"],
        "montant": [1, 2], "region": ["r", "r"], "pertinent": ["1", "0"],
    }).to_csv(chemin, index=False)

    df = charger_annotations(chemin)
    assert df["pertinent"].tolist() == [1, 0]


def test_charger_annotations_supporte_separateur_point_virgule(tmp_path):
    """Excel en français exporte par défaut avec ';' comme séparateur,
    pas ',' — le chargement doit détecter ça automatiquement.
    """
    chemin = tmp_path / "annotation.csv"
    pd.DataFrame({
        "requete": ["a", "a"], "rang": [1, 2], "objet": ["x", "y"],
        "montant": [1, 2], "region": ["r", "r"], "pertinent": ["1", "0"],
    }).to_csv(chemin, index=False, sep=";")

    df = charger_annotations(chemin)
    assert df["pertinent"].tolist() == [1, 0]
    assert list(df.columns) == ["requete", "rang", "objet", "montant", "region", "pertinent"]


def test_charger_annotations_supporte_excel(tmp_path):
    """L'annotation peut aussi être faite dans un fichier Excel (.xlsx),
    plus confortable que le CSV brut pour certains utilisateurs.
    """
    chemin = tmp_path / "annotation.xlsx"
    pd.DataFrame({
        "requete": ["a", "a"], "rang": [1, 2], "objet": ["x", "y"],
        "montant": [1, 2], "region": ["r", "r"], "pertinent": [1, 0],
    }).to_excel(chemin, index=False)

    df = charger_annotations(chemin)
    assert df["pertinent"].tolist() == [1, 0]


def test_charger_annotations_excel_leve_erreur_si_lignes_vides(tmp_path):
    chemin = tmp_path / "annotation.xlsx"
    pd.DataFrame({
        "requete": ["a", "a"], "rang": [1, 2], "objet": ["x", "y"],
        "montant": [1, 2], "region": ["r", "r"], "pertinent": [1, None],
    }).to_excel(chemin, index=False)

    with pytest.raises(ValueError, match="non annotée"):
        charger_annotations(chemin)


def test_comparer_annotation_vs_motscles_accord_parfait():
    requete = RequeteTest(nom="test", requete="x", mots_cles_pertinence=["sécurité"])
    df_annote = pd.DataFrame({
        "requete": ["test", "test"],
        "objet": ["sécurité informatique", "travaux voirie"],
        "pertinent": [1, 0],  # humain d'accord avec les mots-clés
    })
    resultat = comparer_annotation_vs_motscles(df_annote, [requete])
    assert resultat.iloc[0]["taux_accord"] == 1.0
    assert resultat.iloc[0]["faux_positifs_motscles"] == 0
    assert resultat.iloc[0]["faux_negatifs_motscles"] == 0


def test_comparer_annotation_vs_motscles_detecte_faux_positif():
    """Le mot-clé dit pertinent mais l'humain juge que non (ex: homonyme,
    comme le cas SSI = Sécurité Incendie découvert au bloc 3).
    """
    requete = RequeteTest(nom="test", requete="x", mots_cles_pertinence=["ssi"])
    df_annote = pd.DataFrame({
        "requete": ["test"],
        "objet": ["Système de Sécurité Incendie (SSI) du bâtiment"],
        "pertinent": [0],  # l'humain sait que ce n'est pas de la cybersécurité
    })
    resultat = comparer_annotation_vs_motscles(df_annote, [requete])
    assert resultat.iloc[0]["faux_positifs_motscles"] == 1
    assert resultat.iloc[0]["taux_accord"] == 0.0
