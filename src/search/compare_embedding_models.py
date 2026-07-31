"""
Compare le modèle d'embeddings actuel (MiniLM, léger) à un modèle
multilingue plus grand, sur les 4 thèmes difficiles — réponse concrète à
la piste notée dans docs/limitations.md ("tester un modèle d'embeddings
plus grand"), suite à l'échec observé sur "espaces verts" (0,0 des deux
côtés, BM25 et pipeline complet), où l'hypothèse posée était que MiniLM
ne capture pas assez bien la proximité sémantique entre "maintenance des
parcs municipaux" et "espaces verts"/"tonte"/"élagage".

Modèle plus grand choisi : paraphrase-multilingual-mpnet-base-v2
(~278M paramètres, contre ~118M pour MiniLM ; même famille
sentence-transformers, donc comparaison à méthodologie égale, seule la
taille du modèle change).

Lancement : python -m src.search.compare_embedding_models

ATTENTION avant de lancer :
- télécharge un nouveau modèle (~1 Go) au premier lancement — accès
  internet complet nécessaire, comme pour MiniLM en son temps ;
- réencode tous les marchés candidats avec le modèle plus grand (pas de
  réutilisation possible du cache MiniLM existant : dimensions
  différentes) — plus lent que MiniLM par texte, sur un CPU sans GPU
  garanti, l'ordre de grandeur attendu est de plusieurs fois le temps
  déjà mesuré pour MiniLM (voir docs/limitations.md, section "Temps de
  réponse") ;
- chaque modèle utilise un fichier de cache dédié (jamais partagé entre
  modèles : mélanger des vecteurs de dimensions différentes dans le même
  cache casserait silencieusement les calculs de similarité).
"""

import pandas as pd

from src.quality.loader import load_current_only
from src.search.engine import search
from src.search.evaluation import (
    REQUETES_TEST_DIFFICILES,
    evaluer_requete,
    taille_echantillon_pour,
)

MODELE_ACTUEL = "paraphrase-multilingual-MiniLM-L12-v2"
MODELE_PLUS_GRAND = "paraphrase-multilingual-mpnet-base-v2"

CACHE_PAR_MODELE = {
    MODELE_ACTUEL: "data/processed/embeddings_cache.npz",
    MODELE_PLUS_GRAND: "data/processed/embeddings_cache_mpnet.npz",
}


def evaluer_theme_pour_modele(df: pd.DataFrame, requete_test, model_name: str) -> dict:
    sample = df.sample(n=taille_echantillon_pour(requete_test.nom), random_state=42)
    result, _ = search(
        sample,
        requete_test.requete,
        top_k=20,
        cache_embeddings_path=CACHE_PAR_MODELE[model_name],
        embedding_model_name=model_name,
    )
    metriques = evaluer_requete(result, requete_test, k_values=[5, 10], df_corpus=sample)
    metriques["modele"] = model_name
    return metriques


def main():
    df = load_current_only(
        "data/raw/decp.parquet",
        columns=["uid", "objet", "montant", "acheteur_region_nom"],
    )
    df = df.dropna(subset=["objet"])

    toutes_lignes = []
    for requete_test in REQUETES_TEST_DIFFICILES:
        print(f"Thème : {requete_test.nom} — requête : {requete_test.requete}")
        for model_name in (MODELE_ACTUEL, MODELE_PLUS_GRAND):
            print(f"  Modèle : {model_name} ...")
            ligne = evaluer_theme_pour_modele(df, requete_test, model_name)
            toutes_lignes.append(ligne)
            print(f"    P@5={ligne['precision@5']}  P@10={ligne['precision@10']}  NDCG@5={ligne['ndcg@5']}")
        print()

    resultats = pd.DataFrame(toutes_lignes)
    print("=" * 70)
    print("Résumé complet")
    print("=" * 70)
    print(resultats.to_string(index=False))

    resultats.to_csv("data/processed/comparaison_modeles_embeddings.csv", index=False)
    print("\nRésultats sauvegardés dans data/processed/comparaison_modeles_embeddings.csv")


if __name__ == "__main__":
    main()
