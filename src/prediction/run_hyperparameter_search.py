"""
Recherche d'hyperparamètres à budget augmenté pour Random Forest
(`offresRecues`) — réponse à la piste notée dans docs/decisions.md /
docs/limitations.md ("augmenter le budget de recherche : plus
d'itérations, dataset complet plutôt qu'un sous-échantillon de 50 000").

Contexte : la recherche initiale (15 itérations, sous-échantillon de
50 000, cv=3 interne) donnait un résultat contre-intuitif — le modèle
sélectionné restait moins bon que la configuration par défaut, même
avec un scorer corrigé en espace réel. Non tranché à l'époque : budget
trop limité, ou paramètres par défaut déjà bons ? Ce script tranche avec
un budget nettement plus grand, sur un vrai jeu de test tenu à l'écart
de la recherche (RandomizedSearchCV réentraîne son meilleur modèle sur
tout ce qu'on lui donne — lui donner le jeu de test complet aurait biaisé
la comparaison en sa faveur).

Lancement (budget par défaut, voir constantes ci-dessous pour ajuster) :
    python -m src.prediction.run_hyperparameter_search

Paramètres par défaut : n_iter=60 (x4 vs 15 initial), sous-échantillon
de 200 000 (x4 vs 50 000) — pas le dataset complet (656 023 lignes
valides) : avec jusqu'à 300 arbres et 60 x 3 = 180 entraînements de forêt
au total, l'ordre de grandeur attendu sur cette machine (12 coeurs) est
de plusieurs dizaines de minutes à quelques heures — un compromis
budget/temps assumé plutôt qu'un choix caché. TAILLE_ECHANTILLON peut
être augmenté jusqu'à la taille du dataset complet si le temps
disponible le permet.
"""

from __future__ import annotations

import time

from src.prediction.features import construire_matrice_features
from src.prediction.models import (
    SCORER_ESPACE_REEL,
    evaluer_predictions,
    rechercher_hyperparametres,
    separer_train_test,
)
from src.quality.loader import load_current_only

COLONNES = [
    "uid", "offresRecues", "montant", "dureeMois", "codeCPV",
    "procedure", "nature", "acheteur_region_nom", "donneesActuelles",
]

TAILLE_ECHANTILLON = 200_000
N_ITER = 60


def main():
    print("Chargement des données...")
    df = load_current_only("data/raw/decp.parquet", columns=COLONNES)

    print("Construction des features...")
    X, y, _feature_cols = construire_matrice_features(df)
    print(f"Dataset exploitable : {len(X):,} lignes")

    taille = min(TAILLE_ECHANTILLON, len(X))
    idx_echantillon = X.sample(n=taille, random_state=42).index
    X_sample, y_sample = X.loc[idx_echantillon], y.loc[idx_echantillon]
    print(f"Sous-échantillon utilisé : {len(X_sample):,} lignes")

    # Jeu de test tenu à l'écart AVANT la recherche — RandomizedSearchCV
    # (refit=True par défaut) réentraîne son meilleur modèle sur tout ce
    # qu'on lui passe en entrée. Lui passer ce jeu de test aurait biaisé
    # la comparaison finale en faveur du modèle "optimisé" (évalué sur des
    # données qu'il aurait déjà vues à l'entraînement).
    X_train, X_test, y_train, y_test = separer_train_test(X_sample, y_sample)
    print(f"Train : {len(X_train):,} — Test (tenu à l'écart de la recherche) : {len(X_test):,}\n")

    print("=" * 60)
    print(f"Recherche d'hyperparamètres — n_iter={N_ITER} (budget x4 vs 15 initial)")
    print("=" * 60)
    t0 = time.perf_counter()
    resultat_recherche = rechercher_hyperparametres(X_train, y_train, n_iter=N_ITER, scoring=SCORER_ESPACE_REEL)
    duree = time.perf_counter() - t0
    print(f"Terminé en {duree:.1f}s ({duree / 60:.1f} min)")
    print(f"Meilleurs paramètres : {resultat_recherche['meilleurs_parametres']}")
    print(f"Meilleur score en validation croisée interne (R² espace réel) : {resultat_recherche['meilleur_score_cv']}")

    print("\n" + "=" * 60)
    print("Comparaison finale sur le jeu de test tenu à l'écart")
    print("=" * 60)

    from sklearn.ensemble import RandomForestRegressor

    from src.prediction.models import RANDOM_STATE

    modele_defaut = RandomForestRegressor(n_estimators=100, max_depth=15, random_state=RANDOM_STATE, n_jobs=-1)
    modele_defaut.fit(X_train, y_train)
    metriques_defaut = evaluer_predictions(y_test.values, modele_defaut.predict(X_test))
    print(f"Random Forest par défaut  — R² réel : {metriques_defaut['r2_reel']}")

    modele_optimise = resultat_recherche["modele"]
    metriques_optimise = evaluer_predictions(y_test.values, modele_optimise.predict(X_test))
    print(f"Random Forest optimisé    — R² réel : {metriques_optimise['r2_reel']} (budget x4, n_iter={N_ITER})")

    print("\nÀ documenter dans docs/limitations.md une fois ce résultat obtenu :")
    if metriques_optimise["r2_reel"] > metriques_defaut["r2_reel"]:
        print("- le modèle optimisé BAT le défaut : le budget initial (15 iter, 50 000")
        print("  lignes) était bien le facteur limitant, pas les paramètres par défaut.")
    else:
        print("- le modèle optimisé reste moins bon (ou égal) malgré un budget x4 : ça")
        print("  renforce l'hypothèse que la configuration par défaut était déjà une")
        print("  bonne configuration pour ce problème, plutôt qu'un budget de recherche")
        print("  insuffisant.")


if __name__ == "__main__":
    main()
