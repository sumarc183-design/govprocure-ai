"""
Dashboard GovProcure AI — assemble les blocs 1 (qualité), 2 (anomalies)
et 3 (recherche) dans une interface Streamlit unique.

Lancement : streamlit run src/dashboard/app.py

Choix volontairement simple, cohérent avec la roadmap (bloc 4 =
assemblage, pas nouvelle recherche technique) : une seule page avec des
onglets plutôt que plusieurs pages Streamlit séparées, pour éviter de
dupliquer le chargement des données entre pages.
"""

import sys
from pathlib import Path

import pandas as pd
import streamlit as st

# Permet d'importer src.* quand on lance `streamlit run` depuis la racine du projet
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from src.anomaly.detection import detect_isolation_forest, detect_lof
from src.quality.audit import run_full_audit
from src.quality.loader import load_current_only
from src.search.engine import search

DATA_PATH = "data/raw/decp.parquet"

st.set_page_config(page_title="GovProcure AI", layout="wide")


@st.cache_data
def charger_donnees_qualite() -> pd.DataFrame:
    """Charge les colonnes nécessaires à l'audit qualité.

    Mis en cache : sans ça, Streamlit recharge le fichier Parquet complet
    à chaque interaction (chaque clic relance le script), ce qui serait
    beaucoup trop lent sur un fichier de 3M lignes.
    """
    from src.quality.loader import CORE_COLUMNS

    return load_current_only(DATA_PATH, columns=CORE_COLUMNS)


@st.cache_data
def charger_donnees_recherche() -> pd.DataFrame:
    """Charge les colonnes nécessaires à la recherche (sous-ensemble plus léger)."""
    df = load_current_only(
        DATA_PATH,
        columns=["uid", "objet", "montant", "acheteur_region_nom", "dureeMois", "codeCPV"],
    )
    return df.dropna(subset=["objet"])


def onglet_qualite():
    st.header("Qualité des données")
    st.caption(
        "Score par colonne : 1 = complet et sans anomalie structurelle détectée. "
        "Voir docs/limitations.md pour la méthodologie complète."
    )

    with st.spinner("Chargement et audit des données..."):
        df = charger_donnees_qualite()
        reports = run_full_audit(df)

    lignes = []
    for col, r in sorted(reports.items(), key=lambda x: x[1].quality_score):
        lignes.append({
            "Colonne": col,
            "Score qualité": r.quality_score,
            "% manquant": round(r.missing_rate * 100, 2),
            "Problèmes détectés": "; ".join(r.issues) if r.issues else "-",
        })

    st.dataframe(pd.DataFrame(lignes), use_container_width=True, hide_index=True)
    st.caption(f"{len(df):,} marchés analysés (version actuelle uniquement).")


def onglet_anomalies():
    st.header("Détection d'anomalies")
    st.caption(
        "⚠️ Une anomalie détectée n'est jamais une preuve de fraude — "
        "uniquement une priorisation de dossiers à examiner."
    )

    taille_echantillon = st.slider(
        "Taille de l'échantillon analysé", 1_000, 50_000, 20_000, step=1_000,
        help="LOF est coûteux en mémoire sur de gros volumes (voir docs/limitations.md) "
        "— on limite volontairement la taille ici.",
    )

    if st.button("Lancer la détection"):
        with st.spinner("Chargement des données..."):
            df = load_current_only(
                DATA_PATH, columns=["uid", "montant", "dureeMois", "codeCPV"]
            )
            df = df.dropna(subset=["montant", "codeCPV"])
            sample = df.sample(n=min(taille_echantillon, len(df)), random_state=42)

        with st.spinner("Isolation Forest..."):
            result_if = detect_isolation_forest(sample, contamination=0.05)
        with st.spinner("Local Outlier Factor..."):
            result_lof = detect_lof(sample, contamination=0.05)

        n_accord = (result_if["is_anomaly_iforest"] & result_lof["is_anomaly_lof"]).sum()
        n_union = (result_if["is_anomaly_iforest"] | result_lof["is_anomaly_lof"]).sum()
        taux_accord = round(n_accord / n_union, 3) if n_union > 0 else 0.0

        col1, col2, col3 = st.columns(3)
        col1.metric("Anomalies (Isolation Forest)", int(result_if["is_anomaly_iforest"].sum()))
        col2.metric("Anomalies (LOF)", int(result_lof["is_anomaly_lof"].sum()))
        col3.metric("Taux d'accord entre les 2 méthodes", f"{taux_accord*100:.1f}%")

        st.caption(
            "Un faible taux d'accord est normal et attendu : les deux méthodes "
            "détectent des types d'anomalies différents (voir docs/decisions.md)."
        )

        st.subheader("Top 10 anomalies (Isolation Forest)")
        top_if = result_if.nsmallest(10, "anomaly_score_iforest")
        st.dataframe(
            top_if[["uid", "montant", "dureeMois", "montant_ratio_mediane_cpv"]],
            use_container_width=True, hide_index=True,
        )


def onglet_recherche():
    st.header("Recherche de marchés")
    st.caption(
        "Recherche hybride : filtres stricts (région, montant) + BM25 + "
        "embeddings sémantiques, fusionnés par Reciprocal Rank Fusion. "
        "Voir docs/decisions.md pour l'architecture complète."
    )

    requete = st.text_input(
        "Requête en langage naturel",
        placeholder="ex : marchés de cybersécurité en Île-de-France de montant élevé",
    )
    taille_echantillon = st.slider(
        "Taille de l'échantillon de recherche", 5_000, 200_000, 100_000, step=5_000,
    )

    if requete and st.button("Rechercher"):
        with st.spinner("Chargement des données..."):
            df = charger_donnees_recherche()
            sample = df.sample(n=min(taille_echantillon, len(df)), random_state=42)

        with st.spinner("Recherche en cours (le premier lancement télécharge le modèle d'embeddings)..."):
            result, explication = search(sample, requete, top_k=20)

        st.subheader("Critères appliqués")
        col1, col2, col3 = st.columns(3)
        col1.metric("Région filtrée", explication["region_filtree"] or "—")
        col2.metric("Montant élevé demandé", "Oui" if explication["montant_eleve_qualitatif"] else "Non")
        col3.metric("Marchés après filtre strict", explication["n_marches_apres_filtre_strict"])
        st.caption(f"Texte de recherche interprété : « {explication['texte_recherche']} »")

        st.subheader(f"Résultats ({len(result)})")
        if len(result) == 0:
            st.info("Aucun résultat. Essayez une requête moins restrictive.")
        else:
            colonnes_affichees = [c for c in ["objet", "montant", "acheteur_region_nom", "rrf_score"] if c in result.columns]
            st.dataframe(result[colonnes_affichees], use_container_width=True, hide_index=True)

            csv = result[colonnes_affichees].to_csv(index=False).encode("utf-8")
            st.download_button(
                "Exporter en CSV", csv, "resultats_recherche.csv", "text/csv",
            )


def main():
    st.title("GovProcure AI")
    st.caption("Plateforme d'analyse des marchés publics — données ouvertes de la commande publique")

    tab1, tab2, tab3 = st.tabs(["Qualité des données", "Détection d'anomalies", "Recherche"])
    with tab1:
        onglet_qualite()
    with tab2:
        onglet_anomalies()
    with tab3:
        onglet_recherche()


if __name__ == "__main__":
    main()
