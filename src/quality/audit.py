"""
Audit qualité des données DECP.

Produit un score de qualité par variable, basé sur les règles identifiées
lors de l'audit initial du dataset :
- valeurs manquantes ;
- dates aberrantes (hors plage plausible) ;
- montants incohérents (négatifs, nuls, ou extrêmes) ;
- catégories mal normalisées (casse incohérente) ;
- cohérence du champ donneesActuelles / modification_id.

Usage :
    python -m src.quality.audit --input data/raw/decp.parquet
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass, field
from pathlib import Path

import pandas as pd

from src.quality.loader import load_raw

# Plage de dates jugée plausible pour une notification de marché.
# À ajuster si le projet doit couvrir une période différente.
DATE_MIN_PLAUSIBLE = pd.Timestamp("2000-01-01")
DATE_MAX_PLAUSIBLE = pd.Timestamp("2027-12-31")

# Seuil au-delà duquel un montant est considéré comme suspect (en euros).
MONTANT_MAX_PLAUSIBLE = 1_000_000_000  # 1 milliard d'euros


@dataclass
class ColumnQualityReport:
    column: str
    missing_rate: float
    n_missing: int
    n_total: int
    issues: list[str] = field(default_factory=list)

    @property
    def quality_score(self) -> float:
        """Score simple entre 0 et 1 : 1 = complet et sans anomalie détectée.

        Pénalise les valeurs manquantes et chaque type de problème détecté.
        C'est un score de départ volontairement simple ; à affiner avec
        des pondérations métier si besoin.
        """
        score = 1.0 - self.missing_rate
        score -= 0.05 * len(self.issues)
        return max(0.0, round(score, 3))


def audit_missing_values(df: pd.DataFrame) -> dict[str, ColumnQualityReport]:
    n_total = len(df)
    reports = {}
    for col in df.columns:
        n_missing = int(df[col].isna().sum())
        reports[col] = ColumnQualityReport(
            column=col,
            missing_rate=round(n_missing / n_total, 4) if n_total else 0.0,
            n_missing=n_missing,
            n_total=n_total,
        )
    return reports


def audit_dates(df: pd.DataFrame, reports: dict[str, ColumnQualityReport]) -> None:
    for col in ("dateNotification", "datePublicationDonnees"):
        if col not in df.columns:
            continue
        dates = df[col].dropna()
        n_aberrant = int(
            ((dates < DATE_MIN_PLAUSIBLE) | (dates > DATE_MAX_PLAUSIBLE)).sum()
        )
        if n_aberrant > 0:
            reports[col].issues.append(
                f"{n_aberrant} dates hors plage plausible "
                f"[{DATE_MIN_PLAUSIBLE.date()}, {DATE_MAX_PLAUSIBLE.date()}]"
            )


def audit_montant(df: pd.DataFrame, reports: dict[str, ColumnQualityReport]) -> None:
    if "montant" not in df.columns:
        return
    montant = df["montant"].dropna()
    n_negatif_ou_nul = int((montant <= 0).sum())
    n_extreme = int((montant > MONTANT_MAX_PLAUSIBLE).sum())

    if n_negatif_ou_nul > 0:
        reports["montant"].issues.append(
            f"{n_negatif_ou_nul} montants négatifs ou nuls"
        )
    if n_extreme > 0:
        reports["montant"].issues.append(
            f"{n_extreme} montants > {MONTANT_MAX_PLAUSIBLE:,.0f} € (à vérifier)"
        )


def audit_categories(df: pd.DataFrame, reports: dict[str, ColumnQualityReport]) -> None:
    """Détecte les catégories qui varient seulement par la casse/espaces
    (ex: 'Marché', 'MARCHE', 'marché' pour la colonne nature).
    """
    for col in ("nature", "procedure", "type"):
        if col not in df.columns:
            continue
        values = df[col].dropna().astype(str)
        n_variantes_distinctes = values.nunique()
        n_variantes_normalisees = (
            values.str.lower().str.strip().str.normalize("NFKD")
            .str.encode("ascii", errors="ignore").str.decode("ascii")
            .nunique()
        )
        if n_variantes_distinctes > n_variantes_normalisees:
            reports[col].issues.append(
                f"{n_variantes_distinctes} variantes distinctes se réduisent à "
                f"{n_variantes_normalisees} après normalisation (casse/accents) — "
                "à normaliser avant tout groupby"
            )


def audit_versioning(df: pd.DataFrame, reports: dict[str, ColumnQualityReport]) -> None:
    """Vérifie la cohérence du mécanisme donneesActuelles / uid.

    Rappel : plusieurs lignes peuvent partager un même uid si plusieurs
    titulaires sont associés (cotraitance) — ce n'est pas un doublon.
    On vérifie ici qu'il n'y a pas d'incohérence dans le flag lui-même.
    """
    if "donneesActuelles" not in df.columns:
        return
    n_null = int(df["donneesActuelles"].isna().sum())
    if n_null > 0:
        reports["donneesActuelles"].issues.append(
            f"{n_null} valeurs nulles pour un champ censé être booléen"
        )


def run_full_audit(df: pd.DataFrame) -> dict[str, ColumnQualityReport]:
    reports = audit_missing_values(df)
    audit_dates(df, reports)
    audit_montant(df, reports)
    audit_categories(df, reports)
    audit_versioning(df, reports)
    return reports


def print_report(reports: dict[str, ColumnQualityReport]) -> None:
    print(f"{'Colonne':<30} {'Score':>6} {'% Manquant':>12}  Problèmes détectés")
    print("-" * 100)
    for col, r in sorted(reports.items(), key=lambda x: x[1].quality_score):
        issues_str = "; ".join(r.issues) if r.issues else "-"
        print(f"{col:<30} {r.quality_score:>6.2f} {r.missing_rate*100:>11.2f}%  {issues_str}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Audit qualité des données DECP")
    parser.add_argument(
        "--input", type=str, default="data/raw/decp.parquet",
        help="Chemin vers le fichier Parquet DECP",
    )
    args = parser.parse_args()

    df = load_raw(args.input)
    print(f"Dataset chargé : {df.shape[0]:,} lignes, {df.shape[1]} colonnes\n")

    reports = run_full_audit(df)
    print_report(reports)


if __name__ == "__main__":
    main()
