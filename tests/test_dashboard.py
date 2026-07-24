"""
Test de fumée pour le dashboard Streamlit.

Ne teste PAS le rendu réel de l'interface (nécessite un vrai contexte
Streamlit + navigateur, hors périmètre de pytest) — vérifie uniquement
que le module s'importe sans erreur, ce qui attrape déjà la plupart des
erreurs de syntaxe, d'import cassé, ou de dépendance manquante.

Le test manuel de l'interface elle-même (rendu, interactions, onglet
recherche avec le vrai modèle d'embeddings) est documenté comme étape à
valider localement, voir docs/limitations.md.
"""

import importlib


def test_dashboard_app_imports_without_error():
    """Si ce test échoue, l'app plantera au lancement de `streamlit run`."""
    import src.dashboard.app  # noqa: F401
    importlib.reload(src.dashboard.app)
