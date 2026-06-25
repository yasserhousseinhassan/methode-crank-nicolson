# -*- coding: utf-8 -*-
"""
============================================================================
Méthode de Crank-Nicolson — Résolution de l'équation de la chaleur 1D
============================================================================

Résolution numérique de l'équation parabolique de la chaleur

        ∂u/∂t = ∂²u/∂x²,   x ∈ (0, 1),  t > 0

avec conditions aux limites de Dirichlet homogènes  u(0,t) = u(1,t) = 0
et condition initiale  u(x,0) = sin(π x).

Solution analytique exacte :  u(x,t) = exp(-π² t) sin(π x).

Trois schémas aux différences finies sont implémentés et comparés :
    - Explicite (Euler progressif)    — conditionnellement stable (r ≤ 1/2)
    - Implicite (Euler rétrograde)    — inconditionnellement stable
    - Crank-Nicolson (θ = 1/2)        — inconditionnellement stable, O(Δt²+Δx²)

Auteur : Yasser Houssein Hassan
Dépendances : numpy, scipy, matplotlib
============================================================================
"""

import sys
import numpy as np
from scipy.linalg import solve_banded

# Sortie console en UTF-8 (symboles mathématiques : π, σ, ∞, Δ...)
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

# trapezoid (NumPy >= 2) avec repli sur trapz (versions antérieures)
_trapz = getattr(np, "trapezoid", getattr(np, "trapz", None))


# ---------------------------------------------------------------------------
# Solution analytique de référence
# ---------------------------------------------------------------------------
def solution_exacte(x: np.ndarray, t: float) -> np.ndarray:
    r"""u(x, t) = exp(-π² t) sin(π x)."""
    return np.exp(-np.pi**2 * t) * np.sin(np.pi * x)


def condition_initiale(x: np.ndarray) -> np.ndarray:
    r"""u(x, 0) = sin(π x)."""
    return np.sin(np.pi * x)


# ---------------------------------------------------------------------------
# Résolution d'un système tridiagonal (algorithme de Thomas via solve_banded)
# ---------------------------------------------------------------------------
def _resoudre_tridiagonal(sub, diag, sup, b):
    r"""Résout A x = b où A est tridiagonale, en O(n).

    sub, diag, sup : sous-diagonale, diagonale, sur-diagonale (longueur n).
    """
    n = len(diag)
    ab = np.zeros((3, n))
    ab[0, 1:] = sup[:-1]   # sur-diagonale
    ab[1, :] = diag        # diagonale principale
    ab[2, :-1] = sub[1:]   # sous-diagonale
    return solve_banded((1, 1), ab, b)


# ---------------------------------------------------------------------------
# 1. Schéma explicite (Euler progressif)
# ---------------------------------------------------------------------------
def schema_explicite(nx: int, nt: int, T: float = 0.1):
    r"""u^{n+1}_i = u^n_i + r (u^n_{i+1} - 2 u^n_i + u^n_{i-1}),  r = Δt/Δx².

    Stable seulement si r ≤ 1/2 (condition CFL parabolique).
    """
    dx = 1.0 / nx
    dt = T / nt
    r = dt / dx**2
    x = np.linspace(0, 1, nx + 1)
    u = condition_initiale(x)

    for _ in range(nt):
        u_new = u.copy()
        u_new[1:-1] = u[1:-1] + r * (u[2:] - 2 * u[1:-1] + u[:-2])
        u_new[0] = u_new[-1] = 0.0  # Dirichlet homogène
        u = u_new
    return x, u, r


# ---------------------------------------------------------------------------
# 2. Schéma implicite (Euler rétrograde)
# ---------------------------------------------------------------------------
def schema_implicite(nx: int, nt: int, T: float = 0.1):
    r"""-r u^{n+1}_{i-1} + (1+2r) u^{n+1}_i - r u^{n+1}_{i+1} = u^n_i.

    Inconditionnellement stable pour tout r > 0.
    """
    dx = 1.0 / nx
    dt = T / nt
    r = dt / dx**2
    x = np.linspace(0, 1, nx + 1)
    u = condition_initiale(x)

    m = nx - 1  # inconnues internes
    diag = (1 + 2 * r) * np.ones(m)
    sub = -r * np.ones(m)
    sup = -r * np.ones(m)

    for _ in range(nt):
        u[1:-1] = _resoudre_tridiagonal(sub, diag, sup, u[1:-1])
        u[0] = u[-1] = 0.0
    return x, u, r


# ---------------------------------------------------------------------------
# 3. Schéma de Crank-Nicolson (θ = 1/2)
# ---------------------------------------------------------------------------
def schema_crank_nicolson(nx: int, nt: int, T: float = 0.1):
    r"""Moyenne des schémas explicite et implicite :

        -r/2 u^{n+1}_{i-1} + (1+r) u^{n+1}_i - r/2 u^{n+1}_{i+1}
            =  r/2 u^n_{i-1} + (1-r) u^n_i + r/2 u^n_{i+1}

    Inconditionnellement stable, précision O(Δt² + Δx²).
    """
    dx = 1.0 / nx
    dt = T / nt
    r = dt / dx**2
    x = np.linspace(0, 1, nx + 1)
    u = condition_initiale(x)

    m = nx - 1
    diag = (1 + r) * np.ones(m)
    sub = -r / 2 * np.ones(m)
    sup = -r / 2 * np.ones(m)

    for _ in range(nt):
        ui = u[1:-1]
        # Second membre (partie explicite)
        b = (1 - r) * ui
        b[1:] += r / 2 * ui[:-1]
        b[:-1] += r / 2 * ui[1:]
        # Les termes de bord sont nuls (Dirichlet homogène)
        u[1:-1] = _resoudre_tridiagonal(sub, diag, sup, b)
        u[0] = u[-1] = 0.0
    return x, u, r


# ---------------------------------------------------------------------------
# Mesure de l'erreur (norme L∞ et L²)
# ---------------------------------------------------------------------------
def erreurs(x: np.ndarray, u_num: np.ndarray, T: float) -> dict:
    u_ex = solution_exacte(x, T)
    e = u_num - u_ex
    return {
        "err_Linf": np.max(np.abs(e)),
        "err_L2": np.sqrt(_trapz(e**2, x)),
    }


# ---------------------------------------------------------------------------
# Démonstration comparative
# ---------------------------------------------------------------------------
def _demonstration():
    nx, nt, T = 50, 1000, 0.1

    print("=" * 68)
    print("Comparaison des schémas — équation de la chaleur 1D")
    print(f"  nx = {nx}, nt = {nt}, T = {T}")
    print("=" * 68)

    for nom, fonc in [("Explicite      ", schema_explicite),
                      ("Implicite      ", schema_implicite),
                      ("Crank-Nicolson ", schema_crank_nicolson)]:
        x, u, r = fonc(nx, nt, T)
        err = erreurs(x, u, T)
        cfl = "OK" if (nom.strip() != "Explicite" or r <= 0.5) else "INSTABLE (r>1/2)"
        print(f"\n  {nom} | r = {r:.4f} ({cfl})")
        print(f"      Erreur L∞ = {err['err_Linf']:.3e}")
        print(f"      Erreur L² = {err['err_L2']:.3e}")

    # Étude de convergence du schéma de Crank-Nicolson
    print("\n" + "=" * 68)
    print("Étude de convergence (Crank-Nicolson) — ordre attendu ≈ 2")
    print("=" * 68)
    err_prec = None
    for nx in [10, 20, 40, 80]:
        nt = nx**2  # on garde Δt ~ Δx pour observer l'ordre 2 global
        x, u, _ = schema_crank_nicolson(nx, nt, T)
        e = erreurs(x, u, T)["err_Linf"]
        ordre = "" if err_prec is None else f"ordre ≈ {np.log2(err_prec / e):.2f}"
        print(f"  nx = {nx:3d} | erreur L∞ = {e:.3e}  {ordre}")
        err_prec = e


if __name__ == "__main__":
    _demonstration()
