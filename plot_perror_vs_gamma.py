"""
p_error vs gamma for several tree sizes N, on one grid.

The paper's scaling is  L = gamma * sqrt(N):  the packet's energy width ~1/L =
1/(gamma*sqrt(N)) must sit inside the good-transmission window |E| < 1/(16*sqrt(N)).
Because BOTH scale as 1/sqrt(N), the error is essentially a function of gamma
alone -- so the curves for different N should collapse onto one another.  This
script simulates each N to show that.

Definition used here:

    p_error(gamma, N) = 1 - P_right(t_run)

measured on a *transmit* instance (NAND tree = 1, so the packet should fully
transmit; whatever fails to reach the right half is error).  All the N below have
even depth n = log2(N), for which the all-ones leaf assignment evaluates to
NAND = 1 -- a constructive transmit instance that avoids the 2^(2^n) brute force
of `bits_for_target` (infeasible for n = 16).

Run:

    python plot_perror_vs_gamma.py                 # full set incl. N=65536 (slow)
    python plot_perror_vs_gamma.py 4 16 256        # quick subset

Outputs  perror_vs_gamma.png  and  perror_vs_gamma.npz  in the repo root.
"""

from __future__ import annotations

import sys
import time

import numpy as np

import nand_tree_physics as P


# gamma grid (same for every N so the curves are directly comparable)
GAMMAS = np.array([2, 3, 4, 6, 8, 10, 12, 16, 20, 24, 28, 32], dtype=float)

# N values -> tree depth n = log2(N); all even here (all-ones => NAND = 1)
DEFAULT_NS = [16, 256, 1024]


def p_error(n: int, gamma: float) -> float:
    """1 - P_right(t_run) for a transmit (NAND=1) instance at this (n, gamma)."""
    N = 2 ** n
    L = P.L_from_gamma(gamma, n)
    M = P.default_M(L)
    bits = [1] * N                      # even n  ->  NAND tree = 1
    g = P.build_graph(n, M, L, bits)
    assert g.nand_value == 1, f"expected transmit instance, got NAND={g.nand_value}"
    psi0 = P.initial_packet(g, L)
    psi = P.evolve_to(g.H, psi0, P.t_run(L))
    return 1.0 - P.p_right(psi, g)


def compute(Ns):
    curves = {}
    for N in Ns:
        n = int(round(np.log2(N)))
        if 2 ** n != N:
            raise ValueError(f"N={N} is not a power of two")
        if n % 2 != 0:
            raise ValueError(f"N={N} has odd depth n={n}; all-ones is not NAND=1")
        errs = np.empty_like(GAMMAS)
        for i, gamma in enumerate(GAMMAS):
            t0 = time.time()
            errs[i] = p_error(n, float(gamma))
            print(f"  N={N:6d} (n={n:2d})  gamma={gamma:5.1f} -> "
                  f"p_error={errs[i]:.4e}   [{time.time() - t0:5.1f}s]",
                  flush=True)
        curves[N] = errs
    return curves


def make_plot(curves, path="perror_vs_gamma.png"):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(8, 5.5))
    floor = 1e-6                         # keep zeros visible on the log axis
    for N in sorted(curves):
        y = np.clip(curves[N], floor, None)
        ax.loglog(GAMMAS, y, marker="o", lw=1.8, ms=5,
                  label=f"N = {N}  (n = {int(round(np.log2(N)))})")

    # Paper's asymptotic  p_error = O(sqrt(1/gamma))  (eq. 4.7 discussion).
    # Anchored to the curves' mean at the largest gamma so it overlays the data
    # tail; only the slope (-1/2 on log-log) is the physical claim.
    tail = float(np.mean([curves[N][-1] for N in curves]))
    gmax = GAMMAS[-1]
    ax.loglog(GAMMAS, tail * np.sqrt(gmax / GAMMAS), color="k", ls="--", lw=1.5,
              label=r"$\propto 1/\sqrt{\gamma}$  (paper asymptotic)")

    ax.axvline(16, color="0.5", ls=":", lw=1.2)
    ax.text(16, ax.get_ylim()[1], r"  $\gamma = 16$", color="0.4",
            va="top", ha="left", fontsize=9)
    ax.set_xlabel(r"$\gamma$   (packet length $L = \gamma\sqrt{N}$)")
    ax.set_ylabel(r"$P_{\mathrm{error}}$")
    ax.set_title("NAND-tree quantum walk: $P_{\\mathrm{error}}$ vs $\\gamma$ "
                 "(transmit instance)")
    ax.grid(True, which="both", alpha=0.3)
    ax.legend(title="tree size")
    fig.tight_layout()
    fig.savefig(path, dpi=130)
    print(f"\nwrote {path}")


def main(argv):
    Ns = [int(a) for a in argv[1:]] or DEFAULT_NS
    print(f"gamma grid: {GAMMAS.tolist()}")
    print(f"N values:   {Ns}\n")
    t0 = time.time()
    curves = compute(Ns)
    np.savez("perror_vs_gamma.npz", gammas=GAMMAS,
             **{f"N_{N}": curves[N] for N in curves})
    make_plot(curves)
    print(f"total {time.time() - t0:.1f}s")


if __name__ == "__main__":
    main(sys.argv)
