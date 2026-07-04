"""
Physics core for the Hamiltonian NAND-tree quantum algorithm.

Implements the continuous-time quantum walk of

    Farhi, Goldstone, Gutmann, "A Quantum Algorithm for the Hamiltonian
    NAND Tree", quant-ph/0702144.

The graph is a 1-D "runway" (sites r = -M .. M) whose site r = 0 is attached
to the root of a perfectly bifurcating binary tree of depth n (N = 2**n
leaves).  Each leaf optionally carries one extra "oracle" node: the edge is
present iff that leaf's input bit is 1.  The Hamiltonian is H = -A, minus the
adjacency matrix of the whole graph.

A right-moving wave packet of length L, peaked in energy at E = 0, starts on
the left of the runway.  After time t_run = L/2 it has either transmitted to
the right (NAND tree evaluates to 1, transmission coefficient T(0) = 1) or
reflected back to the left (NAND tree evaluates to 0, T(0) = 0).  Measuring the
projector onto the right half of the runway reads out the answer.

This module is pure simulation -- no plotting.  Run it directly to execute a
self-test that checks the quantum measurement against a classical NAND
evaluation for several instances.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import scipy.sparse as sp
from scipy.sparse.linalg import expm_multiply


# --------------------------------------------------------------------------
# Classical NAND tree
# --------------------------------------------------------------------------
def nand(a: int, b: int) -> int:
    """Logical NAND of two bits."""
    return 1 - (a & b)


def nand_tree_value(bits):
    """Evaluate the NAND tree bottom-up.

    `bits` is a length-N=2**n list of leaf input values (0/1), where a 1 means
    the leaf is connected to its oracle node.  Returns (root_value, levels)
    where `levels[k]` is the list of node values at tree level k (level n is
    the leaves, level 0 is the root attached to the runway).
    """
    N = len(bits)
    n = int(round(np.log2(N)))
    if 2 ** n != N:
        raise ValueError(f"len(bits)={N} is not a power of two")

    levels = [None] * (n + 1)
    levels[n] = [int(b) & 1 for b in bits]          # leaves
    for level in range(n - 1, -1, -1):              # combine pairs upward
        above = levels[level + 1]
        levels[level] = [nand(above[2 * i], above[2 * i + 1])
                         for i in range(len(above) // 2)]
    root_value = levels[0][0]
    return root_value, levels


def bits_for_target(n: int, target: int):
    """Return a leaf assignment (length 2**n) whose NAND tree equals `target`.

    Brute-force search (fine for N = 2**n <= 16).  Falls back to all-0 / all-1
    if no exact match is found (should not happen for valid targets).
    """
    N = 2 ** n
    for code in range(2 ** N):
        bits = [(code >> i) & 1 for i in range(N)]
        if nand_tree_value(bits)[0] == target:
            return bits
    return [target] * N


# --------------------------------------------------------------------------
# Graph + Hamiltonian
# --------------------------------------------------------------------------
@dataclass
class Graph:
    """The full graph, its Hamiltonian, and indexing helpers for plotting."""
    n: int
    M: int
    L: int
    bits: list
    dim: int
    H: sp.spmatrix

    # runway: r values (sorted) and the matching state indices
    runway_r: np.ndarray
    runway_idx: np.ndarray            # index into state vector for each runway_r

    node0_index: int                  # state index of runway site r = 0
    right_indices: np.ndarray         # runway sites r >= 1  (measurement region)
    left_indices: np.ndarray          # runway sites r <= 0
    tree_indices: np.ndarray          # all tree + oracle nodes

    # for the tree panel: per-node (state index, x, y, kind)
    tree_nodes: list = field(default_factory=list)
    tree_edges: list = field(default_factory=list)   # (i, j) into tree_nodes

    # classical answer (cached)
    nand_value: int = 0


def build_graph(n: int, M: int, L: int, bits) -> Graph:
    """Build the runway + tree + oracle graph and its Hamiltonian H = -A."""
    N = 2 ** n
    bits = [int(b) & 1 for b in bits]
    if len(bits) != N:
        raise ValueError(f"expected {N} bits, got {len(bits)}")

    # Assign an integer state index to every node, keyed by a label.
    index = {}

    def idx(label):
        if label not in index:
            index[label] = len(index)
        return index[label]

    edges = []  # list of (label_a, label_b)

    # Runway chain r = -M .. M
    for r in range(-M, M):
        edges.append((("R", r), ("R", r + 1)))
    for r in range(-M, M + 1):
        idx(("R", r))  # make sure every runway site exists even if isolated

    # Tree: root ("T",0,0) attached to runway site 0
    edges.append((("R", 0), ("T", 0, 0)))
    for level in range(n):
        for pos in range(2 ** level):
            parent = ("T", level, pos)
            for child_pos in (2 * pos, 2 * pos + 1):
                edges.append((parent, ("T", level + 1, child_pos)))

    # Oracle nodes on connected leaves
    for leaf_pos in range(N):
        if bits[leaf_pos] == 1:
            edges.append((("T", n, leaf_pos), ("O", leaf_pos)))

    # Materialise indices for all referenced labels
    for a, b in edges:
        idx(a)
        idx(b)
    dim = len(index)

    # Hamiltonian H = -A  (real symmetric)
    rows, cols, data = [], [], []
    for a, b in edges:
        ia, ib = index[a], index[b]
        rows += [ia, ib]
        cols += [ib, ia]
        data += [-1.0, -1.0]
    H = sp.coo_matrix((data, (rows, cols)), shape=(dim, dim)).tocsr()

    # Runway bookkeeping
    runway_r = np.arange(-M, M + 1)
    runway_idx = np.array([index[("R", int(r))] for r in runway_r])
    node0_index = index[("R", 0)]
    right_indices = np.array([index[("R", int(r))] for r in range(1, M + 1)])
    left_indices = np.array([index[("R", int(r))] for r in range(-M, 1)])
    tree_indices = np.array(
        [i for lbl, i in index.items() if lbl[0] in ("T", "O")]
    )

    # Tree panel layout (its own local coordinates; root near origin).
    # Leaves spread horizontally; internal node x = mean of its children.
    leaf_x = {pos: (pos - (N - 1) / 2.0) * 1.0 for pos in range(N)}
    node_x = {(n, pos): leaf_x[pos] for pos in range(N)}
    for level in range(n - 1, -1, -1):
        for pos in range(2 ** level):
            node_x[(level, pos)] = 0.5 * (
                node_x[(level + 1, 2 * pos)] + node_x[(level + 1, 2 * pos + 1)]
            )

    tree_nodes = []          # dicts: index, x, y, kind, value
    node_to_local = {}       # label -> position in tree_nodes

    def add_node(label, x, y, kind, value=None):
        node_to_local[label] = len(tree_nodes)
        tree_nodes.append(dict(index=index[label], x=x, y=y,
                               kind=kind, value=value))

    # runway stub: site 0 marker below the root
    add_node(("R", 0), 0.0, -1.0, "runway", None)

    _, levels = nand_tree_value(bits)
    for level in range(n + 1):
        y = 1.0 + level                 # root at y=1, leaves at y=1+n
        for pos in range(2 ** level):
            kind = "leaf" if level == n else "tree"
            add_node(("T", level, pos), node_x[(level, pos)], y, kind,
                     value=levels[level][pos])
    for leaf_pos in range(N):
        if bits[leaf_pos] == 1:
            add_node(("O", leaf_pos), leaf_x[leaf_pos], 2.0 + n, "oracle", 1)

    tree_edges = []
    for a, b in edges:
        if a in node_to_local and b in node_to_local:
            tree_edges.append((node_to_local[a], node_to_local[b]))

    root_value, _ = nand_tree_value(bits)

    return Graph(
        n=n, M=M, L=L, bits=bits, dim=dim, H=H,
        runway_r=runway_r, runway_idx=runway_idx,
        node0_index=node0_index,
        right_indices=right_indices, left_indices=left_indices,
        tree_indices=tree_indices,
        tree_nodes=tree_nodes, tree_edges=tree_edges,
        nand_value=root_value,
    )


# --------------------------------------------------------------------------
# Initial packet and time evolution
# --------------------------------------------------------------------------
def initial_packet(graph: Graph, L: int) -> np.ndarray:
    """Right-moving packet  <r|psi0> = i**r / sqrt(L)  for -L+1 <= r <= 0."""
    psi = np.zeros(graph.dim, dtype=complex)
    for r in range(-L + 1, 1):
        psi[graph.runway_idx[r + graph.M]] = (1j ** r) / np.sqrt(L)
    nrm = np.linalg.norm(psi)
    if nrm > 0:
        psi /= nrm
    return psi


def propagate(H, psi0, t_max, num_frames):
    """Whole trajectory: returns array (num_frames, dim) and the time grid."""
    times = np.linspace(0.0, t_max, num_frames)
    states = expm_multiply(-1j * H, psi0,
                           start=0.0, stop=t_max, num=num_frames, endpoint=True)
    return times, states


def evolve_to(H, psi0, t):
    """Single state exp(-iHt)|psi0>."""
    return expm_multiply(-1j * t * H, psi0)


# --------------------------------------------------------------------------
# Observables
# --------------------------------------------------------------------------
def runway_profile(psi, graph: Graph) -> np.ndarray:
    """|<r|psi>|^2 indexed in the same order as graph.runway_r."""
    return np.abs(psi[graph.runway_idx]) ** 2


def p_right(psi, graph: Graph) -> float:
    return float(np.sum(np.abs(psi[graph.right_indices]) ** 2))


def p_left(psi, graph: Graph) -> float:
    return float(np.sum(np.abs(psi[graph.left_indices]) ** 2))


def p_tree(psi, graph: Graph) -> float:
    if len(graph.tree_indices) == 0:
        return 0.0
    return float(np.sum(np.abs(psi[graph.tree_indices]) ** 2))


# --------------------------------------------------------------------------
# Defaults
# --------------------------------------------------------------------------
# Paper's scaling coefficient in  L = gamma * sqrt(N).  The good-transmission
# window is |E| < 1/(16*sqrt(N)); since the packet's energy width is ~1/L, clean
# transmit/reflect contrast needs gamma >> 16.  gamma = 16 is the borderline
# value that still gives good contrast for n <= 4.
default_gamma = 16.0


def L_from_gamma(gamma: float, n: int, min_L: int = 8) -> int:
    """Packet length  L = gamma * sqrt(N),  N = 2**n  (the paper's scaling).

    Floored at min_L so the packet stays wide enough to be visible/meaningful.
    """
    return max(min_L, int(round(gamma * np.sqrt(2 ** n))))


def default_L(n: int, alpha: float = default_gamma, min_L: int = 24) -> int:
    """Packet length ~ alpha * sqrt(N), but at least min_L to stay visible.

    Thin wrapper over `L_from_gamma` (with alpha playing the role of gamma) that
    keeps a larger visibility floor for the self-test defaults.
    """
    return max(min_L, L_from_gamma(alpha, n, min_L=1))


def default_M(L: int) -> int:
    """Runway half-length large enough that nothing reflects off the far wall."""
    return int(round(3.2 * L))


def t_run(L: int) -> float:
    return L / 2.0


def default_t_max(L: int) -> float:
    """Run a bit past t_run so transmission/reflection is fully visible."""
    return 1.0 * L


# --------------------------------------------------------------------------
# Self-test
# --------------------------------------------------------------------------
def _run_instance(n, bits, L=None, M=None, verbose=True):
    L = L or default_L(n)
    M = M or default_M(L)
    g = build_graph(n, M, L, bits)
    psi0 = initial_packet(g, L)
    psi = evolve_to(g.H, psi0, t_run(L))
    pr = p_right(psi, g)
    classical = g.nand_value
    if verbose:
        print(f"  n={n} N={2**n} L={L} M={M} bits={bits} "
              f"-> NAND={classical}  P_right={pr:.3f}")
    return classical, pr


def _self_test():
    print("Self-test: quantum P_right(t_run) vs classical NAND value\n")
    rng = np.random.default_rng(0)
    failures = 0
    cases = []
    # all-0 and all-1 for n = 1..3
    for n in (1, 2, 3):
        N = 2 ** n
        cases.append((n, [0] * N))
        cases.append((n, [1] * N))
        cases.append((n, bits_for_target(n, 0)))
        cases.append((n, bits_for_target(n, 1)))
    # random instances
    for n in (2, 3):
        N = 2 ** n
        for _ in range(3):
            cases.append((n, list(rng.integers(0, 2, size=N))))

    for n, bits in cases:
        classical, pr = _run_instance(n, bits)
        predicted_high = classical == 1
        ok = (pr > 0.5) == predicted_high
        if not ok:
            failures += 1
            print("    ^ MISMATCH")
    print(f"\n{'PASS' if failures == 0 else 'FAIL'}: "
          f"{len(cases) - failures}/{len(cases)} instances agree.")
    return failures == 0


if __name__ == "__main__":
    _self_test()
