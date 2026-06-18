# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## What this is

An interactive simulation of the continuous-time quantum walk algorithm from
Farhi, Goldstone, Gutmann, *A Quantum Algorithm for the Hamiltonian NAND Tree*
(quant-ph/0702144; included as `quantum_algorithm_nand_tree_paper.pdf`). A
right-moving wave packet on a 1-D runway scatters off a binary tree encoding a
NAND instance: it **transmits** iff the tree evaluates to 1 and **reflects** iff
it evaluates to 0. Measuring the projector onto the right half of the runway at
`t_run = L/2` reads out the answer.

## Commands

```bash
pip install -r requirements.txt   # numpy, scipy, matplotlib

python nand_tree_demo.py          # interactive Tk dashboard
python nand_tree_physics.py       # headless self-test (no plotting)
```

The self-test (`_self_test` in `nand_tree_physics.py`) is the test suite: it
checks that the quantum measurement `P_right(t_run)` agrees with a classical
bottom-up NAND evaluation for all-0, all-1, forced, and random instances at
depths n=1..3. There is no separate test runner — run the module directly. To
exercise one instance, call `_run_instance(n, bits)` from that module.

## Architecture

Two layers, with a clean Tk-free boundary so the physics and plotting can run
headless:

- **`nand_tree_physics.py`** — pure simulation, no plotting. Builds the graph
  (runway sites `r=-M..M` + binary tree + per-leaf oracle nodes), the sparse
  Hamiltonian `H = -A` (minus the adjacency matrix), and the initial packet
  `<r|psi0> = i^r/sqrt(L)`. The whole trajectory `exp(-iHt)|psi0>` comes from a
  single `scipy.sparse.linalg.expm_multiply` call. Also evaluates the NAND tree
  classically (`nand_tree_value`, `bits_for_target`) for the verdict and the
  force-NAND buttons. Key type: the `Graph` dataclass, which carries `H` plus
  all the index bookkeeping (runway/left/right/tree state indices, tree-panel
  node layout) that the UI needs.

- **`nand_tree_demo.py`** — two sub-layers:
  - `PlotModel` owns the matplotlib `Figure`, the simulation state, and all
    drawing. **It has no Tk dependency** and is safe to render headless under
    Agg. It precomputes all `NUM_FRAMES` frames in `resimulate()`.
  - `DemoApp` is a thin native Tk control panel (sliders, buttons, leaf
    toggles) wrapping `PlotModel` via `FigureCanvasTkAgg`.

### Key design points

- **Resimulate vs. replay.** Changing `n`, `L`, `M`, or any leaf bit triggers a
  full `resimulate()` (rebuild graph + re-propagate). The time scrubber and Play
  only step through the precomputed `profiles`/`pr_t`/etc. arrays via
  `set_frame()`, so playback never recomputes and stays smooth. Slider motion is
  debounced (`_schedule_resim`, 150 ms) to coalesce drags into one resim.

- **Parameter scaling matters physically.** The algorithm only works when the
  packet's energy width `~1/L` sits inside the good-transmission window
  `|E| < 1/(16·sqrt(N))`, i.e. `L >> 16·sqrt(N)` (`default_L`, alpha=16). The
  runway half-length `M` (`default_M` ≈ 3.2·L) must be large enough that nothing
  reflects off the far wall within the run time. `bits_for_target` brute-forces
  over all `2^(2^n)` leaf assignments, so it is only viable for small n (n≤4 in
  the UI).

- **Index discipline.** Every graph node gets an integer state index via the
  `idx()` closure in `build_graph`; the runway/tree observables and the tree
  panel all slice the state vector through the index arrays stored on `Graph`.
  When adding nodes or observables, route them through that same indexing.
