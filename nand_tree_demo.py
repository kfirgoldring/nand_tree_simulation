"""
Interactive demo of the Hamiltonian NAND-tree quantum algorithm
(Farhi, Goldstone, Gutmann, quant-ph/0702144).

Run:

    python nand_tree_demo.py

A right-moving wave packet starts on the LEFT of the runway and evolves under
H = -A (minus the adjacency matrix of runway + binary tree + oracle nodes).
At node 0 it meets a tree encoding a NAND instance.  It TRANSMITS to the right
iff the NAND tree evaluates to 1, and REFLECTS iff it evaluates to 0.  At
t_run = L/2 we measure the projector onto the right half of the runway.

The window has the matplotlib plots on the right and a native Tk control panel
on the left:

  * Sliders:  n (tree depth, N=2**n leaves),  gamma (sets packet length via the
    paper's scaling L = gamma*sqrt(N)),  M/L (runway half-length as a multiple of
    L),  and the time scrubber.
  * Play / Pause:  animate the precomputed propagation.
  * All 0 / All 1 / force NAND=1 / force NAND=0:  quick instances.
  * Leaf toggles:  click a leaf box to flip its input bit (green = 1 = edge
    present).
  * Show wave:  toggle the light-blue Re<r|psi> overlay on the runway.

Changing n, L, M, or any leaf bit re-runs the simulation; the time scrubber and
Play only replay the precomputed frames, so playback stays smooth.

The plotting/simulation core (`PlotModel`) has no Tk dependency, so it can be
rendered head-less for testing; `DemoApp` is the thin Tk layer on top.
"""

from __future__ import annotations

import numpy as np
from matplotlib.figure import Figure
from matplotlib.gridspec import GridSpec

import nand_tree_physics as P


# colours
C_PROB = "#1f77b4"      # |psi|^2 on runway
C_WAVE = "#9ecae1"      # Re(psi)
C_RIGHT = "#2ca02c"     # right / transmit
C_LEFT = "#d62728"      # left / reflect
C_TREE = "#7f7f7f"      # tree probability
NUM_FRAMES = 200


# --------------------------------------------------------------------------
# Plot + simulation core (no Tk; safe to render head-less under Agg)
# --------------------------------------------------------------------------
class PlotModel:
    """Owns the figure, the simulation state, and all drawing."""

    def __init__(self, fig: Figure):
        self.fig = fig
        self.n = 2
        self.gamma = P.default_gamma          # L = gamma * sqrt(N)
        self.L = P.L_from_gamma(self.gamma, self.n)
        self.m_ratio = 2.6
        self.bits = P.bits_for_target(self.n, 1)   # start with a transmit case
        self.frame = 0
        self.show_wave = True

        self._build_axes()
        self.resimulate()

    # ------------------------------------------------------------ figure
    def _build_axes(self):
        gs = GridSpec(2, 2, figure=self.fig,
                      left=0.08, right=0.95, top=0.93, bottom=0.09,
                      height_ratios=[1.25, 1.0], width_ratios=[2.2, 1.0],
                      hspace=0.32, wspace=0.24)
        self.ax_runway = self.fig.add_subplot(gs[0, :])
        self.ax_time = self.fig.add_subplot(gs[1, 0])
        self.ax_tree = self.fig.add_subplot(gs[1, 1])
        self.ax_wave = self.ax_runway.twinx()

        self.fig.suptitle("Hamiltonian NAND tree — continuous-time quantum walk",
                          fontsize=13, fontweight="bold")

        # runway
        self.ax_runway.set_xlabel("runway site  r")
        self.ax_runway.set_ylabel(r"$|\langle r|\psi\rangle|^2$", color=C_PROB)
        self.ax_wave.set_ylabel(r"Re$\langle r|\psi\rangle$", color=C_WAVE)
        self.ax_wave.tick_params(axis="y", colors=C_WAVE)
        self.ax_runway.tick_params(axis="y", colors=C_PROB)
        self.prob_line, = self.ax_runway.plot([], [], color=C_PROB, lw=1.8,
                                              zorder=3)
        self.wave_line, = self.ax_wave.plot([], [], color=C_WAVE, lw=0.8,
                                            alpha=0.9, zorder=1)
        self._fill = None

        # time panel
        self.ax_time.set_xlabel("time  t")
        self.ax_time.set_ylabel("probability")
        self.ax_time.set_ylim(-0.02, 1.02)
        self.pr_line, = self.ax_time.plot([], [], color=C_RIGHT, lw=2,
                                          label="P right (transmit)")
        self.pl_line, = self.ax_time.plot([], [], color=C_LEFT, lw=2,
                                          label="P left (reflect)")
        self.ptree_line, = self.ax_time.plot([], [], color=C_TREE, lw=1.2,
                                             ls="--", label="P in tree")
        self.tnow_line = self.ax_time.axvline(0, color="k", lw=1.0, alpha=0.6)
        self.trun_line = self.ax_time.axvline(0, color="purple", lw=1.2,
                                              ls=":", alpha=0.9,
                                              label="t_run = L/2")
        self.ax_time.legend(loc="upper left", fontsize=8)

        # tree panel
        self.ax_tree.set_xticks([])
        self.ax_tree.set_yticks([])

    # ------------------------------------------------------------ sim
    def resimulate(self):
        self.M = max(int(round(self.m_ratio * self.L)), self.L + 5)
        g = P.build_graph(self.n, self.M, self.L, self.bits)
        psi0 = P.initial_packet(g, self.L)
        # adapt run time to M so nothing reflects off the far wall
        t_max = max(P.t_run(self.L) * 1.05, (0.85 * self.M - self.L / 2) / 2.0)
        times, states = P.propagate(g.H, psi0, t_max, NUM_FRAMES)

        self.graph = g
        self.times = times
        self.t_max = t_max
        self.profiles = np.abs(states[:, g.runway_idx]) ** 2
        self.waves = np.real(states[:, g.runway_idx])
        self.pr_t = np.sum(np.abs(states[:, g.right_indices]) ** 2, axis=1)
        self.pl_t = np.sum(np.abs(states[:, g.left_indices]) ** 2, axis=1)
        self.ptree_t = (np.sum(np.abs(states[:, g.tree_indices]) ** 2, axis=1)
                        if len(g.tree_indices) else np.zeros(NUM_FRAMES))
        self.tree_state_idx = np.array([nd["index"] for nd in g.tree_nodes])
        self.tree_probs = np.abs(states[:, self.tree_state_idx]) ** 2
        self.idx_trun = int(np.argmin(np.abs(times - P.t_run(self.L))))

        self._setup_static_axes()
        self.frame = 0
        self.set_frame(0)

    def _setup_static_axes(self):
        g = self.graph
        r = g.runway_r

        self.ax_runway.set_xlim(r[0], r[-1])
        pmax = max(self.profiles.max(), 1e-6)
        self.ax_runway.set_ylim(0, 1.15 * pmax)
        wmax = max(np.abs(self.waves).max(), 1e-6)
        self.ax_wave.set_ylim(-1.2 * wmax, 1.2 * wmax)

        for art in list(self.ax_runway.patches):
            art.remove()
        for ln in list(self.ax_runway.lines):
            if ln is not self.prob_line:
                ln.remove()
        for txt in list(self.ax_runway.texts):
            txt.remove()
        self.ax_runway.axvspan(0.5, r[-1], color=C_RIGHT, alpha=0.07, zorder=0)
        self.ax_runway.axvspan(r[0], 0.5, color=C_LEFT, alpha=0.05, zorder=0)
        self.ax_runway.axvline(0, color="k", lw=1.0, alpha=0.5, zorder=2)
        self.ax_runway.text(0.5 * r[-1], 1.07 * pmax,
                            "measurement region (r > 0)",
                            color=C_RIGHT, ha="center", fontsize=9)

        self.ax_time.set_xlim(0, self.t_max)
        self.pr_line.set_data(self.times, self.pr_t)
        self.pl_line.set_data(self.times, self.pl_t)
        self.ptree_line.set_data(self.times, self.ptree_t)
        self.trun_line.set_xdata([P.t_run(self.L), P.t_run(self.L)])

        self._draw_tree_static()

    def _draw_tree_static(self):
        g = self.graph
        ax = self.ax_tree
        ax.clear()
        ax.set_xticks([])
        ax.set_yticks([])
        ax.set_title("tree + oracle  (size/colour = probability)")
        xs = [nd["x"] for nd in g.tree_nodes]
        ys = [nd["y"] for nd in g.tree_nodes]
        ax.set_xlim(min(xs) - 1, max(xs) + 1)
        ax.set_ylim(min(ys) - 1.0, max(ys) + 1.0)

        for i, j in g.tree_edges:
            a, b = g.tree_nodes[i], g.tree_nodes[j]
            ax.plot([a["x"], b["x"]], [a["y"], b["y"]],
                    color="0.6", lw=1.0, zorder=1)
        for nd in g.tree_nodes:
            if nd["kind"] in ("tree", "leaf") and nd["value"] is not None:
                ax.text(nd["x"], nd["y"] - 0.32, str(nd["value"]),
                        ha="center", va="top", fontsize=7, color="0.3")
            if nd["kind"] == "runway":
                ax.text(nd["x"], nd["y"] - 0.1, "runway r=0", ha="center",
                        va="top", fontsize=8, color="k")
        self.tree_scatter = ax.scatter(xs, ys, s=60, c=np.zeros(len(xs)),
                                       cmap="viridis", vmin=0, vmax=1,
                                       edgecolors="k", linewidths=0.5, zorder=3)

    # ------------------------------------------------------------ per-frame
    def set_frame(self, f):
        self.frame = f = int(np.clip(f, 0, NUM_FRAMES - 1))
        g = self.graph
        r = g.runway_r

        prof = self.profiles[f]
        self.prob_line.set_data(r, prof)
        self.wave_line.set_data(r, self.waves[f])
        self.wave_line.set_visible(self.show_wave)
        self.ax_wave.set_visible(self.show_wave)
        if self._fill is not None:
            self._fill.remove()
        self._fill = self.ax_runway.fill_between(r, 0, prof, color=C_PROB,
                                                 alpha=0.25, zorder=2)

        self.tnow_line.set_xdata([self.times[f], self.times[f]])

        tp = self.tree_probs[f]
        norm = tp / max(tp.max(), 1e-9)
        self.tree_scatter.set_array(norm)
        self.tree_scatter.set_sizes(40 + 400 * norm)

    # ------------------------------------------------------------ readouts
    def set_show_wave(self, on: bool):
        self.show_wave = bool(on)
        self.set_frame(self.frame)

    def verdict(self):
        """(headline_text, colour, detail_text) for the verdict card."""
        nand = self.graph.nand_value
        pr_run = self.pr_t[self.idx_trun]
        reads = 1 if pr_run > 0.5 else 0
        ok = "✓" if reads == nand else "✗"
        if nand == 1:
            headline, colour = "NAND = 1   →   TRANSMIT", C_RIGHT
        else:
            headline, colour = "NAND = 0   →   REFLECT", C_LEFT
        detail = (f"measured  P_right(t_run) = {pr_run:.3f}\n"
                  f"reads NAND = {reads}   {ok}    "
                  f"(predicted |T(0)|² = {nand})")
        return headline, colour, detail

    def params_text(self):
        g = self.graph
        f = self.frame
        return (f"n = {self.n}    N = {2**self.n} leaves    "
                f"γ = {self.gamma:g}    L = γ√N = {self.L}    "
                f"M = {self.M}    dim = {g.dim}\n"
                f"t_run = L/2 = {P.t_run(self.L):.1f}        "
                f"current t = {self.times[f]:.1f}\n"
                f"P_right = {self.pr_t[f]:.3f}    "
                f"P_left = {self.pl_t[f]:.3f}    "
                f"P_tree = {self.ptree_t[f]:.3f}")


# --------------------------------------------------------------------------
# Tk control panel + embedded figure
# --------------------------------------------------------------------------
def run():
    import tkinter as tk
    from tkinter import ttk
    from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg

    app = DemoApp(tk, ttk, FigureCanvasTkAgg)
    app.root.mainloop()


class DemoApp:
    PANEL_W = 330           # control-panel width in px
    PLAY_MS = 40            # animation tick interval

    def __init__(self, tk, ttk, FigureCanvasTkAgg):
        self.tk = tk
        self.ttk = ttk
        self.playing = False
        self._resim_job = None
        self._play_job = None
        self._syncing = False

        self.root = tk.Tk()
        self.root.title("Hamiltonian NAND Tree — quantum walk demo")
        self.root.geometry("1320x820")

        # figure + model
        self.fig = Figure(figsize=(11, 8), dpi=100)
        self.model = PlotModel(self.fig)

        # layout: control panel (left) | canvas (right)
        self.panel = ttk.Frame(self.root, width=self.PANEL_W, padding=10)
        self.panel.pack(side=tk.LEFT, fill=tk.Y)
        self.panel.pack_propagate(False)

        canvas_frame = ttk.Frame(self.root)
        canvas_frame.pack(side=tk.RIGHT, fill=tk.BOTH, expand=True)
        self.canvas = FigureCanvasTkAgg(self.fig, master=canvas_frame)
        self.canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True)

        self._build_panel()
        self._refresh_readouts()
        self.canvas.draw()

    # ----------------------------------------------------------- panel
    def _build_panel(self):
        tk, ttk = self.tk, self.ttk
        pad = dict(padx=4, pady=3)

        # ----- verdict card
        card = ttk.LabelFrame(self.panel, text="result", padding=8)
        card.pack(fill=tk.X, pady=(0, 8))
        self.lbl_verdict = tk.Label(card, text="", font=("Segoe UI", 13, "bold"),
                                    anchor="w", justify="left")
        self.lbl_verdict.pack(fill=tk.X)
        self.lbl_detail = tk.Label(card, text="", font=("Consolas", 9),
                                   anchor="w", justify="left", fg="#333")
        self.lbl_detail.pack(fill=tk.X, pady=(4, 0))

        # ----- parameters
        self.lbl_params = tk.Label(self.panel, text="", font=("Consolas", 9),
                                   anchor="w", justify="left", fg="#333")
        self.lbl_params.pack(fill=tk.X, pady=(0, 8))

        # ----- sliders
        sl = ttk.LabelFrame(self.panel, text="parameters", padding=8)
        sl.pack(fill=tk.X, pady=(0, 8))
        sl.columnconfigure(1, weight=1)
        self.s_n = self._add_slider(sl, 0, "n (depth)", 1, 4,
                                    self.model.n, self._on_n, fmt="{:.0f}")
        self.s_gamma = self._add_slider(sl, 1, "γ  (L=γ√N)", 2, 40,
                                        self.model.gamma, self._on_struct,
                                        fmt="{:.1f}")
        self.s_M = self._add_slider(sl, 2, "M / L", 1.8, 4.0,
                                    self.model.m_ratio, self._on_struct,
                                    fmt="{:.1f}")
        self.s_t = self._add_slider(sl, 3, "time", 0, NUM_FRAMES - 1,
                                    0, self._on_time, fmt="{:.0f}", live=True)

        # ----- playback + show wave
        row = ttk.Frame(self.panel)
        row.pack(fill=tk.X, pady=(0, 8))
        self.b_play = ttk.Button(row, text="▶ Play", command=self._toggle_play)
        self.b_play.pack(side=tk.LEFT)
        self.b_restart = ttk.Button(row, text="⟲ Restart",
                                    command=self._restart)
        self.b_restart.pack(side=tk.LEFT, padx=(6, 0))
        self.wave_var = tk.BooleanVar(value=True)
        ttk.Checkbutton(row, text="show wave", variable=self.wave_var,
                        command=self._on_wave).pack(side=tk.RIGHT)

        # ----- instances
        inst = ttk.LabelFrame(self.panel, text="instances", padding=8)
        inst.pack(fill=tk.X, pady=(0, 8))
        for c in range(2):
            inst.columnconfigure(c, weight=1)
        ttk.Button(inst, text="All 0",
                   command=lambda: self._set_bits([0] * (2 ** self.model.n))
                   ).grid(row=0, column=0, sticky="ew", **pad)
        ttk.Button(inst, text="All 1",
                   command=lambda: self._set_bits([1] * (2 ** self.model.n))
                   ).grid(row=0, column=1, sticky="ew", **pad)
        ttk.Button(inst, text="force NAND = 1",
                   command=lambda: self._set_bits(
                       P.bits_for_target(self.model.n, 1))
                   ).grid(row=1, column=0, sticky="ew", **pad)
        ttk.Button(inst, text="force NAND = 0",
                   command=lambda: self._set_bits(
                       P.bits_for_target(self.model.n, 0))
                   ).grid(row=1, column=1, sticky="ew", **pad)

        # ----- leaf toggles
        self.leaf_frame = ttk.LabelFrame(
            self.panel, text="leaf inputs  (green = 1, click to toggle)",
            padding=8)
        self.leaf_frame.pack(fill=tk.X)
        self.leaf_buttons = []
        self._build_leaf_buttons()

    def _add_slider(self, parent, row, label, lo, hi, init, cb, fmt, live=False):
        tk, ttk = self.tk, self.ttk
        ttk.Label(parent, text=label).grid(row=row, column=0, sticky="w",
                                           padx=(0, 6), pady=2)
        val = tk.StringVar(value=fmt.format(init))
        scale = ttk.Scale(parent, from_=lo, to=hi, value=init,
                          orient=tk.HORIZONTAL)
        scale.grid(row=row, column=1, sticky="ew", pady=2)
        lab = ttk.Label(parent, textvariable=val, width=5, anchor="e")
        lab.grid(row=row, column=2, sticky="e", padx=(6, 0))
        scale._fmt = fmt
        scale._valvar = val
        scale.configure(command=lambda v, s=scale, c=cb, lv=live:
                        self._slider_changed(s, c, lv))
        return scale

    def _slider_changed(self, scale, cb, live):
        scale._valvar.set(scale._fmt.format(float(scale.get())))
        if self._syncing:
            return
        cb(live)

    def _build_leaf_buttons(self):
        tk = self.tk
        for b in self.leaf_buttons:
            b.destroy()
        self.leaf_buttons = []
        N = 2 ** self.model.n
        per_row = 8
        for i in range(N):
            r, c = divmod(i, per_row)
            b = tk.Button(self.leaf_frame, width=2, text=str(self.model.bits[i]),
                          command=lambda k=i: self._toggle_leaf(k))
            b.grid(row=r, column=c, padx=2, pady=2)
            self.leaf_buttons.append(b)
        self._refresh_leaf_buttons()

    def _refresh_leaf_buttons(self):
        for i, b in enumerate(self.leaf_buttons):
            on = self.model.bits[i] == 1
            b.configure(text=str(self.model.bits[i]),
                        bg=(C_RIGHT if on else "#dddddd"),
                        activebackground=(C_RIGHT if on else "#cccccc"),
                        fg=("white" if on else "black"))

    # ----------------------------------------------------------- callbacks
    def _on_n(self, _live):
        self._schedule_resim(structural_n=True)

    def _on_struct(self, _live):
        self._schedule_resim()

    def _on_time(self, _live):
        self.model.set_frame(int(round(float(self.s_t.get()))))
        self._refresh_readouts()
        self.canvas.draw_idle()

    def _on_wave(self):
        self.model.set_show_wave(self.wave_var.get())
        self.canvas.draw_idle()

    def _toggle_leaf(self, k):
        self.model.bits[k] = 1 - self.model.bits[k]
        self._refresh_leaf_buttons()
        self._do_resim()

    def _set_bits(self, bits):
        self.model.bits = list(bits)
        self._refresh_leaf_buttons()
        self._do_resim()

    def _toggle_play(self):
        self.playing = not self.playing
        self.b_play.configure(text="❚❚ Pause" if self.playing else "▶ Play")
        if self.playing:
            self._play_step()
        elif self._play_job is not None:
            self.root.after_cancel(self._play_job)
            self._play_job = None

    def _restart(self):
        # stop playback and rewind the animation to the first frame
        if self.playing:
            self._toggle_play()
        self.model.set_frame(0)
        self._sync_time_slider(0)
        self._refresh_readouts()
        self.canvas.draw_idle()

    def _play_step(self):
        if not self.playing:
            return
        nxt = (self.model.frame + 1) % NUM_FRAMES
        self.model.set_frame(nxt)
        self._sync_time_slider(nxt)
        self._refresh_readouts()
        self.canvas.draw_idle()
        self._play_job = self.root.after(self.PLAY_MS, self._play_step)

    # ----------------------------------------------------------- resim
    def _schedule_resim(self, structural_n=False):
        # debounce: coalesce rapid slider motion into one resim
        if self._resim_job is not None:
            self.root.after_cancel(self._resim_job)
        self._resim_job = self.root.after(
            150, lambda: self._do_resim(structural_n))

    def _do_resim(self, structural_n=False):
        self._resim_job = None
        m = self.model
        if structural_n:
            new_n = int(round(float(self.s_n.get())))
            if new_n != m.n:
                m.n = new_n
                m.bits = P.bits_for_target(m.n, 1)
                self._build_leaf_buttons()
        m.gamma = round(float(self.s_gamma.get()), 1)
        m.m_ratio = round(float(self.s_M.get()), 1)
        m.L = P.L_from_gamma(m.gamma, m.n)   # L = gamma * sqrt(N)
        m.resimulate()
        self._sync_time_slider(0)
        self._refresh_readouts()
        self.canvas.draw_idle()

    # ----------------------------------------------------------- helpers
    def _sync_slider(self, scale, value):
        self._syncing = True
        scale.set(value)
        scale._valvar.set(scale._fmt.format(value))
        self._syncing = False

    def _sync_time_slider(self, frame):
        self._sync_slider(self.s_t, frame)

    def _refresh_readouts(self):
        headline, colour, detail = self.model.verdict()
        self.lbl_verdict.configure(text=headline, fg=colour)
        self.lbl_detail.configure(text=detail)
        self.lbl_params.configure(text=self.model.params_text())


def main():
    run()


if __name__ == "__main__":
    main()
