"""Figures showing how ACO changed the search (PNG and PDF), from the CSV tables only.

Needs matplotlib (``pip install -e .[analysis]``). Run scripts/make_tables.py first (and
scripts/evaluate_runs.py for the accuracy figures). Writes to results/aggregate/analysis/figures/.

Colours: ToT+ACO blue, ToT-only orange, single-call baseline grey (colour-blind-safe pair, one
fixed colour per configuration in every figure).
"""

from __future__ import annotations

import argparse
import csv
from collections import defaultdict
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
BLUE, ORANGE, GREY, INK, MUTED, GRID = "#2a78d6", "#eb6834", "#898781", "#0b0b0b", "#52514e", "#e1e0d9"
GREEN = "#1baf7a"
COLOR = {"tot_aco": BLUE, "tot_only": ORANGE, "baseline": GREY, "sampling_control": GREEN}
LABEL = {"tot_aco": "ToT+ACO", "tot_only": "ToT-only", "baseline": "single call",
         "sampling_control": "sampling control"}
LEVEL_COLORS = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#6250d6"]

plt.rcParams.update(
    {
        "figure.dpi": 130,
        "savefig.dpi": 200,
        "font.size": 9,
        "axes.edgecolor": "#c3c2b7",
        "axes.labelcolor": MUTED,
        "xtick.color": MUTED,
        "ytick.color": MUTED,
        "text.color": INK,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "axes.grid": True,
        "grid.color": GRID,
        "grid.linewidth": 0.6,
        "axes.axisbelow": True,
        "legend.frameon": False,
    }
)


def read(path: Path):
    with path.open(encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def save(fig, out: Path, name: str) -> None:
    out.mkdir(parents=True, exist_ok=True)
    fig.savefig(out / f"{name}.png", bbox_inches="tight")
    fig.savefig(out / f"{name}.pdf", bbox_inches="tight")
    plt.close(fig)
    print(f"wrote {(out / name).relative_to(ROOT)}.png/.pdf")


def by_config_it(rows, key):
    """{config: (iterations, mean over seeds, SD over seeds)} for one numeric column."""
    acc = defaultdict(lambda: defaultdict(list))
    for r in rows:
        v = float(r[key]) if r[key] not in ("", "nan") else np.nan
        acc[r["config"]][int(r["it"])].append(v)
    out = {}
    for cfg, d in acc.items():
        its = sorted(d)
        out[cfg] = (np.array(its), np.array([np.nanmean(d[i]) for i in its]), np.array([np.nanstd(d[i], ddof=1) if len(d[i]) > 1 else 0 for i in its]))
    return out


def line_panel(ax, series, ylabel, title, pct=True):
    for cfg in ("tot_only", "tot_aco"):
        x, m, s = series[cfg]
        ax.plot(x, m * (100 if pct else 1), color=COLOR[cfg], lw=2, marker="o", ms=4, label=LABEL[cfg])
        ax.fill_between(x, (m - s) * (100 if pct else 1), (m + s) * (100 if pct else 1), color=COLOR[cfg], alpha=0.18, lw=0)
    ax.set_xlabel("iteration")
    ax.set_ylabel(ylabel)
    ax.set_title(title, loc="left", fontsize=9.5, color=INK)
    ax.set_xticks(range(1, 7))


def fig_verified_share(tables: Path, out: Path) -> None:
    per_it = read(tables / "per_iteration.csv")
    diff = read(tables / "verified_share_difference.csv")
    fig, axes = plt.subplots(1, 3, figsize=(11, 3.3))
    line_panel(axes[0], by_config_it(per_it, "verified_trace_share"), "traces scored 1.0 by the verifier (%)",
               "(a) share of verified traces, all tasks")
    axes[0].legend(loc="upper left")
    line_panel(axes[1], by_config_it(per_it, "verified_trace_share_after_first_success"), "traces scored 1.0 (%)",
               "(b) only tasks that already had a verified trace")
    x = np.array([int(r["it"]) for r in diff])
    d = np.array([float(r["difference"]) for r in diff]) * 100
    lo = np.array([float(r["ci_low"]) for r in diff]) * 100
    hi = np.array([float(r["ci_high"]) for r in diff]) * 100
    axes[2].axhline(0, color="#c3c2b7", lw=1)
    axes[2].plot(x, d, color=BLUE, lw=2, marker="o", ms=4)
    axes[2].fill_between(x, lo, hi, color=BLUE, alpha=0.18, lw=0)
    axes[2].set_xlabel("iteration")
    axes[2].set_ylabel("ACO minus ToT-only (pp)")
    axes[2].set_title("(c) difference, 95% CI over tasks", loc="left", fontsize=9.5)
    axes[2].set_xticks(range(1, 7))
    fig.suptitle("ACO concentrates the swarm on verified answers; uniform ToT-only stays flat", x=0.01, ha="left", fontsize=10.5)
    fig.tight_layout()
    save(fig, out, "verified_share_per_iteration")


def fig_scores(tables: Path, out: Path) -> None:
    per_it = read(tables / "per_iteration.csv")
    fig, axes = plt.subplots(1, 2, figsize=(7.6, 3.3))
    line_panel(axes[0], by_config_it(per_it, "best_score"), "verifier score", "(a) best trace of the iteration", pct=False)
    line_panel(axes[1], by_config_it(per_it, "mean_score"), "verifier score", "(b) mean over the 8 traces", pct=False)
    axes[0].legend(loc="lower right")
    fig.tight_layout()
    save(fig, out, "scores_per_iteration")


def fig_first_success(tables: Path, out: Path) -> None:
    per_it = read(tables / "per_iteration.csv")
    paths = by_config_it(per_it, "cum_distinct_paths")
    fig, axes = plt.subplots(1, 3, figsize=(11, 3.3))
    line_panel(axes[0], by_config_it(per_it, "cum_tasks_with_verified"), "tasks with a verified trace (%)", "(a) cumulative: first verified trace")
    axes[0].legend(loc="lower right")
    line_panel(axes[1], by_config_it(per_it, "cum_tasks_with_exact"), "tasks with an exact-match trace (%)", "(b) cumulative: first exact-match trace (oracle)")
    line_panel(axes[2], paths, "distinct paths per task (of 243)", "(c) distinct paths explored so far", pct=False)
    fig.tight_layout()
    save(fig, out, "first_success_and_paths")


def fig_pheromone(tables: Path, out: Path) -> None:
    ph = read(tables / "pheromone.csv")
    ph = [r for r in ph if r["subset"] == "after_success"]
    # heatmap: rows = 15 edges, columns = iteration (state before the update of that iteration; 7 = final)
    its = sorted({int(r["it"]) for r in ph})
    rows_keys = sorted({(int(r["level"]), int(r["branch_idx"]), r["branch"]) for r in ph})
    cells = defaultdict(list)
    for r in ph:
        cells[(rows_keys.index((int(r["level"]), int(r["branch_idx"]), r["branch"])), its.index(int(r["it"])))].append(float(r["mean_probability"]))
    grid = np.full((len(rows_keys), len(its)), np.nan)
    for (i, j), v in cells.items():
        grid[i, j] = float(np.mean(v))
    fig, (ax, ax2) = plt.subplots(1, 2, figsize=(11, 4.6), gridspec_kw={"width_ratios": [1.5, 1]})
    im = ax.imshow(grid, cmap="Blues", vmin=0.2, vmax=0.8, aspect="auto")
    ax.grid(False)
    ax.set_xticks(range(len(its)))
    ax.set_xticklabels([str(i) if i <= 6 else "end" for i in its])
    ax.set_yticks(range(len(rows_keys)))
    ax.set_yticklabels([f"L{l}  {b}" for l, _, b in rows_keys], fontsize=7.5)
    for i in range(grid.shape[0]):
        for j in range(grid.shape[1]):
            if not np.isnan(grid[i, j]):
                ax.text(j, i, f"{grid[i, j]:.2f}", ha="center", va="center", fontsize=6.5,
                        color="#ffffff" if grid[i, j] > 0.52 else INK)
    for k in range(1, 5):
        ax.axhline(3 * k - 0.5, color="#fcfcfb", lw=2)
    ax.set_xlabel("iteration (state the agents sample from; 'end' = after the last update)")
    ax.set_title("(a) mean branch probability, tasks that already had a deposit (uniform = 0.33)", loc="left", fontsize=9.5)
    fig.colorbar(im, ax=ax, fraction=0.035, pad=0.02)
    # entropy per level
    for lvl in range(1, 6):
        rr = [r for r in ph if int(r["level"]) == lvl and int(r["branch_idx"]) == 0]
        acc = defaultdict(list)
        for r in rr:
            acc[int(r["it"])].append(float(r["mean_norm_entropy"]))
        xs = sorted(acc)
        ax2.plot(xs, [np.mean(acc[i]) for i in xs], color=LEVEL_COLORS[lvl - 1], lw=2, marker="o", ms=3.5, label=f"L{lvl}")
    ax2.axhline(1.0, color="#c3c2b7", lw=1)
    ax2.set_ylim(0.5, 1.02)
    ax2.set_xlabel("iteration")
    ax2.set_ylabel("normalised entropy (1 = uniform)")
    ax2.set_xticks(range(1, 8))
    ax2.set_xticklabels([str(i) if i <= 6 else "end" for i in range(1, 8)])
    ax2.set_title("(b) how far each level moved from uniform", loc="left", fontsize=9.5)
    ax2.legend(ncol=5, loc="lower left", fontsize=8)
    fig.tight_layout()
    save(fig, out, "pheromone_heatmap_and_entropy")


def fig_pheromone_vs_quality(tables: Path, out: Path) -> None:
    rows = read(tables / "pheromone_vs_quality.csv")
    final = [r for r in rows if r["kind"] == "final"]
    traj = [r for r in rows if r["kind"] == "iteration"]
    fig, axes = plt.subplots(1, 3, figsize=(11.5, 3.5))
    # (a) mass on best branch per level, 3 seeds as dots + mean bar
    ax = axes[0]
    xs = np.arange(1, 6)
    means = []
    for lvl in xs:
        vals = [float(r["mass_on_best_branch"]) for r in final if int(r["level"]) == lvl]
        means.append(np.mean(vals))
        ax.scatter([lvl] * len(vals), vals, color=BLUE, s=14, zorder=3)
    ax.bar(xs, means, color=BLUE, alpha=0.35, width=0.55)
    ax.axhline(1 / 3, color=ORANGE, lw=1.5, ls="--")
    ax.text(0.6, 1 / 3 + 0.015, "chance (1/3)", color=ORANGE, ha="left", fontsize=8)
    ax.set_xticks(xs)
    ax.set_xticklabels([f"L{l}" for l in xs])
    ax.set_ylabel("final probability on the best branch")
    ax.set_ylim(0, 0.8)
    ax.set_title("(a) final probability on the best branch", loc="left", fontsize=9.5)
    # (b) best vs worst top-1
    ax = axes[1]
    for off, key, color, lab in ((-0.17, "top1_is_best", BLUE, "most probable = best"), (0.17, "top1_is_worst", ORANGE, "most probable = worst")):
        m = [np.mean([float(r[key]) for r in final if int(r["level"]) == lvl]) * 100 for lvl in xs]
        ax.bar(xs + off, m, width=0.32, color=color, label=lab)
    ax.axhline(100 / 3, color="#c3c2b7", lw=1, ls="--")
    ax.set_xticks(xs)
    ax.set_xticklabels([f"L{l}" for l in xs])
    ax.set_ylabel("tasks (%)")
    ax.set_title("(b) the pheromone's favourite branch is ...", loc="left", fontsize=9.5)
    ax.set_ylim(0, 80)
    ax.legend(loc="upper right", fontsize=8)
    # (c) trajectory
    ax = axes[2]
    for lvl in xs:
        acc = defaultdict(list)
        for r in traj:
            if int(r["level"]) == lvl:
                acc[int(r["it"])].append(float(r["mass_on_best_branch"]))
        its = sorted(acc)
        ys = [np.mean(acc[i]) for i in its]
        ax.plot(its, ys, color=LEVEL_COLORS[lvl - 1], lw=2, marker="o", ms=3.5)
        ax.text(its[-1] + 0.08, ys[-1], f"L{lvl}", color=INK, fontsize=8, va="center")
    ax.set_xlim(1.8, 6.5)
    ax.axhline(1 / 3, color="#c3c2b7", lw=1, ls="--")
    ax.set_xlabel("iteration")
    ax.set_ylabel("probability on the best branch")
    ax.set_xticks(range(2, 7))
    ax.set_title("(c) over the iterations", loc="left", fontsize=9.5)
    n = [r["tasks"] for r in final if int(r["level"]) == 1]
    fig.suptitle(
        "Does the pheromone point at branches that really work? Reference: per-branch verified rate in the independent "
        f"ToT-only runs; only informative tasks with a deposit (L1: {min(n)}–{max(n)} tasks per seed)",
        x=0.01, ha="left", fontsize=9.5)
    fig.tight_layout()
    save(fig, out, "pheromone_vs_branch_quality")


def fig_accuracy(main: Path, out: Path) -> None:
    table = read(main / "table_main.csv")
    cats = read(main / "per_category.csv")
    # (a) selection rules: ToT-only vs ACO
    rules = ["verified_last3", "verified_last2", "verified_vote_last3", "best_of_all", "early_stop",
             "best_of_last3", "best_of_last2", "last_iteration", "majority_vote"]
    names = {"verified_last3": "verified last 3\n(headline)", "verified_last2": "verified last 2",
             "verified_vote_last3": "verified vote\nlast 3", "best_of_all": "best of\nall",
             "early_stop": "early stop", "best_of_last3": "best of\nlast 3", "best_of_last2": "best of\nlast 2",
             "last_iteration": "last iteration\n(v1 rule)", "majority_vote": "majority vote\n(no verifier)"}
    get = {(r["config"], r["rule"]): r for r in table}
    fig, axes = plt.subplots(1, 2, figsize=(14, 4.2), gridspec_kw={"width_ratios": [1.35, 1]})
    ax = axes[0]
    x = np.arange(len(rules))
    for off, cfg in ((-0.2, "tot_only"), (0.2, "tot_aco")):
        m = [float(get[(cfg, r)]["exact_match_mean"]) * 100 for r in rules]
        s = [float(get[(cfg, r)]["exact_match_sd"]) * 100 for r in rules]
        ax.bar(x + off, m, width=0.38, color=COLOR[cfg], label=LABEL[cfg], yerr=s, error_kw={"lw": 1, "capsize": 2, "ecolor": MUTED})
    base = float(get[("baseline", "single_call")]["exact_match_mean"]) * 100
    ax.axhline(base, color=GREY, lw=1.5, ls="--")
    ax.text(-0.45, base + 1.2, f"single-call baseline {base:.1f}%", ha="left", color=MUTED, fontsize=8,
            bbox={"facecolor": "#fcfcfb", "edgecolor": "none", "alpha": 0.9, "pad": 1.5}, zorder=5)
    ctrl = float(get[("sampling_control", "best_of_samples")]["exact_match_mean"]) * 100
    ax.axhline(ctrl, color=GREEN, lw=1.5, ls="--")
    ax.text(len(rules) - 0.45, ctrl + 1.2, f"sampling control (best of 48, verifier) {ctrl:.1f}%", ha="right", color=MUTED,
            fontsize=8, bbox={"facecolor": "#fcfcfb", "edgecolor": "none", "alpha": 0.9, "pad": 1.5}, zorder=5)
    ax.set_ylim(0, 72)
    ax.set_xticks(x)
    ax.set_xticklabels([names[r] for r in rules], fontsize=6.5)
    ax.set_ylabel("exact match (%), mean ± SD over 3 seeds")
    ax.set_title("(a) the same logged runs under different final-selection rules", loc="left", fontsize=9.5)
    ax.legend(loc="upper right", ncol=2)
    # (b) per category headline
    ax = axes[1]
    order = ["2p1d", "2p2d", "2p3d", "2p4d", "2p5d", "3p1d", "4p1d", "5p1d", "6p1d", "7p1d"]
    cget = {(r["config"], r["rule"], r["category"]): float(r["exact_match_mean"]) * 100 for r in cats}
    x = np.arange(len(order))
    for off, cfg, rule in ((-0.3, "baseline", "single_call"), (-0.1, "sampling_control", "best_of_samples"),
                           (0.1, "tot_only", "verified_last3"), (0.3, "tot_aco", "verified_last3")):
        ax.bar(x + off, [cget[(cfg, rule, c)] for c in order], width=0.19, color=COLOR[cfg], label=LABEL[cfg])
    ax.set_xticks(x)
    ax.set_xticklabels(order, fontsize=7.5)
    ax.set_xlabel("people × days")
    ax.set_ylabel("exact match (%)")
    ax.set_title("(b) per category, headline rule (40 tasks each)", loc="left", fontsize=9.5)
    ax.set_ylim(0, 100)
    ax.legend(loc="upper right", ncol=2, fontsize=8)
    fig.tight_layout()
    save(fig, out, "accuracy_by_rule_and_category")


def fig_sampling_control(main: Path, tables: Path, out: Path) -> None:
    """ToT structure against 'sampling plus verifier': accuracy, tasks with a verified candidate, diversity."""
    table = read(main / "table_main.csv")
    get = {(r["config"], r["rule"]): r for r in table}
    rows = [("baseline", "single_call"), ("sampling_control", "best_of_samples"),
            ("tot_only", "verified_last3"), ("tot_aco", "verified_last3")]
    div = read(tables / "candidate_diversity.csv")
    cfgs = [c for c, _ in rows]
    fig, axes = plt.subplots(1, 3, figsize=(13.5, 4.0))

    ax = axes[0]
    for i, (cfg, rule) in enumerate(rows):
        r = get[(cfg, rule)]
        seeds = [float(r[k]) * 100 for k in ("seed_0", "seed_1", "seed_2") if r.get(k) not in (None, "")]
        ax.bar(i, float(r["exact_match_mean"]) * 100, width=0.6, color=COLOR[cfg])
        ax.scatter([i] * len(seeds), seeds, color=INK, s=9, zorder=5)
        ax.text(i, float(r["exact_match_mean"]) * 100 + 2.2, f"{float(r['exact_match_mean']) * 100:.1f}", ha="center", fontsize=8.5)
    ax.axhline(float(get[("verifier_solver", "verifier_best_slot")]["exact_match_mean"]) * 100, color=MUTED, lw=1, ls=":")
    ax.text(3.4, float(get[("verifier_solver", "verifier_best_slot")]["exact_match_mean"]) * 100 + 1.5,
            "verifier as solver 92.8%", ha="right", fontsize=8, color=MUTED)
    ax.set_xticks(range(4))
    ax.set_xticklabels(["single call", "sampling control\n(48 samples)", "ToT-only", "ToT+ACO"], fontsize=8)
    ax.set_ylim(0, 100)
    ax.set_ylabel("exact match (%), dots = seeds")
    ax.set_title("(a) accuracy (control: best of 48 by verifier; ToT: headline rule)", loc="left", fontsize=9)

    def per_cfg(metric):
        return {c: [float(r[metric]) for r in div if r["config"] == c] for c in cfgs if c != "baseline"}

    ax = axes[1]
    v = per_cfg("tasks_with_a_verified_candidate")
    for i, cfg in enumerate(cfgs[1:]):
        ax.bar(i, np.mean(v[cfg]) * 100, width=0.6, color=COLOR[cfg])
        ax.scatter([i] * len(v[cfg]), np.array(v[cfg]) * 100, color=INK, s=9, zorder=5)
        ax.text(i, np.mean(v[cfg]) * 100 + 2, f"{np.mean(v[cfg]) * 100:.1f}", ha="center", fontsize=8.5)
    ax.set_xticks(range(3))
    ax.set_xticklabels(["sampling control", "ToT-only", "ToT+ACO"], fontsize=8)
    ax.set_ylim(0, 100)
    ax.set_ylabel("% of tasks")
    ax.set_title("(b) tasks with a verified candidate among the 48", loc="left", fontsize=9)

    ax = axes[2]
    d = per_cfg("mean_distinct_answers_per_task")
    for i, cfg in enumerate(cfgs[1:]):
        ax.bar(i, np.mean(d[cfg]), width=0.6, color=COLOR[cfg])
        ax.scatter([i] * len(d[cfg]), d[cfg], color=INK, s=9, zorder=5)
        ax.text(i, np.mean(d[cfg]) + 1.0, f"{np.mean(d[cfg]):.1f}", ha="center", fontsize=8.5)
    ax.set_xticks(range(3))
    ax.set_xticklabels(["sampling control", "ToT-only", "ToT+ACO"], fontsize=8)
    ax.set_ylabel("distinct answers per task")
    ax.set_ylim(0, max(max(x) for x in d.values()) * 1.2)
    ax.set_title("(c) how many different answers the 48 candidates contain", loc="left", fontsize=9)
    fig.tight_layout()
    save(fig, out, "sampling_control_comparison")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--tables", default=str(ROOT / "results" / "aggregate" / "analysis" / "tables"))
    parser.add_argument("--main", default=str(ROOT / "results" / "aggregate" / "main"))
    parser.add_argument("--out", default=str(ROOT / "results" / "aggregate" / "analysis" / "figures"))
    args = parser.parse_args()
    tables, main_dir, out = Path(args.tables), Path(args.main), Path(args.out)
    fig_verified_share(tables, out)
    fig_scores(tables, out)
    fig_first_success(tables, out)
    fig_pheromone(tables, out)
    fig_pheromone_vs_quality(tables, out)
    fig_accuracy(main_dir, out)
    fig_sampling_control(main_dir, tables, out)


if __name__ == "__main__":
    main()
