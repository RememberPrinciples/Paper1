"""
Demo simulation for edge-side node-budget allocation and cooperative verification
under a history-curve + Jaccard-similarity heuristic.

This integrated version exports both numerical CSV files and SVG figures.

Run:
    python edge_budget_similarity_demo.py

Default outputs:
    ./edge_budget_similarity_svg_output/
or, in this notebook/container:
    /mnt/data/edge_budget_similarity_svg_output/
"""
import math
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


# ----------------------------
# Global configuration
# ----------------------------
SEED = 7
MODEL_NAMES = ["Fast", "Balanced", "Accurate", "Diverse"]
OLD_MODELS = ["Fast", "Balanced", "Accurate"]
NEW_MODEL = "Diverse"
NEW_MODEL_JOIN_T = 80

N_CURVE = 48        # maximum recorded draft-node count for each N_acc curve
T_VER = 60.0        # ms, target-model verification time per round
PAYLOAD_MB = 0.08   # toy communication payload per draft node

TRUE_PARAMS = {
    "Fast":     {"A": 5.2, "b": 0.23, "draft_base": 1.10, "delta": 2.0, "Bmean": 100},
    "Balanced": {"A": 6.8, "b": 0.16, "draft_base": 1.70, "delta": 2.2, "Bmean": 100},
    "Accurate": {"A": 8.5, "b": 0.10, "draft_base": 2.60, "delta": 2.5, "Bmean": 100},
    "Diverse":  {"A": 6.2, "b": 0.18, "draft_base": 1.45, "delta": 2.3, "Bmean": 100},
}

PAIR_J = {
    ("Fast", "Balanced"): 0.58,
    ("Fast", "Accurate"): 0.33,
    ("Fast", "Diverse"): 0.10,
    ("Balanced", "Accurate"): 0.38,
    ("Balanced", "Diverse"): 0.22,
    ("Accurate", "Diverse"): 0.27,
}

TRUE_J = {}
for a in MODEL_NAMES:
    for b in MODEL_NAMES:
        if a == b:
            TRUE_J[(a, b)] = 1.0
        else:
            TRUE_J[(a, b)] = PAIR_J.get((a, b), PAIR_J.get((b, a)))


# ----------------------------
# Simulator primitives
# ----------------------------
def true_g(model: str, n: int) -> float:
    """
    Synthetic ground-truth E[N_acc] curve.
    This is only used by the demo simulator. In a real system, g_hat should be
    estimated from historical target-model verification logs.
    """
    p = TRUE_PARAMS[model]
    return 1.0 + p["A"] * (1.0 - np.exp(-p["b"] * n))


def sample_network(active_models, rng):
    """
    Generate a dynamic network and compute state:
    - load changes the draft time per node;
    - bandwidth changes the communication time per node.
    """
    state = {}
    for m in active_models:
        p = TRUE_PARAMS[m]
        load = float(np.clip(rng.beta(2, 3), 0, 1))
        bandwidth = float(np.clip(rng.lognormal(np.log(p["Bmean"]), 0.25), 40, 220))
        draft_ms = p["draft_base"] * (1.0 + 0.7 * load)
        comm_ms = PAYLOAD_MB / bandwidth * 1000.0
        state[m] = {
            "load": load,
            "B": bandwidth,
            "c": draft_ms + comm_ms,
            "delta": p["delta"],
        }
    return state


def draft_time(model: str, n: int, state: dict) -> float:
    """Draft + communication time for model m when it receives n nodes."""
    return state[model]["delta"] + n * state[model]["c"]


def iter_time(alloc: dict, state: dict) -> float:
    """
    One speculative-decoding round time.
    Multiple draft models run in parallel, so the waiting time is the slowest
    draft+communication branch plus one target-model verification pass.
    """
    return max(draft_time(m, n, state) for m, n in alloc.items()) + T_VER


def init_estimates(rng):
    """
    Initialize online estimates.

    Old models:
        Have historical N_acc curves and pairwise Jaccard similarity estimates.

    New model:
        Has a weak generic N_acc prior but zero similarity to all old models.
        This encodes the cold-start rule: unknown similarity is treated as
        maximally diverse, so the model is explored when residual budget exists.
    """
    g_hat = {m: np.zeros(N_CURVE + 1) for m in MODEL_NAMES}
    g_cnt = {m: np.zeros(N_CURVE + 1, dtype=int) for m in MODEL_NAMES}

    for m in OLD_MODELS:
        for n in range(1, N_CURVE + 1):
            samples = true_g(m, n) + rng.normal(0, 0.25, size=20)
            g_hat[m][n] = max(0.1, float(samples.mean()))
            g_cnt[m][n] = 20

    for n in range(1, N_CURVE + 1):
        g_hat[NEW_MODEL][n] = 1.0 + 3.0 * (1.0 - np.exp(-0.10 * n))
        g_cnt[NEW_MODEL][n] = 0

    j_hat, j_cnt = {}, {}
    for a in MODEL_NAMES:
        for b in MODEL_NAMES:
            if a == b:
                j_hat[(a, b)], j_cnt[(a, b)] = 1.0, 999
            elif a in OLD_MODELS and b in OLD_MODELS:
                j_hat[(a, b)] = float(np.clip(TRUE_J[(a, b)] + rng.normal(0, 0.04), 0, 1))
                j_cnt[(a, b)] = 20
            else:
                j_hat[(a, b)] = 0.0
                j_cnt[(a, b)] = 0

    return g_hat, g_cnt, j_hat, j_cnt


def eta(model: str, n: int, state: dict, g_hat: dict) -> float:
    """
    Estimated single-model accepted-token throughput:
        estimated E[N_acc] / estimated one-round time
    """
    return g_hat[model][n] / (draft_time(model, n, state) + T_VER)


def single_opt(active_models, state: dict, g_hat: dict):
    """
    Find the single-model best pair (model, node count) under the current
    network state.
    """
    best = (-1.0, None, None)
    for m in active_models:
        for n in range(1, N_CURVE + 1):
            val = eta(m, n, state, g_hat)
            if val > best[0]:
                best = (val, m, n)
    return best[1], best[2], best[0]


def no_delay_cap(model: str, primary_draft_time: float, state: dict) -> int:
    """
    Maximum nodes model can draft without exceeding the primary model's
    draft+communication completion time.
    """
    cap = math.floor((primary_draft_time - state[model]["delta"]) / state[model]["c"])
    return max(0, min(N_CURVE, cap))


# ----------------------------
# Policies
# ----------------------------
def proposed_policy(active_models, budget: int, state: dict, g_hat: dict, g_cnt: dict, j_hat: dict, cold_init: int = 6):
    """
    Proposed history-curve + similarity + no-delay budget allocation policy.

    Case 1:
        If current budget is lower than the best single-model node count,
        allocate all nodes to the best model under that current budget.

    Case 2:
        Otherwise, allocate the best model its own optimal node count first.
        Then allocate residual nodes to models with the lowest Jaccard similarity
        to the primary model, subject to the no-delay constraint.
    """
    primary, n_star, _ = single_opt(active_models, state, g_hat)

    if budget < n_star:
        best_at_budget = max(active_models, key=lambda m: eta(m, budget, state, g_hat))
        return {best_at_budget: budget}, best_at_budget

    alloc = {primary: n_star}
    residual = budget - n_star
    primary_draft_t = draft_time(primary, n_star, state)

    candidates = [m for m in active_models if m != primary]
    candidates.sort(key=lambda m: (j_hat[(primary, m)], m))

    for m in candidates:
        if residual <= 0:
            break

        cap = no_delay_cap(m, primary_draft_t, state)
        if cap <= 0:
            continue

        if g_cnt[m].sum() == 0:
            n_pref = cold_init
        else:
            _, n_pref, _ = single_opt([m], state, g_hat)

        n_alloc = int(min(residual, n_pref, cap))
        if n_alloc > 0:
            alloc[m] = n_alloc
            residual -= n_alloc

    return alloc, primary


def best_all_policy(active_models, budget: int, state: dict, g_hat: dict):
    """
    Baseline: allocate all current budget to the single model with the highest
    estimated throughput under exactly this budget.
    """
    m = max(active_models, key=lambda x: eta(x, budget, state, g_hat))
    return {m: budget}, m


def primary_only_policy(active_models, budget: int, state: dict, g_hat: dict):
    """
    Baseline: choose the best single-model pair and leave residual budget unused.
    """
    primary, n_star, _ = single_opt(active_models, state, g_hat)
    if budget < n_star:
        primary = max(active_models, key=lambda x: eta(x, budget, state, g_hat))
        n_star = budget
    return {primary: min(n_star, budget)}, primary


def random_residual_policy(active_models, budget: int, state: dict, g_hat: dict, g_cnt: dict, rng):
    """
    Baseline: choose the same primary as primary_only, then randomly assign
    residual budget to other models under the no-delay constraint.
    """
    alloc, primary = primary_only_policy(active_models, budget, state, g_hat)
    residual = budget - sum(alloc.values())
    if residual <= 0:
        return alloc, primary

    primary_draft_t = draft_time(primary, alloc[primary], state)
    candidates = [m for m in active_models if m != primary]
    rng.shuffle(candidates)

    for m in candidates:
        if residual <= 0:
            break

        cap = no_delay_cap(m, primary_draft_t, state)
        if cap <= 0:
            continue

        n_pref = 6 if g_cnt[m].sum() == 0 else single_opt([m], state, g_hat)[1]
        n_alloc = int(min(residual, n_pref, cap))

        if n_alloc > 0:
            alloc[m] = n_alloc
            residual -= n_alloc

    return alloc, primary


# ----------------------------
# Synthetic fused verification model
# ----------------------------
def fused_expected(alloc: dict, primary: str) -> float:
    """
    Synthetic expected accepted-token count after draft-tree fusion.

    This is the demo's placeholder for target-model verification.
    In a real experiment, this part should be replaced by actual LLM verification.
    """
    if len(alloc) == 1:
        m, n = next(iter(alloc.items()))
        return true_g(m, n)

    gain = true_g(primary, alloc[primary])

    for m, n in alloc.items():
        if m == primary:
            continue
        gain += 0.52 * (1.0 - TRUE_J[(primary, m)]) * true_g(m, n)

    secondary = [m for m in alloc if m != primary]
    for i in range(len(secondary)):
        for j in range(i + 1, len(secondary)):
            a, b = secondary[i], secondary[j]
            gain -= 0.12 * TRUE_J[(a, b)] * min(true_g(a, alloc[a]), true_g(b, alloc[b]))

    return max(0.0, gain)


def observe(alloc: dict, primary: str, state: dict, rng):
    """
    Generate one observed round result:
    - N_acc is sampled around the synthetic fused expectation;
    - time_ms is the parallel draft waiting time plus target verification.
    """
    acc = max(0.0, float(rng.normal(fused_expected(alloc, primary), 0.35)))
    return acc, iter_time(alloc, state)


def update_online(alloc: dict, g_hat: dict, g_cnt: dict, j_hat: dict, j_cnt: dict, rng):
    """
    Online update after the proposed policy is executed.

    g_hat update:
        Updates the model-specific N_acc curve at the allocated node count.

    j_hat update:
        Updates pairwise Jaccard similarity estimates for models jointly used
        in the same round.
    """
    for m, n in alloc.items():
        obs = max(0.0, float(rng.normal(true_g(m, n), 0.35)))
        c = g_cnt[m][n]
        g_hat[m][n] = (g_hat[m][n] * c + obs) / (c + 1)
        g_cnt[m][n] = c + 1

    used = list(alloc.keys())
    for i in range(len(used)):
        for j in range(i + 1, len(used)):
            a, b = used[i], used[j]
            obs_j = float(np.clip(rng.normal(TRUE_J[(a, b)], 0.06), 0, 1))
            for x, y in [(a, b), (b, a)]:
                c = j_cnt[(x, y)]
                j_hat[(x, y)] = (j_hat[(x, y)] * c + obs_j) / (c + 1)
                j_cnt[(x, y)] = c + 1


# ----------------------------
# Export helpers
# ----------------------------
def matrix_from_jhat(j_hat: dict) -> np.ndarray:
    """Convert pairwise similarity dictionary to a square matrix."""
    mat = np.zeros((len(MODEL_NAMES), len(MODEL_NAMES)))
    for i, a in enumerate(MODEL_NAMES):
        for j, b in enumerate(MODEL_NAMES):
            mat[i, j] = j_hat[(a, b)]
    return mat


def save_similarity_matrix_svg(matrix: np.ndarray, title: str, out_path: Path, vmin=None, vmax=None):
    """Save one Jaccard-similarity matrix as an SVG heatmap."""
    plt.figure(figsize=(6.2, 5.2))
    im = plt.imshow(matrix, vmin=vmin, vmax=vmax)
    plt.xticks(range(len(MODEL_NAMES)), MODEL_NAMES, rotation=30, ha="right")
    plt.yticks(range(len(MODEL_NAMES)), MODEL_NAMES)
    plt.title(title)

    for i in range(matrix.shape[0]):
        for j in range(matrix.shape[1]):
            plt.text(j, i, f"{matrix[i, j]:.2f}", ha="center", va="center")

    plt.colorbar(im, fraction=0.046, pad=0.04)
    plt.tight_layout()
    plt.savefig(out_path, format="svg")
    plt.close()


def save_rolling_throughput_svg(df: pd.DataFrame, out_path: Path, rolling_window: int = 12):
    """Save rolling-throughput curves for all policies."""
    plt.figure(figsize=(8.8, 5.2))

    for policy in sorted(df["policy"].unique()):
        d = df[df["policy"] == policy].sort_values("t").copy()
        d["rolling_throughput"] = d["throughput"].rolling(rolling_window, min_periods=1).mean()
        plt.plot(d["t"], d["rolling_throughput"], label=policy)

    plt.axvline(
        NEW_MODEL_JOIN_T,
        linestyle="--",
        linewidth=1.5,
        label=f"new model joins (t={NEW_MODEL_JOIN_T})",
    )
    plt.xlabel("Round t")
    plt.ylabel(f"Rolling mean throughput (window={rolling_window})")
    plt.title("Rolling throughput of different budget-allocation policies")
    plt.legend()
    plt.tight_layout()
    plt.savefig(out_path, format="svg")
    plt.close()


def save_avg_throughput_bar_svg(summary: pd.DataFrame, out_path: Path):
    """Save average-throughput bar chart."""
    ordered = summary.sort_values("avg_throughput", ascending=False)

    plt.figure(figsize=(7.2, 4.8))
    plt.bar(ordered.index.tolist(), ordered["avg_throughput"].values)
    plt.ylabel("Average throughput")
    plt.title("Average throughput by policy")
    plt.xticks(rotation=20, ha="right")
    plt.tight_layout()
    plt.savefig(out_path, format="svg")
    plt.close()


def export_results(df: pd.DataFrame, summary: pd.DataFrame, j_initial: np.ndarray, j_after_entry: np.ndarray, j_final: np.ndarray, out_dir):
    """
    Export CSV files and SVG figures.
    """
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    j_delta = j_final - j_initial

    df.to_csv(out_dir / "simulation_detail.csv", index=False)
    summary.round(6).to_csv(out_dir / "policy_summary.csv")

    save_rolling_throughput_svg(df, out_dir / "rolling_throughput.svg")
    save_avg_throughput_bar_svg(summary, out_dir / "avg_throughput_bar.svg")

    save_similarity_matrix_svg(
        j_initial,
        "Initial similarity matrix (cold start)",
        out_dir / "similarity_matrix_initial.svg",
        vmin=0.0,
        vmax=1.0,
    )
    save_similarity_matrix_svg(
        j_after_entry,
        f"Similarity matrix after early exploration (t={NEW_MODEL_JOIN_T + 10})",
        out_dir / "similarity_matrix_after_entry.svg",
        vmin=0.0,
        vmax=1.0,
    )
    save_similarity_matrix_svg(
        j_final,
        "Final similarity matrix",
        out_dir / "similarity_matrix_final.svg",
        vmin=0.0,
        vmax=1.0,
    )
    save_similarity_matrix_svg(
        j_delta,
        "Similarity-matrix change: final - initial",
        out_dir / "similarity_matrix_delta.svg",
    )


# ----------------------------
# Main simulation
# ----------------------------
def main(T: int = 160, out_dir="edge_budget_similarity_svg_output"):
    rng = np.random.default_rng(SEED)
    g_hat, g_cnt, j_hat, j_cnt = init_estimates(rng)

    j_initial = matrix_from_jhat(j_hat).copy()
    j_after_entry = None
    rows = []

    for t in range(1, T + 1):
        active = list(OLD_MODELS)
        if t >= NEW_MODEL_JOIN_T:
            active.append(NEW_MODEL)

        state = sample_network(active, rng)
        budget = int(
            rng.choice(
                [6, 8, 12, 16, 24, 32, 40],
                p=[0.08, 0.12, 0.20, 0.20, 0.20, 0.14, 0.06],
            )
        )

        policies = {
            "proposed": proposed_policy(active, budget, state, g_hat, g_cnt, j_hat),
            "best_all": best_all_policy(active, budget, state, g_hat),
            "primary_only": primary_only_policy(active, budget, state, g_hat),
            "random_residual": random_residual_policy(active, budget, state, g_hat, g_cnt, rng),
        }

        for name, (alloc, primary) in policies.items():
            acc, time_ms = observe(alloc, primary, state, rng)
            primary_t = draft_time(primary, alloc[primary], state)
            no_delay_ok = all(
                draft_time(m, n, state) <= primary_t + 1e-9
                for m, n in alloc.items()
                if m != primary
            )

            rows.append(
                {
                    "t": t,
                    "policy": name,
                    "budget": budget,
                    "primary": primary,
                    "alloc": str(alloc),
                    "N_acc": acc,
                    "time_ms": time_ms,
                    "throughput": acc / time_ms,
                    "models_used": len(alloc),
                    "uses_new": int(NEW_MODEL in alloc),
                    "no_delay_ok": int(no_delay_ok),
                }
            )

            # Only the proposed online policy updates the historical estimates.
            # This reflects deployment: the algorithm learns from its own executed allocations.
            if name == "proposed":
                update_online(alloc, g_hat, g_cnt, j_hat, j_cnt, rng)

        if t == NEW_MODEL_JOIN_T + 10:
            j_after_entry = matrix_from_jhat(j_hat).copy()

    if j_after_entry is None:
        j_after_entry = matrix_from_jhat(j_hat).copy()

    j_final = matrix_from_jhat(j_hat).copy()

    df = pd.DataFrame(rows)
    summary = df.groupby("policy").agg(
        avg_N_acc=("N_acc", "mean"),
        avg_time_ms=("time_ms", "mean"),
        avg_throughput=("throughput", "mean"),
        avg_models_used=("models_used", "mean"),
        no_delay_rate=("no_delay_ok", "mean"),
        new_use_rate_after_entry=("uses_new", lambda s: s[df.loc[s.index, "t"] >= NEW_MODEL_JOIN_T].mean()),
    )
    summary["speedup_vs_best_all"] = summary["avg_throughput"] / summary.loc["best_all", "avg_throughput"]
    summary = summary.sort_values("avg_throughput", ascending=False)

    export_results(df, summary, j_initial, j_after_entry, j_final, out_dir)

    print(summary.round(4))
    print("\nFinal cold-start pair similarity estimates:")
    for pair in [("Fast", "Diverse"), ("Balanced", "Diverse"), ("Accurate", "Diverse")]:
        print(f"{pair}: est={j_hat[pair]:.3f}, count={j_cnt[pair]}, true={TRUE_J[pair]:.3f}")

    print("\nExample proposed allocations after new model entry:")
    cols = ["t", "budget", "primary", "alloc", "N_acc", "time_ms", "throughput"]
    print(df[(df.policy == "proposed") & (df.t >= 150)][cols].to_string(index=False))
    print(f"\nSVG and CSV outputs saved to: {out_dir}")


if __name__ == "__main__":
    main()
