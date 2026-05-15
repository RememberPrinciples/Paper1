"""
Directional-redundancy demo for edge-side node-budget allocation and
cooperative verification under dynamic network states.

Compared with the symmetric Jaccard version, this script uses an asymmetric
coverage/redundancy matrix:

    rho[i -> j] = |P_i ∩ P_j| / |P_i|

where P_i is the path set of draft tree i. The row model is the candidate
model; the column model is the already-selected/reference model.

If one model's draft tree completely contains another model's draft tree, e.g.

    P_Subset ⊂ P_Accurate,

then:
    rho[Subset -> Accurate] = 1.0
    rho[Accurate -> Subset] = |P_Subset| / |P_Accurate| ∈ (0, 1)

Therefore, Subset has no novelty when Accurate has already been selected, but
Accurate can still provide additional novelty when Subset is the primary model.

Run:
    python edge_budget_directional_redundancy_demo.py

Default outputs:
    ./edge_budget_directional_redundancy_output/
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

# "Subset" is the intentionally added containment model.
# Its draft-tree path set is fully contained by "Accurate".
MODEL_NAMES = ["Fast", "Balanced", "Accurate", "Subset", "Diverse"]
OLD_MODELS = ["Fast", "Balanced", "Accurate", "Subset"]
NEW_MODEL = "Diverse"
NEW_MODEL_JOIN_T = 80

N_CURVE = 48        # maximum recorded draft-node count for each N_acc curve
T_VER = 60.0        # ms, target-model verification time per round
PAYLOAD_MB = 0.08   # toy communication payload per draft node

TRUE_PARAMS = {
    "Fast":     {"A": 5.2, "b": 0.23, "draft_base": 1.10, "delta": 2.0, "Bmean": 100},
    "Balanced": {"A": 6.8, "b": 0.16, "draft_base": 1.70, "delta": 2.2, "Bmean": 100},
    "Accurate": {"A": 8.5, "b": 0.10, "draft_base": 2.60, "delta": 2.5, "Bmean": 100},
    # This model is fast and locally useful, but its draft paths are assumed to
    # be a subset of Accurate's paths. Therefore it is redundant after Accurate.
    "Subset":  {"A": 4.8, "b": 0.24, "draft_base": 0.95, "delta": 1.8, "Bmean": 105},
    "Diverse": {"A": 6.2, "b": 0.18, "draft_base": 1.45, "delta": 2.3, "Bmean": 100},
}

# Optional quality factor used only in the synthetic fusion oracle.
# In a real experiment this should be replaced by target-model verification.
MODEL_QUALITY = {
    "Fast": 0.95,
    "Balanced": 1.00,
    "Accurate": 1.05,
    "Subset": 0.90,
    "Diverse": 1.00,
}

# Approximate number of high-probability root-to-node paths in each model's
# draft tree. These are only used to construct a directional redundancy matrix.
PATH_SIZE = {
    "Fast": 18,
    "Balanced": 22,
    "Accurate": 28,
    "Subset": 10,
    "Diverse": 20,
}

# Symmetric overlap counts |P_i ∩ P_j|. Directionality is obtained by dividing
# by the row model's path-set size.
OVERLAP_COUNT = {
    ("Fast", "Balanced"): 10,
    ("Fast", "Accurate"): 8,
    ("Fast", "Subset"): 7,
    ("Fast", "Diverse"): 2,

    ("Balanced", "Accurate"): 9,
    ("Balanced", "Subset"): 6,
    ("Balanced", "Diverse"): 4,

    # Containment pair:
    # every Subset path is contained in Accurate, so overlap = |P_Subset|.
    ("Accurate", "Subset"): 10,
    ("Accurate", "Diverse"): 6,

    ("Subset", "Diverse"): 1,
}


def overlap_count(a: str, b: str) -> int:
    if a == b:
        return PATH_SIZE[a]
    return OVERLAP_COUNT.get((a, b), OVERLAP_COUNT.get((b, a)))


TRUE_RHO = {}
for a in MODEL_NAMES:
    for b in MODEL_NAMES:
        TRUE_RHO[(a, b)] = overlap_count(a, b) / PATH_SIZE[a]

# This exact containment relation is kept noise-free in online observations.
EXACT_CONTAINMENT_DIRECTION = ("Subset", "Accurate")


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
        Have historical N_acc curves and directional redundancy estimates.

    New model:
        Has a weak generic N_acc prior but zero directional redundancy to all
        old models. This encodes cold-start optimistic diversity.
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

    rho_hat, rho_cnt = {}, {}
    for a in MODEL_NAMES:
        for b in MODEL_NAMES:
            if a == b:
                rho_hat[(a, b)], rho_cnt[(a, b)] = 1.0, 999
            elif a in OLD_MODELS and b in OLD_MODELS:
                if (a, b) == EXACT_CONTAINMENT_DIRECTION:
                    rho_hat[(a, b)] = 1.0
                else:
                    rho_hat[(a, b)] = float(np.clip(TRUE_RHO[(a, b)] + rng.normal(0, 0.035), 0, 1))
                rho_cnt[(a, b)] = 20
            else:
                # Cold-start: unknown row->column coverage is treated as zero,
                # so the candidate is considered maximally novel until observed.
                rho_hat[(a, b)] = 0.0
                rho_cnt[(a, b)] = 0

    return g_hat, g_cnt, rho_hat, rho_cnt


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


def redundancy_to_selected(model: str, selected_models: list, rho_matrix: dict) -> float:
    """
    Directional redundancy of a candidate model relative to the already-selected set.

    Approximation used here:
        rho(model -> selected set) = max_s rho(model -> s)

    The exact set version would be:
        |P_model ∩ (union_s P_s)| / |P_model|
    but max-over-selected is sufficient for this demo and keeps the online state
    pairwise.
    """
    if not selected_models:
        return 0.0
    return max(rho_matrix[(model, s)] for s in selected_models)


def sample_directional_redundancy(row_model: str, col_model: str, rng) -> float:
    """Sample an observed directional redundancy value."""
    true_val = TRUE_RHO[(row_model, col_model)]
    if (row_model, col_model) == EXACT_CONTAINMENT_DIRECTION:
        return 1.0
    return float(np.clip(rng.normal(true_val, 0.045), 0, 1))


# ----------------------------
# Policies
# ----------------------------
def proposed_policy(active_models, budget: int, state: dict, g_hat: dict, g_cnt: dict, rho_hat: dict, cold_init: int = 6):
    """
    Proposed history-curve + directional-redundancy + no-delay allocation policy.

    Case 1:
        If current budget is lower than the best single-model node count,
        allocate all nodes to the best model under that current budget.

    Case 2:
        Otherwise, allocate the best model its own optimal node count first.
        Then allocate residual nodes iteratively to the least redundant candidate
        relative to the already-selected set, subject to the no-delay constraint.

    Candidates with redundancy >= 1 are skipped because they bring no novel
    draft paths relative to the selected set.
    """
    primary, n_star, _ = single_opt(active_models, state, g_hat)

    if budget < n_star:
        best_at_budget = max(active_models, key=lambda m: eta(m, budget, state, g_hat))
        return {best_at_budget: budget}, best_at_budget

    alloc = {primary: n_star}
    selected = [primary]
    residual = budget - n_star
    primary_draft_t = draft_time(primary, n_star, state)

    while residual > 0:
        best_candidate = None

        for m in active_models:
            if m in selected:
                continue

            rho = redundancy_to_selected(m, selected, rho_hat)
            if rho >= 0.999:
                # Fully covered by the selected draft set; no novelty.
                continue

            cap = no_delay_cap(m, primary_draft_t, state)
            if cap <= 0:
                continue

            if g_cnt[m].sum() == 0:
                n_pref = cold_init
            else:
                _, n_pref, _ = single_opt([m], state, g_hat)

            n_alloc = int(min(residual, n_pref, cap))
            if n_alloc <= 0:
                continue

            # First minimize directional redundancy; then prefer higher current
            # single-model throughput as a tie-breaker.
            score = (rho, -eta(m, n_alloc, state, g_hat), m)
            if best_candidate is None or score < best_candidate["score"]:
                best_candidate = {
                    "model": m,
                    "n_alloc": n_alloc,
                    "score": score,
                }

        if best_candidate is None:
            break

        m = best_candidate["model"]
        alloc[m] = best_candidate["n_alloc"]
        selected.append(m)
        residual -= best_candidate["n_alloc"]

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

    This baseline does not use the directional redundancy matrix, so it may
    allocate residual nodes to a fully contained model.
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

    Directional redundancy controls marginal novelty:
        novelty(m | selected) = 1 - rho[m -> selected]

    If m's paths are fully covered by the selected set, novelty = 0 and m adds
    no expected accepted tokens. This encodes the containment case requested by
    the user.

    In a real experiment, this function should be replaced by actual target-
    model verification on the fused draft tree.
    """
    if len(alloc) == 1:
        m, n = next(iter(alloc.items()))
        return true_g(m, n)

    gain = true_g(primary, alloc[primary])
    selected = [primary]

    # Dict insertion order is policy order: primary first, then residual models.
    for m, n in alloc.items():
        if m == primary:
            continue

        rho = redundancy_to_selected(m, selected, TRUE_RHO)
        novelty = max(0.0, 1.0 - rho)

        # Residual nodes are not always as valuable as primary nodes. The factor
        # 0.52 is a synthetic complementarity coefficient, not a measured LLM value.
        gain += 0.52 * novelty * MODEL_QUALITY[m] * true_g(m, n)

        # A small verification-opportunity cost discourages filling residual
        # budget with low-novelty candidates.
        gain -= 0.06 * rho * true_g(m, n)
        selected.append(m)

    return max(0.0, gain)


def observe(alloc: dict, primary: str, state: dict, rng):
    """
    Generate one observed round result:
    - N_acc is sampled around the synthetic fused expectation;
    - time_ms is the parallel draft waiting time plus target verification.
    """
    acc = max(0.0, float(rng.normal(fused_expected(alloc, primary), 0.35)))
    return acc, iter_time(alloc, state)


def update_online(alloc: dict, g_hat: dict, g_cnt: dict, rho_hat: dict, rho_cnt: dict, rng):
    """
    Online update after a policy is executed.

    g_hat update:
        Updates the model-specific N_acc curve at the allocated node count.

    rho_hat update:
        Updates directional redundancy estimates for both directions of each
        pair jointly used in the same round.
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
            for row, col in [(a, b), (b, a)]:
                obs_rho = sample_directional_redundancy(row, col, rng)
                c = rho_cnt[(row, col)]
                rho_hat[(row, col)] = (rho_hat[(row, col)] * c + obs_rho) / (c + 1)
                rho_cnt[(row, col)] = c + 1


# ----------------------------
# Export helpers
# ----------------------------
def matrix_from_rho(rho_hat: dict) -> np.ndarray:
    """Convert pairwise directional redundancy dictionary to a square matrix."""
    mat = np.zeros((len(MODEL_NAMES), len(MODEL_NAMES)))
    for i, a in enumerate(MODEL_NAMES):
        for j, b in enumerate(MODEL_NAMES):
            mat[i, j] = rho_hat[(a, b)]
    return mat


def save_redundancy_matrix_svg(matrix: np.ndarray, title: str, out_path: Path, vmin=None, vmax=None):
    """
    Save one directional redundancy matrix as an SVG heatmap.

    Row i, column j means rho[i -> j]:
        fraction of row model's draft paths already covered by the column model.
    """
    plt.figure(figsize=(7.0, 5.7))
    im = plt.imshow(matrix, vmin=vmin, vmax=vmax)
    plt.xticks(range(len(MODEL_NAMES)), MODEL_NAMES, rotation=30, ha="right")
    plt.yticks(range(len(MODEL_NAMES)), MODEL_NAMES)
    plt.xlabel("reference / already-selected model")
    plt.ylabel("candidate model")
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
    plt.title("Rolling throughput under directional-redundancy allocation")
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


def save_containment_demo_svg(rho_final: np.ndarray, out_path: Path):
    """
    A small focused figure that highlights the asymmetric containment relation:
        rho[Subset -> Accurate] = 1
        rho[Accurate -> Subset] in (0, 1)
    """
    subset_idx = MODEL_NAMES.index("Subset")
    accurate_idx = MODEL_NAMES.index("Accurate")

    labels = ["Subset -> Accurate", "Accurate -> Subset"]
    values = [
        rho_final[subset_idx, accurate_idx],
        rho_final[accurate_idx, subset_idx],
    ]

    plt.figure(figsize=(6.8, 4.2))
    plt.bar(labels, values)
    plt.ylim(0, 1.05)
    plt.ylabel("Directional redundancy")
    plt.title("Asymmetric containment relation")
    for idx, val in enumerate(values):
        plt.text(idx, val + 0.03, f"{val:.2f}", ha="center")
    plt.tight_layout()
    plt.savefig(out_path, format="svg")
    plt.close()


def export_results(df: pd.DataFrame, summary: pd.DataFrame, rho_initial: np.ndarray, rho_after_entry: np.ndarray, rho_final: np.ndarray, out_dir):
    """
    Export CSV files and SVG figures.
    """
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    rho_delta = rho_final - rho_initial

    df.to_csv(out_dir / "simulation_detail.csv", index=False)
    summary.round(6).to_csv(out_dir / "policy_summary.csv")

    save_rolling_throughput_svg(df, out_dir / "rolling_throughput.svg")
    save_avg_throughput_bar_svg(summary, out_dir / "avg_throughput_bar.svg")

    save_redundancy_matrix_svg(
        rho_initial,
        "Initial directional redundancy matrix",
        out_dir / "directional_redundancy_matrix_initial.svg",
        vmin=0.0,
        vmax=1.0,
    )
    save_redundancy_matrix_svg(
        rho_after_entry,
        f"Directional redundancy after early exploration (t={NEW_MODEL_JOIN_T + 10})",
        out_dir / "directional_redundancy_matrix_after_entry.svg",
        vmin=0.0,
        vmax=1.0,
    )
    save_redundancy_matrix_svg(
        rho_final,
        "Final directional redundancy matrix",
        out_dir / "directional_redundancy_matrix_final.svg",
        vmin=0.0,
        vmax=1.0,
    )
    save_redundancy_matrix_svg(
        rho_delta,
        "Directional-redundancy change: final - initial",
        out_dir / "directional_redundancy_matrix_delta.svg",
    )
    save_containment_demo_svg(rho_final, out_dir / "containment_asymmetry_demo.svg")


# ----------------------------
# Main simulation
# ----------------------------
def main(T: int = 160, out_dir="edge_budget_directional_redundancy_output"):
    rng = np.random.default_rng(SEED)

    policy_names = ["proposed", "best_all", "primary_only", "random_residual"]

    # Each policy owns an independent online state. This avoids the earlier
    # issue where baselines inherited the proposed method's exploration history.
    online_states = {}
    for name in policy_names:
        # Use identical seeds for identical initial states.
        init_rng = np.random.default_rng(SEED)
        online_states[name] = init_estimates(init_rng)

    rho_initial = matrix_from_rho(online_states["proposed"][2]).copy()
    rho_after_entry = None
    rows = []

    for t in range(1, T + 1):
        active = list(OLD_MODELS)
        if t >= NEW_MODEL_JOIN_T:
            active.append(NEW_MODEL)

        # Same environment and budget are used for all policies in this round.
        state = sample_network(active, rng)
        budget = int(
            rng.choice(
                [6, 8, 12, 16, 24, 32, 40],
                p=[0.08, 0.12, 0.20, 0.20, 0.20, 0.14, 0.06],
            )
        )

        for name in policy_names:
            g_hat, g_cnt, rho_hat, rho_cnt = online_states[name]

            if name == "proposed":
                alloc, primary = proposed_policy(active, budget, state, g_hat, g_cnt, rho_hat)
            elif name == "best_all":
                alloc, primary = best_all_policy(active, budget, state, g_hat)
            elif name == "primary_only":
                alloc, primary = primary_only_policy(active, budget, state, g_hat)
            elif name == "random_residual":
                alloc, primary = random_residual_policy(active, budget, state, g_hat, g_cnt, rng)
            else:
                raise ValueError(f"Unknown policy: {name}")

            acc, time_ms = observe(alloc, primary, state, rng)
            primary_t = draft_time(primary, alloc[primary], state)
            no_delay_ok = all(
                draft_time(m, n, state) <= primary_t + 1e-9
                for m, n in alloc.items()
                if m != primary
            )

            alloc_order = list(alloc.keys())
            contained_waste = int(
                "Accurate" in alloc_order
                and "Subset" in alloc_order
                and alloc_order.index("Accurate") < alloc_order.index("Subset")
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
                    "uses_subset": int("Subset" in alloc),
                    "contained_waste": contained_waste,
                    "no_delay_ok": int(no_delay_ok),
                }
            )

            # Every policy learns only from its own executed allocations.
            update_online(alloc, g_hat, g_cnt, rho_hat, rho_cnt, rng)

        if t == NEW_MODEL_JOIN_T + 10:
            rho_after_entry = matrix_from_rho(online_states["proposed"][2]).copy()

    if rho_after_entry is None:
        rho_after_entry = matrix_from_rho(online_states["proposed"][2]).copy()

    rho_final = matrix_from_rho(online_states["proposed"][2]).copy()

    df = pd.DataFrame(rows)
    summary = df.groupby("policy").agg(
        avg_N_acc=("N_acc", "mean"),
        avg_time_ms=("time_ms", "mean"),
        avg_throughput=("throughput", "mean"),
        avg_models_used=("models_used", "mean"),
        no_delay_rate=("no_delay_ok", "mean"),
        subset_use_rate=("uses_subset", "mean"),
        contained_waste_rate=("contained_waste", "mean"),
        new_use_rate_after_entry=("uses_new", lambda s: s[df.loc[s.index, "t"] >= NEW_MODEL_JOIN_T].mean()),
    )
    summary["speedup_vs_best_all"] = summary["avg_throughput"] / summary.loc["best_all", "avg_throughput"]
    summary = summary.sort_values("avg_throughput", ascending=False)

    export_results(df, summary, rho_initial, rho_after_entry, rho_final, out_dir)

    print(summary.round(4))

    subset_idx = MODEL_NAMES.index("Subset")
    accurate_idx = MODEL_NAMES.index("Accurate")
    print("\nAsymmetric containment check in final proposed matrix:")
    print(f"rho[Subset -> Accurate] = {rho_final[subset_idx, accurate_idx]:.3f}")
    print(f"rho[Accurate -> Subset] = {rho_final[accurate_idx, subset_idx]:.3f}")
    print(f"True rho[Subset -> Accurate] = {TRUE_RHO[('Subset', 'Accurate')]:.3f}")
    print(f"True rho[Accurate -> Subset] = {TRUE_RHO[('Accurate', 'Subset')]:.3f}")

    print("\nFinal cold-start pair redundancy estimates for Diverse in proposed policy:")
    proposed_rho = online_states["proposed"][2]
    proposed_cnt = online_states["proposed"][3]
    for pair in [("Diverse", "Fast"), ("Diverse", "Balanced"), ("Diverse", "Accurate")]:
        print(f"rho{pair}: est={proposed_rho[pair]:.3f}, count={proposed_cnt[pair]}, true={TRUE_RHO[pair]:.3f}")

    print("\nExample proposed allocations after new model entry:")
    cols = ["t", "budget", "primary", "alloc", "N_acc", "time_ms", "throughput", "contained_waste"]
    print(df[(df.policy == "proposed") & (df.t >= 150)][cols].to_string(index=False))
    print(f"\nSVG and CSV outputs saved to: {out_dir}")


if __name__ == "__main__":
    main()
