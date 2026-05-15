"""
Demo simulation for edge-side node-budget allocation and cooperative verification
under a history-curve + Jaccard-similarity heuristic.

Run:
    python edge_budget_similarity_demo.py
"""
import math
from collections import defaultdict
import numpy as np
import pandas as pd

SEED = 7
rng = np.random.default_rng(SEED)

MODEL_NAMES = ["Fast", "Balanced", "Accurate", "Diverse"]
OLD_MODELS = ["Fast", "Balanced", "Accurate"]
NEW_MODEL = "Diverse"
NEW_MODEL_JOIN_T = 80
N_CURVE = 48
T_VER = 60.0       # ms, one target-model verification pass within edge node budget
PAYLOAD_MB = 0.08  # toy communication payload per draft node

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


def true_g(model: str, n: int) -> float:
    """True single-model E[N_acc] curve used only by the simulator."""
    p = TRUE_PARAMS[model]
    return 1.0 + p["A"] * (1.0 - np.exp(-p["b"] * n))


def sample_network(active, local_rng):
    """Dynamic load and bandwidth state."""
    state = {}
    for m in active:
        p = TRUE_PARAMS[m]
        load = float(np.clip(local_rng.beta(2, 3), 0, 1))
        bandwidth = float(np.clip(local_rng.lognormal(np.log(p["Bmean"]), 0.25), 40, 220))
        draft_ms = p["draft_base"] * (1.0 + 0.7 * load)
        comm_ms = PAYLOAD_MB / bandwidth * 1000.0
        state[m] = {"load": load, "B": bandwidth, "c": draft_ms + comm_ms, "delta": p["delta"]}
    return state


def draft_time(model: str, n: int, state) -> float:
    return state[model]["delta"] + n * state[model]["c"]


def iter_time(alloc: dict, state) -> float:
    return max(draft_time(m, n, state) for m, n in alloc.items()) + T_VER


def init_estimates(local_rng):
    """Old models have history; the new model has a weak generic prior and zero pair similarity."""
    g_hat = {m: np.zeros(N_CURVE + 1) for m in MODEL_NAMES}
    g_cnt = {m: np.zeros(N_CURVE + 1, dtype=int) for m in MODEL_NAMES}
    for m in OLD_MODELS:
        for n in range(1, N_CURVE + 1):
            samples = true_g(m, n) + local_rng.normal(0, 0.25, size=20)
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
                j_hat[(a, b)] = float(np.clip(TRUE_J[(a, b)] + local_rng.normal(0, 0.04), 0, 1))
                j_cnt[(a, b)] = 20
            else:
                j_hat[(a, b)] = 0.0  # cold-start: optimistic diversity
                j_cnt[(a, b)] = 0
    return g_hat, g_cnt, j_hat, j_cnt


def eta(model: str, n: int, state, g_hat) -> float:
    return g_hat[model][n] / (draft_time(model, n, state) + T_VER)


def single_opt(active, state, g_hat):
    best = (-1.0, None, None)
    for m in active:
        for n in range(1, N_CURVE + 1):
            val = eta(m, n, state, g_hat)
            if val > best[0]:
                best = (val, m, n)
    return best[1], best[2], best[0]


def no_delay_cap(model: str, primary_draft_time: float, state) -> int:
    cap = math.floor((primary_draft_time - state[model]["delta"]) / state[model]["c"])
    return max(0, min(N_CURVE, cap))


def proposed_policy(active, budget, state, g_hat, g_cnt, j_hat, cold_init=6):
    primary, n_star, _ = single_opt(active, state, g_hat)
    if budget < n_star:
        best_at_budget = max(active, key=lambda m: eta(m, budget, state, g_hat))
        return {best_at_budget: budget}, best_at_budget

    alloc = {primary: n_star}
    residual = budget - n_star
    primary_draft_t = draft_time(primary, n_star, state)
    candidates = [m for m in active if m != primary]
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


def best_all_policy(active, budget, state, g_hat):
    m = max(active, key=lambda x: eta(x, budget, state, g_hat))
    return {m: budget}, m


def primary_only_policy(active, budget, state, g_hat):
    primary, n_star, _ = single_opt(active, state, g_hat)
    if budget < n_star:
        primary = max(active, key=lambda x: eta(x, budget, state, g_hat))
        n_star = budget
    return {primary: min(n_star, budget)}, primary


def random_residual_policy(active, budget, state, g_hat, g_cnt, local_rng):
    alloc, primary = primary_only_policy(active, budget, state, g_hat)
    residual = budget - sum(alloc.values())
    if residual <= 0:
        return alloc, primary
    primary_draft_t = draft_time(primary, alloc[primary], state)
    cand = [m for m in active if m != primary]
    local_rng.shuffle(cand)
    for m in cand:
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


def fused_expected(alloc: dict, primary: str) -> float:
    """Toy expected fused accepted-token model with subadditive complementary gain."""
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


def observe(alloc, primary, state, local_rng):
    acc = max(0.0, float(local_rng.normal(fused_expected(alloc, primary), 0.35)))
    return acc, iter_time(alloc, state)


def update_online(alloc, g_hat, g_cnt, j_hat, j_cnt, local_rng):
    for m, n in alloc.items():
        obs = max(0.0, float(local_rng.normal(true_g(m, n), 0.35)))
        c = g_cnt[m][n]
        g_hat[m][n] = (g_hat[m][n] * c + obs) / (c + 1)
        g_cnt[m][n] = c + 1
    used = list(alloc.keys())
    for i in range(len(used)):
        for j in range(i + 1, len(used)):
            a, b = used[i], used[j]
            obs_j = float(np.clip(local_rng.normal(TRUE_J[(a, b)], 0.06), 0, 1))
            for x, y in [(a, b), (b, a)]:
                c = j_cnt[(x, y)]
                j_hat[(x, y)] = (j_hat[(x, y)] * c + obs_j) / (c + 1)
                j_cnt[(x, y)] = c + 1


def main(T=160):
    local_rng = np.random.default_rng(SEED)
    g_hat, g_cnt, j_hat, j_cnt = init_estimates(local_rng)
    rows = []
    for t in range(1, T + 1):
        active = list(OLD_MODELS)
        if t >= NEW_MODEL_JOIN_T:
            active.append(NEW_MODEL)
        state = sample_network(active, local_rng)
        budget = int(local_rng.choice([6, 8, 12, 16, 24, 32, 40], p=[0.08, 0.12, 0.20, 0.20, 0.20, 0.14, 0.06]))

        policies = {
            "proposed": proposed_policy(active, budget, state, g_hat, g_cnt, j_hat),
            "best_all": best_all_policy(active, budget, state, g_hat),
            "primary_only": primary_only_policy(active, budget, state, g_hat),
            "random_residual": random_residual_policy(active, budget, state, g_hat, g_cnt, local_rng),
        }
        for name, (alloc, primary) in policies.items():
            acc, time_ms = observe(alloc, primary, state, local_rng)
            primary_t = draft_time(primary, alloc[primary], state)
            no_delay_ok = all(draft_time(m, n, state) <= primary_t + 1e-9 for m, n in alloc.items() if m != primary)
            rows.append({
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
            })
            if name == "proposed":
                update_online(alloc, g_hat, g_cnt, j_hat, j_cnt, local_rng)

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
    print(summary.sort_values("avg_throughput", ascending=False).round(4))
    print("\nFinal cold-start pair similarity estimates:")
    for pair in [("Fast", "Diverse"), ("Balanced", "Diverse"), ("Accurate", "Diverse")]:
        print(f"{pair}: est={j_hat[pair]:.3f}, count={j_cnt[pair]}, true={TRUE_J[pair]:.3f}")
    print("\nExample proposed allocations after new model entry:")
    print(df[(df.policy == "proposed") & (df.t >= 150)][["t", "budget", "primary", "alloc", "N_acc", "time_ms", "throughput"]].to_string(index=False))


if __name__ == "__main__":
    main()
