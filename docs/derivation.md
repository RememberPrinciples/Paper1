# Derivation Notes

## 2026-06-04

The first formal system-model draft for the journal extension is maintained in:

- `docs/System_Model_and_Algorithm.tex`

Current scope:

- Task profiling by entropy and reasoning intensity.
- Distributed speculative decoding system state.
- Per-iteration latency model.
- Expected accepted-token model.
- Per-request latency-per-valid-token minimization.
- Deterministic D-PROMISE enumeration solver.
- Initial journal extension toward MAB-based online acceptance profiling.

Next derivation focus:

- Formal MAB arm definition.
- Reward and regret formulation.
- Non-stationary acceptance-rate estimation.
- Fallback action modeling.

## 2026-06-04

Updated `docs/System_Model_and_Algorithm.tex` with a focused mathematical model for the journal problem:

- Decode-stage TPOT is the true optimization objective.
- Prefill latency is ignored.
- The observed context of request `t` is the task type, terminal load vector, and wireless rate vector.
- The MAB action is the device-length pair `(d_i, gamma)`.
- The unknown parameter is the task-conditioned token acceptance rate `alpha_{i,tau}`.
- Acceptance rate is not treated as the reward; it is the unknown component inside the structured TPOT loss.
- Exploration is implemented through an optimistic UCB estimate of `alpha_{i,tau}`.
- After the decode stage, the selected draft model's observed accepted/proposed token ratio updates the corresponding task-device acceptance estimate with temporal weighting.

## 2026-06-05

Updated `docs/System_Model_and_Algorithm.tex` according to the simplified initial modeling requirements:

- The task type is now represented by a generic finite set `mathcal{T}`; no concrete task taxonomy is assumed.
- The optimization problem is stated as an oracle TPOT minimization problem with unknown `alpha_{i,tau}^star`.
- The online solution is a structured UCB algorithm, where only task-device acceptance rates are learned.
- Added an optimism proposition based on Hoeffding's inequality.
- Added a gap-free cumulative regret bound that is sublinear in the request horizon `T`.
- Kept non-stationary discounted/sliding-window updates as a later extension rather than the main theorem setting.

## 2026-06-05

Moved the detailed MAB derivation and analysis into an appendix in `docs/System_Model_and_Algorithm.tex`:

- Main text now keeps the system model, oracle optimization problem, solver summary, and performance statement concise.
- Appendix provides step-by-step analysis:
  1. structured unknown reward,
  2. monotonicity and optimism,
  3. Hoeffding-UCB confidence bound,
  4. algorithmic derivation,
  5. regret analysis,
  6. non-stationary extension note.
- Recompiled successfully after the appendix update.

## 2026-06-05

Updated `docs/System_Model_and_Algorithm.tex` according to the communication and inference-speed modeling review:

- Removed the active device set `mathcal{D}_{act}(t)` and assumed all devices in `mathcal{D}` are schedulable.
- Replaced the ad hoc memory-load power-law model with a profiled bandwidth-degradation abstraction:
  `b_i^mem(t)=beta_i(L_i(t)) b_i^0`.
- Modeled draft-token latency as a Roofline-style compute-plus-memory term:
  `Phi_i(L_i(t))=c_i^d+V_i^d/b_i^mem(t)`.
- Clarified that the wireless latency model should use effective application-layer throughput `B_i(t)`.
- Kept the Shannon-type expression only as a physical-layer abstraction or upper-bound reference, with an efficiency factor `rho_i(t)`.
- Judged `payload / effective_rate + additive_delay` reasonable for 5G/WiFi smart-office or smart-factory settings as a first-order model, but only when the rate and delay terms are measured or calibrated under the relevant deployment conditions.
- Marked newly modified text in blue in the LaTeX document.

## 2026-06-10

Updated `docs/System_Model_and_Algorithm.tex` according to the latest modeling consistency review:

- Unified the decision epoch, objective, algorithm update, and regret horizon at the request level.
- Kept speculative rounds only as internal components of a request-level decode process.
- Defined request-level decode latency as expected number of speculative rounds times round latency:
  `T_decode^req = Y_t T_iter / G(alpha, gamma)`.
- Clarified that TPOT remains `T_iter / G(alpha, gamma)` after dividing by request output length, but regret is accumulated per request.
- Replaced proposed-token statistics with verified-token statistics:
  only draft tokens actually checked by the target model are counted in `V_{i,tau}`.
- Excluded draft tokens after the first rejected token in a speculative round from the acceptance-rate estimator because their outcomes are censored.
- Updated Structured UCB to use `C_{i,tau}/V_{i,tau}` and confidence radius based on `V_{i,tau}`.
- Expanded the communication payload model so terminal uploads include token IDs plus token-level distribution information needed by speculative decoding, represented as `S_tok = S_id + S_dist`.

## 2026-07-05

Reviewed the current two-layer MAB model and corrected the main consistency issues in `docs/System_Model_and_Algorithm.tex`:

- Formalized the bandit objective:
  - acceptance rate `alpha_{i,tau}` is a structured unknown parameter, not the reward;
  - loss is TPOT `J_{i,g}=T_i^round(g|t)/G(alpha_{i,tau},g)`;
  - equivalent reward is effective token rate `R_{i,g}=G(alpha_{i,tau},g)/T_i^round(g|t)`.
- Added request-level loss regret:
  `R_J(T)=sum_t [J^star_{a_t}(t)-J^star_{a_t^star}(t)]`,
  and latency-weighted regret with output length `Y_t^out`.
- Added target-only fallback action `a_0` with TPOT `J_0(t)=T^tar(t)`.
- Fixed UCB cold-start division-by-zero by evaluating `C/V` and the confidence radius only when `V_{i,tau}>0`; unverified pairs use optimistic `alpha_bar=1` and cold-start length.
- Unified edge and terminal objectives to use effective token count `G=1+A`; the entropy-aware dynamic drafting rate is now `(1+A_l)/T_round`.
- Made `g_t^plan` the terminal dynamic drafting upper bound, so the edge action is no longer ignored by the terminal policy.
- Fixed the first-token low-acceptance stopping bug by appending `z_1,H_1` before returning length one.
- Tightened censored feedback semantics: standard speculative decoding only updates from the accepted prefix plus the first rejected draft token; tokens after the first rejection are counterfactual/censored.
- Added request-level decode latency `T_req ~= Y_t^out T_round/G`.
- Clarified that target verification latency includes the `g` draft positions plus one bonus/fallback position.
- Clarified `S_dist` as protocol-dependent payload and constrained the entropy-acceptance curve fitting window/least-squares problem.

## 2026-07-05

Updated the MAB objective wording after multi-agent review of the user's `T/G` reward suggestion:

- Adopted `T/G` as the main bandit optimization variable, but stated it as a cost-type reward / TPOT cost to be minimized.
- Kept the structured `alpha_{i,tau}` learner: UCB is still applied to the shared acceptance parameter, not to independent per-`(i,g)` empirical costs.
- Replaced the optimistic TPOT notation with `underline{J}_{i,g}` to emphasize that substituting `overline{alpha}` yields a lower cost estimate.
- Clarified that standard maximization-reward notation can use `R=-J`, where `J=T/G`.
- Changed terminal entropy-aware stopping from maximizing `G/T` to minimizing `T/G`.
- Added observed TPOT metric `tilde{J}_t` for reporting while keeping UCB updates based on token-level acceptance labels.

## 2026-07-09

Workspace maintenance update:

- Moved model and dataset download scripts from `experiments/` to `download_scripts/`.
- Moved Llama-family local model directories from `experiments/Model/` to `/root/autodl-tmp/Model/`.
- Removed Llama latency experiment scripts, result directories, and related HuggingFace cache entries from `experiments/`.
- Cleared historical files from `report/` per the current cleanup request.
- Added `docs/Makefile` and `docs/README.md` so `System_Model_and_Algorithm.tex` can be rebuilt with `make -C docs` when a XeLaTeX-capable TeX installation is available.
