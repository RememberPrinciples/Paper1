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
