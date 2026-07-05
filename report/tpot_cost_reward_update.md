# TPOT Cost-Type Reward 建模更新报告

日期：2026-07-05

## 1. 用户建议

用户指出：既然此前报告中推荐变量包含 $T/G$，后续多臂赌博机算法应采用 $T/G$ 作为 reward；同时，算法仍必须记录并学习各个草稿模型在不同任务上的接受率 $\alpha_{i,\tau}$。

## 2. 多智能体审阅结论

两个子智能体给出的核心结论一致：

- $T/G$ 是平均每有效 token 时延，数值越小越好。
- 标准 MAB 中 reward 通常越大越好，因此严格术语下 $T/G$ 更适合称为 cost、loss 或 TPOT cost。
- 若用户希望使用 reward 一词，应写成 cost-type reward，并明确算法最小化它。
- 若需要标准最大化 reward 记号，可定义 $R=-T/G$。
- 不建议把当前算法改为普通 per-arm cost-UCB，因为那会放弃共享 $\alpha_{i,\tau}$ 的结构化学习优势。

最终采用口径：

$$
J_{i,g}(t)
=
\frac{T_i^{\mathrm{round}}(g\mid t)}
{G(\alpha_{i,\tau_t},g)}
$$

作为本文的 cost-type reward / TPOT cost，并以最小化累计 $J$ 为目标。

## 3. 保留的结构化学习机制

本次没有改成“直接对每个 $(i,g)$ 学一个独立 $T/G$ 均值”的普通 bandit。

原因：

- 不同规划长度 $g$ 之间可以通过共享 $\alpha_{i,\tau}$ 共享样本。
- $T_i^{\mathrm{round}}(g\mid t)$ 随负载、信道、队列状态变化，直接平均 $T/G$ 会混入系统状态噪声。
- 终端动态停止后实际草稿长度 $g_t$ 可能小于规划长度 $g_t^{\mathrm{plan}}$，直接学习 per-arm observed cost 会和动作定义纠缠。

因此当前算法仍是 structured cost-UCB：

1. 对每个任务-草稿模型对 $(i,\tau)$ 维护 $C_{i,\tau}$ 和 $V_{i,\tau}$。
2. 用已验证 token 估计 $\widehat{\alpha}_{i,\tau}=C_{i,\tau}/V_{i,\tau}$。
3. 构造 $\overline{\alpha}_{i,\tau}$。
4. 将 $\overline{\alpha}_{i,\tau}$ 代入 $G(\alpha,g)$，形成 TPOT cost 的乐观下界：

$$
\underline{J}_{i,g}(t)
=
\frac{T_i^{\mathrm{round}}(g\mid t)}
{G(\overline{\alpha}_{i,\tau_t}(t),g)}.
$$

5. 选择 $\underline{J}_{i,g}(t)$ 最小的动作，或选择 target-only fallback。

## 4. 已修改内容

- `docs/System_Model_and_Algorithm.tex`
  - 将边侧选择章节改为“基于接受率置信界的结构化 TPOT 优化”。
  - 将 $T/G$ 明确为 cost-type reward / TPOT cost。
  - 将标准最大化 reward 表示为 $R=-J$。
  - 将 $\overline{J}_{i,g}$ 改为 $\underline{J}_{i,g}$，表示乐观 TPOT cost 下界。
  - 保留并强化 $\alpha_{i,\tau}$ 的学习地位。
  - 将终端动态停止从最大化 $G/T$ 改为最小化 $T/G$。
  - 增加观测 TPOT 指标 $\widetilde{J}_t$，仅用于实验报告，不混入 $C,V$。

- `docs/derivation.md`
  - 追加本次关于 $T/G$ cost-type reward 的推导日志。

- `RESEARCH_LOG.md`
  - 同步当前算法方向和 reward/regret 口径。

- `report/mab_model_algorithm_review.md`
  - 更新此前报告中的 reward/loss 叙述，避免继续把 $G/T$ 写成主 reward。

- `.ai_rules`
  - 增加规则：每次完成任务后都要在 `report/` 中写入或更新报告，方便用户通过 report 文件夹查看工作结果。

## 5. 当前最终表述

本文采用 cost-sensitive structured UCB。算法不是直接学习每个动作的独立 reward，而是学习任务-草稿模型接受率 $\alpha_{i,\tau}$；再将其上置信界代入 TPOT cost

$$
J=\frac{T}{G}
$$

形成乐观低成本估计，并选择 cost 最小的动作。$T/G$ 是本文的主 bandit 优化变量；$\alpha_{i,\tau}$ 是必须同时学习的结构化隐变量。

