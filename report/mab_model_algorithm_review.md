# 多臂赌博机建模与算法审阅报告

日期：2026-07-05

## 1. 审阅范围与多智能体分工

本次审阅覆盖以下文件：

- `docs/System_Model_and_Algorithm.tex`
- `docs/derivation.md`
- `RESEARCH_LOG.md`
- `conference_paper.txt`
- `current_answer.md`

多智能体分工：

- Explorer Pauli：专门审阅多臂赌博机 reward/regret 定义及其自洽性。
- Explorer Sagan：审阅系统模型、算法伪代码、符号一致性和建模缺口。
- 主线程：整合结论，修正 `docs/System_Model_and_Algorithm.tex`、`docs/derivation.md` 和 `RESEARCH_LOG.md`。

## 2. 当前多臂赌博机到底以什么作为 reward/regret

修正前，主稿中没有正式命名 reward 和 regret。经过后续讨论，当前口径进一步收敛为：使用 $T/G$ 作为 cost-type reward / TPOT cost，并最小化它；若需要标准“越大越好”的 reward 记号，则使用 $-T/G$。

当前已在 `docs/System_Model_and_Algorithm.tex` 中明确：

- 学习对象：任务-草稿模型平均接受率 $\alpha_{i,\tau}$。
- 该接受率是未知结构化参数，不是 reward。
- 协同推测解码候选的真实 cost-type reward 为：

$$
J^*_{i,g}(t)
=
\frac{T_i^{\mathrm{round}}(g\mid t)}
{G(\alpha^*_{i,\tau_t},g)}.
$$

- 若使用标准最大化 reward 记号，可写为负 TPOT：

$$
R^*_{i,g}(t)
=
-J^*_{i,g}(t).
$$

- 请求级 loss regret 为：

$$
\mathcal{R}_J(T)
=
\sum_{t=1}^{T}
\left[
J^*_{a_t}(t)-J^*_{a_t^*}(t)
\right].
$$

- 若关心总解码时延差异，则使用输出长度加权：

$$
\mathcal{R}_{\mathrm{lat}}(T)
=
\sum_{t=1}^{T}
Y_t^{\mathrm{out}}
\left[
J^*_{a_t}(t)-J^*_{a_t^*}(t)
\right].
$$

因此，本文多臂赌博机最合适的表述是：结构化 cost-UCB 学习 $\alpha_{i,\tau}$，并用其上置信界构造乐观 TPOT cost 下界，选择该下界最小的动作。

## 3. 是否有更好的 reward/regret 变量

结论：有。不要使用接受率本身作为 reward。

推荐变量：

1. 主 cost-type reward：$\frac{T_{\mathrm{round}}(g,t)}{G(\alpha,g)}$  
   这是 TPOT，和会议稿 D-PROMISE 的目标一致，也更适合做 latency minimization。

2. 标准最大化 reward：$-\frac{T_{\mathrm{round}}(g,t)}{G(\alpha,g)}$  
   若论文需要使用“reward 越大越好”的标准 MAB 记号，使用负 TPOT。

3. 主 regret：请求级 TPOT regret  
   适合理论分析和 oracle comparison。

4. 实验 regret/性能指标：输出长度加权 latency regret  
   更接近真实 serving 的总时延差异。

不推荐变量：

- $\alpha$：只衡量模型对齐程度，忽略负载、通信和验证时延。
- $\frac{A(\alpha,g)}{T}$ 或直接用 $\frac{G(\alpha,g)}{T}$ 作为主优化量：前者缺少 bonus/fallback token，后者虽与 $T/G$ 等价但会和本稿的 TPOT minimization 口径不一致。

## 4. 已发现并修正的问题

| 问题 | 修正 |
|---|---|
| UCB 冷启动时先计算 $C/V$，但 $V=0$ 会除零 | 先判断 $V=0$；未验证任务-模型对使用 $\overline{\alpha}=1$ 和冷启动长度 |
| 边侧用 $T/G$，终端动态起草用 $A/T$，目标不一致 | 统一为端到端有效 token 数 $G=1+A$，终端速率改为 $(1+A_\ell)/T_{\mathrm{round}}$ |
| 主稿没有正式 reward/regret | 新增 reward、loss、oracle action、请求级 regret 和 latency-weighted regret |
| 缺少 target-only fallback | 新增动作 $a_0$，fallback TPOT 为 $J_0(t)=T^{\mathrm{tar}}(t)$ |
| $g_t^{\mathrm{plan}}$ 被边侧选择后，终端算法没有真正使用 | 将 $g_t^{\mathrm{plan}}$ 改为终端动态起草的本轮上限 |
| 第一个 token 触发低接受率阈值时返回非法 $z_{1:1}$ | 首 token 特判时先加入 $z_1,H_1$，再返回长度 1 |
| 标准 speculative decoding 下反馈集合定义过宽 | 明确只用 accepted prefix 加首个 rejected token；之后 token 为 counterfactual/censored |
| 请求级总时延没有闭合 | 新增 $T_{\mathrm{req}}\approx Y_{\mathrm{out}}T_{\mathrm{round}}/G$ |
| 验证时延没有说明 bonus/fallback 位置 | 明确 $T^{\mathrm{ver}}(g,t)$ 包含 $g$ 个草稿位置和 1 个 bonus/fallback 位置 |
| $S_{\mathrm{dist}}$ 未定义清楚 | 说明其由验证协议决定，可为 sampled-token probability、top-k logits 或压缩分布 |
| 熵接受率乘积公式缺少条件概率解释 | 明确 $a_r$ 是给定前缀已接受条件下的 token 接受概率 |
| 熵曲线窗口集合固定写成 $M$ 个点 | 改为最近至多 $M$ 个点，并定义 $M_{\min}$ |
| 线性拟合未约束预测值在 `[0,1]` | 增加观测点上的区间约束 |
| 设备不可用或 $B_i(t)=0$ 未处理 | 明确跳过该设备或令候选时延为 $+\infty$ |
| 任务 profiling 过于抽象 | 补充说明当前作为给定模块，会议版本可用任务熵和 reasoning intensity 分类 |

## 5. 仍需后续理论或实验支撑的点

以下不是本次可直接“修公式”解决的问题，但需要在后续论文中补齐：

- 严格 regret 上界：当前已定义 regret，但还未证明 sublinear bound。
- 选择性观测偏差：动态停止会改变被观测 token 的熵分布，后续需要分析其对 $\alpha$ 和 $a(H)$ 估计的影响。
- token-level IID 假设：UCB 半径仍以 verified token 数作为样本数，需要说明有效样本数或相关性修正。
- 非平稳性：真实 LLM serving 中接受率、负载和信道可能漂移，后续可使用 sliding-window/discounted UCB。
- 任务分类误差：当前将误差吸收到预测标签下的统计中，更严格模型可扩展为 contextual bandit。
- 会议稿 `conference_paper.txt` 中存在 uplink/downlink 术语不一致；该文件像 PDF 提取文本，本次未直接修改。

## 6. 修改文件

- `docs/System_Model_and_Algorithm.tex`
- `docs/derivation.md`
- `RESEARCH_LOG.md`
- `report/mab_model_algorithm_review.md`
