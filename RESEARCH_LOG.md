# SCI/TMC 论文研究路线图

## 当前论文目标

- 目标期刊级别：IEEE Transactions on Mobile Computing (TMC) 或同等级 SCI 期刊。
- 研究主题：面向 LLM 推理的边缘-终端协同推测解码。
- 当前核心想法：边缘侧部署目标模型，终端侧提供异构草稿模型；请求到达并分类后，边缘节点在任务-草稿模型接受率未知、终端负载变化、无线状态变化的条件下，用多臂赌博机选择协同推理草稿模型，并由被选草稿模型根据 token 熵自适应决定草稿序列长度。
- 当前算法方向：边侧使用 UCB 学习不同任务下各草稿模型的平均 token 接受率，并按系统状态最小化平均有效 token 时延 $T/G$；这里 $T/G$ 是 cost-type reward，若使用标准“reward 越大越好”记号，可等价写为 $-T/G$。终端侧维护接受率随 token 熵单调递减的线性曲线，用于动态起草停止。

## 当前工作目标

- 将现有的请求级 TPOT + structured UCB 模型升级为具备 TMC 期刊级完整性的系统与算法贡献。
- 第一优先级：先强化数学模型，再扩展完整论文写作。
- 当前主要源文件：
  - `docs/derivation.md`
  - `docs/System_Model_and_Algorithm.tex`
- `docs/System_Model_and_Algorithm.tex` 已改为中文 LaTeX 建模文档，后续该文件正文、标题、定理环境名称、算法标题和说明文字均使用中文书写。

## 当前模型状态

- 已将请求级调度框架重构为两层决策：边侧服务器选择 target-only fallback 或草稿模型-规划长度 $(d_i,g_t^{\mathrm{plan}})$，终端侧草稿模型在该规划长度上限内自适应生成长度为 $g_t$ 的草稿序列。
- 旧版固定草稿长度 TPOT 基线为：
$$
J_{i,\gamma}(t)=\frac{T_{\mathrm{iter}}(d_i,\gamma\mid S(t))}{G(\alpha,\gamma)}.
$$
- 新版边侧模型选择指标为乐观平均有效 token 时延：
$$
\underline{J}_{i,g}(t)=
\frac{
T_i^{\mathrm{round}}(g\mid t)
}{
G(\overline{\alpha}_{i,\tau_t}(t),g)
},
$$
并在 $\{J_0(t)\}\cup\{\underline{J}_{i,g}(t)\}$ 中选择最小值。$\underline{J}_{i,g}(t)$ 是由 $\overline{\alpha}_{i,\tau}$ 诱导的乐观 TPOT cost 下界，不是对每个 $(i,g)$ 直接估计的普通 reward 均值。
- prefill latency 在当前问题范围内与动作无关，因此暂不纳入优化目标。
- 终端草稿模型推理时延已采用基于 profiling 的内存带宽退化抽象。
- 无线通信时延使用有效上行吞吐率建模，并考虑 token ID 与 token 分布信息的上传负载：
$$
S_{\mathrm{tok}} = S_{\mathrm{id}} + S_{\mathrm{dist}}.
$$
- 接受率学习已经修正为只使用目标模型实际验证过的 draft token。
- 每轮推测解码中首个 rejected draft token 之后的 token 被视为 censored observation，不进入接受率估计。
- 当前在线求解器为针对任务-草稿模型平均接受率的 UCB 草稿模型选择器，并已加入 target-only fallback 动作 $a_0$。
- 已加入熵感知动态起草：每个 $(i,\tau)$ 维护单调递减线性曲线
$$
a_{i,\tau}(H)=
\left[\theta_{i,\tau,0}-\theta_{i,\tau,1}H\right]_{[0,1]},
\quad \theta_{i,\tau,1}\geq 0.
$$
- 草稿模型逐 token 计算分布熵，根据预测接受率和预测 TPOT cost $\widehat{J}_\ell=T^{\mathrm{round}}(\ell\mid t)/\widehat{G}_\ell$ 动态停止起草；边侧规划长度 $g_t^{\mathrm{plan}}$ 作为终端本轮动态起草上限。
- 请求结束后，使用目标模型实际验证过的 token 更新平均接受率，并使用最近 $10$ 个验证样本拟合修正熵-接受率线性曲线。
- `docs/System_Model_and_Algorithm.tex` 当前可通过 XeLaTeX 编译生成 PDF。

## 距离 TMC 级别的主要差距

### 1. 理论闭环

- 当前主线已经从固定草稿长度 structured UCB 切换为“边侧 UCB 选草稿模型 + 终端熵感知动态起草”。
- 已在主稿中补充 cost-type reward/regret 定义：接受率不是最终 reward，而是结构化未知参数；本文 MAB 优化量为 $J=T/G$，即越小越好的 TPOT cost-type reward；若需要标准 reward，可写为 $R=-J$。请求级 regret 为相对 oracle 动作的 TPOT cost 差。
- 仍需进一步证明严格 regret 上界或 oracle comparison，并界定熵曲线估计误差、动态停止策略与选择性观测对 regret 的影响。

### 1.1 当前主线：基于草稿分布熵的动态草稿长度

- 当前思想：边侧多臂赌博机学习不同任务下各草稿模型的平均接受率，终端侧学习草稿模型在任务下的接受率与 draft token 分布熵之间的函数关系。
- 调度思想：每轮起草时，草稿模型根据已生成 token 的分布熵动态决定是否继续起草，而不是预先固定草稿长度 $\gamma$。
- 潜在优势：更贴近 speculative decoding 的实际逐 token 起草过程，有机会减少低置信 token 的无效通信与验证开销。
- 已有实验观察：接受率与草稿 token 分布熵之间存在明显相关性，因此该方向具备作为期刊主线的潜力。
- 候选参数化形式：将每个草稿模型在任务 $\tau$ 下的接受率建模为熵的线性递减函数：
$$
\alpha_{i,\tau}(H)=\left[\theta_{i,\tau,0}-\theta_{i,\tau,1}H\right]_{[0,1]},
\quad \theta_{i,\tau,1}\geq 0,
$$
其中 $H$ 是 draft token 的分布熵，$\theta_{i,\tau,0}$ 和 $\theta_{i,\tau,1}$ 是需要在线学习的参数，$[\cdot]_{[0,1]}$ 表示截断到 $[0,1]$。
- 当前状态：已作为 `docs/System_Model_and_Algorithm.tex` 的新版主线写入，并通过 XeLaTeX 编译验证。

### 2. censored feedback 学习

- 新模型中，平均接受率和熵-接受率曲线都只能使用目标模型实际验证过的 token。
- 需要进一步说明首个 rejected token 之后的 proposed tokens 为什么不能作为拒绝样本。
- 需要分析近 $10$ 次窗口拟合在 censored feedback 下是否会产生选择偏差。

### 3. target-only fallback

- 主稿已加入 fallback action $a_0$，fallback TPOT 为纯目标模型解码时的 $J_0(t)=T^{\mathrm{tar}}(t)$。
- 后续仍需在实验中验证 fallback 被触发的条件，并将其作为 baseline 与调度动作一起汇报。

### 4. 目标模型验证时延

- 当前 $T_{\mathrm{verify}}$ 仍然过粗。
- 需要将其细化为与草稿长度、上下文长度、边缘侧负载和 batching 状态有关的 profiling 函数。
- 候选形式：
$$
T_{\mathrm{verify}}(\gamma,H_t,L_{\mathrm{edge}}(t)).
$$

### 5. 任务 profiling 与任务不确定性

- 当前任务类型 $\tau_t$ 只是抽象有限标签。
- 需要定义任务如何 profiling 或分类。
- 需要考虑分类器不确定性、任务标签噪声或上下文特征。
- 候选扩展：使用任务 embedding、entropy、reasoning intensity 或分类置信度构造 contextual bandit。

### 6. 非平稳性

- stationary IID acceptance 假设对真实 LLM serving 场景过强。
- 需要引入 discounted UCB 或 sliding-window UCB。
- 如果将非平稳性作为主贡献，regret analysis 应包含 drift 或 variation-budget 项。

### 7. 系统与实验支撑

- 需要获得 draft latency、communication latency、verification latency 和 acceptance behavior 的 profiling 数据。
- 需要在终端异构性、无线 trace、任务类型和草稿长度等维度上进行实验。
- 需要 ablation study 证明每个模型组件确实有贡献。

## 完整论文仍缺失的组成部分

- 完整论文题目与贡献定位。
- Introduction 中相对会议版本的清晰增量说明。
- Related work：
  - speculative decoding；
  - edge/mobile LLM inference；
  - distributed inference；
  - online learning 与 bandits；
  - wireless-aware computation offloading。
- 完整的 TMC 级系统模型。
- 带 fallback action 的最终优化问题。
- 熵感知动态起草的理论性能分析。
- censored-feedback 下平均接受率与熵曲线的在线学习算法。
- 主要理论保证。
- 算法复杂度分析。
- 实验设置。
- baselines。
- 真实 profiling 或 trace-driven profiling 方法。
- 结果图与结果表。
- ablation studies。
- limitations 与 discussion。
- 目标期刊模板下的最终 LaTeX 论文。

## 需要包含的 baselines

- 纯目标模型解码。
- 随机选择终端与草稿长度。
- latency-only greedy selection。
- acceptance-only greedy selection。
- 已知接受率的 deterministic oracle。
- 会议版本 D-PROMISE 风格 profiling scheduler。
- 固定草稿长度的 structured UCB。
- 不含熵感知动态停止的平均接受率 UCB。
- 本文提出的边侧 UCB + 熵感知动态起草 learner。
- 可选：Thompson sampling 或 discounted UCB 变体。

## 近期待办任务

1. 在 `docs/derivation.md` 中同步写入新的两层决策数学模型。
2. 为边侧 UCB 有效接受 token 速率选择规则补充 regret 或 oracle comparison 推导。
3. 明确 $\widehat{g}_{i,\tau}(t)$ 的初始化、更新和误差界。
4. 细化 $T^{\mathrm{ver}}(g,t)$ 为关于草稿长度、上下文长度和边缘侧负载的 profiling 函数。
5. 进一步推导 target-only fallback 与协同推测解码的选择条件，并在实验中验证。
6. 基于已有实验数据拟合 $a_{i,\tau}(H)$，检验线性递减假设的拟合优度、置信区间和跨任务稳定性。
7. 评估近 $10$ 次窗口拟合是否过短，并比较 $M=10,20,50$ 的稳定性与响应速度。
8. 设计仿真与 profiling 流程，并确定所需数据集或 trace。
9. 决定非平稳性是作为主算法贡献还是扩展讨论。

## 当前优先级

- 暂时不要开始润色完整论文。
- 先将 `docs/derivation.md` 与新版 `docs/System_Model_and_Algorithm.tex` 对齐。
- 下一步优先补齐新模型的理论分析，而不是继续润色文字。
