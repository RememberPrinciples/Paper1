# 最近三次需要缓存的回答结果

## 回答 9：重构 System_Model_and_Algorithm 为两层草稿模型选择与动态起草模型

已将 `docs/System_Model_and_Algorithm.tex` 从旧版“边侧选择 $(d_i,\gamma)$ 的固定草稿长度 structured UCB”重构为新版两层模型：

1. 边侧服务器在请求到达并分类后，用 UCB 选择协同推理草稿模型 $d_i$。
2. 被选草稿模型在终端侧根据每个 draft token 的生成熵动态决定停止位置，上传动态长度草稿序列。

新版边侧状态为：

$$
S(t)=\{\tau_t,\mathbf{L}(t),\mathbf{B}(t)\},
$$

其中 $\tau_t$ 是任务类型，$\mathbf{L}(t)$ 是终端负载向量，$\mathbf{B}(t)$ 是上行有效吞吐率向量。

边侧服务器对每个草稿模型-任务类型对 $(i,\tau)$ 维护平均接受率：

$$
\widehat{\alpha}_{i,\tau}(t)
=
\frac{C_{i,\tau}(t)}{V_{i,\tau}(t)}.
$$

并构造 UCB 乐观接受率：

$$
\overline{\alpha}_{i,\tau}(t)
=
\min
\left\{
1,\;
\widehat{\alpha}_{i,\tau}(t)
+
r_{i,\tau}(t)
\right\},
$$

其中

$$
r_{i,\tau}(t)
=
\sqrt{
\frac{
\log\!\left(2N|\mathcal{T}|t^2/\delta\right)
}
{2V_{i,\tau}(t)}
}.
$$

在“草稿模型会产出长度为 $\widehat{g}_{i,\tau_t}(t)$ 的草稿序列”的假设下，期望被接受 draft token 数为：

$$
A(\alpha,g)
=
\sum_{\ell=1}^{g}\alpha^\ell
=
\frac{\alpha(1-\alpha^g)}{1-\alpha}.
$$

边侧选择指标改为平均每秒有效接受 draft token 数：

$$
U_i(t)
=
\frac{
A(\overline{\alpha}_{i,\tau_t}(t),\widehat{g}_{i,\tau_t}(t))
}
{
T_i^{\mathrm{round}}(\widehat{g}_{i,\tau_t}(t)\mid t)
}.
$$

选择规则为：

$$
i_t
=
\underset{i\in[N]}{\mathrm{arg\,max}}\;U_i(t).
$$

终端动态起草部分加入了 token 熵：

$$
H_\ell
=
-\sum_{v\in\mathcal{V}}
q_{i_t,\ell}(v)\log q_{i_t,\ell}(v).
$$

每个 $(i,\tau)$ 维护单调递减线性接受率曲线：

$$
a_{i,\tau}(H;\theta_{i,\tau})
=
\mathrm{clip}_{[0,1]}
\left(
\theta_{i,\tau,0}
-
\theta_{i,\tau,1}H
\right),
\quad
\theta_{i,\tau,1}\geq 0.
$$

给定动态草稿长度 $\ell$，预测接受 draft token 数为：

$$
\widehat{A}_{\ell}
=
\sum_{k=1}^{\ell}
\prod_{r=1}^{k}\widehat{a}_r.
$$

预测有效接受 token 速率为：

$$
\widehat{R}_{\ell}
=
\frac{\widehat{A}_{\ell}}
{T_{i_t}^{\mathrm{round}}(\ell\mid t)}.
$$

动态停止规则包括两类条件：

$$
\widehat{a}_\ell < \eta_{\mathrm{acc}},
$$

或

$$
\widehat{R}_{\ell}
<
\widehat{R}_{\ell-1}
-
\epsilon_{\mathrm{stop}}.
$$

最终选择：

$$
g_t
=
\underset{1\leq m\leq \ell}{\mathrm{arg\,max}}\;\widehat{R}_m.
$$

请求结束后，边侧服务器只使用目标模型实际验证过的 token 更新统计量：

$$
C_{i_t,\tau_t}
\leftarrow
C_{i_t,\tau_t}
+
\sum_{o\in\mathcal{O}_t}Y_o,
$$

$$
V_{i_t,\tau_t}
\leftarrow
V_{i_t,\tau_t}
+
|\mathcal{O}_t|.
$$

首个 rejected token 之后未验证的 proposed tokens 不进入统计量。熵-接受率曲线使用最近 $M=10$ 个验证样本拟合修正：

$$
\min_{\theta_0,\theta_1}
\sum_{(H_m,Y_m)\in\mathcal{W}_{i,\tau}(t)}
\left(
Y_m-\theta_0+\theta_1H_m
\right)^2,
\quad
0\leq\theta_0\leq 1,\;\theta_1\geq 0.
$$

然后用平滑更新：

$$
\theta_{i,\tau}(t+1)
=
(1-\rho_\theta)\theta_{i,\tau}(t)
+
\rho_\theta\widetilde{\theta}_{i,\tau}(t).
$$

同时更新预测平均草稿长度：

$$
\widehat{g}_{i_t,\tau_t}(t+1)
=
(1-\rho_g)\widehat{g}_{i_t,\tau_t}(t)
+
\rho_g g_t.
$$

文档已通过 XeLaTeX 编译，输出：

```text
docs/System_Model_and_Algorithm.pdf
```

同时已更新 `RESEARCH_LOG.md`，将“熵感知动态草稿长度”从候选方向提升为当前主线，并将下一步任务调整为补齐新模型的理论分析、fallback action、验证时延 profiling 和实验设计。

## 回答 8：BanditSpec 如何实现多臂赌博机

BanditSpec 把 speculative decoding 中的“超参数配置选择”建模为多臂赌博机。这里的 arm 不是固定只能是草稿模型，而是一个候选 speculative decoding 配置：

$$
S=\{S_1,S_2,\ldots,S_K\}.
$$

每个 $S_i$ 可以是一个 draft model，也可以是一种 speculation length、tree structure，或其他 speculative decoding hyperparameter specification。第 $t$ 轮选择一个 arm：

$$
I_t\in[K],
$$

然后用配置 $S_{I_t}$ 调用 speculative decoding subroutine：

$$
X_{I_t,t}=\mathrm{SpecDecSub}(pt_{t-1},P,S_{I_t},L).
$$

这里 $P$ 是 target model，$L$ 是最大 speculation length，$X_{I_t,t}$ 是本轮被接受的 token 序列加 bonus token。BanditSpec 将本轮 reward 定义为接受 token 序列长度：

$$
Y_{I_t,t}=\mathrm{len}(X_{I_t,t}),
\qquad
Y_{I_t,t}\in[1,L+1].
$$

因此，arm 越好，平均每轮接受 token 数越多，需要的 target verification 轮数越少，整体 stopping time 越短。

在 stochastic setting 下，它假设每个 arm 有固定但未知的平均接受长度：

$$
\mathbb{E}[Y_{I_t,t}\mid H_{t-1},I_t=i]=\mu_i.
$$

最优 arm 是：

$$
i^\star=\arg\max_{i\in[K]}\mu_i.
$$

BanditSpec 的目标不是普通固定 horizon 下最大化累计 reward，而是最小化 stopping time regret，即和始终使用最优配置 $S_{i^\star}$ 相比，多用了多少 speculative decoding rounds。

它提出两个具体算法。

**第一，UCBSpec。**

UCBSpec 先 round-robin 试每个 arm 一次：

$$
I_t=t,\quad t\le K.
$$

之后统计每个 arm 被选择的次数：

$$
n_{i,t}=\sum_{s=1}^{t}\mathbf{1}\{I_s=i\},
$$

以及该 arm 的经验平均接受长度：

$$
\hat{\mu}_{i,t}
=
\frac{\sum_{s=1}^{t}Y_{I_s,s}\mathbf{1}\{I_s=i\}}{n_{i,t}}.
$$

然后构造置信半径 $cr_{i,t}$，形成 UCB index：

$$
\mathrm{UCB}_{i,t}=\hat{\mu}_{i,t}+cr_{i,t}.
$$

每轮选择：

$$
I_{t+1}=\arg\max_{i\in[K]}\mathrm{UCB}_{i,t}.
$$

直觉是：如果某个配置历史接受长度高，$\hat{\mu}_{i,t}$ 高，会被 exploitation；如果某个配置尝试次数少，$cr_{i,t}$ 大，会被 exploration。

**第二，EXP3Spec。**

EXP3Spec 面向 adversarial/non-stationary setting。它维护一个概率分布：

$$
p_t\in\Delta[K].
$$

每轮按概率采样 arm：

$$
I_t\sim p_t.
$$

它不是直接最大化 accepted length，而是把“没有达到最大接受长度”的部分定义为 loss：

$$
\ell_{i,t}=\frac{L+1-Y_{i,t}}{L}.
$$

由于 bandit feedback 只能观察被选中的 arm，所以 EXP3Spec 用重要性加权构造 loss estimator：

$$
\hat{Z}_{i,t}
=
\mathbf{1}\{i=I_t\}
\frac{L+1-Y_{I_t,t}}{L\,p_{t,i}}.
$$

然后用指数权重更新概率：

$$
p_{t,i}
=
\frac{
\exp\left(-\eta_t\sum_{s=1}^{t-1}\hat{Z}_{i,s}\right)
}{
\sum_{j=1}^{K}
\exp\left(-\eta_t\sum_{s=1}^{t-1}\hat{Z}_{j,s}\right)
}.
$$

其中学习率采用类似：

$$
\eta_t=\sqrt{\frac{\log K}{tK}}.
$$

所以，BanditSpec 和 HedgeSpec 的根本区别是反馈形式：

BanditSpec 每轮只观察被选 arm 的反馈：

$$
(I_t,Y_{I_t,t}).
$$

HedgeSpec 则试图用 target verified trajectory 反事实评估所有 drafters，得到 full-information loss vector：

$$
f_t=[f_t[1],\ldots,f_t[N]].
$$

因此 BanditSpec 更像经典 MAB：需要 exploration-exploitation；而 HedgeSpec 更像 full-information expert learning：每轮可以更新所有 expert。

## 回答 7：混合任务到达时 HedgeSpec 的乒乓效应问题

如果任务是混合到达的，而且系统只维护一个全局 drafter 权重向量：

$$
p_t=(p_t[1],\ldots,p_t[N]),
$$

那么确实可能出现乒乓效应。比如 Python 请求连续来了几轮，Python drafter 的权重升高；接着 Math 请求来了几轮，Math drafter 的权重升高；再切回 Python，权重又被拉回来。这个现象不是实现细节，而是模型目标本身导致的。

标准 Hedge/NormalHedge 在无上下文形式下竞争的是一个 best fixed expert in hindsight：

$$
\min_{i\in[N]}\sum_{t=1}^{T} f_t[i].
$$

也就是说，它假设存在一个在整个序列上平均最好的 drafter。如果请求序列本身是多模态的，例如 Python、Math、SQL、MedicalQA 混合到达，那么最优策略通常不是固定一个 drafter，而是一个随上下文切换的策略：

$$
\pi^\star(x_t)\in[N],
$$

其中 $x_t$ 表示当前 prompt、已生成上下文、任务类型或运行时状态。此时如果仍然用一个全局 $p_t$，学习器会在不同任务分布之间来回追赶，表现为权重震荡。

更合理的处理方式有四类。

第一，**每个请求维护独立的 Hedge/NormalHedge 状态**。也就是对每个 request/session 单独维护：

$$
p_{r,h},
$$

其中 $r$ 是请求编号，$h$ 是该请求内部的 speculative decoding chunk/token 轮次。这样一个 Python 请求内部会逐渐偏向 Python drafter，一个 Math 请求内部会逐渐偏向 Math drafter，彼此不互相污染。代价是每个请求都存在冷启动。

第二，**使用全局先验 + 请求内在线适应**。全局层维护一个先验分布 $p_0$，新请求到达时用它初始化：

$$
p_{r,1}=p_0.
$$

然后请求内部根据 target verification feedback 更新 $p_{r,h}$。请求结束后，可以把该请求的统计信息缓慢回写到全局先验，而不是直接覆盖：

$$
\log p_0[i]\leftarrow (1-\rho)\log p_0[i]+\rho \log \bar p_r[i].
$$

这里 $\rho$ 应该较小，用于避免全局先验被最近几个任务快速拉偏。

第三，**做 contextual Hedge / contextual expert selection**。把选择概率写成上下文相关形式：

$$
p_t[i]=\pi_\theta(i\mid x_t),
$$

其中 $x_t$ 可以包括 prompt embedding、已生成 token 的统计特征、draft entropy、任务标签、请求来源、历史接受率等。这样系统学习的是“在什么上下文下选择哪个 drafter”，而不是一个无条件全局权重。

第四，**允许切换但惩罚过度切换**。如果一个请求内部确实是混合任务，例如先写 SQL 再解释医学文本，可以用 tracking-the-best-expert 或 fixed-share Hedge：

$$
w_{i,t+1}
=
(1-\alpha)\tilde w_{i,t+1}
+
\frac{\alpha}{N-1}\sum_{j\ne i}\tilde w_{j,t+1},
$$

其中 $\alpha$ 是切换率。它允许策略在不同专家之间切换，但不会因为一两个 chunk 的短期波动就剧烈振荡。

因此，混合到达场景下不建议把 HedgeSpec 实现成一个单一全局 learner。更稳妥的系统设计是：

$$
\text{全局先验}
\rightarrow
\text{请求内 Hedge/NormalHedge}
\rightarrow
\text{可选的上下文路由或慢速全局回写}.
$$

这样既能避免静态任务分类器的误路由，又能避免全局在线权重在混合流量中来回乒乓。

## 回答 9：接受率上置信界中为什么有 \(N\)，以及推导依据

接受率上置信界中的 \(N\) 表示候选草稿模型或终端设备数量，也就是 bandit 问题中的 arm 数。它出现在

$$
r_{i,\tau}(t)
=
\sqrt{
\frac{\log(2N|\mathcal{T}|t^2/\delta)}
{2V_{i,\tau}(t)}
}
$$

的分子里，是因为这里使用的不是只对单个 \((i,\tau)\) 和单个时刻成立的置信界，而是一个 uniform high-probability bound，需要同时覆盖：

1. 所有设备或草稿模型 \(i=1,\ldots,N\)；
2. 所有任务类型 \(\tau\in\mathcal{T}\)；
3. 所有在线决策时刻 \(t\)。

具体推导来自 Hoeffding inequality 加 union bound。对固定的 \((i,\tau,t)\)，如果已验证 draft token 的接受标签看成 Bernoulli 样本，则经验接受率满足

$$
\Pr\left(
\left|\widehat{\alpha}_{i,\tau}(t)-\alpha_{i,\tau}\right|
\ge r
\right)
\le
2\exp\left(-2V_{i,\tau}(t)r^2\right).
$$

如果希望所有 \(N|\mathcal{T}|\) 个任务-设备对、所有时刻都以高概率成立，可以把单个事件的失败概率设为

$$
\frac{\delta}{N|\mathcal{T}|t^2}.
$$

令

$$
2\exp\left(-2V_{i,\tau}(t)r^2\right)
=
\frac{\delta}{N|\mathcal{T}|t^2},
$$

解得

$$
r
=
\sqrt{
\frac{\log(2N|\mathcal{T}|t^2/\delta)}
{2V_{i,\tau}(t)}
}.
$$

因此，\(N\) 的作用是为“所有设备同时成立”付出的 union bound 代价。设备越多，需要同时覆盖的置信事件越多，置信半径越保守。

这个推导本质上基于经典 Hoeffding-UCB/UCB1 分析，不是 speculative decoding 领域某一篇专门论文独有的公式。最合适引用的是：

1. P. Auer, N. Cesa-Bianchi, and P. Fischer, "Finite-time Analysis of the Multiarmed Bandit Problem," Machine Learning, 2002.
2. T. Lattimore and C. Szepesvari, Bandit Algorithms, Cambridge University Press, 2020.

更准确地说，Auer et al. 2002 是 UCB1 的经典 finite-time MAB 来源；本文这里的形式是在 Hoeffding confidence radius 基础上，对任务类型 \(|\mathcal{T}|\)、设备数 \(N\) 和时间 \(t\) 做 uniform high-probability union bound 后得到的变体。

## Algorithm 1 English Version for PPT Screenshot

```latex
\begin{algorithm}[t]
\caption{UCB-Based Draft Model Selection at the Edge Server}
\label{alg:edge_ucb_selection_en}
\begin{algorithmic}[1]
\Require Task type $\tau_t$; system state $\{\mathbf{L}(t),\mathbf{B}(t)\}$; acceptance statistics $\{C_{i,\tau},V_{i,\tau}\}$; cold-start lengths $\{g_{i,\tau}^{0}\}$
\Ensure Selected draft model $d_{i_t}$ and planned draft length $g_t^{\mathrm{plan}}$
\State $\overline{J}_{\min}\gets+\infty$
\For{$i=1,\ldots,N$}
    \State Compute the empirical acceptance rate $\widehat{\alpha}_{i,\tau_t}(t)=C_{i,\tau_t}(t)/V_{i,\tau_t}(t)$
    \State Compute the UCB acceptance estimate $\overline{\alpha}_{i,\tau_t}(t)=\min\{1,\widehat{\alpha}_{i,\tau_t}(t)+r_{i,\tau_t}(t)\}$
    \State Compute the load-dependent draft-token latency $\Phi_i(L_i(t))$
    \If{$V_{i,\tau_t}(t)=0$}
        \State $\mathcal{G}_i(t)\gets\{g_{i,\tau_t}^{0}\}$
    \Else
        \State $\mathcal{G}_i(t)\gets\{1,\ldots,g_{\max}\}$
    \EndIf
    \For{$g\in\mathcal{G}_i(t)$}
        \State Compute the round latency $T_i^{\mathrm{round}}(g\mid t)$ using $B_i(t)$ and $L_i(t)$
        \State Compute the optimistic TPOT $\overline{J}_{i,g}(t)=T_i^{\mathrm{round}}(g\mid t)/G(\overline{\alpha}_{i,\tau_t}(t),g)$
        \If{$\overline{J}_{i,g}(t)<\overline{J}_{\min}$}
            \State $\overline{J}_{\min}\gets\overline{J}_{i,g}(t)$
            \State $i_t\gets i$, $g_t^{\mathrm{plan}}\gets g$
        \EndIf
    \EndFor
\EndFor
\State \Return $d_{i_t}$, $g_t^{\mathrm{plan}}$
\end{algorithmic}
\end{algorithm}
```

For a compact PPT screenshot, the equations used in Algorithm 1 can be placed below the algorithm as a small notation block:

```latex
\[
r_{i,\tau_t}(t)=
\sqrt{\frac{\log(2N|\mathcal{T}|t^2/\delta)}{2V_{i,\tau_t}(t)}},
\quad
G(\alpha,g)=\sum_{\ell=0}^{g}\alpha^\ell,
\quad
\overline{J}_{i,g}(t)=
\frac{T_i^{\mathrm{round}}(g\mid t)}
{G(\overline{\alpha}_{i,\tau_t}(t),g)}.
\]
```
