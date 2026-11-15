# 4 Robot (Imitation) Learning

```text
行为克隆（Behavioral Cloning）提供了一个天然的平台，可以基于真实环境交互数据进行学习，而无需设计任何奖励函数（reward function）；同时在处理多模态演示数据集（multimodal demonstration datasets）时，生成式模型（generative models）相比逐点策略（point-wise policies）具备更优的效果。
```



从人类演示（human demonstrations）中学习，是第 3 节所讨论的强化学习（RL）流程一套具备实用性的替代方案。事实上，尤其是在现实机器人领域，在线探索（online exploration）通常成本高昂，并且存在潜在安全风险；而设计（稠密）奖励信号（dense reward signals）是一项稳定性差、高度依赖特定任务的工作。此外，就连任务成功检测本身，往往都需要定制化的测量设备；同时回合式训练（episodic training）要求能够可靠地重置环境——以上这些因素，都加大了在实体硬件上大规模训练强化学习算法的难度。行为克隆（Behavioral Cloning, BC）把控制问题转化为一个模仿学习（imitation learning）问题，从而避开了上述限制：它利用预先采集的专家演示数据，为学到的自主行为提供基准约束。最为重要的一点是，通过学习模仿，自主系统能够自然而然地遵循隐含编码在中的目标、偏好以及成功判定标准。

形式化地，设 $D = \{\tau^{(i)}\}_{i=1}^{N}$ 为一组专家轨迹（expert trajectories）集合，其中 $\tau^{(i)} = \{(o_t, a_t)\}_{t=0}^{T_i}$ 代表集合 $D$ 中第 $i$ 条长度为 $T_i$ 的轨迹；$o_t \in \mathcal{O}$ 表示观测（observations，例如图像与本体感知信息的组合），$a_t \in \mathcal{A}$ 为专家动作（expert actions）。

一般而言，观测 $o \in \mathcal{O}$ 同时包含图像信息与本体感知（proprioceptive）信息；而动作 $a \in \mathcal{A}$ 代表机器人需要执行的控制指令，例如关节位形（joint configuration）。请注意，与第 3 节不同：在模仿学习（imitation learning）语境下，$D$ 指代一个离线数据集，它收集了 $N$ 条长度为 $T_i$、不含奖励的人类专家轨迹 $\tau^{(i)}$，而非环境动力学（environment dynamics）。同样，本节中的 $\tau^{(i)}$ 是由观测–动作对构成、长度为 $T_i$ 的轨迹，关键特点是完全不包含任何奖励信息。图 18 以图像形式展示了遥操作 SO‑100 机械臂六条关节驱动量的平均变化趋势。可以看到，在录制回合（episodes）过程中，本体感知状态与相机画面同步采集，得到一套统一、高帧率的图像与关节遥操作混合数据。图 19 展示了同一数据集里的 $(o_t, a_t)$ 对，同时给出人类专家执行的动作以及与之对应的观测。原则上，各条专家轨迹 $\tau^{(i)}$ 的长度可以互不相同：为达成同一个目标，演示过程可能呈现多模态策略（multi-modal strategies），由此产生多种截然不同的行为。

<p align="center">
  <img src="images/figure_19.png">
</p>

<p align="center">
  <strong>图 19</strong>｜在数据集 <code>lerobot/svla_so101_pickplace</code> 内某条录制轨迹上采样得到的观测与动作对。观测同时包含本体感知（proprioceptive）与视觉信息，并与由人类专家操控的主机器人（leader robot）的位形同步记录；这为基于观测回归动作提供了完整信息。
</p>

行为克隆（Behavioral Cloning, BC）（Pomerleau, 1988）旨在通过学习从观测到动作的映射来生成合成行为；在最直观的形式化定义下，该问题可以当作一个监督学习（supervised learning）问题高效求解：给定任意风险函数 $L: \mathcal{A} \times \mathcal{A} \mapsto \mathbb{R},\; L(a, a')$，学习（确定性）映射 $f: \mathcal{O} \mapsto \mathcal{A}$，满足 $a_t = f(o_t)$。

$$
\min_{f}\; \mathbb{E}_{(o_t,a_t)\sim p(\cdot)} \, L\big(a_t,\, f(o_t)\big), \tag{18}
$$


通常我们假设专家的观测‑动作联合分布（joint observation-action distribution）$p: \mathcal{O} \times \mathcal{A} \mapsto [0,1]$ 是未知的，这一点与经典监督学习（Supervised Learning, SL）框架保持一致<sup>3</sup>。但与标准监督学习的假设不同：数据集 $D$ 中的采样（也就是底层分布 $p$ 的实现样本）并不满足独立同分布（i.i.d.），因为专家演示是以轨迹形式依次采集得到的。在实际操作中，可以通过打乱顺序、非序列化地取用样本，也就是对数据集 $D$ 内样本做混洗（shuffling），在一定程度上缓解该问题；借此可以利用蒙特卡洛估计（MC estimates）近似得到分布 $p$ 下的期望风险，不过这类估计一般而言精度会有所下降。另一种降低在非独立同分布样本上做回归所带来负面影响的方案，是交替执行行为克隆与数据采集（Ross 等人，2011，DAgger），通过迭代方式合并多份数据集。但本文仅考虑仅有一份离线轨迹数据集 $D$、无法继续采集新数据的场景，因此 DAgger 不在我们的讨论范围内。尽管从非独立同分布（non-i.i.d.）数据中学习存在固有的难点，但行为克隆（BC）范式在机器人领域具备多项工程优势。首先，训练过程是离线（offline）进行的，能够天然适配专家演示数据；它将动作约束在模仿的范围内，从根源上杜绝机器人执行危险动作，大幅降低探索带来的风险。其次，行为克隆完全不需要设计奖励，因为演示本身就已经体现了人类的意图。不再依赖奖励，也就规避了纯粹基于奖励的强化学习中容易出现的目标错位（misalignment）与投机取巧（specification gaming，也叫 reward hacking / 奖励破解）风险（Heess 等人，2017）。第三，由于专家轨迹中隐含了终止条件（terminal conditions），任务成功判定与环境重置都已经内嵌在数据集之中。最后，实验结果表明：随着跨任务、跨本体、跨环境采集的演示数据体量不断增加，行为克隆的效果可以自然地随之提升。不过从原理上来说，行为克隆最多只能复刻出和示教者水平相当的行为，无法修正人类做出的次优决策。这一局限在序列决策（sequential decision-making）任务中尤为棘手——造成该问题的原因，或是数据采集成本高昂，或是人类本身的示教表现就属于次优。即便如此，很多机器人应用依然可以借助成本相对可控的流程采集高质量的人工示教轨迹，这也证明在这类场景下选用行为克隆是合理的。

<p align="center">
  <img src="images/figure_20.png">
</p>

<p align="center">
  <strong>图 20</strong>｜逐点策略（point-wise policies）存在两处局限：（A）协变量偏移（covariate shift）；（B）对多模态示教数据拟合效果差。（A）微小的预测误差就可能让策略进入分布外状态，形成恶性循环，最终导致任务失败。（B）场景中伸向目标物体存在两种可行方式——先从左侧靠近或是先从右侧靠近，二者效果等价，因此在人类示教数据集里出现的概率相当，最终形成多模态示教样本。

</p>


<p align="center">
  <img src="images/figure_21.png">
</p>

<p align="center">
  <strong>图 21</strong>｜直观来看，单一隐变量模型（single latent model）中的隐变量（latent variable）可以包含当前执行任务的相关信息；这会直接造成：对于两组不同任务，同一组观测‑动作对（observation-action pair）对应的似然（likelihood）并不相同。例如：（A）抓取方块时，夹爪张开幅度较大对应的似然，应当高于小幅张开；而（B）推动方块时，情况则恰好相反。

</p>

尽管在概念上十分简洁优美，但通过求解式 (18) 得到的点估计策略（point-estimate policies）$f: \mathcal{O} \mapsto \mathcal{A}$，存在两个已知缺陷：（1）误差累积（compounding errors）（Ross 等人，2011）；（2）难以拟合多模态分布（multimodal distributions）（Florence 等人，2022；Ke 等人，2020）。图 20 展示了这两个和显式策略学习相关的核心问题（Florence 等人，2022）。除数据集 $D$ 本身具有时序性之外，由协变量偏移（covariate shift）引发的误差累积可能带来严重后果：即便只是很小的预测误差 $0 < \|\mu(o_t) - a_t\| \le \epsilon$，也会很快让策略进入分布外（out-of-distribution）状态，生成结果的置信度随之下降，进而造成误差不断累积（图 20，左）。此外，点估计策略通常无法学习多模态目标；而在解决现实机器人任务的人类示教数据中，多模态目标十分常见——为实现同一个目标，可以存在多条效果等价的轨迹（例如对称抓取，见图 20，右）。具体而言，单模态回归器（unimodal regressors）倾向于对各个模态取平均，输出的控制指令犹豫不决，甚至存在安全隐患（Florence 等人，2022）。为解决多模态拟合效果差这一问题，Florence 等人（2022）提出不去显式学习预测函数 $f: a = f(o)$，而是学习数据集 $D$ 的样本背后的生成模型（generative model）$p(o,a)$。


## 4.1 A (Concise) Introduction to Generative Models

生成式模型（Generative Models, GMs）旨在学习采集数据背后用于生成样本的随机过程；通常的实现方式是拟合一个概率分布，用来逼近未知的数据分布 $p$。遵循生成式模型领域的行文习惯，记 $p(x) \leftarrow P(x),\; x \sim p$。

在行为克隆场景下，未知的数据分布 $p$ 可以表示专家在 $(o,a)$ 对上的联合分布。因此，给定由 $N$ 组样本对构成的有限数据集 $D = \{(o,a)_i\}_{i=0}^{N}$，将其作为模仿学习的目标（并假设样本独立同分布），生成式模型希望学到一个参数化分布 $p_\theta(o,a)$，满足两点要求：
（1）新采样得到的样本 $(o,a) \sim p_\theta(\cdot)$ 和数据集 $D$ 内的样本风格相近；
（2）对于无法直接观测的原始分布 $p$，在它那些已经被观测到的区域上赋予较高的似然。

基于似然的学习为同时达成上述两个目标提供了一套具备理论依据的训练目标，因此在生成式模型中得到广泛使用（Prince, 2023）。

<p align="center">
  <img src="images/figure_22.png">
</p>

<p align="center">
  <strong>图 21</strong>｜（A）机器人应用中的隐变量模型（latent variable model）控制着可观测变量 $(o,a)$ 与不可观测隐变量（latent variable）之间的相互影响。（B）变分自编码器（VAEs）借助变分推断（variational inference）对精确隐变量模型做近似。

</p>


### 4.1.1 Variational Auto-Encoders

生成式模型（Generative Models, GMs）中一种常用的归纳偏置（inductive bias）假定：样本 $(o,a)$ 受到一个不可观测的隐变量 $z \in \mathcal{Z}$ 的影响，由此可得：

$$
p(o,a) = \int_{\mathrm{supp}(\mathcal{Z})} p(o,a \mid z)\, p(z) \tag{19}
$$

式 (19) 的含义：在隐变量空间的支撑集（support，$\mathrm{supp}(\mathcal{Z})$）上对隐变量 $z$ 积分，从而得到观测‑动作对的边缘分布 $p(o,a)$；其中 $p(z)$ 是隐变量的先验分布，$p(o,a\mid z)$ 为给定隐变量 $z$ 条件下观测与动作的条件分布。


直观而言，对于机器人任务中的观测‑动作对 $(o,a)$，$z$ 可以理解为人类示教者正在执行任务的某种高层表征。在这种设定下，将 $p(o,a)$ 看作完整联合分布 $p(o,a,z)$ 在 $\mathrm{supp}(\mathcal{Z})$ 上求边缘分布，能够自然刻画不同任务对观测‑动作对似然的影响。图 21 通过（A）抓取任务与（B）推动任务直观展示该思想：当靠近目标物体时，使夹爪张开的动作对应的似然（$q_6$ 越大，夹爪张开幅度越大）会随任务发生变化——在抓取任务中该似然较高，而在推动任务中较低。尽管隐空间 $\mathcal{Z}$ 的结构通常远比全部实际任务构成的集合更加丰富，式 (19) 依然提供了一套可靠框架，用于学习以那些不可观测但具备相关性的因素为条件的联合分布。图 22 在机器人应用场景下展示了这套隐变量框架：真实的、以 $z$ 为条件的生成过程会为每一组 $(o,a)$ 对赋予似然 $p((o,a)\mid z)$。借助贝叶斯定理（Bayes’ theorem），我们可以由似然 $p_\theta(o,a\mid z)$、先验 $p_\theta(z)$ 以及证据（evidence）$p_\theta(o,a)$，在 $\mathrm{supp}(\mathcal{Z})$ 上重建后验分布 $q_\theta(z\mid o,a)$。变分自编码器（VAEs）使用近似后验 $q_\phi(z\mid o,a)$ 逼近式 (19) 给出的隐变量模型，同时对参数化似然 $p_\theta(o,a\mid z)$ 的参数做回归（图 22）。

给定包含 $N$ 组独立同分布（i.i.d.）观测‑动作对的数据集 $D$，在参数 $\theta$ 下所有数据点的对数似然（在贝叶斯语境下，也称为证据 $p_\theta(D)$）可以写为：

$$
\begin{align}
\log p_\theta(\mathcal{D})
&= \log \sum_{i=0}^{N} p_\theta\big((o,a)_i\big) \tag{20} \\
&= \log \sum_{i=0}^{N} \int_{\mathrm{supp}(\mathcal{Z})} p_\theta\big((o,a)_i \mid z\big)\, p(z) \tag{21} \\
&= \log \sum_{i=0}^{N} \int_{\mathrm{supp}(\mathcal{Z})} \frac{q_\theta\big(z \mid (o,a)_i\big)}{q_\theta\big(z \mid (o,a)_i\big)} \cdot p_\theta\big((o,a)_i \mid z\big)\, p(z) \tag{22} \\
&= \log \sum_{i=0}^{N} \mathbb{E}_{z \sim q_\theta(\bullet \mid (o,a)_i)} \left[ \frac{p(z)}{q_\theta\big(z \mid (o,a)_i\big)} \cdot p_\theta\big((o,a)_i \mid z\big) \right], \tag{23}
\end{align}
$$

公式释义：
式 (20)：数据集 $\mathcal{D}$ 在参数 $\theta$ 下的对数似然，等于对全部样本点的似然之和取对数。
式 (21)：代入隐变量模型，对隐变量 $z$ 在其支撑集上积分（边缘化）。
式 (22)：引入提议分布 $q_\theta\big(z \mid (o,a)_i\big)$，做恒等变形，为重要性采样铺垫。
式 (23)：将积分改写为关于分布 $q_\theta\big(z \mid (o,a)_i\big)$ 的期望形式。


其中，我们在式 (20) 中代入了式 (19)；在式 (21) 里乘上恒等因子 $1=\dfrac{q_\theta\big(z\mid(o,a)_i\big)}{q_\theta\big(z\mid(o,a)_i\big)}$；并在式 (23) 使用了期望的定义。

在一种特殊情形下：若假设分布是可计算的（tractable），则 $p_\theta(\mathcal{D})$ 通常同样可计算；此时对 $\log p_\theta(\mathcal{D})$ 关于 $\theta$ 求极大，就为推断生成模型的未知参数 $\theta$（点估计）提供了一个自然目标。遗憾的是，如果使用神经网络这类逼近器对分布 $p$ 建模，式 (23) 几乎无法直接计算，针对高维、非结构化数据时更是如此。

在关于变分自编码器（VAEs）的开创性工作中，Kingma 与 Welling（2013）提出两项核心贡献，用以从非结构化数据中学习复杂隐变量生成式模型：（1）给出式 (23) 的一个可计算变分下界，将其作为优化目标，联合学习似然与后验；（2）采用高容量函数逼近器来建模似然 $p_\theta(o,a\mid z)$ 以及近似后验分布 $q_\phi(z\mid o,a) \approx q_\theta(z\mid o,a)$。

具体来说，对式 (23) 应用延森不等式（Jensen’s inequality，$\log \mathbb{E}[\bullet] \ge \mathbb{E}[\log(\bullet)]$），就可以推导出式 (23) 的下界，也就是证据下界（Evidence LOwer Bound，ELBO），结果如下：

$$
\begin{align}
\log p_\theta(\mathcal{D})
&\ge \sum_{i=0}^{N}\left(
\mathbb{E}_{z\sim q_\theta(\bullet\mid (o,a)_i)}
\big[\log p_\theta\big((o,a)_i \mid z\big)\big]
+
\mathbb{E}_{z\sim q_\theta(\bullet\mid (o,a)_i)}
\left[\log\left(\frac{p(z)}{q_\theta\big(z \mid (o,a)_i\big)}\right)\right]
\right) \tag{24}\\
&= \sum_{i=0}^{N}\left(
\mathbb{E}_{z\sim q_\theta(\bullet\mid (o,a)_i)}
\big[\log p_\theta\big((o,a)_i \mid z\big)\big]
-
D_{\mathrm{KL}}\big[q_\theta\big(z \mid (o,a)_i\big) \parallel p(z)\big]
\right) \tag{25}
\end{align}
$$

公式释义：
式 (24)：数据集对数似然的下界，第一项为重构期望，第二项为对数比值的期望。
式 (25)：利用 KL 散度（Kullback–Leibler divergence）改写，也就是标准证据下界（ELBO）：
- $\mathbb{E}_{z\sim q_\theta(\bullet\mid (o,a)_i)}\big[\log p_\theta\big((o,a)_i \mid z\big)\big]$：重构项，衡量在给定隐变量 $z$ 时重建观测‑动作样本的对数似然；
- $D_{\mathrm{KL}}\big[q_\theta\big(z \mid (o,a)_i\big) \parallel p(z)\big]$：正则项，衡量近似后验与隐变量先验之间的 KL 散度。


真实后验 $q_\theta(z \mid o,a)$ 通常无法直接求解，这使得我们不能计算式 (25) 中的期望项与 KL 散度项。因此 Kingma 和 Welling（2013）提出使用近似后验 $q_\phi(z \mid o,a)$ 推导证据下界（ELBO），从而得到最终可求解的 ELBO 目标函数：

$$
\mathrm{ELBO}_{\mathcal{D}}(\theta,\phi)
= \sum_{i=0}^{N}
\left(
\mathbb{E}_{z\sim q_\phi(\bullet\mid (o,a)_i)}
\big[\log p_\theta\big((o,a)_i \mid z\big)\big]
-
D_{\mathrm{KL}}\big[q_\phi\big(z \mid (o,a)_i\big) \parallel p(z)\big]
\right) \tag{26}
$$

公式释义：
式 (26) 就是在数据集 $\mathcal{D}$ 上、同时以生成参数 $\theta$ 与推断参数 $\phi$ 为变量的证据下界目标：
- $\mathbb{E}_{z\sim q_\phi(\bullet\mid (o,a)_i)}\big[\log p_\theta\big((o,a)_i \mid z\big)\big]$：重构似然期望，鼓励从采样得到的隐变量 $z$ 恢复出观测‑动作样本；
- $D_{\mathrm{KL}}\big[q_\phi\big(z \mid (o,a)_i\big) \parallel p(z)\big]$：KL 正则项，用来约束编码器输出的近似后验向预先设定的隐先验靠拢。

在训练时我们最大化该 ELBO，同步优化编码器参数 $\phi$（近似后验）和解码器参数 $\theta$（条件似然）。

由延森不等式（Jensen’s inequality）可知，最大化证据下界（ELBO）等价于最大化数据的对数似然，由此得到一个自然且可求解的优化目标。具体而言，式 (26) 内的期望可以基于学习得到的分布，采用蒙特卡洛估计（MC estimates）进行估算；而 KL 散度项通常能够求出闭式解，需要满足两个条件：（1）将 \(q_\phi\) 建模为高斯分布：\(q_\phi(z|o,a)=\mathcal{N}\big(\mu_\phi(o,a),\Sigma_\phi(o,a)\big)\)，其中均值向量 \(\mu_\phi(o,a)\) 和协方差矩阵 \(\Sigma_\phi(o,a)\) 均为可学习参数；（2）为隐空间设置标准高斯先验：\(p(z)=\mathcal{N}(\boldsymbol{0},\mathbf{I})\)。

如果转而最小化负证据下界（negative ELBO），就可以直观理解变分自编码器（VAE）的学习动态。该目标能够分解为物理意义清晰的两项（不失一般性，下文仅考虑单个样本 \((o,a)\sim \mathcal{D}\)）：

$$
\min_{\theta,\phi}-\mathrm{ELBO}_{(o,a)\sim\mathcal{D}}(\theta,\phi)
=\min_{\theta,\phi}\mathbf{L}^{\mathrm{rec}}(\theta)+\mathbf{L}^{\mathrm{reg}}(\phi), \tag{27}
$$
$$
\mathbf{L}^{\mathrm{rec}}(\theta)
=\mathbb{E}_{z\sim q_\phi(\bullet\mid o,a)}\big[\log p_\theta(o,a\mid z)\big], \tag{28}
$$
$$
\mathbf{L}^{\mathrm{reg}}(\phi)
=D_{\mathrm{KL}}\big[q_\phi(z\mid o,a)\parallel p(z)\big]. \tag{29}
$$

公式释义：
式 (27) 表示最小化负证据下界（negative ELBO）等价于最小化重构损失与正则损失之和，优化参数为生成参数 $\theta$ 和推断参数 $\phi$；
- $\mathbf{L}^{\mathrm{rec}}(\theta)$（式 28）：重构损失项，是隐变量 $z$ 采自近似后验 $q_\phi(\bullet\mid o,a)$ 时对数条件似然的期望，驱动模型根据隐变量重建观测‑动作对；
- $\mathbf{L}^{\mathrm{reg}}(\phi)$（式 29）：KL 正则项，代表近似后验 $q_\phi(z\mid o,a)$ 与隐先验 $p(z)$ 之间的 KL 散度，约束编码器输出的后验分布向预设隐先验对齐。

训练时最小化该联合目标，同步更新编码器参数 $\phi$ 与解码器参数 $\theta$。

对于任意给定的 $(o,a)$ 对，式 (28) 中的期望项一般通过蒙特卡洛估计（MC estimates）进行计算，于是得到

$$
-\mathbb{E}_{z\sim q_\phi(\bullet\mid o,a)}\big[\log p_\theta(o,a\mid z)\big]
=\mathbf{L}^{\mathrm{rec}}
\approx -\frac{1}{n}\sum_{i=0}^{n}\log p_\theta(o,a\mid z_i).
$$

公式释义：
该式为重构损失的蒙特卡洛近似形式；
- $\mathbb{E}_{z\sim q_\phi(\bullet\mid o,a)}\big[\log p_\theta(o,a\mid z)\big]$：从近似后验 $q_\phi(\bullet\mid o,a)$ 采样隐变量 $z$，求取对数条件似然的期望；
- $\mathbf{L}^{\mathrm{rec}}$：重构损失（reconstruction loss）；
- $-\dfrac{1}{n}\displaystyle\sum_{i=0}^{n}\log p_\theta(o,a\mid z_i)$：采用 $n$ 个独立采样样本 $z_i$，用样本均值对期望做蒙特卡洛估计，以此近似重构损失。

在实际计算时，我们通过有限数量的隐变量采样来逼近原本难以直接求解的期望。

假设 $p_\theta(o,a\mid z)$ 由各向同性高斯分布（isotropic Gaussian distribution）参数化，该分布的均值为 $\mu_\theta(z)\in\mathbb{R}^d$，方差为 $\sigma^2$，则对数似然可以简化为：

$$
\log p(o,a\mid z_i)
= -\frac{1}{2\sigma^2}\left\|(o,a)-\mu_\theta(z_i)\right\|_2^2
-\frac{d}{2}\log\big(2\pi\sigma^2\big)
\implies
\mathbf{L}^{\mathrm{rec}}
\approx \frac{1}{n}\sum_{i=0}^{n}\left\|(o,a)-\mu_\theta(z_i)\right\|_2^2
$$

公式释义：
该式为各向同性高斯假设下对数似然与重构损失的化简结果；
- $\displaystyle -\frac{1}{2\sigma^2}\left\|(o,a)-\mu_\theta(z_i)\right\|_2^2$：平方误差项，衡量网络输出均值 $\mu_\theta(z_i)$ 与真实观测‑动作对 $(o,a)$ 之间的 $L_2$ 距离；
- $\displaystyle -\frac{d}{2}\log\big(2\pi\sigma^2\big)$：高斯分布的常数熵项，与输入无关，优化时常可忽略；
- $\displaystyle \mathbf{L}^{\mathrm{rec}} \approx \frac{1}{n}\sum_{i=0}^{n}\left\|(o,a)-\mu_\theta(z_i)\right\|_2^2$：重构损失（reconstruction loss）最终退化为对采样得到的隐变量 $z_i$ 求取平均平方误差。

由此可见，在固定方差的各向同性高斯假设下，最大化对数似然等价于最小化观测‑动作对与解码器预测均值之间的均方误差。

在实际应用中，通常使用参数化分布（例如高斯分布）来近似学习得到的似然 $p_\theta(o,a\mid z)$；该分布的参数由从 $\mu_\theta(z)$ 导出的可学习系数向量给出，其中 $z \sim p(\cdot)$。在这一框架下，训练变分自编码器（VAE）等价于同时完成两项任务：（1）通过最小化重构损失 $\mathbf{L}^{\mathrm{rec}}$，对数据集 $\mathcal{D}$ 中的样本进行重构，这是回归任务中经典的有监督学习目标；（2）通过最小化正则项 $\mathbf{L}^{\mathrm{reg}}$，对隐表示施加正则约束。正则项的作用是实现信息压缩：若按惯例选取先验 $p(z)=\mathcal{N}(0,\mathbf{I})$（见式 29），该正则器会约束后验分布，进而限制 $q_\phi(z\mid o,a)$ 的表达能力。

