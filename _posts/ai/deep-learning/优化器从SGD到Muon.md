---
title: 优化器的前世今生：从 SGD 到 Muon
date: 2026-04-06
tags: []
categories: [Survey]
mathjax: true
---

从 SGD 出发，经过 Momentum、Adam、AdamW，一直走到 2024 年的 Muon——这条演进路线的背后，其实是一个反复被追问的同一个问题：**怎么让梯度下降既走得快，又走得对？**

<!-- more -->

---

## 为什么 SGD 不够用

> 论文：[Large-Scale Machine Learning with Stochastic Gradient Descent](https://leon.bottou.org/publications/pdf/compstat-2010.pdf)（Bottou，COMPSTAT 2010）

SGD 的问题其实很早就被发现了。1986 年 Rumelhart 提出反向传播时，Momentum 就已经一并提出——但当时神经网络还很浅、参数量很小，SGD 勉强够用。整个 1990 年代到 2000 年代，深度学习处于寒冬，主流是 SVM、决策树这类不需要梯度下降的方法，SGD 的局限自然也不紧迫。

真正的转折点是 **2012 年 AlexNet**。ImageNet 竞赛证明深层网络有效，大家开始训练真正意义上的大模型，此前被掩盖的缺陷一下子都暴露了出来。Bottou 在 2010 年写下这篇论文时，这种需求还没有真正爆发，但他已经预判到“大规模训练”会成为主流，提前把 SGD 的问题系统梳理清楚。两年后，需求真的来了，Adam 的作者们也就有了现成的理论基础可以引用。

神经网络的训练本质上是：给定损失函数 $L(\theta)$，找让 $L$ 最小的参数 $\theta$。朴素的梯度下降是：

$$\theta_{t+1} = \theta_t - \eta \nabla L(\theta_t)$$

这在凸函数上没问题。但神经网络的 loss landscape 是**高维非凸**的：

**非凸**是指 loss 函数的形状不是简单的碗。凸函数只有一个最低点，从任何地方往下走都能找到它；非凸函数则到处都是坑——局部最小值、鞍点、平坦区域。从某个位置往下走，可能掉进一个局部最低点出不来，而那不是真正的最优解。

**高维**是指参数空间的维度极高——现代神经网络的参数可能有几十亿个。高维会带来两个反直觉的性质。第一，**鞍点极其普遍**。在高维空间里，一个点在某些方向上是局部最低、在另一些方向上是局部最高，这种情况比比皆是，SGD 很容易在鞍点附近徘徊。第二，**不同维度的梯度尺度差异巨大**。参数空间里有的方向非常陡，有的方向几乎平坦，同一个学习率在陡的方向走太猛，在平的方向又走太慢。

![凸函数与非凸函数对比](https://raw.githubusercontent.com/wnma3mz/blog_posts/master/imgs/opt/loss_landscape.svg)

这两个性质叠加，带来了训练中两个具体的麻烦：

**麻烦一：梯度有噪声。** 实际训练用的是 mini-batch，每次只看一小批数据，梯度是真实梯度的有噪声估计。噪声大的时候，每一步走的方向都在抖，很难稳定收敛。Momentum 解决的就是这个：把历史梯度做指数加权平均，抑制噪声、积累方向：

$$m_t = \beta m_{t-1} + (1-\beta) g_t, \quad \theta_{t+1} = \theta_t - \eta m_t$$

**麻烦二：不同参数的梯度尺度差异巨大。** 在 embedding 层，常见词（“的”、“是”）每个 batch 都会被更新，梯度很大；罕见词可能几千个 step 才更新一次，梯度极小。如果用同一个学习率 $\eta$，要么常见词更新过猛，要么罕见词更新太慢——参数之间的梯度尺度可以差几个数量级，一个全局学习率根本覆盖不了。

![SGD 的困境：同一学习率无法适应不同参数](https://raw.githubusercontent.com/wnma3mz/blog_posts/master/imgs/opt/sgd_problem.svg)

这就引出了 Adam 要做的事：**给每个参数单独估计一个合适的步长**。

---

## Adam：自适应学习率

> 论文：[Adam: A Method for Stochastic Optimization](https://arxiv.org/abs/1412.6980)（Kingma & Ba，ICLR 2015）

Adam 引入了两个量来解决前面两个麻烦：**一阶矩 $m_t$** 负责积累梯度方向、抑制噪声，**二阶矩 $v_t$** 负责估计每个参数的梯度尺度、自适应步长。两个问题是独立的，所以用两个量分别解决。

这两条路此前都有人单独走过——

- **只用 $m_t$（Momentum）**：解决了梯度噪声问题，但步长全局固定，梯度尺度差异完全没有解决。
- **只用 $v_t$（RMSProp，Hinton 2012）**：解决了梯度尺度问题，但没有 Momentum，在 mini-batch 噪声大时方向会乱抖，收敛不稳定。

Adam 的核心贡献，就是把这两个已经存在的思路合在一起，再加上偏差修正，变成一个开箱即用的优化器。

符号说明：$g_t = \nabla L(\theta_t)$ 是第 $t$ 步的梯度，即 loss 对当前参数的偏导数。

**一阶矩 $m_t$**（梯度的指数加权平均，负责积累方向）：

$$m_t = \beta_1 m_{t-1} + (1 - \beta_1) g_t$$

**二阶矩 $v_t$**（梯度平方的指数加权平均，负责估计每个参数的梯度尺度）：

$$v_t = \beta_2 v_{t-1} + (1 - \beta_2) g_t^2$$

这里 $g_t^2$ 是**逐元素平方**（$g_t \odot g_t$），不是向量外积 $g_t g_t^\top$。区别很重要：逐元素平方的结果与 $g_t$ 形状完全相同，仍是 $n$ 个数；而外积 $g_t g_t^\top$ 是一个 $n \times n$ 的矩阵。所以 $v_t$ 的存储大小和参数量一样，只有 $n$ 个数。

这也是为什么 $v_t$ 虽然名叫“二阶矩”，但它实际上只捕捉了每个参数自己的梯度尺度，没有包含参数之间的相关性。这个局限后面还会回来。

然后用 $\sqrt{v_t}$ 来归一化更新步长：

$$\theta_{t+1} = \theta_t - \eta \cdot \frac{m_t}{\sqrt{v_t} + \epsilon}$$

$\sqrt{v_t}$ 是这个参数历史梯度的 RMS（均方根），用它做分母，相当于把每个参数的更新量归一化到同一尺度：梯度一直大的参数，$\sqrt{v_t}$ 大，步长被压小；梯度一直小的参数，$\sqrt{v_t}$ 小，步长被放大。

还有一个细节。训练初期 $m_t$ 和 $v_t$ 都从零开始，前几步会严重低估真实值。展开来看，第 $t$ 步时 $m_t = (1-\beta_1)\sum_{\tau=1}^t \beta_1^{t-\tau} g_\tau$，其系数之和是 $(1-\beta_1^t)$，而不是 1，也就是说整体被缩小了 $(1-\beta_1^t)$ 倍。Adam 就用这个因子做修正，把低估补回来：

$$\hat{m}_t = \frac{m_t}{1 - \beta_1^t}, \quad \hat{v}_t = \frac{v_t}{1 - \beta_2^t}$$

默认超参数 $\beta_1 = 0.9$，$\beta_2 = 0.999$，$\epsilon = 10^{-8}$，这套参数在绝大多数任务上开箱即用。Adam 一经提出就统治优化器领域将近十年：不需要仔细调参，收敛稳定，几乎对所有任务都够用。

![SGD 震荡 vs Adam 平稳收敛](https://raw.githubusercontent.com/wnma3mz/blog_posts/master/imgs/opt/sgd_vs_adam.svg)

### Adam 的存储代价

这些好处不是免费的。与 SGD 相比，Adam 额外引入了两份和参数等大的数组：

| 优化器 | 需要存储的内容 | 显存倍数 |
|--------|--------------|--------|
| SGD | 参数 $\theta$ | 1× |
| SGD + Momentum | 参数 $\theta$ + $m_t$ | 2× |
| Adam | 参数 $\theta$ + $m_t$ + $v_t$ | 3× |

以 GPT-3（1750 亿参数，FP32 存储）为例，仅优化器状态就需要约 2100 GB，相当于 26 张 A100（80GB）专门用来存 $m_t$ 和 $v_t$。这解释了为什么大模型训练需要几百张卡，也解释了为什么省内存方向的研究在 2018 年之后变得如此紧迫。

### Adam 并不总是优于 SGD

Adam 不是 SGD 的“升级版”，两者是不同的权衡。Adam 的自适应步长在梯度稀疏、参数尺度差异大的任务上优势明显——NLP、推荐系统、embedding 层都是这类场景。但在**图像分类**等任务上，研究者反复观察到：Adam 训练时 loss 下降很快，但最终测试精度往往不如精心调过学习率的 SGD + Momentum。

原因在于，$\sqrt{v_t}$ 的归一化压缩了不同参数之间的梯度尺度差异。这当然会让训练更稳定，但也会让模型“过于平等地”对待每个参数方向：梯度大的方向（通常对应更重要的特征）被压小，梯度小的方向被放大。SGD 没有这一步归一化，更忠实地沿着 loss 下降最快的方向走，所以在 loss landscape 比较规则的任务上，反而可能找到更好的最小值。

**Adam 收敛快但不一定收敛好，SGD 收敛慢但有时候落点更佳。** 这也是 AdamW 出现后依然有人用 SGD 训 ResNet 的原因。

---

## 两个独立的批判

### 工程 bug：AdamW

> 论文：[Decoupled Weight Decay Regularization](https://arxiv.org/abs/1711.05101)（Loshchilov & Hutter，ICLR 2019）

训练神经网络时，为了防止过拟合，通常会加上 L2 正则：在 loss 上加一项 $\frac{\lambda}{2} \|\theta\|^2$，让参数不要太大。L2 正则对梯度的影响是每次更新时多了一项 $\lambda \theta$，把参数往零拉一点，这个操作又叫**权重衰减（weight decay）**。

在 SGD 里，L2 正则和权重衰减**完全等价**：

$$\theta_{t+1} = \theta_t - \eta (g_t + \lambda \theta_t) = (1 - \eta\lambda)\theta_t - \eta g_t$$

但在 Adam 里，这个等价关系悄悄被打破了。回忆 Adam 的标准更新：

$$\theta_{t+1} = \theta_t - \eta \cdot \frac{m_t}{\sqrt{v_t} + \epsilon} \quad \text{（原始 Adam）}$$

加入 L2 正则后，梯度变成 $g_t + \lambda\theta_t$，这一项被一起送进归一化流程：

$$\theta_{t+1} = \theta_t - \eta \cdot \frac{m_t + \lambda \theta_t}{\sqrt{v_t} + \epsilon} \quad \text{（Adam + L2，有问题）}$$

正则化强度被每个参数自己的 $\sqrt{v_t}$ 缩放了——梯度大的参数正则化被压小，梯度小的参数正则化被放大。这和“统一强度地把参数往零拉”完全不一样。

修复很简单：把权重衰减从梯度里**解耦**出来，直接作用在参数上，绕过归一化：

$$\theta_{t+1} = \theta_t - \eta \cdot \frac{m_t}{\sqrt{v_t} + \epsilon} - \eta\lambda \theta_t \quad \text{（AdamW）}$$

改动只有一行，但效果差异显著。今天几乎所有 LLM 的训练都用 AdamW 而不是 Adam。

### 理论缺陷：AMSGrad

> 论文：[On the Convergence of Adam and Beyond](https://arxiv.org/abs/1904.09237)（Reddi et al.，ICLR 2018）

AdamW 修的是工程 bug。同年，另一篇论文从理论层面发现了更根本的问题：**Adam 在某些情况下理论上不收敛**。

问题出在 $v_t$ 的“遗忘”机制上。$v_t$ 是梯度平方的滑动平均，只保留近期的梯度信息——这在大多数时候是好事，能快速适应梯度分布的变化。但在**非平稳**情况下会导致收敛失败。

所谓非平稳，是指梯度的分布随时间变化而不稳定。举个具体例子：训练语言模型时，“量子纠缠”这个词极少出现，对应的 embedding 权重在绝大多数 batch 里梯度都接近 0，偶尔遇到一个集中讨论量子物理的 batch，梯度突然变成 10。这不是异常值，是完全合理的训练信号，只是分布极不均匀。

$v_t$ 的更新公式是 $v_t = \beta_2 v_{t-1} + (1-\beta_2) g_t^2$，展开后是历史梯度平方的加权平均，越久远权重越小：

$$v_t = (1-\beta_2)\sum_{\tau=1}^{t} \beta_2^{t-\tau} g_\tau^2$$

假设前 999 步梯度都是 0.01，第 1000 步梯度突然变成 10，取 $\beta_2 = 0.999$：

- $v_{999} \approx (0.01)^2 = 0.0001$（长期被小梯度主导，积累很小）
- $v_{1000} = 0.999 \times 0.0001 + 0.001 \times 100 \approx 0.1$

第 1000 步实际步长：$\dfrac{\eta}{\sqrt{0.1}} \approx \dfrac{\eta}{0.316}$

理想步长应该是：$\dfrac{\eta}{\sqrt{100}} = \dfrac{\eta}{10}$

两者相差 **30 倍**。问题就在这里：$v_t$ 记录的是过去的历史。当这一步梯度突然变大时，$v_t$ 来不及反应。等它更新完，这一步的参数已经用错误的步长更新过了，很可能直接跨过最优点。

AMSGrad 的修复思路是让 $v_t$ **只能增不能减**，用历史最大值来归一化：

$$\hat{v}_t = \max(\hat{v}_{t-1}, v_t), \quad \theta_{t+1} = \theta_t - \eta \cdot \frac{\hat{m}_t}{\sqrt{\hat{v}_t} + \epsilon}$$

这在理论上有收敛保证，但步长只能越来越小，实践中收敛速度往往慢于 Adam。AMSGrad 的价值不在于被广泛使用，而在于它揭示了 Adam 的理论缺陷，促使研究者开始认真问：**Adam 的 $v_t$ 到底在估计什么，它估计得准吗？**

### 两个批判的共同指向

AdamW 和 AMSGrad 是从两个方向在批评 Adam，但它们最后都指向同一件事：**$v_t$ 对“该用多大步长”的估计，本身就很粗糙。**

神经网络的参数不是 $n$ 个独立的标量，而是组织成矩阵的。一个 $1000 \times 1000$ 的权重矩阵，每一行对应一个输出特征，每一列对应一个输入特征——理想的步长估计应该能感知到“这一行整体需要大幅更新，那一列只需要微调”这种矩阵层面的结构。

$v_t$ 做不到这一点。它为每个参数独立存了一个数，只知道这个元素自己的历史梯度大小，感知不到行与行、列与列之间的曲率关系。100 万个参数就有 100 万个 $v_t$ 值，数量看起来不少，但它描述的始终只是“每个元素单独有多大”，而不是“这些元素作为一个矩阵整体应该怎么更新”。

从数学上精确描述这个差距，需要从最优更新公式推导起。在当前参数 $\theta_t$ 附近，对 loss 做二阶泰勒展开：

$$L(\theta_t + \Delta\theta) \approx L(\theta_t) + g_t^\top \Delta\theta + \frac{1}{2} \Delta\theta^\top H \Delta\theta$$

其中 $H = \nabla^2 L$ 是 **Hessian 矩阵**，第 $(i,j)$ 个元素描述的是“参数 $j$ 变化时，参数 $i$ 的梯度怎么变”。对 $\Delta\theta$ 求最优解：

$$\Delta\theta^* = -H^{-1} g_t$$

这就是**牛顿法**：用 Hessian 的逆乘以梯度，作为实际的更新方向。

这里引入一个术语：**预条件（preconditioning）**。普通梯度下降直接用 $g_t$ 作为更新方向；如果在更新之前先用一个矩阵 $P$ 变换一下梯度——$\Delta\theta = -P \cdot g_t$——$P$ 就叫**预条件子（preconditioner）**。牛顿法里 $P = H^{-1}$，它的作用是：曲率大的方向步长被压小，曲率小的方向步长被放大，每一步都针对当前 loss landscape 的形状量身定制。

打个比方。普通梯度下降像在山上蒙眼走路，只知道脚下哪个方向是下坡；预条件则像先拿到一张地形图，知道哪里是陡坡、哪里是缓坡，于是在陡的地方小步走，在缓的地方大步走。$H^{-1}$ 就是那张最理想的地形图。

那 Adam 的 $v_t$ 扮演的是什么角色？$v_t \approx \mathbb{E}[g_t^2]$ 是梯度的逐元素均方。注意外积矩阵 $g_t g_t^\top$ 的第 $(i,i)$ 个对角元素恰好是 $g_{t,i}^2$，因此 $v_t$ 正好对应 Fisher 信息矩阵 $F = \mathbb{E}[g_t g_t^\top]$ 的对角线：

$$v_t \approx \text{diag}(F) = \text{diag}\!\left(\mathbb{E}[g_t g_t^\top]\right)$$

因此 Adam 的实际更新相当于：

$$\Delta\theta_{\text{Adam}} \approx -\,\text{diag}(F)^{-1/2} \cdot g_t$$

对比牛顿法的理想更新 $\Delta\theta^* = -H^{-1} g_t$，差距一目了然：**Adam 用 $\text{diag}(F)^{-1/2}$ 替代了 $H^{-1}$**。前者只有 $n$ 个数，后者是完整的 $n \times n$ 矩阵——Adam 把 $n^2$ 个信息压缩成了 $n$ 个，丢掉的正是所有非对角元素，即参数之间梯度相关性的全部信息。

**那为什么不直接用完整的 Hessian？** 因为存不下。GPT-3 有 1750 亿参数，Hessian 就有 $1750\text{亿} \times 1750\text{亿}$ 个元素，存储这个矩阵需要的显存比地球上所有存储设备加起来还多。

所以问题变成了：**既然完整的 $H$ 用不了，能不能用一个比对角近似更好、但又负担得起的近似？**

这把研究者引向了两条岔路，出发点相同——Adam 的对角近似离理想的 $H^{-1}$ 差太远——但选择了不同的权衡方向：

- **省内存方向**：$v_t$ 的对角近似已经是 $O(n)$ 了，但就连这个也太贵。能不能用更低秩的结构来近似 $H^{-1}$，把存储进一步压下去，让本来根本跑不起来的大模型能跑起来？
- **更准确方向**：不压缩内存，愿意付出更多代价，在参数矩阵的行空间和列空间上分别做曲率估计，捕捉参数之间的结构性相关性，做比对角近似更好的 $H^{-1}$ 近似。

---

## 岔路一：Adafactor——显存不够，先让模型能跑起来

> 论文：[Adafactor: Adaptive Learning Rates with Sublinear Memory Cost](https://arxiv.org/abs/1801.04014)（Shazeer & Stern，ICML 2018）

看到这里会有一个疑问：前面刚说完 $v_t$ 的对角近似已经太粗糙，Adafactor 反而要把它压缩得更小——不是在进一步丢精度吗？

确实如此。但 Adafactor 要解决的根本不是“怎么优化得更好”，而是一个更现实的工程问题：**模型太大，显存不够，训练根本跑不起来。** 前面提到 Adam 的存储是参数的 3 倍。到了 2018 年，T5 的参数量已经达到几十亿，光是 $v_t$ 就要占用数 GB 显存，这成了一个很现实的瓶颈。研究者做了一个明确的取舍：**宁愿让优化器估得更粗糙，也要先让模型能跑起来。**

对于形状为 $m \times n$ 的权重矩阵，Adam 存储的 $v_t$ 同样是 $m \times n$ 的。Adafactor 观察到 $v_t$ 的行和列之间存在冗余，把它分解成两个向量的外积来近似：

$$V_t \approx r_t s_t^\top$$

其中 $r_t \in \mathbb{R}^m$ 是行方向的缩放因子，$s_t \in \mathbb{R}^n$ 是列方向的缩放因子。存储从 $O(mn)$ 降到了 $O(m + n)$——对一个 $1000 \times 1000$ 的矩阵，内存从 100 万个数压缩到 2000 个，节省了 500 倍。

具体更新规则是：

$$r_t = \beta_2 r_{t-1} + (1-\beta_2)(G_t^2 \mathbf{1}_n), \quad s_t = \beta_2 s_{t-1} + (1-\beta_2)(\mathbf{1}_m^\top G_t^2)$$

$$\hat{V}_t = \frac{r_t s_t^\top}{\mathbf{1}_m^\top r_t}, \quad \theta_{t+1} = \theta_t - \eta \cdot \frac{G_t}{\sqrt{\hat{V}_t}}$$

$r_t$ 累积了每一行的梯度能量，$s_t$ 累积了每一列的梯度能量，两者外积给出对 $v_t$ 的低秩近似。Adafactor 还去掉了一阶矩 $m_t$（直接用原始梯度），进一步省了一份内存，代价是训练初期噪声更大，需要配合学习率预热（warmup）来稳定。

这个方案成了 Google T5 系列的标配优化器。它的价值不在于收敛更快，而在于**让本来训不起来的模型变得可训**。

Adafactor 沿着“省内存”的方向已经做到了极限——再压缩下去就什么信息都没了。如果不想省内存，而是想把曲率估计做得**更准**，该怎么走？

---

## 岔路二：走向真正的矩阵曲率

### Shampoo——在矩阵的行列空间上做曲率估计

> 论文：[Shampoo: Preconditioned Stochastic Tensor Optimization](https://arxiv.org/abs/1802.09568)（Gupta et al.，ICML 2018）

Adam 的 $v_t$ 把每个参数当成独立的标量——每个元素自己估自己的步长，完全不知道自己和邻居的关系。Shampoo 的观察是：**权重本来就是矩阵，矩阵的行之间、列之间天然存在关联，这些关联本身就是曲率信息，应该被利用起来。**

什么叫“行列之间的关联是曲率信息”？可以这样想。一个权重矩阵 $W \in \mathbb{R}^{m \times n}$，每一行对应一个输出特征，每一列对应一个输入特征。如果某两行对应的输出特征在训练中总是一起变化，也就是梯度方向高度一致，就说明这两个方向上的 loss 曲率是耦合的。调整其中一行时，另一行也应该跟着调整，不能各自独立估步长。这类“哪些方向是耦合的”的信息，正是 Adam 的 $v_t$ 捕捉不到的。

Shampoo 用梯度的外积来累积这个信息。当梯度 $G_t$ 到来时，$G_t G_t^\top$ 是一个 $m \times m$ 的矩阵，第 $(i,j)$ 个元素描述第 $i$ 个输出特征的梯度和第 $j$ 个输出特征的梯度有多相关。类似地，$G_t^\top G_t$ 是 $n \times n$ 的矩阵，捕捉列方向（输入特征之间）的关联。

Shampoo 把历史上所有步的这两个矩阵累加起来：

$$L_t = \sum_{\tau=1}^t G_\tau G_\tau^\top \in \mathbb{R}^{m \times m}, \quad R_t = \sum_{\tau=1}^t G_\tau^\top G_\tau \in \mathbb{R}^{n \times n}$$

更新规则是：

$$W_{t+1} = W_t - \eta \cdot L_t^{-1/4} G_t R_t^{-1/4}$$

为什么是 $-1/4$ 次方？这里有一点矩阵代数。

Shampoo 用 $L_t \otimes R_t$（Kronecker 积）来近似整个权重矩阵的 Hessian。$L_t$ 捕捉行方向的曲率，$R_t$ 捕捉列方向的曲率，Kronecker 积再把两者组合起来。在这个近似下，理想预条件子 $H^{-1}$ 对应的最优操作，正好是左乘 $L_t^{-1/4}$、右乘 $R_t^{-1/4}$。这里的 $-1/4$ 来自 Kronecker 积的代数性质，可以理解成两个矩阵各承担一半的逆，再开一次根号。

直觉上，$L_t^{-1/4}$ 对梯度做左乘，在行方向做预条件：行方向曲率大（$L_t$ 对应特征值大），步长被压小；$R_t^{-1/4}$ 对梯度做右乘，在列方向做同样的事。两边同时归一化，比 Adam 只在每个元素上独立归一化要准确得多。

举个具体的例子：假设某个权重矩阵的前两行对应的特征在训练中一直高度相关（梯度方向总是一致），$L_t$ 的左上角 $2\times 2$ 子矩阵就会有很大的值。$L_t^{-1/4}$ 作用后，这两行的更新会被整体压小，避免在这个方向上走过猛。这正是 Adam 做不到的：Adam 只能分别压小这两行里每个元素的步长，看不到“这两行应该被整体对待”。

### Shampoo 的代价与价值

代价很明显：需要额外存储 $L_t$（$m \times m$）和 $R_t$（$n \times n$）两个矩阵，并定期计算它们的 $-1/4$ 次幂。以 LLM 中常见的 $4096 \times 4096$ 投影矩阵为例，$L_t$ 和 $R_t$ 各有约 1680 万个数，合计是 $v_t$（同样 1680 万个数）的 2 倍。加上参数 $\theta$ 和动量 $m_t$，Shampoo 的总存储大约是参数量的 **5×**，而 Adam 只需要 3×。Google 后来做了大量工程优化（分布式 Shampoo），才让它在实际大模型训练中可用。

但 Shampoo 的理论意义很明确：**在矩阵空间做预条件，比在标量空间做对角近似效果更好。** 这个洞察直接启发了后来的 Muon。

Shampoo 留下了一个悬念：矩阵预条件的方向是对的，但能不能用更轻量的方式实现同样的效果，不需要真的去算那些大矩阵的幂次？

---

## Sophia：把 Hessian 对角直接估出来

> 论文：[Sophia: A Scalable Stochastic Second-Order Optimizer for Language Model Pre-training](https://arxiv.org/abs/2305.14342)（Liu et al.，2023）

Shampoo 走的是“行列空间做预条件”的路，代价是要存和计算大矩阵的分数次幂。另一条路是：能不能直接把 $H^{-1}$ 的对角估出来，然后用它替代 Adam 的 $v_t^{-1/2}$？

回忆一下差距到底在哪里。Adam 用的是 $\text{diag}(F)^{-1/2}$，也就是 Fisher 信息矩阵的对角；而理想的牛顿法要的是 $H^{-1}$。很多时候，$F$ 和 $H$ 的对角差得不算太远，但在 loss landscape 曲率差异很大的地方，比如 LLM 的某些层，两者会明显偏开：有的方向曲率很高，Adam 把步长估大了，一步就越过最优点；有的方向曲率很低，Adam 又把步长估小了，收敛会非常慢。

Sophia 的思路很直接：**直接估 $\text{diag}(H)$，然后用它来归一化梯度。**

### 怎么估 Hessian 的对角

直接算 $H = \nabla^2 L$ 要对每个参数分别求二阶导，计算量是梯度的 $n$ 倍，完全不可行。Sophia 用的是 **Hutchinson 估计量**。

随机采样一个向量 $u \sim \mathcal{N}(0, I)$，构造：

$$\hat{h} = u \odot (H u)$$

为什么 $\hat{h}$ 的期望恰好是 $\text{diag}(H)$？展开第 $i$ 个分量：

$$\mathbb{E}[\hat{h}_i] = \mathbb{E}\!\left[u_i \cdot (Hu)_i\right] = \mathbb{E}\!\left[u_i \sum_j H_{ij} u_j\right] = \sum_j H_{ij}\, \mathbb{E}[u_i u_j]$$

由于 $u$ 的各分量独立且方差为 1，有 $\mathbb{E}[u_i u_j] = \mathbf{1}[i = j]$。求和只剩 $j = i$ 那一项：

$$\mathbb{E}[\hat{h}_i] = H_{ii}$$

随机向量的独立性把所有非对角项全部消掉，只留下对角元素——这就是等价的原因。

关键在于，$Hu$ 不需要显式计算 $H$。它只是一次 Hessian-向量积，可以通过两次反传实现，计算量和一次梯度反传差不多。每隔 $k$ 步（典型值 $k=10$）做一次这样的估计，再用指数加权平均平滑：

$$h_t = \beta_2 h_{t-k} + (1-\beta_2) \hat{h}_t$$

额外开销约 5%（每 10 步多一次梯度量级的计算），但换来的是比 $v_t$ 更准确的曲率信息。

### 为什么还需要 Clipping

直接用 $h_t$ 归一化梯度有一个问题：神经网络的 loss landscape 是非凸的，$H$ 的对角可能出现**负值**——在鞍点附近，某些方向的曲率是负的，$h_t < 0$ 意味着用这个方向的“曲率”来归一化会让步长方向反转。

Sophia 的解法是把更新量裁剪（clip）到 $[-1, 1]$：

$$\theta_{t+1} = \theta_t - \eta \cdot \text{clip}\!\left(\frac{m_t}{\max(\gamma h_t,\, \epsilon)},\ 1\right)$$

当 $h_t < 0$ 或 $h_t$ 太小时，clip 被触发，退回到动量 SGD 的行为。实验显示，clip 触发率极低（约 1–5%），说明 Hessian 对角在大多数情况下是可靠的正数——clip 只是一个安全网，不影响主要更新。

### 效果

在 GPT-2 规模的模型上，Sophia 比 Adam **少用一半步数就能达到相同的验证 loss**。背后的原因很直观：Adam 的 $v_t$ 在曲率差异大的地方系统性地估错步长，Sophia 用更准的 $h_t$ 纠正了这个偏差，每一步都走在更对的方向上。

Sophia 的存储开销和 Adam 相当（$\theta$、$m_t$、$h_t$ 各一份），工程实现也不复杂（PyTorch 原生支持 Hessian-向量积）。它是目前最接近“真正用上二阶信息”的实用优化器。

---

## 旁注：Lion——极端压缩方向的尝试

> 论文：[Symbolic Discovery of Optimization Algorithms](https://arxiv.org/abs/2302.06675)（Chen et al.，2023）

Adafactor 试图压缩 $v_t$，Lion 的压缩更极端：**把整个更新量压成符号（sign）。**

Adam 的更新是 $\frac{m_t}{\sqrt{v_t} + \epsilon}$——一个带尺度信息的向量。Lion 直接扔掉尺度，只留方向：

$$\theta_{t+1} = \theta_t - \eta \cdot \text{sign}(m_t)$$

每个参数的更新量只有 $\pm\eta$ 两种取值。

**省了什么：** 完全不需要 $v_t$，存储从 Adam 的 3× 降到 2×，也省去了 $\sqrt{\cdot}$ 和除法运算。

**代价是什么：** 没有对梯度尺度的任何自适应，不同参数的更新量全部一样大。在梯度尺度差异极大的任务上（比如 NLP 中的 embedding 层），这个强制“平等”会让训练不稳定，需要更大的 weight decay 来补偿。

Lion 是 Google Brain 用程序搜索（evolutionary algorithm）发现的，不是人工推导出来的，而是在大量候选更新规则里跑出来的。它在大规模视觉和视觉-语言预训练上表现不错（ImageNet fine-tuning +2%），但在 NLP 任务上的提升并不稳定。**它更大的价值是启发性的**：把更新量压缩到符号，性能却没有崩，说明梯度的方向信息可能比尺度信息更重要。这个洞察，正是 Muon 的出发点。

---

## Muon：矩阵正交化，而不是逐元素归一化

> 论文：[Muon: An optimizer for hidden layers in neural networks](https://arxiv.org/abs/2409.20325)（Kosson et al.，2024）

### 核心想法：把梯度矩阵“正交化”

Shampoo 的更新 $L_t^{-1/4} G_t R_t^{-1/4}$ 在行方向和列方向分别对梯度做缩放。要理解它在做什么，需要用 SVD 来看梯度矩阵的结构。

对梯度矩阵 $G$ 做奇异值分解：$G = U \Sigma V^\top$，其中 $U$、$V$ 是正交矩阵，$\Sigma$ 是对角矩阵，对角线上的值 $\sigma_1 \geq \sigma_2 \geq \cdots \geq 0$ 叫**奇异值**。可以把 $G$ 理解为：沿着 $V$ 的列方向读入信息，经过 $\Sigma$ 缩放，再沿着 $U$ 的列方向输出。奇异值越大，说明这个方向的梯度分量越强。

Shampoo 的预条件操作本质上是把不同奇异值方向的分量“归一化”——曲率大的方向步长压小，曲率小的方向步长放大。把这个思路推到极致：如果把**所有奇异值都压成 1**，就是把梯度矩阵投影到最近的正交矩阵 $UV^\top$ 上。

这个操作背后的直觉是：**梯度里不同奇异值的大小，反映的是各方向信号强弱的不均匀，但这不一定等于真实的重要性差异。正交化做的，就是把这种不均匀性消掉，让每个更新方向被更平等地对待。**

Muon 的更新规则：

$$W_{t+1} = W_t - \eta \cdot \text{orthogonalize}(M_t)$$

其中 $M_t$ 是梯度的 Nesterov 动量，$\text{orthogonalize}$ 把 $M_t$ 投影到最近的正交矩阵。

### 怎么算正交化

直接做 SVD 分解然后取 $UV^\top$，计算量是 $O(\min(m,n)^2 \cdot \max(m,n))$，对 $4096 \times 4096$ 的矩阵来说代价可观。Muon 用的是 **Newton-Schulz 迭代**，不需要显式 SVD：

$$X_0 = G / \|G\|_F, \quad X_{k+1} = \frac{3}{2} X_k - \frac{1}{2} X_k X_k^\top X_k$$

这个迭代的思路类似于用牛顿法求矩阵的“平方根倒数”：如果把 $X_k X_k^\top$ 当作某种“当前尺度的平方”，每次迭代都在把这个平方往单位矩阵方向拉近。可以证明，当初始矩阵的奇异值落在 $(0, \sqrt{3})$ 范围内时，迭代收敛，极限恰好是 $UV^\top$——所有奇异值都被压成了 1。迭代只需要矩阵乘法，收敛速度是三次方，通常 5 步以内就足够精确。计算量比 Shampoo 的分数次幂低得多。

### 和 Shampoo 的关系

Shampoo 做的是 $L_t^{-1/4} G_t R_t^{-1/4}$，把梯度的奇异值缩放为 $\sigma_i^{-1/2} \cdot \sigma_j^{-1/2}$ 的倍数（其中 $\sigma_i$、$\sigma_j$ 是 $L_t$、$R_t$ 的奇异值）。Muon 做的是把所有奇异值压成 1。两者本质上都是在消除梯度奇异值分布的不均匀，只是 Shampoo 的消除是“按比例缩放”，Muon 的消除是“全部统一”。Muon 更激进，但也更简单——不需要维护 $L_t$ 和 $R_t$，不需要计算矩阵的分数次幂。

### 效果和适用范围

在 Transformer 的隐藏层（linear 投影矩阵）上，Muon 的收敛速度和 Sophia 相当，显存开销接近 Adam（只需额外存一份动量矩阵）。它对 embedding 层和 output 层不适用（这两层的梯度结构不满足矩阵正交化的假设），实际使用时通常是 Muon 处理隐藏层权重、AdamW 处理其余参数的混合方案。

---

## 回望这条路

从 SGD 到 Muon，每一步都是对同一个问题的更深一层回答：**怎么用梯度信息来估计“该往哪走、走多大步”？**

SGD 说：直接走，学习率全局固定。问题：不同参数需要不同步长。

Adam 说：为每个参数单独估一个步长（$v_t$），用历史梯度平方来归一化。问题：$v_t$ 只是 Fisher 信息矩阵的对角，丢掉了参数之间的全部相关信息；理论上不收敛（AMSGrad 的批判）；weight decay 实现有 bug（AdamW 的修复）。

Adafactor 说：显存不够，先让模型能跑起来——把 $v_t$ 进一步压缩成行列外积。代价是精度更粗，换来的是 T5 级别的大模型能被训练。

Shampoo 说：参数是矩阵，步长估计应该在矩阵的行列空间上做，而不是逐元素独立。在行列方向累积曲率，用 $L_t^{-1/4}$ 和 $R_t^{-1/4}$ 来预条件。代价是存储和计算大矩阵的分数次幂。

Sophia 说：与其间接地用 Fisher 近似 Hessian，不如直接估 Hessian 的对角——Hutchinson 估计量只需要额外 5% 的计算量，换来的是更准的曲率信息，步数减半。

Lion 说：方向比尺度重要——把更新量压成 sign，性能没有崩。

Muon 说：Shampoo 的思路对，但不需要那么重——直接把梯度矩阵正交化（奇异值全压成 1），用 Newton-Schulz 迭代替代分数次幂，更快、更简单、效果相当。

这条路上反复出现一个模式：**每个批评最后都指向同一件事——曲率信息不够。** 而后面的每次改进，本质上都在尝试用更低的代价拿到更多曲率信息。Adam 用 $n$ 个数近似 $n^2$ 的 Hessian；Adafactor 用 $m+n$ 个数近似 $mn$ 的 $v_t$；Sophia 直接估 $\text{diag}(H)$；Shampoo 和 Muon 则在矩阵空间里做结构化近似。

另一个模式也很明显：**理论上最优的方案往往不可用，可用的方案总是在精度和代价之间做权衡。** 完整 Hessian 存不下，完整 Fisher 矩阵也存不下，Shampoo 的大矩阵到了超大模型上同样难以为继。每一代优化器，其实都是在当时的硬件和模型规模约束下，把这个权衡往前推了一点。

这条路还没有走到尽头。Muon 刚发布，Sophia 还在找落地场景，混合优化器（不同层用不同优化器）的空间才刚刚打开。

**优化器的问题本质上是一个关于“如何用有限信息描述高维曲率”的问题——而这个问题，随着模型越来越大，只会越来越重要。**

---

## 参考文献

- Bottou，L.（2010）。[Large-Scale Machine Learning with Stochastic Gradient Descent](https://leon.bottou.org/publications/pdf/compstat-2010.pdf)。*COMPSTAT 2010*。

- Kingma，D. P. & Ba，J.（2015）。[Adam: A Method for Stochastic Optimization](https://arxiv.org/abs/1412.6980)。*ICLR 2015*。

- Loshchilov，I. & Hutter，F.（2019）。[Decoupled Weight Decay Regularization](https://arxiv.org/abs/1711.05101)。*ICLR 2019*。

- Reddi，S. J.、Kale，S. & Kumar，S.（2018）。[On the Convergence of Adam and Beyond](https://arxiv.org/abs/1904.09237)。*ICLR 2018*。

- Shazeer，N. & Stern，M.（2018）。[Adafactor: Adaptive Learning Rates with Sublinear Memory Cost](https://arxiv.org/abs/1801.04014)。*ICML 2018*。

- Gupta，V.、Koren，T. & Singer，Y.（2018）。[Shampoo: Preconditioned Stochastic Tensor Optimization](https://arxiv.org/abs/1802.09568)。*ICML 2018*。

- Chen，X.、Liang，C.、Huang，D.、Real，E.、Wang，K.、Liu，Y. …… & Le，Q. V.（2023）。[Symbolic Discovery of Optimization Algorithms](https://arxiv.org/abs/2302.06675)。*NeurIPS 2023*。

- Liu，H.、Li，Z.、Hall，D.、Liang，P. & Ma，T.（2023）。[Sophia: A Scalable Stochastic Second-Order Optimizer for Language Model Pre-training](https://arxiv.org/abs/2305.14342)。*arXiv 2023*。

- Kosson，A.、Messmer，B. & Jaggi，M.（2024）。[Muon: An optimizer for hidden layers in neural networks](https://arxiv.org/abs/2409.20325)。*arXiv 2024*。
