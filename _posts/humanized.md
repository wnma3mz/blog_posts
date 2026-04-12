---
title: LLM的推理加速-DFlash：用扩散模型做投机解码
date: 2026-04-12 14:00:00
tags:
  - Speculative Decoding
  - Diffusion
categories: [Survey]
mathjax: true
---

EAGLE-3 的自回归草稿器遇到了速度-质量的对立：想让草稿更准，就得加深模型；模型越深，串行生成的代价越大。DFlash 的回答是：**用块扩散模型替换自回归草稿器**，单次前向传播并行生成整个 block，让草稿成本与投机长度彻底脱钩。相比 EAGLE-3，加速约 **2.5×**，平均接受长度提升约 **2.2×**。

- 论文：[https://arxiv.org/abs/2602.06036](https://arxiv.org/abs/2602.06036)

- 代码：[https://github.com/z-lab/dflash](https://github.com/z-lab/dflash)

<!-- more -->

# 前置知识

- 投机解码：{% post_link ai/llm/inference/LLM的推理加速-投机解码 %}
- EAGLE 系列：{% post_link ai/llm/inference/LLM的推理加速-EAGLE三部曲 %}

---

# EAGLE 的天花板

回顾 EAGLE-3：Draft Model 是 1 层 Decoder，**自回归**地逐个生成草稿 token。投机长度为 $\gamma$ 时，草稿阶段的总耗时是：

$$T_{\text{draft}}^{\text{AR}} = \gamma \cdot t_{\text{step}}$$

这里隐藏着一个矛盾：

- 想让草稿**更准**（提高平均接受长度）→ 需要加深 Draft Model，$t_{\text{step}}$ 变大
- 想让草稿**更快** → 需要减小 $t_{\text{step}}$，只能用更浅的模型

两个目标天然对立，在自回归框架内无法同时满足。EAGLE-3 实测的平均接受长度（AAL）在 Qwen3-8B 上约为 3.4，继续扩大训练数据或加深模型，收益都趋于饱和。

**如果草稿生成不再是串行的呢？**

把 $\gamma$ 个 token 一次性并行生成出来，草稿成本就变成：

$$T_{\text{draft}}^{\text{parallel}} = t_{\text{parallel}}$$

无论 $\gamma$ 取多大，耗时都是一次前向传播。这样就可以用更深的模型（提升质量），而不必担心延迟随 $\gamma$ 线性增长。

这就是 DFlash 的出发点。接下来的问题是：**什么模型能在单次前向传播中并行生成多个 token？**

---

# 扩散语言模型

答案来自语言扩散模型。图像扩散（DDPM）的核心思路是从噪声出发多步去噪，迁移到语言上，"噪声"变成"被 Mask 的 token"，"去噪"变成"预测被 Mask 的位置"——关键在于，**每步可以并行预测所有被 Mask 的位置**，不受自回归的串行限制。

## LLaDA：从 Mask 出发的扩散语言模型

[LLaDA（2502.09992）](https://arxiv.org/abs/2502.09992) 是一个从头预训练的扩散语言模型，是理解 DFlash 的基础。

### 训练

**加噪：一次采样，不是 T 步链**

标准图像扩散（DDPM）需要走完整的 T 步马尔科夫加噪链（通常 T=1000）。LLaDA 的做法更简单：对每条训练样本，**只采样一个随机时间步** $t \sim \text{Uniform}(0， 1)$，然后一次性按比例 $t$ 把 token 替换为 `[M]`，就得到了训练输入。

本质上，LLaDA 把"扩散"简化成了"随机 mask"，时间步 $t$ 是一个连续值，代表 mask 的比例，没有离散的多步链。

以句子 `The cat sat on the mat` 为例，假设采样到 $t = 0.5$：

```
原句：  The   cat   sat   on   the   mat
加噪：  [M]   cat   [M]   on   the   [M]
```

每条样本独立采样 $t$，因此模型会见过各种各样的 mask 程度——从 $t \approx 0$（几乎不 mask，相当于做完形填空）到 $t \approx 1$（几乎全 mask，相当于从头生成）。

**为什么不像 BERT 固定 15%？**

BERT 的 15% 是经验值，只让模型学会"在大量上下文下猜少数词"。LLaDA 的随机比例让模型同时学会了两件事：$t$ 小时，模型练习的是精细的局部填补；$t$ 大时，模型练习的是在上下文很少时凭借全局语义生成。这两种能力在推理的多步去噪中都会用到——前几步 mask 比例大，后几步 mask 比例小，模型需要应对整个范围。

**预测与 Loss**

将加噪序列输入模型，**一次前向传播**同时预测所有 `[M]` 位置的原始 token。因为没有因果 mask，每个 `[M]` 都能看到序列中所有未被 mask 的 token（双向注意力）。

训练 loss 只在被 mask 的位置计算交叉熵：

$$\mathcal{L}(\theta) = -\mathbb{E}_{t，\， x_t} \left[ \frac{1}{L} \sum_{i:\， x_t^i = \text{M}} \log p_\theta(x_0^i \mid x_t) \right]$$

上例中，loss 只由位置 1（The）、位置 3（sat）、位置 6（mat）贡献；位置 2、4、5 不参与。

### 推理

推理时需要事先确定输出长度 $L$，从一条**全 Mask 序列**出发，经过 $T$ 步去噪逐步恢复。

以生成 6 个 token（`on the mat . It was`）为例，共 $T = 3$ 步：

**Step 1**（$t=1$，全 Mask）：

```
输入：  [M]  [M]  [M]  [M]  [M]  [M]
预测：   on   the  mat   .   It   was
置信：  0.9  0.8  0.9  0.6  0.5  0.5
```

当前步要解锁 $\lfloor L \cdot (1 - s/T) \rfloor = \lfloor 6 \times (2/3) \rfloor = 4$ 个 token，保留置信度最高的 4 个：

```
输出：   on   the  mat   .   [M]  [M]
```

**Step 2**（$t=0.67$）：

```
输入：   on   the  mat   .   [M]  [M]
预测（并行预测所有 [M] 位置）：     It   was
置信：                            0.7  0.8
```

当前步要解锁 $\lfloor 6 \times (1/3) \rfloor = 2$ 个，但只剩 2 个 `[M]`，保留置信度高的 1 个：

```
输出：   on   the  mat   .   [M]  was
```

**Step 3**（$t=0.33$）：

```
输入：   on   the  mat   .   [M]  was
预测（并行预测所有 [M] 位置）：     It
置信：                            0.85
```

最后一步，全部接受：

```
最终：   on   the  mat   .   It   was
```

**置信度策略是推理时的超参，不是训练出来的**

每一步要解锁多少个 token，由公式 $n_{\text{unmask}} = \lfloor L(1 - s/T) \rfloor$ 决定——随时间步 $s$ 动态变化，越到后期解锁得越少，越谨慎。模型输出的 softmax 概率直接作为置信度，阈值由这个公式隐式确定，不需要额外训练任何参数。

这和训练过程的呼应在于：训练时模型见过各种 mask 比例（从 $t\approx 1$ 的"几乎全 mask"到 $t\approx 0$ 的"几乎不 mask"），推理的每一步恰好对应训练时的一个 $t$ 值——模型在高 mask 比例下学会了"大胆猜"，在低 mask 比例下学会了"精细填"，两种能力共同支撑了多步去噪。

论文实验也证实，低置信度重 Mask 策略显著优于随机重 Mask（例如在 GSM8K 上 70.0 vs 21.3）。

![LLaDA 的生成过程（逆向扩散）](https://raw.githubusercontent.com/wnma3mz/blog_posts/master/imgs/LLM的推理加速-DFlash/fig1_llada_generation.svg)

在 2.3T token 上预训练后，LLaDA 8B 在 MMLU（65.5）、GSM8K（78.3）、HumanEval（59.8）上与同规模自回归模型相当，部分任务超越。

**关键特性**：每一步都是**一次前向传播处理整个序列**，所有 `[M]` 并行预测。代价是需要事先确定输出长度，且需要多步迭代（通常 $T = 10 \sim 20$ 步）。

## 块扩散：把自回归和扩散融合起来

纯扩散语言模型有一个实际问题：**生成时需要提前确定输出长度**，不如自回归模型灵活。而且其生成质量（尤其是长序列）通常弱于自回归模型。

[Block Diffusion（BD3-LM，2503.09573）](https://arxiv.org/abs/2503.09573) 提出了一种折中方案：**块间自回归，块内扩散**。

### 核心思路

将序列划分为 $B$ 个 block，每个 block 含 $L'$ 个 token：

- **块间**：自回归——第 $b$ 个 block 以前面所有干净 block 为条件，按顺序逐 block 生成
- **块内**：扩散——每个 block 内部并行预测所有 $L'$ 个 token，不受位置顺序限制

以 12 个 token 分成 3 个 block（每块 4 个）为例：

```
Block 1：the  quick  brown  fox
Block 2：jumps  over  the  lazy
Block 3：dog  sits  down  here
```

### 训练

#### Attention Mask 的精确设计

块扩散的核心在于一套三层组合的 Attention Mask，同时编码了「块内双向」和「块间因果」两种约束。

BD3-LM 把序列中的 token 分为两类：**干净 token**（前面已完成的 block $x^{<b}$）和**加噪 token**（当前 block $x_t^b$）。针对这两类 token 之间的交互，定义了三个子 mask：

- $M_{\text{BD}}$（Block-Diagonal）：加噪 token 之间的自注意力，只能看到同一 block 内的其他加噪 token
- $M_{\text{OBC}}$（Offset Block-Causal）：加噪 token 对干净 token 的跨块注意力，只能看到前面所有干净 block
- $M_{\text{BC}}$（Block-Causal）：干净 token 的注意力，只能看到同 block 内以及前面 block 的 token（标准因果）

完整 Attention Mask 的结构：

$$M_{\text{full}} = \begin{bmatrix} M_{\text{BD}} & M_{\text{OBC}} \\ 0 & M_{\text{BC}} \end{bmatrix}$$

用矩阵示意（3 个 block，每块 3 个 token，加噪 block = Block 2，干净 block = Block 1 + Block 3 已完成部分）：

```
             B1t1 B1t2 B1t3 | B2t1 B2t2 B2t3 | B3t1 B3t2 B3t3
             ← 干净 Block1 → | ← 加噪 Block2 → | ← 干净 Block3 →

B1t1(干净) → [  1    1    1  |  0    0    0   |  0    0    0  ]  ← M_BC: 只看本块
B1t2(干净) → [  1    1    1  |  0    0    0   |  0    0    0  ]
B1t3(干净) → [  1    1    1  |  0    0    0   |  0    0    0  ]
B2t1(加噪) → [  1    1    1  |  1    1    1   |  0    0    0  ]  ← M_OBC(看前块) + M_BD(看本块)
B2t2(加噪) → [  1    1    1  |  1    1    1   |  0    0    0  ]
B2t3(加噪) → [  1    1    1  |  1    1    1   |  0    0    0  ]
B3t1(干净) → [  1    1    1  |  1    1    1   |  1    1    1  ]  ← M_BC: 看所有前块
B3t2(干净) → [  1    1    1  |  1    1    1   |  1    1    1  ]
B3t3(干净) → [  1    1    1  |  1    1    1   |  1    1    1  ]
```

关键点：**加噪 block（Block 2）的每个 token 都能看到所有干净的前置 block（Block 1），以及本 block 内的全部加噪 token（双向）**，但看不到后面的 block。这让块内去噪时能充分利用前文语义，同时块内各位置并行预测。

#### 训练时每个 block 独立采样噪声比例

训练一条样本时，**每个 block 独立采样自己的噪声时间步** $t_b \sim \text{Uniform}(0， 1)$，互不干扰。这样一次前向传播就同时训练了所有 block 的去噪能力，效率比逐 block 分开训练高约 20–25%。

以 3 个 block 的样本为例：

```
Block 1：t₁ = 0.8 → [M]   [M]   brown [M]    （80% 被 mask）
Block 2：t₂ = 0.3 → jumps [M]   the   lazy    （30% 被 mask）
Block 3：t₃ = 0.6 → [M]   sits  [M]   [M]    （60% 被 mask）
```

模型在一次前向传播中，同时以各 block 的干净前置内容为条件，预测三个 block 中被 mask 的位置，loss 在所有被 mask 的位置上累加：

$$\mathcal{L} = \sum_{b=1}^{B} \mathbb{E}_{t_b，\， x_{t_b}^b} \left[ \sum_{i:\， x_{t_b，i}^b = \text{M}} \log p_\theta(x_0^{b，i} \mid x_{t_b}^b，\， x^{<b}) \right]$$

注意这里 $x^{<b}$ 是该 block 前面所有 block 的**干净**版本——训练时模型始终以干净上下文为条件预测当前 block，这与推理时的行为完全一致。

#### 噪声调度：按 block 大小优化裁剪范围

BD3-LM 的另一个细节是 mask 比例并非完全均匀采样，而是对每种 block 大小 $L'$ 单独优化一个「裁剪窗口」$[\beta， \omega]$，在这个区间内均匀采样 $t$，区间外设为 0 或 1。这样可以降低梯度方差，稳定训练。实验发现：

- $L' = 4$：使用 $[0， 1]$（不裁剪）
- $L' = 16$：使用 $[0.3， 0.8]$
- $L' = 128$：使用 $[0.5， 1.0]$（倾向于高噪声）

### 推理：KV Cache 跨 block 传递

推理时逐 block 生成，每个 block 内部走多步扩散去噪，完成后**把该 block 的 KV Cache 缓存起来**，直接传给下一个 block，避免对前置内容重复计算注意力。

$$x_{\text{noisy}}，\; K^b，\; V^b \;\leftarrow\; x_\theta(x_{t}^b，\; K^{1:b-1}，\; V^{1:b-1})$$

以生成 `jumps over the lazy dog sits down here`（2 个 block，每块 4 token）为例：

**生成 Block 1**（无前置上下文）：

```
初始：  [M]  [M]  [M]  [M]

扩散去噪（多步，置信度高的先解锁，解锁后不再重新 mask）：
  步1：  [M]   [M]   the   [M]    （"the" 置信最高，先解锁）
  步2：  jumps [M]   the   lazy   （再解锁两个）
  步3：  jumps over  the   lazy   ← Block 1 完成，缓存 K¹， V¹
```

**生成 Block 2**（以 Block 1 的 KV Cache 为条件）：

```
K¹， V¹ 已缓存（来自 Block 1 的 jumps over the lazy）

初始：  [M]  [M]  [M]  [M]

扩散去噪（每步 Attention 直接复用 K¹， V¹，无需重算 Block 1）：
  步1：  dog  [M]   [M]  [M]
  步2：  dog  sits  [M]  [M]
  步3：  dog  sits  down here  ← Block 2 完成，缓存 K²， V²
```

最终输出：`jumps over the lazy dog sits down here`

**「解锁后不再重新 mask」**（Carry-Over Unmasking）是一个重要约束：块内一旦某个 token 被解锁，后续去噪步骤不会把它重新变回 `[M]`。这与 LLaDA 的「低置信度重 Mask」策略不同——LLaDA 是全序列扩散，需要通过重 Mask 迭代修正；BD3-LM 因为有块间因果依赖，一旦某 block 完成就必须保持稳定，才能作为下一个 block 的可靠上下文。

![Block Diffusion：块间 AR，块内扩散](https://raw.githubusercontent.com/wnma3mz/blog_posts/master/imgs/LLM的推理加速-DFlash/fig2_block_diffusion.svg)

### 与纯扩散的对比

纯扩散模型（如 LLaDA）从全 mask 出发，没有任何上下文，必须多步迭代（T = 10～20 步）才能逐步收敛。块扩散在生成质量和灵活性上比纯扩散更好，但块内仍然是扩散过程，需要若干步去噪。

需要指出的是，**块扩散本身并不等于单步生成**——它只是把扩散的范围从全序列缩小到了一个 block，步数多少取决于具体实现。DFlash 在使用块扩散架构时，会在下一章解释为何能做到单步。

### 关键权衡

- $L'=1$：退化为纯自回归（逐 token 生成，无并行）
- $L'=L$：退化为纯扩散（全序列一次生成，无法动态延伸长度）
- 中间值（$L'=4\sim 16$）：块内并行提速，块间 KV Cache 保持灵活性和生成质量

BD3-LM 在 LM1B 上的困惑度：$L'=4$ 达到 28.23，$L'=16$ 为 30.60，均优于同类扩散模型（MDLM 31.78），且不需要提前固定序列长度。

---

# DFlash：把块扩散变成投机解码的草稿器

回到最初的问题：用扩散模型当草稿器，能一次并行生成 $\gamma$ 个 token，为什么不直接用 LLaDA 这样的纯扩散模型？

问题在于 LLaDA 需要多步去噪（T = 10～20 步），每步都是一次完整前向传播，草稿总成本是 $T \times t_{\text{full}}$，并不比 EAGLE 的串行生成便宜。块扩散同样有多步去噪，直接拿来用也解决不了这个问题。

这里有一个关键洞察：LLaDA 需要多步迭代，根本原因是它在"凭空生成"——从全 mask 出发，没有任何条件，只能靠多步迭代一点点收敛。而投机解码的场景完全不同：每次生成草稿时，**已有完整的前缀（prompt + 已生成内容）作为强条件**，不确定性本来就很低，一步就能给出高质量预测，多步迭代反而是浪费。这和块扩散里"前置 block 已完成、作为当前 block 的条件"在本质上是同一件事。

DFlash 的核心判断是：既然条件这么强，干脆彻底简化——**保留块扩散的 attention 结构**（块内双向让 $\gamma$ 个位置并行预测，块间因果与投机解码的左到右验证对齐），**丢掉扩散的迭代过程**（加噪/去噪在这个场景里没有必要）。

因此 DFlash 的 Draft Model 虽然采用块扩散的 attention 架构，但**训练目标是直接的 token 级交叉熵，推理是单次前向传播直接取 logits**，没有任何迭代去噪步骤。本质上它是一个借用了块扩散 attention mask 的并行预测器，草稿成本真正压缩到 $1 \times t_{\text{draft}}$，与 $\gamma$ 无关。

![投机解码草稿成本对比（γ=8 示意）](https://raw.githubusercontent.com/wnma3mz/blog_posts/master/imgs/LLM的推理加速-DFlash/fig3_draft_cost_compare.svg)

## 架构设计

DFlash 的 Draft Model 是一个 **5 层 Transformer**（Qwen3-Coder-30B 使用 8 层），与 Target Model 共享 token embedding 和 LM Head。

**关键创新：多层特征注入到每层 KV Cache**

![](https://raw.githubusercontent.com/wnma3mz/blog_posts/master/imgs/LLM的推理加速-DFlash/arch.png)

最朴素的想法是把输入直接送进 Draft Model，让它独立完成扩散生成。但 Draft Model 只有 5 层，自身表达能力有限——而 Target Model 跑完 32 层后，中间层的隐向量已经包含了丰富的上下文语义（词性、语义关联、长程依赖等），这些信息 Draft Model 自己推导不出来，Target Model 却已经算好了。

DFlash 的做法是把这些中间结果直接"喂"进 Draft Model，相当于给了它一条捷径。注入到每一层 KV 而不是只注入一次，是为了让这条捷径在每一层都持续有效，不随深度衰减。

以 Target Model 有 32 层、hidden_size=4096、上下文长度为 10 为例，具体步骤如下：

**Step 1：从 Target Model 提取 3 个中间层的隐向量**

在第 2 层到倒数第 3 层之间均匀采样 3 层（例如第 2、16、31 层），每层输出形状为 `[batch， 10， 4096]`，共 3 份。

**Step 2：拼接 + Projection 降维**

把 3 份隐向量沿特征维度拼接：

```
[batch， 10， 4096] × 3  →  concat  →  [batch， 10， 12288]
                          Proj(12288→4096)
                          →  [batch， 10， 4096]   # 融合特征 g
```

Projection 是一个轻量线性层，把三层语义压缩回单份 hidden_size，不引入额外参数量。

**Step 3：注入到 Draft Model 每一层的 KV Cache**

这里的注入方式值得细说。如果只是用一个没有注入的块扩散草稿器，Draft Model 每层的 Attention 只能在 `[M]` tokens 之间互相 attend，外加前置 block 的 KV Cache：

```
无注入时：
  Query:  Q = [q₁， q₂， q₃]        ← 来自 [M][M][M]
  Key:    K = [k₁， k₂， k₃]        ← 来自 [M][M][M] 自身
  Value:  V = [v₁， v₂， v₃]

  每个 [M] 只能 attend 到其他 [M]，看不到 Target 的语义
```

DFlash 把融合特征 $g$ 额外投影成 $K_g$、$V_g$，**前置（prepend）到每一层 KV Cache 的头部**：

```
DFlash 注入后：
  Query:  Q  = [q₁， q₂， q₃]                          ← 不变
  Key:    K  = [k_g₁， k_g₂， k_g₃，  k₁， k₂， k₃]      ← Target特征在前，序列本身在后
  Value:  V  = [v_g₁， v_g₂， v_g₃，  v₁， v₂， v₃]       （论文选择 prepend，append 效果等价，
                                                         softmax 对 K 的排列顺序不敏感）

  每个 [M] 的 Query 打分对象变成 6 个 key，
  Target 语义通过 K/V 通道直接参与每一层的 Attention 权重计算
```

这和 EAGLE-3 把 Target 特征拼在 input embedding 上的做法有本质区别：EAGLE-3 的特征只影响第 1 层输入，随着层加深逐渐被稀释；DFlash 在**每一层**都把 Target 特征放进 K/V，每一层 Attention 都能直接"看到" Target 的语义，深度越深收益越稳定。

**与 EAGLE-3 的对比**

| | EAGLE-3 | DFlash |
|---|---|---|
| 特征注入位置 | 只在 Draft Model 输入端（一次） | Draft Model **每一层** KV Cache |
| 随层深度传播 | 特征信息逐层稀释 | 每层都能直接 attend 到 Target 特征 |
| Draft Model 深度 | 1 层（加深收益有限） | 5 层（深度可扩展） |

正因为每层都有 Target 特征的直接注入，加深 Draft Model 才能持续带来收益，而不会像 EAGLE-3 那样深度提升后接受率趋于饱和。

![DFlash 架构：Target Model 特征注入 Draft Model 各层](https://raw.githubusercontent.com/wnma3mz/blog_posts/master/imgs/LLM的推理加速-DFlash/fig4_architecture.svg)

## 训练：位置权重衰减

训练时，Draft Model 需要预测 block 内 $\gamma$ 个 token。越靠后的位置越难预测，且实际上接受率也更低。DFlash 用**指数衰减权重**来优先优化靠前的位置：

$$\mathcal{L} = \sum_{k=1}^{\gamma} w_k \cdot \text{CE}\!\left(\hat{p}_k，\， p_k^*\right)， \quad w_k = \exp\!\left(-\frac{k-1}{\gamma_{\text{decay}}}\right)$$

| Block 大小 $\gamma$ | $\gamma_{\text{decay}}$ |
| ------------------- | ----------------------- |
| 16                  | 7                       |
| 10                  | 5                       |
| 8                   | 4                       |

训练还使用了**块间稀疏 Attention Mask**：block 内全局 Attention，block 间只能看到前一个 block，确保训练与推理的 Attention 模式完全一致。

训练超参数：6 epochs，AdamW，学习率 $6\times 10^{-4}$，最大序列长度 3072 tokens，每个序列随机采样 512 个 anchor 位置。

## 验证：完全兼容标准投机解码

DFlash 不修改任何验证步骤，直接套用**标准拒绝采样**（Speculative Decoding 原论文的接受算法）：

1. Draft Model **单次前向传播**并行生成 $\gamma$ 个候选 token（一条线性序列）
2. Target Model 并行计算每个位置的 logits
3. 按位从左到右验证，遇到第一个被拒绝的 token 则截断
4. 接受率保证：输出分布与 Target Model 直接自回归完全等价

这是**无损加速**，没有任何质量妥协。

值得对比的是 EAGLE 系列：自回归草稿器每步只能生成一个 token，为了提高覆盖率需要在每个位置分叉，构建一棵草稿树，再用 tree attention 并行验证整棵树。DFlash 一步直接输出 $\gamma$ 个 token，天然是线性序列，不需要树结构，验证流程也更简单。树是自回归草稿器为了弥补串行生成缺陷而引入的补丁，DFlash 从根本上消除了这个问题。

## 为什么 5 层 Draft Model 比 1 层更快？

这是 DFlash 最反直觉的地方。直觉上，5 层应该比 1 层慢。

实际的对比：

| 方法         | 草稿过程                          | 实际延迟（$\gamma=16$）   |
| ------------ | --------------------------------- | ------------------------- |
| EAGLE-3 (1L) | 16 次串行 × 1 层前向              | $16 \times t_{1\text{L}}$ |
| DFlash (5L)  | 1 次并行 × 5 层前向               | $t_{5\text{L}}$           |

由于矩阵乘法的并行性，$t_{5\text{L}} \ll 16 \times t_{1\text{L}}$。论文实测 DFlash 的草稿成本低于 EAGLE-3(16)——用 5 层换来了更低的总延迟和更高的预测质量。

这里有一个值得细想的问题：既然已经有 Target Model 特征注入，用 1 层草稿器不够吗？

在 EAGLE 里，加深草稿器是两难的：1 层快但质量有上限，加到 2 层每步多一层，γ 步下来延迟翻倍，所以 EAGLE 卡在 1 层出不去。DFlash 里这个约束消失了——不管草稿器是 1 层还是 5 层，都只需要 1 次前向传播，加深几乎不增加延迟。既然深度不再是负担，就可以用更多层来做更充分的特征整合：每层变换都能进一步融合 Target 语义和块内各位置的依赖关系，对 $\gamma$ 个位置的联合预测越来越准。5 层比 1 层接受率更高，而延迟几乎不变，自然应该选 5 层。

换句话说，**并行生成把"加深模型"从负担变成了纯收益**。

---

# 实验结果

## 平均接受长度（AAL）

在 Qwen3-8B 上：

| 方法                   | AAL（温度=0） | AAL（温度=1） |
| ---------------------- | ------------- | ------------- |
| EAGLE-3 (budget=16)    | 2.96          | 2.83          |
| EAGLE-3 (budget=60)    | 3.40          | —             |
| **DFlash（block=16）** | **6.49**      | **5.48**      |

DFlash 的 AAL 约为 EAGLE-3 的 **2.2×**。这意味着每轮 Target Model 验证后平均接受的 token 更多，Target Model 调用频率更低。

![平均接受长度对比（Qwen3-8B，温度=0）](https://raw.githubusercontent.com/wnma3mz/blog_posts/master/imgs/LLM的推理加速-DFlash/fig5_aal_bar.svg)

## 端到端加速比

**Transformers 后端：**

| 模型     | DFlash 加速比 | vs EAGLE-3(16) |
| -------- | ------------- | -------------- |
| Qwen3-4B | **4.91×**     | +2.47×         |
| Qwen3-8B | **4.86×**     | +2.40×         |

温度=1（采样模式）下：Qwen3-4B **4.24×**，Qwen3-8B **4.03×**。

**SGLang 后端（B200 GPU，并发=1）：**

| 模型                       | DFlash 加速比 |
| -------------------------- | ------------- |
| Qwen3-4B（数学）           | **8.01×**     |
| Qwen3-8B（数学）           | **5.1×**      |
| Qwen3-Coder-30B            | **3.5×**      |
| LLaMA-3.1-8B（GSM8K）      | **2.4×**      |
| LLaMA-3.1-8B（HumanEval）  | **2.8×**      |

---

# DiffuSpec：无需训练的扩散草稿器

[DiffuSpec（2510.02358）](https://arxiv.org/abs/2510.02358) 是同期把扩散语言模型引入投机解码的工作，核心卖点是**完全不需要额外训练**——直接把一个预训练的扩散语言模型当草稿器插进来用。

## 核心问题：扩散是双向的，验证是单向的

扩散语言模型用双向注意力，一次并行预测多个位置，输出的是对整个序列的联合分布。而投机解码的验证步骤需要的是**从左到右的因果条件概率**：$p(t_k \mid t_1， \ldots， t_{k-1})$。这两者之间有根本性的对齐问题。

DiffuSpec 提出了两个机制来桥接这个 gap：

**因果一致性路径搜索（CPS）**

扩散模型一次前向传播后，对每个位置输出一个候选 token 集合（取 softmax 概率最高的 top-M 个），$\gamma$ 个位置合起来构成一个 token lattice——每个位置有多个候选，所有位置的组合是指数级的。

问题在于：扩散模型选出的联合最优组合，不一定符合自回归的因果条件（即"用前 $k-1$ 个 token 预测第 $k$ 个"）。CPS 在这个 lattice 上做**从左到右的 beam search**（默认 beam=3），对每条路径用两个分数的加权和打分：扩散模型对该 token 的置信度，加上一个小型因果语言模型（n-gram 或轻量 LM）对该路径因果一致性的评分。最终选出分数最高的那条路径作为草稿序列。

以生成 4 个位置的草稿为例：

```
扩散模型输出的候选集合（每个位置 top-3）：
  位置1: [To(0.6)，  For(0.3)，  The(0.1)]
  位置2: [solve(0.7)， find(0.2)， get(0.1)]
  位置3: [x(0.8)，    it(0.1)，   the(0.1)]
  位置4: [，(0.5)，    we(0.3)，   .(0.2)]

CPS beam search（beam=3）从左到右扩展：
  步1: 保留 [To， For， The] 3条路径
  步2: 每条路径扩展，保留综合分最高的3条：
       [To solve， To find， For solve]
  步3: → [To solve x， To solve it， To find x]
  步4: → 最终选出 "To solve x ，"
```

候选集的大小根据每个位置的熵自适应调整——预测越确定的位置候选越少，越不确定的位置候选越多（上限 $M_{\max}=15$），兼顾搜索质量和效率。

**自适应草稿长度（ADL）**

草稿长度 $\gamma$ 在推理过程中动态调整。ADL 用指数移动平均（EMA）追踪两个信号：每轮草稿的有效生成长度 $L^{\text{gen}}$（遇到 EOS 截断）和实际被接受的长度 $L^{\text{acc}}$，然后按如下规则更新下一轮的 $\gamma$：

$$k_{t+1} = \text{clip}\!\left(\lfloor L^{\text{gen}} \rfloor + \delta \cdot \mathbf{1}[L^{\text{acc}} \geq L^{\text{gen}}]，\ k_{\min}，\ k_{\max}\right)$$

接受率高（$L^{\text{acc}} \geq L^{\text{gen}}$）时，$\gamma$ 增加 $\delta$（默认 10 个 token）；接受率低时，$\gamma$ 收缩回 $L^{\text{gen}}$ 附近。默认 $k_{\min}=20$，$k_{\max}=30$，控制器开销为 $O(1)$。

## 实验结果

在 Qwen-2.5-72B 和 LLaMA-2-70B 上测试，无训练情况下：

| 任务 | 加速比 |
|------|--------|
| MT-Bench | 3.09× |
| 翻译 | 3.38× |
| 数学 | 4.02× |
| 摘要 | 2.41× |
| **平均** | **3.08×** |

平均接受长度（MAT）达到 6.99，接近需要额外训练的 EAGLE-2。

## 局限

CPS 搜索引入了额外延迟；草稿长度需要提前指定上限；不同任务的接受率差异较大，ADL 需要仔细调参。本质上，无训练意味着扩散模型和 Target Model 之间没有显式对齐，只靠搜索策略弥合。

---

# SpecDiff-2：用训练解决对齐问题

[SpecDiff-2（2511.00606）](https://arxiv.org/abs/2511.00606) 是对 DiffuSpec 的后续工作，核心判断是：**无训练的 CPS 搜索是治标不治本，扩散草稿器和自回归验证器之间的分布对齐问题应该在训练阶段解决。**

## 两个对齐机制

**Streak-Distillation（训练阶段）**

用一个代理验证器（贪心接受）来给扩散模型提供训练信号：模型生成的草稿，如果验证器连续接受了 $k$ 个 token，就给这条路径正梯度；否则给负梯度。本质是让扩散模型学会"生成验证器愿意接受的序列"。

训练的草稿模型是现成的扩散语言模型（DiffuCoder-7B 或 DiffuLLaMA-7B），在此基础上微调。

**Self-Selection Acceptance（推理阶段）**

推理时并行采样 $K$ 条草稿，用 Target Model 给每条草稿打分，选期望吞吐最高的那条提交验证。用少量额外的并行计算换更高的接受率。

## 实验结果

在 Qwen2.5-72B 和 LLaMA-2-70B 上：

| 模型 | 任务 | 加速比 |
|------|------|--------|
| Qwen2.5-72B | Math-500 | **4.62×** |
| LLaMA-2-70B | 综合 | 3.61× |
| **平均** | | **4.71×** |

相比 DiffuSpec 的 3.08× 提升了约 55%。

## 局限

Streak-Distillation 训练成本较高（约 50k–60k GPU hours）；Self-Selection 需要并行生成 $K$ 条草稿，引入了额外的显存和计算开销；跨 tokenizer 或跨模型族的迁移效果会下降。

---

# 现在的瓶颈与未来方向

## 三者对比

| | DiffuSpec | SpecDiff-2 | DFlash |
|---|---|---|---|
| 是否需要训练 | 否 | 是（蒸馏） | 是（从头训练） |
| 对齐方式 | CPS 搜索（推理时） | Streak-Distillation（训练时） | Target特征直接注入 KV Cache |
| 草稿器深度 | 独立扩散模型 | 独立扩散模型 | 5 层，深度嵌入 Target 语义 |
| 平均加速比 | ~3× | ~4.7× | **~5×（Transformers）/ ~8×（SGLang B200）** |
| 核心思路 | 扩散并行 + 搜索对齐 | 扩散并行 + 训练对齐 | 扩散并行 + 特征融合对齐 |

DiffuSpec 和 SpecDiff-2 把扩散模型当作一个**独立**的草稿器，通过外部手段（搜索或蒸馏）让它和 Target Model 对齐。DFlash 的选择是让草稿器**从设计上就依赖 Target Model**——把 Target 的中间特征注进 Draft 的每一层，对齐问题在架构层面就解决了。

## 瓶颈与可能方向

DFlash 把投机解码的加速比推到了新高，但这条路上仍有几个明显的瓶颈。

**瓶颈一：高并发下加速比大幅下降**

DFlash 的测试主要在并发=1（单请求）的场景下进行，此时 GPU 计算资源充裕，并行草稿生成能发挥出最大收益。但在生产环境的高并发场景下，GPU 已经被多个请求占满，草稿生成和验证都面临资源竞争，加速比会显著缩小。这是所有投机解码方法的共同瓶颈，不只是 DFlash 的问题。

**可能方向**：针对批处理推理专门设计的扩散草稿器，或者结合 Continuous Batching 的流水线调度，让草稿生成和验证在时间轴上错开。

**瓶颈二：草稿长度 γ 需要手动调**

当前 DFlash 的 block size（即 $\gamma$）在训练时就固定了（8、10 或 16）。不同任务、不同模型的最优 $\gamma$ 差异很大：数学推理适合长草稿（接受率高），开放对话适合短草稿（接受率低、长草稿浪费）。训练时固定 $\gamma$ 意味着一个模型无法自适应。

**可能方向**：类似 DiffuSpec 的 ADL，在推理时动态调整 $\gamma$；或者训练时覆盖多种 block size，推理时按任务类型切换。

**瓶颈三：扩散草稿器本身的训练成本**

DFlash 的草稿器需要针对每个 Target Model 单独训练（6 epochs，最大序列长度 3072），不同模型族之间不能直接复用。这意味着每换一个 Target Model 就要重新训练草稿器，部署成本较高。

**可能方向**：草稿器的通用预训练——先在大规模语料上预训练一个扩散基础草稿器，再用少量数据对特定 Target Model 做轻量对齐（类似 SpecDiff-2 的蒸馏，但成本更低）。

**瓶颈四：显存占用增加**

DFlash 的草稿器有 5 层，且每层 KV Cache 都要额外存储 Target Model 提取的融合特征，显存占用高于 EAGLE-3 的 1 层草稿器。在显存紧张的场景（如边缘设备或多路并发）下，这会限制可用的批大小。

**可能方向**：特征压缩（减少注入的 Target 层数）、或者量化草稿器的 KV Cache。

## 更长远：扩散与自回归的深度融合

DFlash 本质上还是"自回归 Target + 扩散草稿"的两阶段结构，Target Model 本身没有变。更激进的方向是让 Target Model 自身也具备并行生成能力——比如把 Target Model 训练成一个混合自回归-扩散模型，在解码阶段直接输出多个 token 的联合分布，彻底消除对单独草稿器的依赖。这个方向目前还处于早期探索阶段（如 D3LLM 等工作），距离生产可用还有较长的路。

---

# 小结

DFlash 的核心洞见只有一句话：**投机解码的草稿器不必是自回归的——只要能快速生成多个候选 token，任何并行生成模型都可以胜任。**

块扩散模型天然满足这个需求：块内并行生成 $\gamma$ 个 token，草稿成本与 $\gamma$ 无关。加上对 Target Model 深层特征的全层 KV Cache 注入，Draft Model 的预测质量大幅提升，平均接受长度从 EAGLE-3 的 3.4 提升到 6.49。

从 EAGLE-1 到 DFlash，推理加速的方向从未变过：**减少 Target Model 的前向次数**。EAGLE 系列通过更好的草稿树做到这一点；DFlash 通过提升草稿质量（更高的 AAL）和降低草稿成本（并行生成）同时发力，实现了新的效率前沿。

---

# 参考文献

- [DFlash（2602.06036）](https://arxiv.org/abs/2602.06036) — 本文主角
- [LLaDA（2502.09992）](https://arxiv.org/abs/2502.09992) — 掩码扩散语言模型
- [Block Diffusion / BD3-LM（2503.09573）](https://arxiv.org/abs/2503.09573) — 块间 AR、块内扩散
- [DiffuSpec（2510.02358）](https://arxiv.org/abs/2510.02358) — 无训练扩散草稿器 + CPS
- [SpecDiff-2（2511.00606）](https://arxiv.org/abs/2511.00606) — Streak-Distillation 对齐训练
- [EAGLE（2401.15077）](https://arxiv.org/abs/2401.15077) — 特征级自回归草稿器
- [EAGLE-2（2406.16858）](https://arxiv.org/abs/2406.16858) — 动态草稿树
- [EAGLE-3（2503.01840）](https://arxiv.org/abs/2503.01840) — 多层特征融合
- [Speculative Decoding（2211.17192）](https://arxiv.org/abs/2211.17192) — 投机解码原论文
- [D3LLM（2601.07568）](https://arxiv.org/abs/2601.07568) — 自回归-扩散深度融合探索
