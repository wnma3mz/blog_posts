---
title: FFN 的前世今生：从 ReLU 到 SwiGLU，再到万亿参数的 MoE
date: 2026-04-08
tags:
  - Transformer
  - FFN
  - MoE
categories: [Survey]
mathjax: true
---

Transformer 的 FFN 层看起来只是一个简单公式，但过去七年真正一路演进的其实是两件事：激活函数怎么选，MLP 结构怎么组织。

这两条线分别从 ReLU、GELU、SwiGLU 和从密集 FFN、MoE 稀疏路由一路推进，最后在 DeepSeek-V3、Mixtral 这类模型里汇合成今天的形态：专家内部用 SwiGLU，外部用路由决定激活哪些专家。

<!-- more -->

---

## 先搞清楚 FFN 在 Transformer 里做什么

Transformer 的每一层由两个子模块组成：Multi-Head Attention 和 Feed-Forward Network（FFN）。Attention 解决的是“哪些 token 应该互相影响”，FFN 解决的是“每个 token 的表示应该怎么被变换”。这两个职责是刻意分离的：Attention 负责跨位置的信息混合，FFN 负责每个位置内部的特征变换，互不干扰，各自并行。

具体来说，原始 Transformer 的 FFN 是一个两层 MLP，中间有一个激活函数：

$$\text{FFN}(x) = W_2 \cdot \text{ReLU}(W_1 x + b_1) + b_2$$

其中 $x \in \mathbb{R}^{d_{\text{model}}}$，$W_1 \in \mathbb{R}^{d_{\text{ff}} \times d_{\text{model}}}$，$W_2 \in \mathbb{R}^{d_{\text{model}} \times d_{\text{ff}}}$，通常 $d_{\text{ff}} = 4 d_{\text{model}}$。

三个部件各司其职：

- **$W_1$（升维）**：把 $d$ 维的 token 表示映射到 $4d$ 维的隐空间，让更多潜在特征“浮出水面”。
- **激活函数**：对 $4d$ 维向量逐元素做非线性变换，决定哪些特征被保留、哪些被压制。这一步是整个 FFN 里**唯一的非线性**。
- **$W_2$（降维）**：把激活后的结果压缩回 $d$ 维，打包成下一层能用的表示。

**为什么是两层 MLP，为什么是 4 倍？**

这两个问题的答案都不来自理论推导，而来自工程积累。两层 MLP 的设计直接继承自 [ConvS2S](https://arxiv.org/abs/1705.03122)（Gehring et al., 2017）——Transformer 作者在“把卷积换成 attention 还能不能工作”这个问题上直接复用了前人的逐位置 MLP 结构，而 1×1 卷积等价于 position-wise 全连接层。4 倍的中间维度则是一个被多个团队反复验证的经验值：Vaswani et al. 2017 原论文做过 d_ff 从 1024 到 4096 的消融（Table 3），结果是越大越好，4 倍是当时参数预算下的性价比最优点。

这个设计在参数分配上有一个不那么显眼但很关键的特性：Attention 的参数量约为 $4d^2$（Q/K/V/O 各一个矩阵），FFN 的参数量约为 $8d^2$，FFN **占整个层参数的约 2/3**。这意味着对 FFN 激活函数的改进，性价比极高——改进的是参数量最大的那个组件。

**这两个组件对应本文的两条演进线：**

- **激活函数**：ReLU 够用吗？为什么换成 GELU、再换成 SwiGLU？为什么现代 FFN 从 2 个矩阵变成了 3 个？
- **MLP 结构**：单个 FFN 的参数量增加就要等比增加计算量，MoE 如何打破这个约束？

两条线到 2024 年的交汇点是：**每个 MoE 专家内部用 SwiGLU**——先把激活函数这件事做好，再在结构层面做稀疏化。

![Transformer 架构中 FFN 与 Attention 的参数占比](https://raw.githubusercontent.com/wnma3mz/blog_posts/master/imgs/ffn/fig1_ffn_overview.svg)

---

## 激活函数存在的意义，以及 ReLU 的问题

> 相关工作：[原始 Transformer](https://arxiv.org/abs/1706.03762)（Vaswani et al., 2017）

### 为什么 FFN 必须有非线性

先想一个问题：如果把 FFN 里的激活函数去掉，这个子层会变成什么？

$$\text{FFN}_{\text{no-act}}(x) = W_2 (W_1 x) = (W_2 W_1) x$$

两个矩阵相乘还是一个矩阵，所以这个两层 FFN 子层会退化成一个线性映射。只有当整个网络都只由这类线性层堆叠而成时，整体才可以折叠成一个线性变换。这里讨论的是 FFN 本身，而不是完整 Transformer；后者还有 attention 里的 softmax、LayerNorm 等非线性。“猫”和“狗”在语义上的差异、“银行（金融机构）”和“银行（河岸）”在语境上的歧义，都说明单靠线性映射不够。

激活函数的作用，就是在每一层强行引入非线性，打破这种退化。

### ReLU 是怎么工作的

ReLU（Rectified Linear Unit）是最直接的非线性：

$$\text{ReLU}(x) = \max(0, x)$$

正值原样通过，负值全部清零。用 FFN 的语言来说：$W_1 x$ 把 token 投影到 $4d$ 维空间，得到 $4d$ 个“候选特征”；ReLU 把其中值为负的特征全部归零，只让正值特征继续传播；$W_2$ 再把这些“幸存特征”压缩回来。

ReLU 的比喻：**一扇单向门**。特征的激活值大于零，门开；小于零，门关，而且是永久性的关死——负值特征对这个 token 的这次前向传播没有任何贡献。

### 感受一下 ReLU 的问题

用一个具体的例子。假设 FFN 某个隐藏神经元对应的语义是“这个词具有动作性”，它的激活值在不同上下文下是这样的：

| 输入 | 激活值（$W_1 x$ 的某个分量） | ReLU 输出 |
| --- | --- | --- |
| “他**奔跑**起来” | +2.1 | +2.1（完整通过） |
| “轻轻地**走**” | +0.3 | +0.3（通过，但很弱） |
| “他**停下**来” | -0.2 | 0（清零） |
| “静止的**雕像**” | -1.8 | 0（清零） |

问题出在后两行：-0.2 和 -1.8 被同等对待，都变成了 0。但这两者在语义上未必等价：-0.2 更像“弱不满足”，-1.8 更像“强烈不满足”。ReLU 把它们都压成 0，这部分区分度就被抹掉了。

更麻烦的是训练时：如果某个神经元在很长一段时间里都落在负半轴（比如这个“动作性”神经元在大量静态描述文本上始终输出负值），ReLU 会让它在这些样本上的梯度为零，参数更新显著变慢，实践中就可能出现有名的 **dying ReLU** 问题。这里的“死亡”更准确地说是“长期不工作”，而不是数学上绝对不可恢复，但对训练依然是个真实问题。

这两个问题的根源是同一个：ReLU 的决策是**二值的、不可逆的**——负值信息被彻底丢弃，而不是被“考虑之后决定不用”。这也是后来 GELU 和 SwiGLU 试图缓解的地方。

---

## 第一次改进：GELU

> 论文：[Gaussian Error Linear Units (GELUs)](https://arxiv.org/abs/1606.08415)（Hendrycks & Gimpel, 2016）

GELU（Gaussian Error Linear Unit）把激活函数从“确定性开关”改成了“概率性加权”。

### 核心公式

$$\text{GELU}(x) = x \cdot \Phi(x)$$

其中 $\Phi(x)$ 是标准正态分布的累积分布函数（CDF）。直觉上，GELU 把输入 $x$ 与“$x$ 大于标准正态随机变量的概率”相乘——正值越大，通过的比例越高；负值虽然不被完全截断，但会按比例衰减。

在实践中通常用近似公式计算：

$$\text{GELU}(x) \approx 0.5x\left(1 + \tanh\!\left(\sqrt{\frac{2}{\pi}}\,(x + 0.044715x^3)\right)\right)$$

### 与 ReLU 的本质差异

| | ReLU | GELU |
| --- | --- | --- |
| 负值处理 | 全部截断为 0 | 按概率衰减，不完全清零 |
| 梯度 | $x < 0$ 时恒为 0 | 处处可微，负区域有小梯度 |
| 单调性 | 单调 | 非单调（在 $x \approx -0.17$ 处有局部极小值） |
| 信息损失 | 有损（负值丢失） | 有损（但损失更平滑） |

用前面“动作性”神经元的例子来感受一下区别：激活值 -0.2 的 token，在 ReLU 下完全被清零；在 GELU 下，$\Phi(-0.2) \approx 0.42$，输出是 $-0.2 \times 0.42 \approx -0.08$——仍然是负的，但不是零。激活值 -1.8 的 token，$\Phi(-1.8) \approx 0.04$，输出约 $-0.07$——被强烈压制，但梯度仍然小幅存在。也正因为如此，GELU 往往能缓解 ReLU 在负半轴上的训练停滞问题。不过把它说成“彻底解决 dying ReLU”就太满了。更稳妥的说法是：它让神经元更不容易因为长期落在负区间而失去学习能力。

GELU 被 BERT、GPT-2、GPT-3 等早期大语言模型广泛采用，成为 2018–2021 年的主流选择。

![ReLU vs GELU 函数图像对比](https://raw.githubusercontent.com/wnma3mz/blog_posts/master/imgs/ffn/fig2_relu_gelu.svg)

---

## 真正的范式转变：从 2 个矩阵到 3 个矩阵

> 论文：[Language Modeling with Gated Convolutional Networks](https://arxiv.org/abs/1612.08083)（Dauphin et al., 2017）；[GLU Variants Improve Transformer](https://arxiv.org/abs/2002.05202)（Noam Shazeer, 2020）

GELU 改进了激活函数的形状，但没有改变 FFN 的结构——仍然是一条路径：升维 → 激活 → 降维。门控线性单元（Gated Linear Unit, GLU）带来了一个更根本的变化：**把单路径变成双路径，用两条并行的线性变换共同决定信息通量**。

### GLU 为什么要引入第二条路径

要理解 GLU 的动机，需要先回到 2017 年的语言模型领域。当时主流方案是**堆叠卷积层**来建模序列——LSTM 虽然能建模长距离依赖，但速度慢、难以并行；卷积快但感受野有限，必须叠很多层才能让信息在序列中传播足够远。

在这种“必须叠深”的压力下，标准的两个矩阵加激活函数的结构开始出问题。问题的根源在于**激活函数的导数**。以当时常用的 tanh 为例，它的输出被压缩在 $(-1, 1)$，导数最大也只有 1，且在输入较大时迅速趋向零。反向传播时，梯度每经过一层激活函数就乘以一个小于 1 的因子，叠个十几二十层之后，梯度就指数级缩小，底层的参数几乎得不到有效更新——这就是梯度消失。

两个矩阵在**浅层**时完全够用：层少，梯度的衰减还没有积累到无法训练的程度。但 Dauphin 等人想要的是更深的卷积网络来竞争 LSTM，深度一上来两矩阵结构就撑不住了。

他们的解法是把激活函数从“乘在信号路径里”变成“用来控制门的”。GLU 引入了第二个矩阵 $V$ 和一条独立的门控路径：

$$\text{GLU}(x) = (xW) \otimes \sigma(xV)$$

其中 $\otimes$ 是逐元素乘法。关键在于左分支 $xW$ 是**纯线性变换**，没有任何激活函数——这条路径的梯度在反向传播时直接通过，不经过任何饱和环节。sigmoid 只用在门控路径上，决定“放多少信号通过”，但它的饱和不再直接卡死主信号的梯度流。

对比当时的另一个候选方案 GTU（Gated Tanh Unit）：

$$\text{GTU}(x) = \tanh(xW) \otimes \sigma(xV)$$

这里两条路径都有饱和激活函数，梯度在两边都会被压缩，深层时仍然会消失。GLU 的本质改进就是把左分支的 tanh 去掉，让它成为一条梯度高速公路，相当于一个**乘法形式的残差连接（multiplicative skip connection）**——可以类比 ResNet 的加法跳连接，只不过这里是逐元素相乘而不是相加。

论文实验验证了这个改进的实质性收益：WikiText-103 上 GLU 达到 37.2 perplexity（vs. LSTM-1024 的 48.7），Google Billion Words 达到 38.1（vs. LSTM 的 39.8）。

### SwiGLU：当前的标准

[Shazeer（2020）](https://arxiv.org/abs/2002.05202)把 GLU 引入 Transformer 的 FFN 层，系统测试了不同的门控激活函数，其中 **SwiGLU** 效果最好：

$$\text{SwiGLU}(x, W, V) = \text{Swish}(xW) \otimes (xV)$$

$$\text{Swish}(x) = x \cdot \sigma(x) = \frac{x}{1 + e^{-x}}$$

与原始 GLU 相比，SwiGLU 可以理解为把其中一支从 sigmoid 门换成了 Swish。这里有个容易混淆的小点：由于逐元素乘法可交换，文献和工程实现里对“两支里谁叫门、谁叫值”的命名并不总是完全一致；真正关键的是，一支带非线性，另一支保持线性。sigmoid 把输出压缩到 $(0, 1)$，调制值永远是正的，相当于只能“放多少”，不能“反转”；Swish 的输出可以为负，允许这一支对某些特征施加负向调制。这给了模型更丰富的特征调控空间。在等参数条件下，SwiGLU 的 perplexity（1.636）比 GELU FFN 基线（1.677）低约 2.4%，在 GLUE、SuperGLUE、SQuAD 上的微调性能也全面领先。

完整的 SwiGLU FFN 结构是：

$$\text{FFN}_{\text{SwiGLU}}(x) = W_2 \cdot (\text{Swish}(xW_1) \otimes (xV))$$

这里有**三个权重矩阵**（$W_1, V, W_2$），而标准 FFN 只有两个。为了维持总参数量不变，需要把中间维度从 $4d$ 缩减到约 $\frac{8}{3}d \approx 2.67d$：

| 结构 | 矩阵数 | 中间维度 | 总参数量 |
| --- | --- | --- | --- |
| 标准 FFN（ReLU/GELU） | 2 | $4d$ | $2 \times d \times 4d = 8d^2$ |
| SwiGLU FFN | 3 | $\frac{8}{3}d \approx 2.67d$ | $3 \times d \times \frac{8}{3}d = 8d^2$ |

第三个矩阵 $V$ 带来的不是参数增加，而是**第二条信息路径**。如果沿用本文的记号，可以把 $xV$ 看成内容分支，把 $\text{Swish}(xW_1)$ 看成调制分支：前者负责“编码什么内容”，后者负责“以多大强度、什么符号去调它”。两者逐元素相乘，相当于给每个 token 配备了一个自适应过滤器，在不同上下文下动态激活不同的特征子空间。这是单路径 ReLU/GELU FFN 做不到的。

![FFN 结构演进：标准 FFN → 门控 FFN（SwiGLU）](https://raw.githubusercontent.com/wnma3mz/blog_posts/master/imgs/ffn/fig3_ffn_structure.svg)

---

## SwiGLU 成为工业标准

> 论文：
> - [PaLM: Scaling Language Modeling with Pathways](https://arxiv.org/abs/2204.02311)（Chowdhery et al., 2022）
> - [Llama 2: Open Foundation and Fine-Tuned Chat Models](https://arxiv.org/abs/2307.09288)（Touvron et al., 2023）
> - [Llama 3](https://arxiv.org/abs/2407.21783)（Meta AI, 2024）

2022 年，Google 发布了 540B 参数的 PaLM，在技术报告中明确写道选用 SwiGLU 激活，并将这一选择归因于 Shazeer 2020 年的工作。此后，几乎所有主流 LLM 都跟进采用了相同设计：

| 模型 | 年份 | 激活函数 | FFN 扩展比 |
| --- | --- | --- | --- |
| GPT-3 | 2020 | ReLU | 4× |
| PaLM | 2022 | SwiGLU | ~4× |
| LLaMA 1 | 2023 | SwiGLU | 2.69× |
| LLaMA 2 | 2023 | SwiGLU | 2.69× |
| LLaMA 3 | 2024 | SwiGLU | 2.67× |
| Qwen 2.5 | 2024 | SwiGLU | 2.67× |
| Gemma 3 | 2025 | SwiGLU | 3–4× |

SwiGLU 的普及揭示了 LLM 架构研究中一个有趣的现象：一旦某个组件被足够大的模型验证，整个领域就会迅速收敛到这个选择。这倒不一定是因为没有替代方案，更常见的原因是**切换激活函数的实验成本极高**，而 SwiGLU 的收益已经被反复验证。

---

## 容量瓶颈：FFN 是知识存储器

在理解 MoE 之前，需要先想清楚：FFN 层在语言模型中究竟承担什么角色？

一个很有影响力的解释视角来自 [Geva et al. (2021)](https://arxiv.org/abs/2012.14913)，他们从实验角度提出：**FFN 层的行为可以部分地看作键值记忆（key-value memory）**。

### FFN 即键值记忆：怎么证明的

把 FFN 拆开看：$W_1 x$ 计算输入与 $W_1$ 每一行的内积，结果再经过激活函数。注意 $W_1$ 的每一行是一个 $d$ 维向量，和 $x$ 做内积，本质上是在测量“输入 $x$ 与这一行向量有多相似”——这很像**键（key）查询**的语义。激活函数则扮演阈值的角色：内积足够大，也就是匹配度够高，才通过；否则就被压制。

被激活的神经元对应 $W_2$ 的一列，这一列向量被按激活值加权叠加进输出——这是**值（value）读取**的语义。整个 FFN 的前向过程因此可以理解为：

1. $W_1 x$：用输入去查询所有键，看哪些模式匹配
2. $\text{act}(W_1 x)$：筛选出匹配的键（激活值为正），抑制不匹配的
3. $W_2 \cdot \text{act}(W_1 x)$：把匹配键对应的值加权叠加，写回输出

Geva 等人的实验对此做了可解释性验证：他们找到了大量“语义清晰的键”——比如某一行 $W_1$ 的权重向量，会和“表示年份的 token”（1990、2003 等）的输入高度匹配，而对应的 $W_2$ 列则在输出空间里把概率质量集中到“数字相关词”上，像一个“见到年份就激活、激活后输出数字知识”的记忆单元。他们在 GPT-2 上系统测试后发现，约 29% 的神经元可以找到对应清晰语言模式的键；浅层 FFN 更多存储语法、词法模式，深层 FFN 更多存储事实性知识。

这个视角至少解释了两件事：
1. 为什么增大 FFN 的中间维度常常能提升模型质量——更大的中间层意味着更多的键值对，可以存储更多模式和知识。
2. 为什么 FFN 的部分知识可以被“编辑”——至少在某些局部事实上，直接修改 $W_2$ 的对应列，就能改变模型对某类输入的输出倾向，这也是“模型编辑（model editing）”技术的一条重要思路。

### 容量的代价

如果接受这个视角，那就有一个自然推论：**中间维度越大，能容纳的键值对越多，模型可存的模式和知识也越多**。

但这里有个铁律：**参数越多，每次前向传播的 FLOP 越多**。FFN 中间维度翻倍，矩阵乘法的计算量也翻倍，每个 token 都跑一遍。当模型规模达到数百亿参数时，这个代价就很难承受——你想让模型“知道更多”，推理成本也会跟着涨上去。

这里有个关键问题要先想清楚：**增加参数但只激活一部分，效果真的还能保持甚至提升吗？**

直觉上这似乎矛盾——一次推理用的参数少了，为什么不退步？答案就在键值记忆这个视角里：不同的输入应该“查到”不同的知识，**大多数知识对某一个具体 token 来说其实是不相关的**。强迫每个 token 都激活全部参数，不是在充分利用容量，反而可能是在引入噪声。稀疏激活更接近“按需检索”：总知识量更大，每次检索也更聚焦。

Switch Transformer 的实验给出了直接证据：固定 FLOP 预算下，用 MoE 把参数量扩大 7 倍后，模型的困惑度不是持平，而是显著下降。“参数多且稀疏”比“参数少但全激活”学得更好。这就是 MoE 的核心逻辑。

既然思路清楚了，还有一条路看似也说得通：既然 W1 的每一行都像一个“知识键”，能不能把已经训练好的 dense 模型直接拆分成多个专家，再微调一下，变成 MoE？

这条路确实有人尝试过，[Komatsuzaki et al.（2022）](https://arxiv.org/abs/2212.05055)把它称为“Sparse Upcycling”——把预训练好的 T5 模型的 MLP 权重复制成多份作为专家初始化，路由器随机初始化，然后继续训练。短期来看，这种做法有一定优势，能比从头训练的 MoE 更快收敛。但 OLMoE（2024）的系统性对比发现：从头训练的 MoE 只需要约 500B token 后就能追上 upcycled 模型，随后反超。

原因也不难理解：**路由器是随机初始化的**，它不知道复制来的各个专家各自擅长什么，整个系统需要从零发现分工。更根本的是，dense 模型的 FFN 权重被训练成“对所有 token 都有用”，而 MoE 专家更理想的状态是“只对某类 token 有用”——这是两个不同的优化目标，fine-tuning 能做的局部修正未必足以弥合这个差距。至少从现有公开结果看，只要训练预算足够大，从头训练往往会更好。

---

## 条件计算的尝试：早期 MoE

> 论文：[Outrageously Large Neural Networks: The Sparsely-Gated Mixture-of-Experts Layer](https://arxiv.org/abs/1701.06538)（Shazeer et al., 2017）

2017 年，Shazeer 等人提出了稀疏门控 MoE 层，将条件计算正式引入大规模语言模型。

### 核心设计

MoE 的基本思路是：把单个 FFN 替换成 $E$ 个独立的 FFN（专家），每次只让 token 通过其中的 $k$ 个：

$$\text{MoE}(x) = \sum_{i=1}^{E} G(x)_i \cdot \text{Expert}_i(x)$$

其中 $G(x)$ 是门控网络，输出 token $x$ 对各专家的权重：

$$G(x) = \text{Softmax}(\text{TopK}(x \cdot W_g, k))$$

$\text{TopK}$ 操作选出得分最高的 $k$ 个专家，其余置零。这样每次前向传播只需要计算 $k$ 个专家，参数利用率变成 $k/E$。

实验结果：在 1-Billion Word benchmark 上，4096 专家的 MoE 模型（68B 参数）比计算量匹配的 LSTM 基线低 **24% perplexity**，达到约 33 perplexity。

### 负载均衡：最棘手的问题

MoE 最难解决的不是路由本身，而是**专家负载不均衡**。如果不加约束，门控网络会倾向于总是选择少数几个“好”专家，导致大部分专家几乎不被使用——这既浪费参数，又会形成计算瓶颈（被频繁选中的专家成为“热点”）。

Shazeer 等人引入了辅助负载均衡损失：

$$\mathcal{L}_{\text{balance}} = \alpha \cdot \sum_{i=1}^{E} f_i \cdot p_i$$

其中 $f_i$ 是专家 $i$ 被分配的 token 比例，$p_i$ 是门控网络对专家 $i$ 的平均得分。这个损失鼓励每个专家被均匀使用，但权重 $\alpha$ 的调整非常敏感——太小则不起作用，太大则干扰主要训练目标。

尽管如此，早期 MoE 的工程复杂性极高，训练不稳定，没有在主流 LLM 中大规模采用。真正让 MoE 进入主流的，是 2021 年的 Switch Transformer。

---

## MoE 简化：Switch Transformer

> 论文：[Switch Transformers: Scaling to Trillion Parameter Models with Simple and Efficient Sparsity](https://arxiv.org/abs/2101.03961)（Fedus et al., 2021）

Switch Transformer 提出了一个看似反直觉的简化：**每个 token 只路由到一个专家（$k=1$）**。

### 为什么 $k=1$ 是对的

Shazeer 的原始设计用 $k=2$ 或更多，是因为担心单专家路由的梯度太稀疏（门控网络只有被选中专家的梯度）。Switch Transformer 的反驳是：在大批量、分布式训练中，每个专家每次都能收到大量 token，梯度稀疏不是问题；而简化到 $k=1$ 带来的工程红利（无需对多专家输出做加权求和）非常可观。

Switch 路由的完整算法：

$$e^* = \arg\max_i (x \cdot W_g)_i$$
$$\text{MoE}(x) = \text{Expert}_{e^*}(x)$$

辅助损失同样保留，但权重更轻。关键创新是引入了**容量因子（capacity factor）**：每个专家在每个 batch 里最多处理 $C = c \cdot \frac{T}{E}$ 个 token（$T$ 是 batch 的 token 数，$c$ 通常设为 1.0–1.25）。超过容量的 token 直接跳过该专家（overflow），保证每个专家的负载有上界。

### 训练效率提升

Switch Transformer 在 T5 的框架上，用相同的 FLOP 预算将模型参数扩展了 7 倍（通过 MoE 层）。Switch-Base 64 专家模型（14.7B 总参数）在步骤效率上达到了 T5-Base（223M 参数）的 **7 倍**——同样的质量阈值，Switch 只需 T5 训练步数的 1/7。这验证了 MoE 的核心价值：**在固定计算预算内，参数越多（知识容量越大），质量越高**。

![MoE 专家路由示意：token 被门控网络分配到不同专家](https://raw.githubusercontent.com/wnma3mz/blog_posts/master/imgs/ffn/fig4_moe_routing.svg)

---

## 精细化的 MoE：DeepSeekMoE

> 论文：[DeepSeekMoE: Towards Ultimate Expert Specialization in Mixture-of-Experts Language Models](https://arxiv.org/abs/2401.06066)（Dai et al., 2024）

Switch Transformer 简化了路由，但 DeepSeekMoE 的作者在分析既有 MoE 架构时指出了两个结构性缺陷，并通过实验给出了证据。

**问题一：知识混杂（Knowledge Hybridity）。** 当专家数量有限（典型值是 8 或 16 个）时，被路由到同一个专家的 token 往往跨越多种截然不同的语义类型——这个专家既要处理数学推理，又要处理代码语法，还要处理日常对话。参数量固定，却需要同时承载差异极大的知识，利用效率自然不高。

**问题二：知识冗余（Knowledge Redundancy）。** 不同专家接收的 token 可能都需要某些通用知识（如基本语法规则），导致多个专家各自在参数里重复学习同一份内容。这部分重叠的知识占据了本可用于专业化的参数容量。

DeepSeekMoE 通过“禁用专家”实验来量化冗余程度：对于每个 token，屏蔽路由概率最高的若干个专家，强制从剩余专家中选择，然后观察模型损失的变化幅度。如果专家之间高度冗余，禁用头部专家后其他专家可以无缝顶替，损失变化不大；如果专家已经高度专业化，禁用任何一个都会造成明显的性能下降。实验结果显示，在相同被禁用比例下，GShard 的损失增幅远小于 DeepSeekMoE——说明 GShard 的专家之间可替代性更强，即冗余程度更高，而 DeepSeekMoE 的每个专家承担的专门职能更强。

DeepSeekMoE 针对这两个问题做了系统性改进。

### 细粒度专家分解

DeepSeekMoE 的第一个洞察：**专家的粒度应该更细**。如果把一个专家拆成更多更小的专家，每个专家负责更窄的能力范围，则专家之间的重叠会更少，整体利用率更高。

具体设计：用 $mN$ 个细粒度专家替换 $N$ 个标准专家（$m$ 是分裂倍数，通常取 4），但 top-$k$ 的选择数量也相应增加到 $mK$，保持每次激活的参数量不变。

### 共享专家与路由专家分离

DeepSeekMoE 的第二个洞察：有些知识是**所有 token 都需要的**（如基本语法、常识），有些知识是**特定 token 才需要的**（如特定领域知识）。把这两类知识混在一起让路由网络选择，会降低效率。

解决方案：引入**共享专家（shared experts）**，始终被所有 token 激活，负责通用知识；其余为**路由专家（routed experts）**，通过 top-$K$ 路由动态选择。

$$\text{DeepSeekMoE}(x) = \sum_{i=1}^{K_s} \text{Expert}_i^{\text{shared}}(x) + \sum_{j \in \text{top-}K_r} G_j(x) \cdot \text{Expert}_j^{\text{routed}}(x)$$

这个设计让模型可以在路由专家之间实现更高程度的专业化，因为通用知识已经被共享专家兜底了。

### 实验结果

DeepSeekMoE 的主要验证在 2B 和 16B 两个尺度上进行（而非更大的规模）：

| 模型 | 激活参数 | Pile Loss |
| --- | --- | --- |
| DeepSeekMoE-2B | 0.3B | **1.808** |
| GShard-2B（对照） | 0.3B | 1.867 |
| Dense-2B（对照） | 2B | 2.060 |

在 16B 尺度上，DeepSeekMoE-16B（2.8B 激活参数）与 LLaMA-2-7B（7B 全激活）性能相当，计算量仅约 40%——相同的效果，花了不到一半的计算。

---

## 工业验证：Mixtral 和 DeepSeek-V3

### Mixtral 8×7B：简洁有效

> 论文：[Mixtral of Experts](https://arxiv.org/abs/2401.04088)（Jiang et al., Mistral AI, 2024-01-08）

Mixtral（2024 年 1 月 8 日发布）采用了最直接的 MoE 设计：8 个专家，每次 top-2 路由，每个专家就是一个标准的 SwiGLU FFN。路由网络使用标准 Softmax，在 8 个专家这个规模下，Softmax 路由本身就有一定的自均衡效果，不需要额外的工程手段——这和 Switch Transformer 并不矛盾：Switch 解决的是数百个专家时的大规模工程稳定性，8 个专家的规模下基础路由已经够用。

$$\text{Mixtral}(x) = \sum_{i \in \text{top-2}} G_i(x) \cdot \text{Expert}_i(x), \quad G(x) = \text{Softmax}(x \cdot W_g)$$

**关键数字**：47B 总参数、13B 活跃参数（约 28% 利用率），在主要基准上超越了拥有 70B 全激活参数的 LLaMA-2 70B：

| 基准 | Mixtral 8×7B（13B 激活） | LLaMA-2 70B（70B 激活） |
| --- | --- | --- |
| MMLU | **70.6%** | 69.9% |
| GSM8K | **58.4%** | 53.6% |
| MBPP（Pass@1） | **60.7%** | 49.8% |

13B 的活跃参数量打败了 70B 的密集模型，这个结果有力地证明了 MoE 的参数效率。

### DeepSeek-V3：256 个专家的精密系统

> 论文：[DeepSeek-V3 Technical Report](https://arxiv.org/abs/2412.19437)（DeepSeek AI, 2024）

DeepSeek-V3 代表了 2024 年 MoE 设计的最高水准，也是 DeepSeekMoE 架构理念的全面实施。

关键参数：
- 总参数：671B（256 个路由专家 + 1 个共享专家）
- 每次激活：8 个路由专家 + 1 个共享专家，约 37B 活跃参数（**6% 利用率**）
- 每个专家：标准 SwiGLU FFN

DeepSeek-V3 特别解决了 MoE 训练中的一个长期痛点：**辅助损失与主任务的冲突**。传统辅助负载均衡损失需要仔细调整权重，权重太小则专家塌陷，权重太大则干扰语言建模。

DeepSeek-V3 提出了**无辅助损失的负载均衡（auxiliary-loss-free load balancing）**：通过动态调整每个专家的偏置项 $b_i$，而不是通过梯度反传的辅助损失来实现负载均衡：

$$G_i(x) = \text{Softmax}(x \cdot w_i + b_i), \quad b_i \leftarrow b_i + \gamma \cdot \text{sign}(\bar{f} - f_i)$$

其中 $\bar{f}$ 是平均负载，$f_i$ 是专家 $i$ 的实际负载，$\gamma$ 是步长。当某个专家过载时，降低其偏置；当某个专家欠载时，提升其偏置。这个机制不经过反向传播，不干扰模型的主训练目标。

**关键数字**：训练成本约 270 万 H800 GPU 小时（约 200 万美元），MMLU-Pro 75.9、MATH-500 90.2。在这些公开 benchmark 上，它的表现已经接近 GPT-4o 和 Claude-3.5-Sonnet。若和同量级密集模型相比，MoE 至少在训练成本上展示出了非常明显的效率优势。

---

## MoE 的规模边界：从大模型到小模型

目前讨论的 MoE 模型都是数十亿到千亿参数级别的，但 MoE 的思路能不能用在更小的模型上——比如总参数 7B、激活参数 1B 这样的规模？

**OLMoE（Allen AI, 2024）**是目前最系统地回答这个问题的工作。它的设计：总参数 6.9B，每层 64 个专家，每 token 激活 8 个（top-8 路由），激活参数约 1.3B。用 5.1 万亿 token 训练后，在 1B 激活参数这个类别里碾压了所有密集基线：

| 模型 | 类型 | MMLU | HellaSwag | ARC-C |
| --- | --- | --- | --- | --- |
| OLMoE-1B-7B | MoE（1.3B 激活/6.9B 总） | **54.1** | **80.0** | **62.1** |
| OLMo-1B | Dense（1B） | 32.1 | 67.5 | 36.4 |
| DCLM-1B | Dense（1B） | 48.5 | 75.1 | 57.6 |
| OLMo-7B | Dense（7B，5× 激活量） | 54.9 | — | — |

OLMoE 用 1.3B 的激活参数量，达到了 OLMo-7B（激活参数量是其 5 倍）的水平，且训练 FLOP 效率约为同等密集模型的 3 倍。专家也确实出现了领域特化——部分专家专门处理数学符号，部分专家专门处理非拉丁字符，证明路由有实质意义。

不过，小型 MoE 有一个固有的工程瓶颈：**必须把所有专家的权重加载进内存，但每次只用其中一小部分**。OLMoE-1B-7B 需要 7B 的 GPU 内存，却只做了 1.3B 的计算——在高吞吐的批量推理场景下，这笔账是划算的（参数量＝知识容量），但在单用户低延迟场景下，相同内存预算不如直接用一个 1.3B 的密集模型。MoE 的工程优势需要足够大的 batch size 才能充分体现。

这也是为什么工业界的 MoE 产品主要集中在大规模推理场景：专家并行在多卡部署下摊薄了路由和通信的开销，越大的模型、越高的吞吐，MoE 相对于密集模型的优势越显著。

---

## 专家真的在专业化吗：MoE Lens 的解剖

> 论文：[MoE Lens: An Expert Is All You Need](https://arxiv.org/abs/2603.05806)（2026）

MoE 设计的核心假设是：不同专家会分化出不同的专长，路由机制把合适的 token 送给合适的专家。但这个假设成立吗？专家真的在专业化，还是大部分工作其实只由少数几个专家承担？

MoE Lens 对 DeepSeekMoE、Qwen 1.5 MoE、OLMoE 三个模型做了系统分析，结论颇为出人意料。

### 路由高度集中

研究者分析了英语、代码、法语三类数据在各专家间的路由分布。结果发现：**极少数专家处理了超过 50% 的路由决策**，大多数专家被激活的频率显著低于均匀分布的基准值（DeepSeekMoE 的均匀基准是 6/64 ≈ 9.4%）。这意味着路由并不均匀——网络实际上已经形成了少数“主力专家”和大量“边缘专家”的分工格局。

### 单个专家就够了？

更关键的发现来自以下实验：对每个 token，只保留路由概率最高的那一个专家，丢弃其余被选中的专家，看性能如何变化。

结果：**从 top-6 降到 top-1，perplexity 仅上升约 5%**。研究者还测量了单专家隐状态 $H^l_1$ 与全加权隐状态 $H^l_6$ 之间的余弦相似度，在所有 27 层上均达到 **0.95 左右**。换句话说，$H^l_1 \approx H^l_6$——top-1 专家一个人就能几乎完整地复现整个 top-k 集成的输出。

### 这意味着什么

这个发现有两层含义，方向相反：

**一方面，这是推理优化的机会。** 如果 top-1 专家的输出就足够好，那么在延迟敏感的场景下可以只激活单个专家，大幅节省计算和内存带宽。对于 64 专家激活 6 的设计，理论上可以压缩到激活 1，推理成本降至原来的 1/6，质量损失仅 5%。

**另一方面，这也提出了一个架构层面的疑问。** 设计 MoE 的初衷是让多个专家协同贡献各自的专业知识，但如果一个专家就能代替六个，是否意味着当前的训练机制还没有充分诱导出真正的专业分化？路由集中和单专家可替代性，可能是同一个现象的两面——少数强专家“包揽”了大部分工作，而其余专家没有被充分利用。

值得注意的是，MoE Lens 的分析对象包括 DeepSeekMoE——一个已经引入了共享专家机制的架构。有人可能会问：共享专家不是已经解决了专业化问题吗？

答案是：两者解决的不是同一个问题。DeepSeekMoE 的共享专家针对的是**知识冗余**——多个路由专家重复学习通用知识（语法、常识等），共享专家把这部分剥离出来，让路由专家“少做重复劳动”。而 MoE Lens 发现的是**路由集中**——即便有了共享专家兜底，路由专家之间的使用频率依然极度不均，少数强专家包揽大多数工作。打个比方：共享专家解决的是“公司有了专职行政，工程师不用兼职打杂”，而路由集中说的是“即便如此，还是有两三个明星员工包揽了大部分核心项目，其余人的专业能力没被充分发挥”。共享专家的存在并没有消除这个现象，在 DeepSeekMoE 上 top-1 专家的隐状态与 top-6 加权输出的余弦相似度仍然达到 0.95。

这是当前所有 MoE 架构共同面对的开放命题。

---

## 动态路由：DTop-p MoE

> 论文：[Sparsity-Controllable Dynamic Top-p MoE for Large Foundation Model Pre-training](https://arxiv.org/abs/2512.13996)（He et al., 2025）

从 Shazeer 2017 到 DeepSeek-V3，所有 MoE 模型都在用 **top-k 路由**：每个 token 固定激活 $k$ 个专家，$k$ 是一个在训练前就设定好的超参数。这个设计有个隐含假设：**所有 token 都需要同样数量的专家参与计算**。

但这个假设是对的吗？“2 + 3 = ？”和“请解释量子纠缠的哲学含义”，真的需要激活同样多的专家吗？

### top-k 的问题，以及 top-p 为什么也不够

一个自然的改进思路是把 top-k 换成 top-p（nucleus sampling）：不再固定专家数量，而是固定一个概率阈值 $p$，把路由得分累加超过 $p$ 的专家都激活。这样简单 token 可能只需要 2 个专家，复杂 token 需要更多。

但固定阈值的 top-p 有个致命问题：**稀疏度不可控**。训练初期路由分布比较平坦，阈值 $p=0.3$ 可能激活 8 个专家；训练到后期路由变得尖锐，同样的 $p=0.3$ 可能只激活 3 个专家。稀疏度的剧烈波动导致梯度不稳定，训练容易崩溃。实验中，将阈值从 0.30 调到 0.35，激活专家数就从 ~8 跳到 >11，超参数极度敏感。

### PI 控制器：让稀疏度成为可控量

DTop-p 的解法来自控制论：用一个**比例-积分（PI）控制器**把概率阈值 $p$ 变成一个动态调整的量，实时追踪目标激活专家数 $T$。

每个训练步骤：
1. 统计当前 batch 的平均激活专家数 $a_t$
2. 计算误差 $e_t = (T - a_t) / N$
3. 更新阈值：$p_{t+1} = p_0 + K_p \cdot e_t + K_i \cdot \sum e_i$

比例项 $K_p \cdot e_t$ 处理即时偏差，积分项 $K_i \cdot \sum e_i$ 消除长期累积偏差——这与工业控制系统中的温度调节、电机转速控制是同一套机制。

实际效果：固定 top-p（$p=0.35$）的激活专家数标准差约为 4；DTop-p 的标准差降至约 1，且在约 18 个训练步内就收敛到目标值。

此外，DTop-p 还引入了**逐层路由归一化**，让每一层根据自己的路由分布特点学习合适的阈值——浅层（提取通用特征）倾向于激活更少专家（约 2-4 个），深层（复杂推理）倾向于激活更多（约 8-10 个）。这是 top-k 完全无法实现的层级差异化。

### 实验结果

在 1.3B 激活参数、6.9B 总参数的 64 专家模型上，训练 100B token 后：

| 方法 | 平均分（13 基准） | vs. Top-k |
| --- | --- | --- |
| Top-k（baseline） | 49.0 | — |
| 固定 Top-p | 49.3 | +0.3 |
| **DTop-p（ours）** | **50.9** | **+1.9** |

+1.9 分的提升在 NLP 基准上是显著的，而且在更多训练数据（300B token）下差距进一步拉大，跨模型规模（0.4B–2.4B dense）和跨专家粒度（4/32 到 16/128）均保持稳定。

DTop-p 指向了 MoE 路由机制的一个更本质的问题：**稀疏度本身应该是被学习的，而不是被固定的**。不同 token 的计算需求不同，强迫它们用相同数量的专家，既是一种浪费，也是一种约束。

这个思路目前还没有进入工业主流，原因并非方法有缺陷，而是时间不够——论文于 2025 年 12 月才挂出，距今不足半年。工业界从论文到实际用于大模型预训练，通常需要 1-2 年的验证和工程化周期；加之主要实验在 6.9B 总参数规模下完成，能否在 DeepSeek-V3 量级（671B、10T+ token）下保持稳定，还需要更大规模的复现。PI 控制器也引入了额外的超参数（$K_p$、$K_i$、目标激活数 $T$），对于本就调参复杂的 MoE 训练是额外的工程负担。大概率会在 2026-2027 年的新一轮训练周期里看到跟进验证。


---

## 现代 LLM 的 FFN 配置全景

综合所有论文，现代 LLM 在 FFN 设计上已经高度收敛：

**密集模型配置（2023–2025）：**

| 模型 | 激活函数 | 扩展比 | 矩阵数 |
| --- | --- | --- | --- |
| LLaMA 2（70B） | SwiGLU | 2.69× | 3 |
| LLaMA 3（405B） | SwiGLU | ~2.67× | 3 |
| Qwen 2.5（72B） | SwiGLU | 2.67× | 3 |
| Gemma 3（27B） | SwiGLU | 3–4× | 3 |

**稀疏 MoE 模型配置（2024）：**

| 模型 | 专家数 | 每次激活 | 总参数 / 激活参数 | 路由策略 |
| --- | --- | --- | --- | --- |
| Mixtral 8×7B | 8 | top-2 | 47B / 13B | 标准 Softmax |
| OLMoE | 64/层 | top-8 | 6.9B / 1.3B | Dropless token-choice |
| DeepSeekMoE-16B | 64+2 共享 | top-6+2s | 16B / 2.8B | 细粒度+共享 |
| DeepSeek-V3 | 256+1 共享 | top-8+1s | 671B / 37B | 无辅助损失均衡 |

---

## 回望这条路

```
标准 FFN + ReLU（Transformer 2017）
    ↓ 问题：硬性截断，负值信息丢失，dying ReLU
GELU（2016 / 2018 广泛采用）
    ↓ 改进：平滑激活，处处可微，梯度不消失
    ↓ 问题：仍是单路径，无自适应特征选择
GLU（Dauphin et al. 2017）→ SwiGLU（Shazeer 2020）
    ↓ 改进：双路径门控，从 2 矩阵变 3 矩阵，自适应特征过滤
    ↓ 成为工业标配：PaLM, Llama, Qwen, Gemma...
    ↓ 问题：参数越多 FLOP 越多，无法解耦
早期 MoE（Shazeer 2017）
    ↓ 改进：稀疏路由，解耦参数量与 FLOP
    ↓ 问题：训练不稳定，工程复杂；upcycling 短期有效但长期不如从头训练
Switch Transformer（2021）
    ↓ 改进：单专家路由简化，容量因子保证稳定，7× 步骤效率
    ↓ 问题：专家知识重叠，利用率不高
Mixtral 8×7B / OLMoE（2024）
    ↓ 验证：简单 top-2 路由在工业规模有效；1B 激活 MoE 超越 7B 密集
DeepSeekMoE（2024）
    ↓ 改进：细粒度专家 + 共享专家分离，16B/2.8B 激活匹配 7B 密集
DeepSeek-V3（2024）
    ↓ 改进：无辅助损失均衡，256 专家工业化，671B/37B 激活匹配 GPT-4o
MoE Lens（2026）
    ↓ 发现：路由高度集中，top-1 专家余弦相似度 0.95，专业化程度不足
DTop-p MoE（2025）
    ↓ 改进：PI 控制器动态调整稀疏度，稳定性↑，平均得分 +1.9
```

这条路线背后有一个贯穿始终的张力：**容量（参数量）和效率（FLOP）**。激活函数的演进试图从质量上提升每个参数的利用率，MoE 则从架构上打破了“参数量＝FLOP”这条旧约束。两个方向最终在现代 LLM 中合流：每个 MoE 专家内部都用 SwiGLU，而整体架构则用稀疏路由控制激活的专家数量。

### 激活函数现在够好了吗？

从实验数据看，**大体可以这么说：在现有 Transformer 架构下，SwiGLU 已经非常接近一个默认选项**。从 2022 年 PaLM 到 2025 年 Gemma 3，几乎所有主流 LLM 都选择了 SwiGLU 或其近亲（GeGLU），公开资料里也还没有出现足够强、足够稳定的新替代者。

但“够好”有前提：**激活函数的设计空间和架构绑定**。目前能预见的三个触发点：

**触发点一：非 Transformer 架构普及。** Mamba、RWKV 等线性序列模型的 FFN-like 模块结构不同，门控机制已经内嵌到序列混合层里，SwiGLU 的双路径设计未必是最优选择。

**触发点二：极端量化或硬件约束。** SwiGLU 的 Swish 函数（$x \cdot \sigma(x)$）需要算指数，在极端低比特量化（INT4 以下）或某些专用芯片上，计算代价可能不可忽视。更“硬件友好”的激活函数（比如 ReLU²）可能重新进入视野。

**触发点三：激活稀疏性成为核心目标。** MoE 解决的是“哪些专家被激活”的粗粒度稀疏问题，但激活函数本身也可以产生细粒度稀疏——ReLU 的输出天然有大量精确的零，适合跳过无效计算。SwiGLU 的输出几乎没有精确零值，在需要激活稀疏性的推理优化场景（如 PowerInfer、Deja Vu 等工作）里反而是劣势。

简单说：**如果你在训练一个标准 Transformer 大模型，通常没必要专门折腾激活函数，SwiGLU 仍然是默认优选。如果架构变了、硬件变了，或者稀疏推理成了首要目标，这个问题就值得重新打开。**

---

## 参考文献

- Vaswani, A., et al. (2017). *Attention Is All You Need*. NeurIPS 2017. [arXiv:1706.03762](https://arxiv.org/abs/1706.03762)
- Gehring, J., et al. (2017). *Convolutional Sequence to Sequence Learning*. ICML 2017. [arXiv:1705.03122](https://arxiv.org/abs/1705.03122)
- Dauphin, Y., et al. (2017). *Language Modeling with Gated Convolutional Networks*. ICML 2017. [arXiv:1612.08083](https://arxiv.org/abs/1612.08083)
- Hendrycks, D., & Gimpel, K. (2016). *Gaussian Error Linear Units (GELUs)*. [arXiv:1606.08415](https://arxiv.org/abs/1606.08415)
- Shazeer, N., et al. (2017). *Outrageously Large Neural Networks: The Sparsely-Gated Mixture-of-Experts Layer*. ICLR 2017. [arXiv:1701.06538](https://arxiv.org/abs/1701.06538)
- Shazeer, N. (2020). *GLU Variants Improve Transformer*. [arXiv:2002.05202](https://arxiv.org/abs/2002.05202)
- Geva, M., et al. (2021). *Transformer Feed-Forward Layers Are Key-Value Memories*. EMNLP 2021. [arXiv:2012.14913](https://arxiv.org/abs/2012.14913)
- Fedus, W., et al. (2021). *Switch Transformers: Scaling to Trillion Parameter Models with Simple and Efficient Sparsity*. JMLR 2022. [arXiv:2101.03961](https://arxiv.org/abs/2101.03961)
- Komatsuzaki, A., et al. (2022). *Sparse Upcycling: Training Mixture-of-Experts from Dense Checkpoints*. [arXiv:2212.05055](https://arxiv.org/abs/2212.05055)
- Chowdhery, A., et al. (2022). *PaLM: Scaling Language Modeling with Pathways*. [arXiv:2204.02311](https://arxiv.org/abs/2204.02311)
- Dai, D., et al. (2024). *DeepSeekMoE: Towards Ultimate Expert Specialization in Mixture-of-Experts Language Models*. [arXiv:2401.06066](https://arxiv.org/abs/2401.06066)
- Jiang, A. Q., et al. (2024). *Mixtral of Experts*. Mistral AI. [arXiv:2401.04088](https://arxiv.org/abs/2401.04088)
- Touvron, H., et al. (2023). *Llama 2: Open Foundation and Fine-Tuned Chat Models*. [arXiv:2307.09288](https://arxiv.org/abs/2307.09288)
- Meta AI. (2024). *The Llama 3 Herd of Models*. [arXiv:2407.21783](https://arxiv.org/abs/2407.21783)
- Muennighoff, N., et al. (2024). *OLMoE: Open Mixture-of-Experts Language Models*. [arXiv:2409.02060](https://arxiv.org/abs/2409.02060)
- Qwen Team. (2024). *Qwen2.5 Technical Report*. [arXiv:2412.15115](https://arxiv.org/abs/2412.15115)
- DeepSeek AI. (2024). *DeepSeek-V3 Technical Report*. [arXiv:2412.19437](https://arxiv.org/abs/2412.19437)
- Gemma Team. (2025). *Gemma 3 Technical Report*. [arXiv:2503.19786](https://arxiv.org/abs/2503.19786)
- Anonymous. (2026). *MoE Lens: An Expert Is All You Need*. [arXiv:2603.05806](https://arxiv.org/abs/2603.05806)
- He, Z., et al. (2025). *Sparsity-Controllable Dynamic Top-p MoE for Large Foundation Model Pre-training*. [arXiv:2512.13996](https://arxiv.org/abs/2512.13996)
