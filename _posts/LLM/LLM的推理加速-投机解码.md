---
title: LLM的推理加速-投机解码
date: 2025-08-30 15:04:12
tags: [NLP, Attention]
categories: [Note]
mathjax: true
---

介绍投机解码

[Fast Inference from Transformers via Speculative Decoding](https://arxiv.org/abs/2211.17192)

<!-- more -->


## 投机解码

### 动机

### 基本设定

在解题场景下，输入输出如下所示（不讨论为什么要用这个输出格式）

模型的尺寸是跟能力成正比的，所以在解题场景下，通常使用 72B 模型

|输入|输出|
|--|--|
|你是一位老师，给你一个题目，请按下面要求解答题目。<br>要求：<br>1. 复述一遍题目<br>2. 给出答案的解题思路分析，最后给出题目的最终答案；<br>3. 如果题目题干不完整，存在信息丢失、错乱等情况，请说明题目不能解答，并给出具体的原因。<br>(2分)女性口腔上皮细胞中染色体组成是____，有____个细胞分子。<br>【题干】女性口腔上皮细胞中染色体组成是____，有____个细胞分子。<br>【解析】xxx<br>【答案】xxx|【题干】女性口腔上皮细胞中染色体组成是____，有____个细胞分子。<br>【解析】xxx<br>【答案】xxx|

将模型的输出简单分为三个部分【题干】、【解析】、【答案】

### 加速动机

输出中每个部分的难易程度是不一样的

【题干】要求重新输出一遍输入的部分内容

【解析】需要更强的逻辑推理能力

所以，【题干】对于 72B 模型是大材小用的。甚至【解析】中也不是所有内容都需要 72B。

### 怎么加速

“容易”的 token （【题干】部分）可以由小尺寸模型代替大尺寸模型生成。

小尺寸模型称为 Draft Model，用于更快速的生成 token。

大尺寸模型称为 Target Model（实际生成的模型），对 Draft Model 生成的 token 进行校验以及纠错

### 细化例子

假设模型需要输出一次计算过程，「1×2+3×4=14」。

7B 模型可能会输出为「1×2+3×4=111」，而错误的 token 只是最后的「111」

72B 模型只需要纠正「111」这个 token，而前面的生成过程可以被小模型代替

## 过程

### 图示

![](https://raw.githubusercontent.com/wnma3mz/blog_posts/master/imgs/LLM的推理加速-EAGLE三部曲/image1.png)


- 每一行表示一次生成
- 绿色表示 Draft Model（小模型）生成的 token 且 Target Model（大模型）接受的 token
- 红色表示 Target Model 拒绝的部分
- 蓝色表示 Target Model 生成的部分



### 耗时统计

在第一行中，实际生成了 5 个 Token。

| 标准的生成过程                      | 投机解码                                                     |
| ----------------------------------- | ------------------------------------------------------------ |
| Target Model 生成 5 个 token 的时间 | Draft Model 生成 5 个 token 的时间 （含拒绝 1 次）<br> Target Model 校验的时间（相对快）<br> Target Model 生成 1 个 token 的时间 |



### 投机解码的加速因素

- Draft Model 生成的速度（模型参数量）
- Draft Model 生成 token 的**平均接受长度**



#### 平均接受长度

- 接受长度：Draft Model 每次推理 **10** （设定的超参数）个 token，Target Model 接受了 7 个 token，那么接受长度为 7。
- 平均接受长度：在推理单个句子的时候，Draft Model 会进行 N 次推理。对每次推理的接受长度取平均。



### 校验

假设 Draft Model 生成「小明」、「跑」、「步」、「很快」

Target Model 会逐个 token 校验，以「跑」为例

- **Draft Model** 在生成「跑」这个 token 时，同时保存生成「跑」这个 token 的概率值 **q**
- **Target Model** forward「跑」以及之前所有 token（「小明」）。模型生成 logits，[seq_len x vocab_size]
  - 取最后一个 token，得到 [1 x vocab_size] 大小的 tensor
- 通过 softmax + 采样方法 得到每个 token 的概率，找到「跑」这个 token 的概率 **p**
  - 采样方法，如 greedy、top_p、top_k
- q <= p（Draft Model 的“信心”小于 Target Model），接受；q > p，拒绝。 



## 理论证明

投机采样是从 p(x) 和 q(x) 联合采样的结果，需要证明最终生成的 token 分布与直接采样 p(x) 分布是完全相同的。

- **p(x)**：这是**目标分布**，也就是我们最终想要采样的真实分布。在大型语言模型（LLM）中，它通常指一个大型、复杂的模型（如GPT-4）的输出分布。
- **q(x)**：这是**提议分布**，也就是用于“推测”的辅助分布。它通常是一个小型、快速的模型，用于快速生成多个候选令牌。



### 全概率公式

在投机采样中，生成的 token 无非两种情况猜对和猜错两种，写作。


$$
P(x=x^{′})=P(\text{guess accepted},x=x^{′})+P(\text{guess rejected},x=x^{′})
$$
把事件“最终采样结果是 x′”分解成了两种互斥的情况：

- **P(guess accepted,x=x′)**：表示“猜对了，且猜对的 token 是 x′”的概率。
- **P(guess rejected,x=x′)**：表示“猜错了，且从目标分布重新采样得到的 token 是 x′”的概率。

这个公式符合概率论中的**全概率公式**，因为任何一个最终的 token x′ 都必然是通过这两种方式之一产生的。

所以，需要证明投机采样的 token 与目标分布是一致的，即 $P(x=x^{′}) = p(x^{′})$

### 接受的概率

$$
P(\text{guess accepted},x=x^{′})=P(x=x^{′} \quad \text{and} \quad \text{guess accepted})
$$

接受的概率是视作是一个联合概率，

- 事件 A：草稿模型 $q(x)$ 提议生成 token x'，事件概率是 $P(A)=q(x')$
- 事件 B：在提议了 x′ 的条件下，这个提议被接受了。这个事件的概率设置为 $\beta$。

根据联合概率公式 $P(A \cap B) = P(A) \cdot P(B|A)$，我们得到：
$$
P(\text{guess accepted},x=x^{′})=q(x')\cdot \beta
$$


这里的 $\beta$ 定义为 $\min\left(1, \frac{p(x')}{q(x')}\right)$

- $p(x') \ge q(x')$：目标模型认为 $x'$ 的概率比草稿模型认为的要高，比率 $\frac{p(x')}{q(x')}$ 将大于等于1。为了保持概率在 $[0, 1]$ 之间，我们取最小值，得到 $\beta = 1$。这表示只要草稿模型猜对了，且目标模型也高度认可这个猜想，我们就**百分之百地接受**这个令牌。
- $p(x') < q(x')$：草稿模型对这个 token 的“猜测”比目标模型更自信。为了纠正这个偏差，我们必须以一个小于 1 的概率来**接受**它。这个概率就是$\frac{p(x')}{q(x')}$。通过这种方式，我们“拒绝”了草稿模型提出的那些过高的概率值，并将其调整回与目标模型一致的水平。

所以，就有

$$
P(\text{guess accepted},x=x^{′})=q(x')\cdot \min\left(1, \frac{p(x')}{q(x')}\right)
$$



### 拒绝的概率

如果拒绝了，那就是需要从目标分布 $p(x^{'})$ 进行采样。其也是一个联合概率，需要在拒绝的情况下 $1-\beta$ 下，进行计算，得到
$$
P(\text{guess rejected},x=x^{′})=(1-\beta)p(x^{'})
$$



### 合并

$$
P(x=x^{′})=P(\text{guess accepted},x=x^{′})+P(\text{guess rejected},x=x^{′})=\beta\cdot p(x^{'}) + (1-\beta)p(x^{'})=p(x^{'})
$$




​		

