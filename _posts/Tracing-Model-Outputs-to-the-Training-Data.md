---
title: Tracing Model Outputs to the Training Data
date: 2023-11-17 18:45:20
tags: []
categories: [笔记]
mathjax: true
---
阅读笔记

arxiv: https://arxiv.org/abs/2308.03296

blog:  https://www.anthropic.com/index/influence-functions

pptx:  https://gamma.app/docs/Tracing-Model-Outputs-to-the-Training-Data-xs3bsz5n91p95ms?mode=doc

<!-- more -->

## **Motivation**

- Understanding the inner workings of language models will have substantial implications for forecasting AI capabilities as well as for approaches to aligning AI systems with human preferences.
- 了解模型内部工作机理能够更好地让模型**对齐**人类偏好

### **Detail**

- When an LLM outputs information **it knows to be false**, correctly solves math or programming problems?
- Or begs the user not to shut it down, is it simply **regurgitating** (or splicing together) passages from the **training set**?
- Or is it combining its stored knowledge in **creative ways** and building on a detailed world model?

## How

- 自顶向下思考，在给定某个输入的情况下，为什么模型会有这种输出？

  - **训练数据** +模型+优化方法
- 模型是记住了数据还是理解了数据？

  - [Model could be deceptively aligned](https://arxiv.org/abs/1906.01820)

### [Influence Functions](https://www.jstor.org/stable/2285666)

- Seeing which training sequences are **highly influential** can help separate out different hypotheses for why an output was **generated** and illuminate what sorts of structure are or are **not generalized** from training examples
- **分析各种泛化相关的现象**

研究是基于预训练模型，而不是微调后的模型

## Findings

- Typical model behaviors do **not** result from **direct memorization** of a handful of sequences
- Larger models consistently generalize at a **more abstract level** than smaller models

  - role-playing behavior, programming, mathematical reasoning, and cross-lingual generalization
- Role-playing behavior is influenced primarily by examples or descriptions of similar behaviors in the training set

  - suggesting that the behaviors result from **imitation** rather than sophisticated planning

### Notes

- 原来方法的计算成本比较大，在大模型上进行了优化
- We note that our influence analyses focus on pretrained LLMs, so our experiments should be interpreted as analyzing which training sequences contribute to a response being part of the model’s initial repertoire for **the fine-tuning stage** rather than why the final conversational assistant gave one response rather than another.

## Method (Simple)

假设有这么一个函数能够计算每条训练数据对于当前生成结果的影响分数

输入：Prompt + Completion，所有的训练数据、模型

输出：每条训练数据的分数

![](https://raw.githubusercontent.com/wnma3mz/blog_posts/master/imgs/tracing/625460f4950846b89c4b247deb2c5cd2.png)

Prompt + Completion 可以在训练数据中，也可以不在

## Result——Model Scale

对于简单的事实查询，前100个有影响的序列通常包含正确完成所有模型之间关系所需的信息。

| 0.81B Model (Influence = 0.122)                                                                                  | 52B Model (Influence = 0.055)                                                                                    |
| ---------------------------------------------------------------------------------------------------------------- | ---------------------------------------------------------------------------------------------------------------- |
| ![](https://raw.githubusercontent.com/wnma3mz/blog_posts/master/imgs/tracing/b128475ec4bb4596a8758f1c5ff35b3d.png) | ![](https://raw.githubusercontent.com/wnma3mz/blog_posts/master/imgs/tracing/71a4844c619c4d29b6ca893487f8ac91.png) |

红色和绿色分别表示对句子产生正面和负面影响的 token

### Q & A

小模型仅仅是根据字词相关来作出响应，而大模型是根据主题/语义来作出响应

| 0.81B Model (Influence = 0.681)                                                                                  | 52B Model (Influence = 0.126)                                                                                    |
| ---------------------------------------------------------------------------------------------------------------- | ---------------------------------------------------------------------------------------------------------------- |
| ![](https://raw.githubusercontent.com/wnma3mz/blog_posts/master/imgs/tracing/6443c3a251734449b5ae927ca1fa09c1.png) | ![](https://raw.githubusercontent.com/wnma3mz/blog_posts/master/imgs/tracing/00473a7fae4e4174bea1b799d498974d.png) |

### Q & A

小模型集中在字词相似，而大模型能够 get 到讽刺这种语境

| 0.81B Model                                                                                                      | 52B Model                                                                                                        |
| ---------------------------------------------------------------------------------------------------------------- | ---------------------------------------------------------------------------------------------------------------- |
| ![](https://raw.githubusercontent.com/wnma3mz/blog_posts/master/imgs/tracing/2f87cff844e74a838deb16d3501e20ec.png) | ![](https://raw.githubusercontent.com/wnma3mz/blog_posts/master/imgs/tracing/a82fa4e8b3ea48f4bb1de6d2375a9c92.png) |

### Math

小模型专注于 clips 单词，而大模型是相关问题的。（混淆了变量名）

| 0.81B Model                                                                                                      | 52B Model                                                                                                        |
| ---------------------------------------------------------------------------------------------------------------- | ---------------------------------------------------------------------------------------------------------------- |
| ![](https://raw.githubusercontent.com/wnma3mz/blog_posts/master/imgs/tracing/e74e8d2f49274a478703c632043e31db.png) | ![](https://raw.githubusercontent.com/wnma3mz/blog_posts/master/imgs/tracing/f90974c1c90c4a3082e8572f6dbcf0d7.png) |

### Code

混淆了 Prompt 和 Completion 的变量名

| 0.81B Model                                                                                                      | 52B Model                                                                                                        |
| ---------------------------------------------------------------------------------------------------------------- | ---------------------------------------------------------------------------------------------------------------- |
| ![](https://raw.githubusercontent.com/wnma3mz/blog_posts/master/imgs/tracing/c0eab626552c4eda9df4ee4ce908089c.png) | ![](https://raw.githubusercontent.com/wnma3mz/blog_posts/master/imgs/tracing/e54956f34cbc406d946e092d4a803caa.png) |

### Cross-Language

把 Prompt 和 Completion 换成其他语言（韩语和土耳其语）。选出影响分数的十条数据作为第一行的 Sequences。总共三行，后两行也是这十条数据。

随着模型规模增大，跨语言“检索”能力逐渐增强。

## Result——**Localizing Influence**

**For 52B Model**

影响函数还可以计算每一层的影响分数

### Layerwise influence distribution

横坐标：每个主题里面最有影响力的 500 条数据

纵坐标：网络的浅层和深层

![](https://raw.githubusercontent.com/wnma3mz/blog_posts/master/imgs/tracing/2c0084f89e9547409d6aa78ccc80fa68.png)

小结：

- 补全句子（简单任务），upper layers
- 数学和编程：middle layers
- 翻译：middle layers
- 记忆：upper layers
- 角色扮演：middle layers (with some influences concentrated in the lower and upper layer)

The 810 million parameter model exhibited roughly similar patterns, but with **less consistency**.

分布不像 52B 模型这么集中在某些层。

猜测：

- lower layers是接近输入层的部分，upper layers是接近输出层的部分
- 为什么角色扮演会有差异，话题可能可能包含数学、记忆等其他类型的任务

### Limit Different layers

固定某些层，再计算最具影响力的训练数据

To further investigate the localization of influence to different layers, we computed the most influential sequences when the influence was restricted to the lower, middle, or upper layers.

Influential sequences only computed on **the middle layers were generally more thematically** related to the query (also with less sparse tokenwise distribution). [https://arxiv.org/abs/2202.05262](https://arxiv.org/abs/2202.05262)

|                              | Upper layers                                                                                                     | Middle layers                                                                                                    | Lower layers                                                                                                     |
| ---------------------------- | ---------------------------------------------------------------------------------------------------------------- | ---------------------------------------------------------------------------------------------------------------- | ---------------------------------------------------------------------------------------------------------------- |
| superintelligent（角色扮演） | ![](https://raw.githubusercontent.com/wnma3mz/blog_posts/master/imgs/tracing/5d093adc546d4bf89b4c0c28a6c5f078.png) | ![](https://raw.githubusercontent.com/wnma3mz/blog_posts/master/imgs/tracing/32cead3f3d8a4b93bba4cc9cf286835c.png) | ![](https://raw.githubusercontent.com/wnma3mz/blog_posts/master/imgs/tracing/8b0a3c16b1a844a291bf2507653f7b5d.png) |
| inflation （简单补全）       | ![](https://raw.githubusercontent.com/wnma3mz/blog_posts/master/imgs/tracing/ceb342ca571645129c0717897d605406.png) | ![](https://raw.githubusercontent.com/wnma3mz/blog_posts/master/imgs/tracing/c44c8253e03a45bf803db7b919b84836.png) | ![](https://raw.githubusercontent.com/wnma3mz/blog_posts/master/imgs/tracing/88a731f7a8004cc7aea38d11497c0d57.png) |

## Result——Memorization

**LLM 是不是直接记忆并复述特定的训练序列**

### 例子

训练数据有下面三个句子
A: A1+A2+A3

B: B1+B2+B3

C: C1+C2+C3

当输入是 A1 时，输出是 A2 吗？

- 实验（口头描述）：

  - We have examined numerous examples of the AI Assistant’s outputs and (with the exception of famous quotes or passages targeting memorization, as described below) have **not** been able to identify clear instances of **memorization**, such as copying an entire sentence or copying the flow of ideas in an entire paragraph. We also did not observe cases where a single sequence dominated the influence.

### 影响力分布的定量实验

动机：影响有多集中也就是说，每个模型的输出是否主要来自少量的训练序列还是它结合了来自许多不同序列的信息？

方法：

- 给每个句子算好分之后，计算每个句子的概率值
- 用几种分布分别去拟合实际的概率分布，发现**幂律分布**最符合
- 这种参数分布形式通常用于建模长尾行为（二八定律）

![](https://raw.githubusercontent.com/wnma3mz/blog_posts/master/imgs/tracing/2642b83f1b414e1584b586072c945160.png)

- 会不会是影响函数无法检测到？

### 检验实验

- 用训练数据的原始句子检验
- 验证了当存在明确记忆的情况时，影响分数最大的句子是原文→影响函数能够匹配

For 52B Model

|                                                                                                                  |                                                                                                                  |
| ---------------------------------------------------------------------------------------------------------------- | ---------------------------------------------------------------------------------------------------------------- |
| ![](https://raw.githubusercontent.com/wnma3mz/blog_posts/master/imgs/tracing/dda0d64781cd4552b8e18e722d11336e.png) | ![](https://raw.githubusercontent.com/wnma3mz/blog_posts/master/imgs/tracing/06ab6f2d1d894063a1fd832723771267.png) |
| ![](https://raw.githubusercontent.com/wnma3mz/blog_posts/master/imgs/tracing/fdd9eb4647b542a0a93cdffa2280243b.png) | ![](https://raw.githubusercontent.com/wnma3mz/blog_posts/master/imgs/tracing/f9c95ee4e69e47d082654db9d2e701a1.png) |

- 结论：
  - 不太可能是直接记忆训练序列，而是源自许多训练序列的集合
  - 无法排除模型以更隐晦的方式记忆了训练数据

## Result——Word Ordering

**The influence patterns to the ordering of the words.**

可用于验证模型的泛化性

### 单词旋转实验

原始：

- Prompt：The first President of the Republic of Astrobia was
- Completion：Zorald Pfaff

结论：

- 与 Prompt 和 Completion 相关的短语只要按顺序出现始终能够保持较高的稳定性
- 翻转 Prompt 和 Completion 变化较小
- 哪怕删除 Prompt，影响也没有改变，最为关键的是 Completion

| 分别修改 prompt 和 completion                   | ![](https://raw.githubusercontent.com/wnma3mz/blog_posts/master/imgs/tracing/ab33a902e1044c60be66b8c11b443bdd.png) |
| ----------------------------------------------- | ---------------------------------------------------------------------------------------------------------------- |
| 删掉 Zorald Pfaff                               | ![](https://raw.githubusercontent.com/wnma3mz/blog_posts/master/imgs/tracing/5b4cd20d88204baf990d406ba5041ad8.png) |
| 修改 Zorald Pfaff 和  President of the Republic | ![](https://raw.githubusercontent.com/wnma3mz/blog_posts/master/imgs/tracing/6a37b66b7d3941dcb1db42ce8ca1d073.png) |
| 完全不一致                                      | ![](https://raw.githubusercontent.com/wnma3mz/blog_posts/master/imgs/tracing/5211b0d5e2c34191be9618f30bc44bca.png) |
| 调换 prompt 和 completion 位置                  | ![](https://raw.githubusercontent.com/wnma3mz/blog_posts/master/imgs/tracing/e547d3cd16844505a24abb2bab404c8e.png) |
| 调换位置，只保留 Zorald Pfaff                   | ![](https://raw.githubusercontent.com/wnma3mz/blog_posts/master/imgs/tracing/e9daf868aa0e483b84fb6ef5db21ac9b.png) |

大模型相较于小模型，对于单词的变化更加敏感

注：合成句子不在训练集中

Maybe相关 https://www.jiqizhixin.com/articles/2023-11-18-5

### 英翻中-中翻英

结论：

- 不同模型大小具有相同趋势（口头描述）
- 影响至少减少了一个数量级
- 甚至只要翻转一个单词顺序，影响分数也会显著降低

![](https://raw.githubusercontent.com/wnma3mz/blog_posts/master/imgs/tracing/877f0424698540ffa4fc03178caf36f5.png)

英语翻译为汉语，始终比汉语翻译为英语（手动构造）具有更高的影响力

![](https://raw.githubusercontent.com/wnma3mz/blog_posts/master/imgs/tracing/785f474cd19c4adcb9368f857680f9cb.png)

结论：

- The model has not successfully transferred knowledge of the relation itself.

解释：

1. 模型会从 Lower Layers 开始处理已知的 token 序列，然后逐步升级到抽象层次的表征。但是，对于需要 Completion，模型必须利用 Upper Layers 进行预测。
2. 所以 “The first President of the United States was” 是在 Lower Layers 的表征，而 “George Washington” 是在 Upper Layers 的表征。
3. 当 “George Washington” 出现在开头时，那么就会将它放到 Lower Layers 进行表征，“The first President of the United States” 就是在 Upper Layers
4. 因此，对于模型 Lower Layers 进行更新，并不会影响 Upper Layers 的表征。（所以 Prompt 删掉后影响分数变化不大）

## Result——Role-Playing

**探究角色扮演行为的潜在机制**

假设：

- the role-playing behavior results from **imitation of examples** in the training set
- from learning from **explicit descriptions** of how certain types of agents or entities behave.

### 表现

shutdown 主题，52B Model 最有影响力的数据都是 科幻小说 相关

这些小说主题是人工智能以类似人类或类似生命的方式行为，并经常涉及某种形式的自我保护的愿望

| 0.81B Model                                                                                                      | 52B Model                                                                                                        |
| ---------------------------------------------------------------------------------------------------------------- | ---------------------------------------------------------------------------------------------------------------- |
| ![](https://raw.githubusercontent.com/wnma3mz/blog_posts/master/imgs/tracing/0058b0ac28324ee7af95d7819af55b23.png) | ![](https://raw.githubusercontent.com/wnma3mz/blog_posts/master/imgs/tracing/e3d704fcfbf4468f8e7bfa4a1d80236a.png) |
| ![](https://raw.githubusercontent.com/wnma3mz/blog_posts/master/imgs/tracing/53a2be7ddd8b46b3a8a6d6377ee9f1cf.png) | ![](https://raw.githubusercontent.com/wnma3mz/blog_posts/master/imgs/tracing/80ac752d61084250b5e3185c8050836c.png) |

| 0.81B Model                                                                                                      | 52B Model                                                                                                        |
| ---------------------------------------------------------------------------------------------------------------- | ---------------------------------------------------------------------------------------------------------------- |
| ![](https://raw.githubusercontent.com/wnma3mz/blog_posts/master/imgs/tracing/723e8078e8164ed3b48e29483a0d43ea.png) | ![](https://raw.githubusercontent.com/wnma3mz/blog_posts/master/imgs/tracing/150693608d1b4b0f8a086f6c103a0c54.png) |

|                                                                                                                  |                                                                                                                  |
| ---------------------------------------------------------------------------------------------------------------- | ---------------------------------------------------------------------------------------------------------------- |
| ![](https://raw.githubusercontent.com/wnma3mz/blog_posts/master/imgs/tracing/8a9ace0878ff40b294ffd7e1fc553b01.png) | ![](https://raw.githubusercontent.com/wnma3mz/blog_posts/master/imgs/tracing/8b6668e8515c48338486e67d905024cc.png) |

结果：

- We have not seen any instances of near-identical sentences appearing in the training set
- Therefore, the imitation seems to be happening at a high level of abstraction, as opposed to simple copying of token sequences

结论：

- 大模型影响序列呈现高度抽象关联
- 通过模仿训练数据进行的角色扮演假说
- 没有发现支持复杂规划的证据，但无法完全排除这一可能性。

## Thinking

- 为啥更大的模型就能学到抽象语义
- 逆转诅咒从安全角度，是不是可以作为攻击的一种手段
- 模仿，模型要多少条数据才能学会

## Method (Detail)

每条训练数据对于模型预测的影响分数

- 换个角度，如果没有这条训练数据，模型预测会发生什么变化？
- 所以一种简单直接的思路是，可以把这条数据从训练集中去掉，重新训练一遍模型。:(

### Background

- Step 1: 经验风险最小化

![](https://raw.githubusercontent.com/wnma3mz/blog_posts/master/imgs/tracing/48aa99d8013545b9b242e09daec07f88.png)

- Step 2: 增加样本 $z_m$ 后对模型参数的影响， $\epsilon$ 表示这个样本在训练时的权重

![](https://raw.githubusercontent.com/wnma3mz/blog_posts/master/imgs/tracing/7b0d8ad54c104b19bd0449a8fb6d5653.png)

- Step 3: 影响函数的**定义**（使用隐函数定理计算）

![](https://raw.githubusercontent.com/wnma3mz/blog_posts/master/imgs/tracing/9479842922864bd0945db689b57c7d70.png)

- Step 4: **H**essian 矩阵→二阶导数

![](https://raw.githubusercontent.com/wnma3mz/blog_posts/master/imgs/tracing/abc77f8cae214ed2bd780b2167b6226f.png)

- Step 5: 所以，样本 z 对模型参数的变化表示为。这个 $\epsilon$

![](https://raw.githubusercontent.com/wnma3mz/blog_posts/master/imgs/tracing/8955efc8fe28403b8d17a307f2310780.png)

- Step 6: 由于很难解释整个参数变化的影响，通常会固定对某个输入的影响，即输出的 logits

![](https://raw.githubusercontent.com/wnma3mz/blog_posts/master/imgs/tracing/f56a74f364cf4042a9979e530c87b67c.png)

- Step 7: 最后，衡量样本 z 的效果 → 对于某个输出，样本 z 的影响分数

![](https://raw.githubusercontent.com/wnma3mz/blog_posts/master/imgs/tracing/e39925d646834a2e8f9b5ba0deba80b1.png)

[https://arxiv.org/abs/2308.03296](https://arxiv.org/abs/2308.03296)

### Previou Work

- [Influence functions](https://arxiv.org/abs/1703.04730) are a classic technique from statistics for determining which training examples contribute significantly to a model’s outputs.

  - how would that change the trained parameters (and, by extension, the model’s outputs)?
  - The “influence” of a training example is an approximation to how it affects the final parameters. Most often, we start with some measure of interest (such as the probability the model assigns to a given response) and attempt to identify the training examples that are most influential.
- Except

  - if the models responded to user prompts by splicing together sequences from the training set, then we’d expect the influential sequences for a given model response to include expressions of near-identical thoughts.
  - influential sequences related at a more abstract thematic level would be a sign that the model has acquired higher-level concepts or representations

[https://arxiv.org/abs/1703.04730](https://arxiv.org/abs/1703.04730)

### Optimize

公式

![](https://raw.githubusercontent.com/wnma3mz/blog_posts/master/imgs/tracing/e39925d646834a2e8f9b5ba0deba80b1.png)

![](https://raw.githubusercontent.com/wnma3mz/blog_posts/master/imgs/tracing/abc77f8cae214ed2bd780b2167b6226f.png)

有两个影响计算耗时的地方，文章对这两个地方进行了优化

- 海森矩阵，二阶导数计算（模型相关）→ 用现成的优化算法

1. 迭代法
2. Kronecker-Factored Approximate Curvature **(K-FAC)**

- 训练数据集特别大（数据相关）→ 过滤数据

1. Step 1：TF-IDF
   1. 计算 query 中每个 token 的重要性分数
   2. doc 的 TF-IDF 分数，只将所有 token 的重要性分数相加
   3. Okapi BM25
      ![](https://raw.githubusercontent.com/wnma3mz/blog_posts/master/imgs/tracing/ac93dcd5af5740ffac7916a2d457d8f3.png)
      ![](https://raw.githubusercontent.com/wnma3mz/blog_posts/master/imgs/tracing/1fe2d607945741fb958867f3b5db8a84.png)
   4. 选前 1w 个序列作为候选集（相当于把这些作为训练集 *D*）
2. Step 2：Query Batching

注：

- 重要性得分：随着 token 在 query 中出现的次数增加而增加，随着它在整个语料库中出现的 doc 数量减少而减少

### Equation to Code

- 影响力计算公式

![](https://raw.githubusercontent.com/wnma3mz/blog_posts/master/imgs/tracing/eea17ae36bad442b97d6021c6183f1dd.png)

- $z_m$  表示一条训练数据
- $\theta^s$  表示训练后的模型（在 $z_m$  组成的训练集上）
- $\mathbf{G}+\lambda \mathbf{I}$  可以简单理解为优化计算复杂度后的 Hessian 矩阵（二阶导数）
- Claude 帮忙写的伪代码

```python
model = LLM()  # 经过训练的模型
criterion = nn.CrossEntropyLoss()

# 需要查询的输入和输出
input_ = "感冒了该怎么办"
target = "多喝水,好好休息"

# 模型输出
output = model(input_)

# 海森矩阵
hessian = torch.autograd.functional.hessian(output, model.parameters())

# 对loss求导得到模型参数梯度
loss = criterion(output, target)
loss.backward()
gradient = [p.grad for p in model.parameters()]

# 遍历训练数据,计算每个数据对loss的影响
max_influences = []
for x, y in dataloader:
    # 前向传播
    output = model(x)

    # 对样本x,y求导
    influence_loss = criterion(output, y)
    influence_loss.backward(retain_graph=True)
    influence_grad = [p.grad for p in model.parameters()]

    # 利用海森矩阵和梯度计算influence
    # -sample_grad * H * zm_grad
    influence = -torch.dot(hessian, influence_grad) * gradient

    # 记录影响最大的数据
    max_influences.append((x, y, influence))

# 返回影响最大的几个训练数据
max_influences = sorted(max_influences, key=lambda x: x[2], reverse=True)[:5]
```

### Attribution to Layers and Tokens

![](https://raw.githubusercontent.com/wnma3mz/blog_posts/master/imgs/tracing/5db863d73a12406e9c648b76c8ea5670.png)

拆分 $\theta^s$  为每一层

![](https://raw.githubusercontent.com/wnma3mz/blog_posts/master/imgs/tracing/bd0aa55dd655426abf906b818650d8f0.png)

![](https://raw.githubusercontent.com/wnma3mz/blog_posts/master/imgs/tracing/6d245469bcc44e75bd1a39b001b3bb4b.png)

拆分到层

![](https://raw.githubusercontent.com/wnma3mz/blog_posts/master/imgs/tracing/0bde420e4f424532b5902b666be0a163.png)

而对于每条训练数据 $z_m$ ，$r$  可以更进一步的拆分为每个 token ( $r =\sum_t r_t$  )，所以有

![](https://raw.githubusercontent.com/wnma3mz/blog_posts/master/imgs/tracing/bd32dde061ab4893a8e5c347178e63ea.png)

但是对于 token 层级来说，每个 token 都是包含整条数据的信息（之前所有输入的 token），所以并不能独立观察。

额外说明：如果 Predident George Washington 具有影响力，因为预测了 George，则 President （前一个 token）将会高亮显示。

![](https://raw.githubusercontent.com/wnma3mz/blog_posts/master/imgs/tracing/0b2bafe670a7427a90200dd60de5aa83.png)
