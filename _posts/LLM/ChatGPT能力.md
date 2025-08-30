---
title: LLM 复杂推理的来源分析以及 LIMA 阅读笔记
date: 2025-08-20 15:52:42
tags: [NLP, Attention]
categories: [Note]
mathjax: true
---

LLM 复杂推理的来源分析：fuyao 博客 + 综述

[拆解追溯 GPT-3.5 各项能力的起源 | Notion](https://yaofu.notion.site/GPT-3-5-360081d91ec245f29029d37b54573756)

[A Survey on In-context Learning](https://arxiv.org/abs/2301.00234v4) 



LIMA 阅读笔记

[LIMA: Less Is More for Alignment](https://arxiv.org/abs/2305.11206)

<!-- more -->

# 拆解 ChatGPT

22 年 12 月：[拆解追溯 GPT-3.5 各项能力的起源 | Notion](https://yaofu.notion.site/GPT-3-5-360081d91ec245f29029d37b54573756)



2022 年 11 月 30 日，ChatGPT 横空出世，它又强又聪明，且跟它说话很好玩，还会写代码。ChatGPT 是怎么变得这么强的？它的各种强大的能力到底从何而来？



## GPT-3 的能力

| 能力           | 说明                                                         | 来源                                                         |
| -------------- | ------------------------------------------------------------ | ------------------------------------------------------------ |
| **语言生成**   | 遵循提示词（prompt），然后生成补全提示词的句子 (completion)。这也是今天人类与语言模型最普遍的交互方式。 | 语言建模的训练目标                                           |
| **世界知识**   | 事实性知识 (factual knowledge) 和常识 (commonsense)。        | 3000 亿单词的训练语料库模型的 1750 亿参数是为了存储知识      |
| **上下文学习** | 遵循给定任务的几个示例，然后为新的测试用例生成解决方案。很重要的一点是，GPT-3虽然是个语言模型，但它的论文几乎没有谈到“语言建模” (language modeling) —— 作者将他们全部的写作精力都投入到了对上下文学习的愿景上，这才是 GPT-3的真正重点。 | 可能来自于同一个任务的数据点在训练时按顺序排列在同一个 batch 中 |



前两项能力的来源是相对明确的。所以，想要讨论 **上下文学习** （ICL） 是怎么来的？

ICL 在这里指的是：使用思维链做**复杂推理能力 / CoT**



## 复杂推理能力的来源代码数据？

![](https://raw.githubusercontent.com/wnma3mz/blog_posts/master/imgs/ChatGPT能力/image1.png)



### Q & A

Q：怎么理解 Instruction tuning？

A：Instruction tuning 有两个版本，一个是 supervised tuning，另一个是 reinforcement learning from human feedback (RLHF). ChatGPT 就是通过 RLHF 得来的



Q：supervised instruction tuning vs. supervised fine tuning（SFT）

A：

- supervised instruction tuning 使用**自然语言**作为任务描述，而supervised fine tuning 使用固定格式或标签作为任务描述。
- supervised fine tuning 的数据集包含**指令**、输入和输出三个部分，而supervised fine tuning 的数据集只包含输入和输出两个部分。
- 我们现在口头表述的 SFT，实际上对应这里的 supervised instruction tuning


![](https://raw.githubusercontent.com/wnma3mz/blog_posts/master/imgs/ChatGPT能力/image2.png)

从上往下看，关注 Instruct GPT Initial，发现 davinci-instruct-beta 和 text-davinci-001 [效果差](https://arxiv.org/pdf/2201.11903v1.pdf) 。 → 复杂推理能力不来源这里

![](https://raw.githubusercontent.com/wnma3mz/blog_posts/master/imgs/ChatGPT能力/image3.png)

而在 code-davinci-002 和 text-davinci-002 中发现了新的能力，这些能力在 GPT-3 Initial 中没有。

它们的区别在于**指令微调（Instruction Tuning）**和**代码训练（Code Training）。**



将能力进行归类

| **指令微调** | **代码训练** |
| ------------ | ------------ |
| 响应人类指令 | 复杂推理能力 |
| 泛化能力     | 长距离依赖   |



进一步解释

| **能力**                 | **说明**                                                     |
| ------------------------ | ------------------------------------------------------------ |
| 响应人类指令             | -                                                            |
| **指令微调 →** 泛化能力  | 指令数量超过一定程度之后**自动出现的**Sanh. et. al. Oct 2021. Multitask Prompted Training Enables Zero-Shot Task GeneralizationWei et. al. Sep 2021. Finetuned Language Models Are Zero-Shot LearnersChung et. al. Oct 2022. Scaling Instruction-Finetuned Language Models |
| **代码训练 →**长距离依赖 | 猜测[语言中的下个词语预测通常是非常局部的，而代码通常需要更长的依赖关系来做一些事情，比如前后括号的匹配或引用远处的函数定义](https://twitter.com/peterjliu/status/1603098202856722432?s=46&t=Gw2wumo4l8kIRaGzXDXyRw)由于面向对象编程中的类继承，代码也可能有助于模型建立编码层次结构的能力 |


### 代码训练 → 复杂推理能力

**注意：相关性 Not 因果性**

发现下面框出来的模型**不能做思维链**，共性是都没有经过代码数据训练

- 最初的 GPT-3 没有接受过代码训练
- text-davinci-001 模型，经过了指令微调，但它的思维链推理的能力非常弱

![](https://raw.githubusercontent.com/wnma3mz/blog_posts/master/imgs/ChatGPT能力/image4.png)



**能做思维链**的共性：都经过了代码数据训练

- [针对代码训练的模型具有很强的语言推理能力](https://arxiv.org/abs/2211.09110)
- code-davinci-002 以及后续变种可以做思维链，并且[效果很强](https://openreview.net/forum?id=yf1icZHC-l9)



弱解释

- [PaLM ](https://arxiv.org/abs/2204.02311)有 5% 的代码训练数据，可以做思维链
- 直觉来说，**面向过程的编程 (procedure-oriented programming)** 跟人类**逐步解决任务**的过程很类似，**面向对象编程 (object-oriented programming)** 跟人类**将复杂任务分解为多个简单任务**的过程很类似



### 悬而未决

**区分代码训练和指令微调效果的最好方法**可能是**比较 code-cushman-001、T5 和 FlanT5**

- 因为它们具有相似的模型大小（110亿 和 120亿），相似的训练数据集 (C4)，它们最大的区别就是有没有在代码上训练过 / 有没有做过指令微调。
- 目前还没有这样的比较。我们把这个留给未来的研究。



## 复杂推理能力是激发还是注入？

能力是否已经存在于 GPT-3 Initial 中，只是通过指令和代码训练**激发**？ 或者是通过指令和代码训练**注入能力？**

主要猜测是**激发**，理由如下



### 数据量

- [OpenAI 的论文报告](https://arxiv.org/abs/2203.02155)的指令数据量大小只有 77K，比预训练数据少了几个数量级
- Flan-PaLM 的指令微调仅为预训练计算的 0.4%



### 基座模型

- code-davinci-002 的基座**可能不是** GPT-3 Initial（初代 GPT-3） 

#### 猜测原因

|                  | 训练数据集的时间                              | 上下文长度（绝对位置编码） |
| ---------------- | --------------------------------------------- | -------------------------- |
| 初代 GPT-3       | 在数据集 C4  的 2016 - 2019 上训练            | 2048                       |
| code-davinci-002 | 21 年才结束，有可能在 C4 的 2019-2021  上训练 | 8192                       |



用绝对位置编码无法直接扩充上下文长度，所以猜测是训练了新模型。

- text-davinci-001 有时甚至比参数量更小的 code-cushman-001 还差。



### 利好 Code 预训练数据

（因为大家看好**激发 + 代码预训练数据**），所以会有利好 Code 预训练数据的相关研究 ：

- [Language Models of Code are Few-Shot Commonsense Learners](https://arxiv.org/abs/2210.07128)
- [在 T5 上继续训练 Code 训练](https://arxiv.org/pdf/2305.07922)，但它没对比 T5 的效果



## ICL ⇋ 对话能力

![](https://raw.githubusercontent.com/wnma3mz/blog_posts/master/imgs/ChatGPT能力/image5.png)

分化了两个模型

- ChatGPT：基于经验的观测结果，ChatGPT 似乎不像 text-davinci-003 那样受到 in-context demonstrations 的**强烈**影响。（？）
- text-davinci-003：**恢复了** text-davinci-002 所牺牲的**上下文学习能力**，根据 InstructGPT的论文，这是来自于强化学习调整阶段混入了语言建模的目标（而不是 RLHF 本身）。

24 年 10 月：[**A Survey on In-context Learning**](https://arxiv.org/abs/2301.00234v4) **—— Analysis**

## 预训练数据的多样性

https://aclanthology.org/2022.naacl-main.380.pdf

#### 语料库的来源

![](https://raw.githubusercontent.com/wnma3mz/blog_posts/master/imgs/ChatGPT能力/image6.png)

说明

- 左侧列是语料库；最上面一行是指标（用 Few-Shot 评测）
- 下划线表示低于平均值（ALL 和 Majority 的平均值），加粗表示高于平均值
- 1.3B 模型

发现：比较不同单一语料库模型（如Blog、Cafe、News）的性能，上下文学习能力**高度依赖于语料库的领域来源**。比如，Blog 54B

#### 语料库的多样性

![](https://raw.githubusercontent.com/wnma3mz/blog_posts/master/imgs/ChatGPT能力/image7.png)

发现：

- 单独使用 Cafe 或  KiN 语料库时未观察到上下文少样本学习能力，但**组合训练两者（KiN+Ency）则使这种能力得以出现**。
- 但并非所有组合都有效（如Cafe+News组合可能表现不佳甚至下降）

#### 语料库的大小

用单一语料库 HyperCLOVA 训练

![](https://raw.githubusercontent.com/wnma3mz/blog_posts/master/imgs/ChatGPT能力/image8.png)

- 数据量很重要：6B → 56B 会有显著提升 
- 多样性很重要：56B → 150B  并不会有显著提升

## 模型架构 —— Attention

https://arxiv.org/pdf/2209.11895

评测分数：上下文中第 500 个 token 的 loss - 第 50 个 token 的 loss，取平均。

### 怎么理解这个评测指标

背景知识：现代语言模型中，语境中后面的标记（token）比前面的标记更容易预测，即随着语境变长，损失（loss）会下降。

→ 利用更长的语境信息，从而提高预测能力的现象，就是所谓的**上下文学习**（In-context learning）



**Loss 的下降意味着模型的预测能力得到提高**（提高了 ICL 能力），即第 500 个 token 的损失低于第 50 个标记的 token（Loss(500) < Loss(50)）。



因此 Loss(500) - Loss(50) 的结果通常为负值。**分数越负，表明模型的 ICL 能力越强**



一条样本有 512 个 token：

- 选择第 **500 个 token** 是因为它接近 512 个 token 的末尾
- 选择第 **50 个 token** 是因为它在语境中足够靠后，足以建立文本的一些基本属性（如语言和文档类型），但又足够靠前，仍接近开头
- 研究表明，选择不同的标记索引并不会改变其结论

10k 条样本，取均值。



Q：为什么是看差值，而不是第 500 个 token 的 loss？第 500 个 token 也是看了前面所有 token，也能说明 ICL？

A：

- 单独看第 500 个 token 的 loss，只能反映模型在该语境长度下的最终预测效果，无法体现语境的增长对预测能力带来了多少提升。
- 如果只看第 500 个 token 的 loss ，大模型无疑会表现出更低的 loss。但这并不能直接说明其**上下文学习**比小模型更强


![](https://raw.githubusercontent.com/wnma3mz/blog_posts/master/imgs/ChatGPT能力/image9.png)

- 如果只有一层，随着训练 token 增加，分数不会显著变化
- 当有 2-3 层 attention 的时候，会在某个时间发生“相变”

### Q & A

Q：为什么是 Attention？

A：

- 文章中有其他地方补充了实验，如果去掉 Attention 效果会变差。
- DeepMind 在 LSTM 和 RNN 中也没发现 ICL 能力，[链接](https://papers.nips.cc/paper_files/paper/2022/file/77c6ccacfd9962e2307fc64680fc5ace-Paper-Conference.pdf)
- （3分）ICL 是 MLP + Attention 共同作用的效果，[链接](https://arxiv.org/pdf/2507.16003 )



# 训练 “ChatGPT”

[LIMA: Less Is More for Alignment](https://arxiv.org/abs/2305.11206)

假设：**模型的知识和能力几乎完全在预训练阶段学习，SFT 只是激发模型能力**

## 实验

### 数据

![](https://raw.githubusercontent.com/wnma3mz/blog_posts/master/imgs/ChatGPT能力/image10.png)

- 多样性：不同来源
- 高质量：手动撰写了 200 条 高质量回复 + 人工挑选 800 条数据

基于 LLaMa 65B 进行训练



### 结果

#### Q & A

![](https://raw.githubusercontent.com/wnma3mz/blog_posts/master/imgs/ChatGPT能力/image11.png)

- 300 条测试数据
- 左图是人工打分，右图是 GPT-4 打分

表现可以堪比 GPT-4（Win+Tie = 43%）



#### Multi-Turn Dialogue

![](https://raw.githubusercontent.com/wnma3mz/blog_posts/master/imgs/ChatGPT能力/image12.png)

- Zero-Shot：在 1k QA 数据上训练的模型评测
- Dialogue-Finetuned：1k + 30 条多轮对话数据训练的模型评测


所以，QA 数据训练的模型是具备对话能力的；加入 30 条对话数据可以“强化” 模型的对话能力
