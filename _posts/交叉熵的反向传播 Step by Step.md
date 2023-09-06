---
title: 交叉熵的反向传播 Step by Step（PyTorch）
date: 2023-09-03 21:49:20
tags: []
categories: [笔记]
mathjax: true
---

本文从公式开始，一步步用Pytorch实现自定义的交叉熵损失函数，最后理解分布式损失函数

<!-- more -->

## Step 1：A Simple Case 

以交叉熵为例，普通的二分类问题
$$
\text {Binary Cross Entropy} = -\left (y \log (\hat {y}) + (1-y) \log (1-\hat {y})\right)
$$
更常见的多分类问题
$$
\text {Multiclass Cross Entropy} = -\sum_{c=0}^{C} y_c \log (\hat {y}_c)
$$
其中，$y$ 表示真实的标签值（通常是one-hot向量），$\hat{y}$ 表示模型的输出，$C$ 表示有多少类别需要分类

在 PyTorch 中调用（本文使用的torch version==2.0.0）

```python
import torch

criterion = torch.nn.CrossEntropyLoss()

outputs = torch.tensor([[0.5, 0.2, 0.3]])   # \hat{y} Shape: bs, C
targets = torch.tensor([0])                 # y       Shape: bs

loss = criterion(outputs, targets)
# tensor(0.9398)
```

这里的`outputs`是模型的输出，`targets`是真实的标签值

## Step 2: 手写

实现`CrossEntropyLoss`需要两个条件

1. 使用 PyTorch 写出损失函数的公式，以及对应的导数
2. 根据公式计算loss，并计算梯度（grad）进行反向传播

### 公式

从$-\sum_{c=0}^{C} y_c \log (\hat {y}_c)$出发

1. 由于模型输出并不是单纯的概率值，因此会对模型输出进行`softmax`计算，得到$\hat{y}$
	$$\text {softmax}(x_i) = \frac {e^{x_i}}{\sum_{j=1}^n e^{x_j}}$$
2. 再以简单的单分类任务来说，
   1. $y$ = [1, 0, 0], $\hat{y}$ = [0.5, 0.2, 0.3] -> log($\hat{y}$)=[-0.9398, -1.2398, -1.1398]
   2. 计算时，只有c=0时，结果有值（-0.9398），其他类别时为0，无需计算
   3. 因此，实际最终结果就是取出对应类别的 $\hat{y}$ 的log值

```python
def MyCrossEntropyLoss(outputs, targets):
    # outputs.shape: bs, C
    # targets.shape: bs
    bs = outputs.size(0)
    outputs = torch.softmax(outputs, dim=1)  # 对预测结果进行softmax操作

    # 使用索引选择对应类别的概率值，并使用负对数函数计算损失
    loss = -torch.log(outputs[range(bs), targets]) # 对每个计算取对应的target
    return loss.mean() # 最终返回结果的平均值

outputs = torch.tensor([[0.5, 0.2, 0.3]])	# \hat{y} Shape: bs, C
targets = torch.tensor([0])	

loss = MyCrossEntropyLoss(outputs, targets)
# tensor(0.9398)
```

#### 梯度

1. 结合前面的计算，每次计算loss时，实际上只有对应类别的计算是有效的，因此将其简化为$-y_c\log(\hat{y}_c)$，而由于$y_c=1$，所以其实只有$-\log(\hat{y}_c)$

2. 这个公式只需要计算log的导数。对于$f(x)=\log(x)$，其导数$f^{'}(x)=\frac{1}{x}$。

3. 因此，这里的梯度很简单，如下所示，其中，$y_c=1$
    $$
    \frac {\partial \text {CrossEntropy}(y, \hat {y})}{\partial \hat {y}_c} = -\frac {y_c}{\hat {y}_c}=-\frac{1}{\hat{y}_c}
    $$
    
4. 而对于模型的输出而言，还有一个SoftMax计算。SoftMax的梯度有

    1. 当 $i = j$ 时：

    $$
    \frac {\partial y_i}{\partial x_i} = \frac {e^{x_i}\left (\sum_{k=1}^n e^{x_k}\right) - e^{x_i} e^{x_i}}{\left (\sum_{k=1}^n e^{x_k}\right)^2} = y_i (1-y_i)
    $$

	2. 当 $i \neq j$ 时：

    $$
    \frac {\partial y_i}{\partial x_j} = -\frac {e^{x_j} e^{x_i}}{\left (\sum_{k=1}^n e^{x_k}\right)^2} = -y_i y_j
    $$
    
5. 由于求导的链式法则，因此将SoftMax的导数乘以交叉熵的导数，就能得到模型输出的梯度。综上，对于模型的输出，交叉熵的梯度仅在目标标签上有效（≠0），所以最后的梯度计算为，

$$-\frac{1}{\hat{y}_c}\times y_c(1-y_c)=y_c-1$$

### autograd

在PyTorch内部有一套完善的求导机制。因此，仅需按照其规则进行书写即可实现前向传播、反向传播（链式求导）。如下所示，在forward填写loss的计算，backward中填写grad的计算即可

```python
import torch

class CrossEntropyLoss(torch.autograd.Function):
    @staticmethod
    def forward(ctx, outputs, targets):
        # outputs：模型的输出
        # targets: 真实标签
        ...
        ctx.save_for_backward(...) # 需要在反向传播中使用的变量
        return loss
    
    @staticmethod
    def backward(ctx, grad_output):
        # grad_output: 反向传播的梯度
        ... = ctx.saved_tensors
        ...
		return new_grad_output
```

#### Forward

这里基本就是把公式照抄过来，但需要额外把梯度计算时用到的变量存一下。

```python
    @staticmethod
    def forward(ctx, outputs, targets):
        bs = outputs.size(0)
        outputs = torch.softmax(outputs, dim=1)  

        loss = -torch.log(outputs[range(bs), targets]) 
        ctx.save_for_backward(outputs, targets)
        return loss.mean()
```

#### Backward

这里的`grad_output` 表示梯度计算的输出。它是一个与前向传播函数的输出形状相同的张量。在反向传播过程中，我们将计算当前函数的导数，乘以 `grad_output` 作为输入。这个值代表了后续节点对当前节点的梯度贡献。

通过将 `grad_output` 乘以导数（也称为雅可比向量积），可以有效地传递梯度信息到较早的节点，从而实现自动微分。

注意，`grad_output` 的形状必须与函数的输出形状一致，否则会引发错误。

```python
    @staticmethod
    def backward(ctx, grad_output):
        outputs, targets = ctx.saved_tensors
        bs = outputs.size(0)
        
        grad_y_pred = outputs.clone()
        grad_y_pred[range(bs), targets] -= 1  # 计算对应类别的梯度
        grad_y_pred /= bs # 取均值
        
        grad_y_pred *= grad_output.item()  # 乘以关于loss的梯度
        
        return grad_y_pred, None
```

### 完整版

```python
import torch

class MyCrossEntropyLossFunction(torch.autograd.Function):
    @staticmethod
    def forward(ctx, outputs, targets):
        bs = outputs.size(0)
        outputs = torch.softmax(outputs, dim=1)  

        loss = -torch.log(outputs[range(bs), targets]) 
        ctx.save_for_backward(outputs, targets)
        return loss.mean()
    
    @staticmethod
    def backward(ctx, grad_output):
        outputs, targets = ctx.saved_tensors
        bs = outputs.size(0)
        
        grad_y_pred = outputs.clone()
        grad_y_pred[range(bs), targets] -= 1  # 只需计算对应标签的梯度
        
        grad_y_pred /= bs # 取均值
        
        grad_y_pred *= grad_output.item()  # 乘以关于loss的梯度
        
        return grad_y_pred, None
  
def MyCrossEntropyLoss(outputs, targets):
    return MyCrossEntropyLossFunction.apply(outputs, targets)

outputs = torch.tensor([[0.5, 0.2, 0.3]], requires_grad=True)
targets = torch.tensor([0])
loss = MyCrossEntropyLoss(outputs, targets)
# tensor(0.9398, grad_fn=<MyCrossEntropyLossFunctionBackward>)
```

结合简单的双层MLP

```python
import torch
import copy

class MLP(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.l1 = torch.nn.Linear(5, 5, bias=False)
        self.l2 = torch.nn.Linear(5, 3, bias=False)

    def forward(self, x):
        return self.l2(self.l1(x))
    
input_ = torch.rand((5, 5))
labels = torch.randint(0, 3, size=(5,))
model1 = MLP()
model2 = copy.deepcopy(model1)

print("="*10, "Before Train", "="*10)
print(model1.l1.weight)
print(model2.l1.weight)

criterion1 = torch.nn.CrossEntropyLoss()
criterion2 = MyCrossEntropyLoss

optim1 = torch.optim.Adam(model1.parameters(), lr=1e-2)
optim2 = torch.optim.Adam(model2.parameters(), lr=1e-2)

outputs1 = model1(copy.deepcopy(input_))
loss1 = criterion1(outputs1, labels)
optim1.zero_grad()
loss1.backward()
optim1.step()

outputs2 = model2(copy.deepcopy(input_))
loss2 = criterion2(outputs2, labels)
optim2.zero_grad()
loss2.backward()
optim2.step()

print("="*10, "After Train", "="*10)
print(model1.l1.weight)
print(model2.l1.weight)

```

## Step 3: 分布式

模型训练速度提升+单卡放不下大模型，由于这两个需求，所以分布式计算中需要重写损失函数的forward和backward，以Megatron-LM的训练代码为例，如下所示。接下来将逐步拆解实现过程（忽略label_smoothing，将其看作0.0）。

https://github.com/NVIDIA/Megatron-LM/blob/main/megatron/core/tensor_parallel/cross_entropy.py

```python
# Copyright (c) 2022, NVIDIA CORPORATION. All rights reserved.

import torch

from megatron.core.parallel_state import (
    get_tensor_model_parallel_group,
    get_tensor_model_parallel_rank,
    get_tensor_model_parallel_world_size,
)

from .utils import VocabUtility


class _VocabParallelCrossEntropy(torch.autograd.Function):
    @staticmethod
    def forward(ctx, vocab_parallel_logits, target, label_smoothing=0.0):

        # Maximum value along vocab dimension across all GPUs.
        logits_max = torch.max(vocab_parallel_logits, dim=-1)[0]
        torch.distributed.all_reduce(
            logits_max, op=torch.distributed.ReduceOp.MAX, group=get_tensor_model_parallel_group()
        )
        # Subtract the maximum value.
        vocab_parallel_logits = vocab_parallel_logits - logits_max.unsqueeze(dim=-1)

        # Get the partition's vocab indecies
        get_vocab_range = VocabUtility.vocab_range_from_per_partition_vocab_size
        partition_vocab_size = vocab_parallel_logits.size()[-1]
        rank = get_tensor_model_parallel_rank()
        world_size = get_tensor_model_parallel_world_size()
        vocab_start_index, vocab_end_index = get_vocab_range(partition_vocab_size, rank, world_size)

        # Create a mask of valid vocab ids (1 means it needs to be masked).
        target_mask = (target < vocab_start_index) | (target >= vocab_end_index)
        masked_target = target.clone() - vocab_start_index
        masked_target[target_mask] = 0

        # Get predicted-logits = logits[target].
        # For Simplicity, we convert logits to a 2-D tensor with size
        # [*, partition-vocab-size] and target to a 1-D tensor of size [*].
        logits_2d = vocab_parallel_logits.view(-1, partition_vocab_size)
        masked_target_1d = masked_target.view(-1)
        arange_1d = torch.arange(start=0, end=logits_2d.size()[0], device=logits_2d.device)
        predicted_logits_1d = logits_2d[arange_1d, masked_target_1d]
        predicted_logits_1d = predicted_logits_1d.clone().contiguous()
        predicted_logits = predicted_logits_1d.view_as(target)
        predicted_logits[target_mask] = 0.0
        # All reduce is needed to get the chunks from other GPUs.
        torch.distributed.all_reduce(
            predicted_logits,
            op=torch.distributed.ReduceOp.SUM,
            group=get_tensor_model_parallel_group(),
        )

        # Sum of exponential of logits along vocab dimension across all GPUs.
        exp_logits = vocab_parallel_logits
        torch.exp(vocab_parallel_logits, out=exp_logits)
        sum_exp_logits = exp_logits.sum(dim=-1)
        torch.distributed.all_reduce(
            sum_exp_logits,
            op=torch.distributed.ReduceOp.SUM,
            group=get_tensor_model_parallel_group(),
        )

        # Loss = log(sum(exp(logits))) - predicted-logit.
        loss = torch.log(sum_exp_logits) - predicted_logits

        # Normalize and optionally smooth logits
        exp_logits.div_(sum_exp_logits.unsqueeze(dim=-1))

        vocab_size = exp_logits.size(-1)
        if label_smoothing > 0:
            """
            We'd like to assign 1 / (K - 1) probability mass to every index that is not the ground truth.
            = (1 - alpha) * y_gt + alpha * mean(y_{i for i != gt})
            = (1 - alpha) * y_gt + (alpha / (K - 1)) * \sum_{i != gt} y_i
            = ((K - 1) * (1 - alpha) / (K - 1)) * y_gt + (alpha / (K - 1)) * \sum_{i != gt} y_i
            = (K * (1 - alpha) - 1) / (K - 1)) * y_gt  + (alpha / (K - 1)) * \sum_{i} y_i
            = (1 - (alpha * K) / (K - 1)) * y_gt + ( (alpha * K) / (K - 1) ) * \sum_{i} y_i / K
            From: https://github.com/NVIDIA/NeMo/blob/main/nemo/collections/common/losses/smoothed_cross_entropy.py
            """
            assert 1.0 > label_smoothing > 0.0
            smoothing = label_smoothing * vocab_size / (vocab_size - 1)

            # Exp logits at this point are normalized probabilities. So we can just take the log to get log-probs.
            log_probs = torch.log(exp_logits)
            mean_log_probs = log_probs.mean(dim=-1)
            loss = (1.0 - smoothing) * loss - smoothing * mean_log_probs

        ctx.label_smoothing, ctx.vocab_size = label_smoothing, vocab_size

        # Store softmax, target-mask and masked-target for backward pass.
        ctx.save_for_backward(exp_logits, target_mask, masked_target_1d)

        return loss

    @staticmethod
    def backward(ctx, grad_output):

        # Retreive tensors from the forward path.
        softmax, target_mask, masked_target_1d = ctx.saved_tensors
        label_smoothing, vocab_size = ctx.label_smoothing, ctx.vocab_size

        # All the inputs have softmax as thier gradient.
        grad_input = softmax
        # For simplicity, work with the 2D gradient.
        partition_vocab_size = softmax.size()[-1]
        grad_2d = grad_input.view(-1, partition_vocab_size)

        # Add the gradient from matching classes.
        arange_1d = torch.arange(start=0, end=grad_2d.size()[0], device=grad_2d.device)

        softmax_update = 1.0 - target_mask.view(-1).float()

        if label_smoothing > 0:
            smoothing = label_smoothing * vocab_size / (vocab_size - 1)
            grad_2d[arange_1d, masked_target_1d] -= (1.0 - smoothing) * softmax_update
            average_grad = 1 / vocab_size
            grad_2d[arange_1d, :] -= smoothing * average_grad
        else:
            grad_2d[arange_1d, masked_target_1d] -= softmax_update

        # Finally elementwise multiplication with the output gradients.
        grad_input.mul_(grad_output.unsqueeze(dim=-1))

        return grad_input, None, None


def vocab_parallel_cross_entropy(vocab_parallel_logits, target, label_smoothing=0.0):
    """
    Performs cross entropy loss when logits are split across tensor parallel ranks

    Arguments:
        vocab_parallel_logits: logits split across tensor parallel ranks
                               dimension is [sequence_length, batch_size, hidden_size]

        target: correct vocab ids of dimseion [sequence_length, micro_batch_size]

        lobal_smoothing: smoothing factor, must be in range [0.0, 1.0)
                         default is no smoothing (=0.0)
    """
    return _VocabParallelCrossEntropy.apply(vocab_parallel_logits, target, label_smoothing)
```

### 为什么 

本文不介绍分布式训练的其他原理，仅关注模型最后的输出，以及损失函数的计算。

前文已经详细介绍了交叉熵的实现，那么为什么到分布式训练的时候，就不能直接复用呢？主要原因在于模型的输出被拆散了。回顾模型的输出，其shape为[`bs`, `C`]，`bs`表示每次训练的数据量`batch size`，`C`表示任务的总的分类数量，在LLM中主要是词表大小`vocab_size`（下文用此代表类别数量）。

对于模型的输出而言，分布式（假设有`word_size`张GPU）主要分两种情况
1. 拆分`bs`，每张卡上有`bs`数据量，那么整合模型的最后输出应为[`bs` * `word_size`, `vocab_size`]
2. 拆分`vocab_size`，单个模型的输出为[`bs`, `vocab_size` // `word_size`]。这种情况需要保证`vocab_size`能够整除`word_size`

因此，在进行loss计算前，需要先同步所有的模型输出，然后再进行loss计算。这里的同步，就是从所有GPU中获取模型的输出。但考虑到通信开销问题，有些操作是可以进行优化的，即无需通信全部内容再计算。重新思考损失的计算过程可以分为如下几步：

1. SoftMax计算：$\exp(x) / \sum(\exp(x)$
2. log的计算：  $-\log[\exp(x) / \sum(\exp(x)]$
3. 索引：根据targets索引矩阵对应位置 $x[..., ...]$

其中，前两者跟最后一个步骤是可以独立运行的。而前两者的计算是耦合的，为节约通信量，对log计算进行变化

1. $-\log[\exp(x) / \sum(\exp(x)]$
2. $-\log[\exp(x)] + \log[\sum\exp(x)]$
3. $\log[\sum\exp(x)] - x$



因此，可以由原始的`func1`推导至分布式的`func2`
```python
import torch

def func1(outputs, targets):
    # 原始版本
    bs = outputs.size(0)
    # Step 0:
    predicted_logits = outputs[range(bs), targets]

    predicted_logits = torch.softmax(predicted_logits, dim=1)
    loss = -torch.log(predicted_logits)
    print(loss)

def func2(outputs, targets):
    # 分布式的简化版本
    bs = outputs.size(0)

    min_c, max_c = 0, 3 # 预先设定的最小和最大值

    # 只选择在该设备上的数值
    target_mask = (targets < min_c) | (targets >= max_c)
    # 由于在分布式中min_c不一定为0，所以需要做一个归一化
    masked_target = targets.clone() - min_c
    masked_target[target_mask] = min_c

    # arange_1d, masked_target 分别对应 上文的range(bs), targets
    arange_1d = torch.arange(start=0, end=bs)
    predicted_logits = outputs[arange_1d, masked_target]
    
    predicted_logits = predicted_logits.clone().contiguous()
    # 对于非该设备上计算的值，置为0。则不进行计算
    predicted_logits[target_mask] = 0.0
    # 先算第二项
    predicted_logits = torch.sum(predicted_logits, dim=-1)

    loss = torch.log(torch.exp(outputs).sum(dim=-1)) - predicted_logits
    print(loss)

outputs = torch.tensor([[0.5, 0.2, 0.3]])
targets = torch.tensor([0])    

func1(outputs, targets)
func2(outputs, targets)
```


### Forward



#### 预处理
一般而言，为了数值稳定性，SoftMax计算前会减去最大值，即 

$$\text {softmax}(x_i) = \frac {e^{x_i}}{\sum_{j=1}^n e^{x_j}} = \frac {e^{x_i - \max(x)}}{\sum_{j=1}^n e^{x_j - \max(x)}}$$

```python
# 每张卡上的模型输出为 vocab_parallel_logits，shape：[bs, seq_len, vocab_size // word_size]
# SoftMax的前置计算，可以无需通信全部参数，分两步进行
# 1. 计算每张卡的最大值
# 2. 再计算所有卡的最大值
# 3. 最后对每卡张减去最大值
# 这样就能避免通信所有的参数
logits_max = torch.max(vocab_parallel_logits, dim=-1)[0]
torch.distributed.all_reduce(
    logits_max, op=torch.distributed.ReduceOp.MAX, group=get_tensor_model_parallel_group()
)
# Subtract the maximum value.
vocab_parallel_logits = vocab_parallel_logits - logits_max.unsqueeze(dim=-1)
```

这里的`seq_len`是LLM中句子的长度，句子的每个位置（token）会计算下一个token的概率（在词表中选），因此，可以理解为每个位置都是一个分类任务，因此`vocab_size`就是分类的数量。即，可以转换为[`bs`*`seq_len`, `vocab_size` // `word_size`]的形式


由于单卡不一定存在所有词表的值，因此需要在这张卡上将非该卡输出位置的值置为0。
```python
# Get the partition's vocab indecies
get_vocab_range = VocabUtility.vocab_range_from_per_partition_vocab_size
partition_vocab_size = vocab_parallel_logits.size()[-1]
rank = get_tensor_model_parallel_rank()
world_size = get_tensor_model_parallel_world_size()
vocab_start_index, vocab_end_index = get_vocab_range(partition_vocab_size, rank, world_size)

# Create a mask of valid vocab ids (1 means it needs to be masked).
target_mask = (target < vocab_start_index) | (target >= vocab_end_index)
masked_target = target.clone() - vocab_start_index
masked_target[target_mask] = 0
```

#### 根据 targets 选择模型真实输出的Logits 

将模型的输出和标签 转换为[`bs`*`seq_len`, `vocab_size` // `word_size`]的形式，以便于计算
```python
logits_2d = vocab_parallel_logits.view(-1, partition_vocab_size)
masked_target_1d = masked_target.view(-1) 

# 选择真实标签的logits，相当于原来的outputs[range(bs), targets]
# 相当于把此步骤置前，而对于后续计算无影响，并减少了后续的通信时间
arange_1d = torch.arange(start=0, end=logits_2d.size()[0], device=logits_2d.device)
predicted_logits_1d = logits_2d[arange_1d, masked_target_1d]
predicted_logits_1d = predicted_logits_1d.clone().contiguous()
```

#### 计算loss

将logits转换为原来的shape，再求和，先得到了最后的第一项，之后再减去log
```python
predicted_logits = predicted_logits_1d.view_as(target)
predicted_logits[target_mask] = 0.0
# All reduce is needed to get the chunks from other GPUs.
torch.distributed.all_reduce(
    predicted_logits,
    op=torch.distributed.ReduceOp.SUM,
    group=get_tensor_model_parallel_group(),
)
```

计算SoftMax
$$\text {softmax}(x_i) = \frac {e^{x_i}}{\sum_{j=1}^n e^{x_j}}$$
```python
exp_logits = vocab_parallel_logits
# 计算每张卡的exp，作为分子
torch.exp(vocab_parallel_logits, out=exp_logits)
# 再计算这张卡上的exp之和
sum_exp_logits = exp_logits.sum(dim=-1)
# 再计算所有卡的exp之和，作为分母
torch.distributed.all_reduce(
    sum_exp_logits,
    op=torch.distributed.ReduceOp.SUM,
    group=get_tensor_model_parallel_group(),
)
```

计算loss
```python
# 第一项是log(sum(exp(logits)))，第二项是最后的logits
loss = torch.log(sum_exp_logits) - predicted_logits

# 为了Backward，需要保存一些变量
exp_logits.div_(sum_exp_logits.unsqueeze(dim=-1))
```


### Backward


#### 预处理

```python
softmax, target_mask, masked_target_1d = ctx.saved_tensors
label_smoothing, vocab_size = ctx.label_smoothing, ctx.vocab_size

# 根据SoftMax计算的结果，忽略无需计算梯度的位置。target_mask中的元素不是0就是1
grad_input = softmax
# For simplicity, work with the 2D gradient.
partition_vocab_size = softmax.size()[-1]
grad_2d = grad_input.view(-1, partition_vocab_size)

# Add the gradient from matching classes.
arange_1d = torch.arange(start=0, end=grad_2d.size()[0], device=grad_2d.device)

softmax_update = 1.0 - target_mask.view(-1).float()
```

#### 梯度计算

```python
# 正常计算位置，就-1，否则将不减（即不做计算）
grad_2d[arange_1d, masked_target_1d] -= softmax_update

# 最后乘上关于loss的梯度
grad_input.mul_(grad_output.unsqueeze(dim=-1))
```