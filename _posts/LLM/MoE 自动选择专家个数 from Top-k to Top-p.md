---
title: MoE 自动选择专家个数 from Top-k to Top-p
date: 2024-01-24 16:41:53
tags: [NLP, Attention, MoE]
categories: [Note]
mathjax: true
---

MoE 自动选择专家个数

<!-- more -->
## 背景

在 MoE 中每次推理需要指定选择专家个数，且每层专家个数完全一致。想到 Nucleus Sampling（Top-p采样），是不是可以把指定专家的数量换成，累计概率值来灵活的选择专家（cumsum）

## 预期

在保证性能的同时，降低激活的专家数量

## 代码
### 原 top-K 实现

```python
def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
    batch_size, sequence_length, hidden_dim = hidden_states.shape
    hidden_states = hidden_states.view(-1, hidden_dim)
    # router_logits: (batch * sequence_length, n_experts)
    router_logits, _ = self.gate(hidden_states)
 
    routing_weights = F.softmax(router_logits, dim=1, dtype=torch.float)
 
    # 从这里选择最大的 N 个专家。输出的两个 tensor shape 为 bs * N
    routing_weights, selected_experts = torch.topk(routing_weights,
                                                   self.top_k,
                                                   dim=-1)
    routing_weights /= routing_weights.sum(dim=-1, keepdim=True)
     
    final_hidden_states = None
    for expert_idx in self.expert_indicies:
        expert_layer = self.experts[expert_idx]
        # 找到对应的专家，并取值
        expert_mask = (selected_experts == expert_idx)
        expert_weights = (routing_weights * expert_mask).sum(dim=-1,
                                                             keepdim=True)
 
        current_hidden_states = expert_layer(hidden_states).mul_(
            expert_weights)
        if final_hidden_states is None:
            final_hidden_states = current_hidden_states
        else:
            final_hidden_states.add_(current_hidden_states)
 
    return tensor_model_parallel_all_reduce(final_hidden_states).view(
        batch_size, sequence_length, hidden_dim)
```

### 新 top-P 实现

```python
def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
    batch_size, sequence_length, hidden_dim = hidden_states.shape
    hidden_states = hidden_states.view(-1, hidden_dim)
    # router_logits: (batch * sequence_length, n_experts)
    router_logits, _ = self.gate(hidden_states)
 
    routing_weights = F.softmax(router_logits, dim=1, dtype=torch.float)
 
    # 用 one-hot 的形式输出所有专家的概率，并做好归一化，即输出 shape 为 bs * num_expert。
    # 没被选中的专家值为 0
    routing_weights = top_p_prob(routing_weights, top_p=0.8)
     
    final_hidden_states = None
    for expert_idx in self.expert_indicies:
        expert_layer = self.experts[expert_idx]
        # 在这里直接取值
        expert_weights = routing_weights[:, expert_idx].unsqueeze(dim=-1)
 
        current_hidden_states = expert_layer(hidden_states).mul_(
            expert_weights)
        if final_hidden_states is None:
            final_hidden_states = current_hidden_states
        else:
            final_hidden_states.add_(current_hidden_states)
 
    return tensor_model_parallel_all_reduce(final_hidden_states).view(
        batch_size, sequence_length, hidden_dim)
```

**top_p_prob** 

```python
# From GPT-4
def top_p_prob(probs: torch.Tensor, top_p: float = 0.8) -> torch.Tensor:
    # 输入输出的 shape 一样（bs * num_expert），概率值从高到低，累计概率超过 top_p 的部分进行归一化，其余部分置 0
 
    # 按照概率值降序排序，同时获取排序后的索引
    sorted_probs, indices = torch.sort(probs, dim=-1, descending=True)
    # 计算累积概率
    cumulative_probs = torch.cumsum(sorted_probs, dim=-1)
 
    # 使累积概率小于top_p的部分为1，其余部分为0
    mask = cumulative_probs < top_p
 
    # 保留最大的一个概率
    mask[..., 1:] = mask[..., :-1].clone()
    mask[..., 0] = 1
 
    # 根据mask来选择需要保留的概率值
    masked_probs = torch.zeros_like(probs)
    masked_probs.scatter_(dim=-1, index=indices, src=sorted_probs * mask.float())
    # 计算概率值的和，归一化的分母
    sum_masked_probs = masked_probs.sum(-1).unsqueeze(-1).repeat(1, probs.shape[-1])
 
    # 需要保留位置的索引，mask为1的位置
    mask_masked = torch.zeros_like(probs)
    mask_masked.scatter_(dim=-1, index=indices, src=mask.float())
 
    # 执行索引操作保证了元素在原始位置，此处更新对应位置的概率值，其他位置为0
    masked_probs = torch.where(mask_masked.bool(), masked_probs / sum_masked_probs, torch.zeros_like(probs))
 
    return masked_probs
```

### 实现思路

在 top_k 的实现中，假设有 torch.tensor([[1,2,3], [2,4,3]])

1. 先找出最大的 top_k 的最大索引和值
   1. 假设 top_k 为2，则有 
   2. routing_weights：tensor([[3, 2],[4, 3]])
   3. selected_experts ：tensor([[2, 1], [1, 2]])
   4. 即输出的 shape 为 (bs, top_k)
2. 对 routing_weights 进行归一化
3. 根据 expert_mask 重新计算哪些专家的值为 0。此时输出的 shape 为 (bs, num_expert)

然而，对于 top_p，每行选出来的专家数是不确定的，因为是根据概率值选出来的。所以需要重新设计这里的输出。

既然这里最终用到的 weight 还是 (bs, num_expert)，那么可以使用 one-hot 的形式来表示专家被选择的情况。即，

- 用 tensor([[0, 0, 1],[0, 1, 1]]) 表示专家的情况选择情况，如果为1，则进行归一化计算，否则直接设置成 0
- 最后专家的权重只需要用 routing_weights[:, expert_idx]来选择，而不需要根据 expert_mask 进行计算。

## 实验

在 Mixtral 上，top_p=0.6 和 0.7。因此，观察模型在每层选择了几个专家。

| layer idx     | top_p=0.6 | top_p=0.7 |
| ------------- | --------- | --------- |
| 0             | 2.545     | 3.357     |
| 15            | 2.605     | 3.342     |
| 31            | 1.908     | 2.425     |
| 所有层取 mean | 2.529     | 3.26      |

在某 1B * 8 实验上，top_p=0.7、0.8、0.9

| layer idx     | top_p=0.7  | top_p=0.8  | top_p=0.9  |
| ------------- | ---------- | ---------- | ---------- |
| 0             | 4.90282023 | 5.88775894 | 6.97505182 |
| 12            | 3.70174125 | 4.63081198 | 5.96097647 |
| 23            | 4.41420775 | 5.08847432 | 5.97912663 |
| 所有层取 mean | 4.34094666 | 5.23354833 | 6.38608094 |