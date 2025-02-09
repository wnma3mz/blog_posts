---
title: Token-Level 的 KVCache 复用
date: 2025-01-25 14:09:10
tags: []
categories: []
mathjax: true
---

在 LLM 中一大应用场景就是进行多轮对话，发起的第二轮对话大多情况下是基于第一轮对话的结果进行的。因此，完全可以缓存第一轮对话的结果，避免重复计算 KV Cache，以加速第二轮 TTFT 时间。

更进一步，在非对话场景可以查找输入的**最长公共前缀序列**，以复用 KV Cache。

主要论文：https://arxiv.org/pdf/2312.07104

<!-- more -->

## 背景

LLM 在生成 token 的时候，需要重新计算前面所有 token 的 attention。且 LLM 生成过程又是 token by token 的，因此每生成一个新的 token 都需要重新计算一次 attention。

以「今天天气真不错」为例，当输入「今天」时

1. 计算「今天」的 KV Cache
2. 生成 token「天气」
3. 计算「今天天气」的 KV Cache
4. 生成 token「真」
5. 计算「今天天气真」的 KV Cache
6. 生成 token「不」
7. 计算「今天天气真不」的 KV Cache
8. 生成 token「错」

这样，每次生成一个 token 都需要重新计算一次 KV Cache，这样效率是很低的。

一种直接的思路是每次缓存生成过程的 KV Cache，下次生成 token 时直接使用缓存的 KV Cache。

1. 计算「今天」的 KV Cache
2. 生成 token「天气」
3. 计算「今天天气」的 KV Cache 时，直接使用缓存的「今天」的 KV Cache，只计算「天气」的 KV Cache
4. 生成 token「真」
5. 计算「今天天气真」的 KV Cache 时，直接使用缓存的「今天天气」的 KV Cache，只计算「真」的 KV Cache
6. 生成 token「不」
7. 计算「今天天气真不」的 KV Cache 时，直接使用缓存的「今天天气真」的 KV Cache，只计算「不」的 KV Cache
8. 生成 token「错」

## 问题

LLM 的一大场景时多轮对话，即会利用前若干轮对话的情况。具体来说，

```mermaid
graph LR
    A[Q1: 今天天气怎么样] --> B[A1: 天气不错]
    B --> C[Q2: 适合穿什么衣服呢]
    C --> D[A2: 短袖]
```

A1 是由 Q1 生成的。A2 是由 Q1+A1+Q2 生成的。由于 Q2 是额外的一个请求，故在生成 A2 时，往往会重新计算 Q1+A1 的 KV Cache。

如果能够缓存 Q1+A1 的 KV Cache，那么生成 A2 时就可以直接使用 Q1+A1 的 KV Cache，只计算 Q2 的 KV Cache。这样可以加速 A2 的 TTFT 时间。

更进一步，哪怕不是多轮对话，只要输入的两个句子有**最长公共前缀**，就可以复用 KV Cache。

## 解决方案

[SGLang: Efficient Execution of Structured Language Model Programs
](https://arxiv.org/pdf/2312.07104) 在工程上对此进行了实现。其核心思想就是使用`Radix Tree`来存储 KV Cache，以实现 KV Cache 的复用，命名为`RadixAttention`。

如下图所示，`RadixAttention` 能够根据输入的内容，查找到之前计算过的 KV Cache，以节约计算时间。并且，对于多个分支的情况也进行了比较好的处理。

![](https://raw.githubusercontent.com/wnma3mz/blog_posts/master/imgs/kvcache/1739091372419.png)

## Radix Tree

`Radix Tree` 是一种树形数据结构，用于查找过程中的字符串。其核心思想是将相同前缀的字符串合并为一个节点，以节约空间。但是放到 LLM 中，需要对里面的一些数据和逻辑进行修改。

- 核心目的：是查找**最长公共前缀**
- 存储内容：KV Cache 的索引，及对应的 token 序列
- 实现功能
    - 插入 token 序列及其对应的 KV Cache 索引
    - 输入 token 序列，查找最长公共前缀，返回对应的 KV Cache 索引，以及前缀长度
    - 删除 token 序列及其对应的 KV Cache 索引（侧重工程上优化，不重点介绍）

对于树结构，需要实现一个最基本的节点结构，如下：

```python
from typing import Dict, List, Optional, Tuple

class Node:
    def __init__(self, request_id: str):
        self.children: Dict[int, Node] = {}
        self.is_end = False
        self.path = None
        self.request_id = request_id

    def __repr__(self):
        return f"Node({self.request_id}): path={self.path}; is_end={self.is_end}"
```

### 插入

在每次生成 token 时，需要将 token 序列及其对应的 KV Cache 索引插入到 `Radix Tree` 中。

而生成 token 其实又分了两种情况，

- 生成第一个 token 时，此时 KV Cache 的长度是大于 1 的
- 生成后续 token 时，此时 KV Cache 的长度是 1

所以，分别实现两个函数 `insert` 和 `append_to_request`。用 `request_id` 来标识 KV Cache 的索引并且也能对应到完整的 token 生成序列。   

```python

class RadixTree:
    def __init__(self):
        self.root = Node(None)  # 根节点
        self.request_id_map: Dict[str, Node] = {}

    def insert(self, input_ids: List[int], request_id: str):
        # 生成第一个 token 时
        node = self.root
        path = []
        for id_ in input_ids:
            if id_ not in node.children:
                node.children[id_] = Node(request_id)
            node = node.children[id_]
            path.append(id_)
            node.path = path[:]
        node.is_end = True
        self.request_id_map[request_id] = node

    def append_to_request(self, input_ids: List[int], request_id: str):
        if request_id not in self.request_id_map:
            self.insert(input_ids, request_id)
            return
        # 对于后续 token 生成，只需要在原有的 KV Cache 上追加即可
        node = self.request_id_map.pop(request_id)
        path = node.path
        node.is_end = False
        for id_ in input_ids:
            if id_ not in node.children:
                node.children[id_] = Node(request_id)
            node = node.children[id_]
            path.append(id_)
            node.path = path[:]
        node.is_end = True
        self.request_id_map[request_id] = node        
```

### 查找

在定下数据结构之后，这个查找是比较容易实现的，只需要遍历 `Radix Tree`，找到最长的公共前缀即可。

主要是需要注意一些边界情况，比如输入的 token 序列不存在于 `Radix Tree` 中，或者输入的 token 序列是 `Radix Tree` 中某个 token 序列的子序列。

```python

    def longest_common_prefix(self, input_ids: List[int]) -> Tuple[Optional[str], int]:
        # 返回最长的公共前缀
        node = self.root
        longest = []
        for id_ in input_ids:
            if id_ not in node.children:
                return node.request_id, len(longest) - 1 if len(longest) > 0 else -1
            node = node.children[id_]
            if node.path is not None and len(node.path) > len(longest):
                longest = node.path[:]
        return node.request_id, len(longest) - 1 if len(longest) > 0 else -1
```

### 删除

由于 KV Cache 会消耗大量的显存/内存，所以需要定期的删除。

这里的删除只能从后往前删，并且结合删除策略，比如删除最近最少使用的 KV Cache。可能还需要做一些其他数据结构上的调整。暂时没实现完整。

## 总结

使用 `Radix Tree` 来存储 KV Cache，可以有效优化 TTFT 的时间，但同时也会增加显存/内存的消耗。

另外有几个注意事项：

- 实现的时候使用了 token ids 进行查找。而对于多模态来说，不同图片的 token ids 序列可能是相同的，所以会在查找的时候出现问题
- 删除策略不够完善，需要进一步优化
- `Radix Tree` 需要结合 KV Cache 一起优化。比如这里在复用了 KV Cache 之后，实际上是复制了一份KV Cache ，导致消耗内存变大。理想情况是直接接着之前的 KV Cache。

对应项目的实现：https://github.com/wnma3mz/tLLM/blob/main/tllm/commons/radix_tree.py

