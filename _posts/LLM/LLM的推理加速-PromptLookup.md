---
title: LLM的推理加速-Prompt Lookup
date: 2025-08-24 21:48:23
tags: [NLP, Attention, Prompt Lookup]
categories: [Note]
mathjax: true
---

[GitHub](https://github.com/apoorvumang/prompt-lookup-decoding/?tab=readme-ov-file)

<!-- more -->

## 背景

[OpenAI 文档](https://platform.openai.com/docs/guides/predicted-outputs?lang=python)

当使用 LLM 重构下面代码的时候（比如想把 `username` 改成 `email`），模型必须 token by token 的输出所有内容。
```js
class User {
  firstName: string = "";
  lastName: string = "";
  username: string = "";
}

export default User;
```

但实际上，有大部分内容都是在输入（Prompt）中，无需模型“动脑子”输出
```js
class User {
  firstName: string = "";
  lastName: string = "";
```

## 实现方式

以 `What is the capital of South Korea?` 输入为例，模型会构建一个查询表

| 2-gram      | 3-speculate tokens |
| ----------- | ------------------ |
| What is     | the capital of     |
| is the      | capital of South   |
| the capital | of South Korea     |

模型在生成过程中，当生成到 `the capital` 时，会命中表中的内容。将 `of South Korea` 拼接到输出后面，并且用模型校验这个输出。

如果模型接受这个输出，那么模型就不需要 token by token 的生成 `of`、`South`、`Korea` 了。

参考：[https://zhuanlan.zhihu.com/p/1920447613800547342](https://zhuanlan.zhihu.com/p/1920447613800547342)

代码上，[GitHub](https://github.com/apoorvumang/prompt-lookup-decoding/?tab=readme-ov-file) 给出了关键代码，如下所示

```python
def find_candidate_pred_tokens(input_ids, max_ngram_size=3, num_pred_tokens=10):
    # input_ids：输入的 token 序列，1 x token_len 的向量
    input_length = input_ids.size(1)

    for ngram_size in range(max_ngram_size, 0, -1):
        # 提取最后 n 个 token作为搜索的 n-gram
        ngram = input_ids[0, -ngram_size:].tolist()

        # 构建滑动窗口，窗口大小为 ngram_size。**滑动窗口相当于构建查询表的第一列**
        windows = input_ids.unfold(dimension=1, size=ngram_size, step=1)

        # 将 ngram 转换为张量，用于比较
        ngram_tensor = torch.tensor(ngram, device=input_ids.device).unsqueeze(0)

        # 找到所有匹配的窗口
        matches = (windows == ngram_tensor).all(dim=2)

        # 获取所有匹配的窗口索引
        match_indices = matches.nonzero(as_tuple=True)[1]

        # 遍历所有匹配的窗口索引，找到一个有效的 continuation。**遍历所有匹配的窗口索引，相当于查询表的第二列**
        for idx in match_indices:
            start_idx = idx + ngram_size
            end_idx = start_idx + num_pred_tokens
            # 确保 continuation 不超过输入长度，并且不与 n-gram 重复
            if end_idx <= input_length and start_idx < input_length - ngram_size:
                return input_ids[0, start_idx:end_idx]

    # 如果没有找到匹配项，返回一个空张量
    return torch.tensor([], dtype=torch.long, device=input_ids.device)
```