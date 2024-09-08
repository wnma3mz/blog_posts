---
title: LLM的推理相关计算公式
date: 2024-09-05 13:45:43
tags: [NLP, Inference, Math]
categories: [Note]
mathjax: true
---

记录 LLM 在推理上的理论计算公式

<!-- more -->


### 参数量

embedding + lm head + transformer block $\times$ N

- Embedding / LM Head 的参数量：vocab_size $\times$ hidden_size
- transformer block 的参数量
    - self-attention：qkvo 四个线性层，需要分别考虑两种 Attention
        - multi-head attention（MHA）：4 $\times$ hidden_size $^2$
        - grouped query attention（GQA）：
            - q 和 o，2 $\times $ hidden_size $^2$
        
            - k 和 v，2 $\times $ hidden_size $^2$ / num_attention_heads $\times $ num_key_value_heads
    - FFN：三个线性层，3 $\times$ hidden_size $\times$ intermediate_size
    - layer norm：2 $\times$ 2 $\times$ hidden_size
    

以 [llama2-13B](https://huggingface.co/meta-llama/Llama-2-13b-hf/blob/main/config.json) 为例，关键参数

```json
  "hidden_size": 5120,
  "intermediate_size": 13824,
  "mlp_bias": false,
  "num_attention_heads": 40,
  "num_hidden_layers": 40,
  "num_key_value_heads": 40,
  "vocab_size": 32000,
```

由于 num_attention_heads == num_key_value_heads，所以是 MHA

- Embedding/LM Head： $32000\times 5120$
- transformer block
    - self-attention：$4\times 5120\times 5120$
    - FFN: $3\times 5120\times 13824$
    - layer norm: $4\times 5120$

综上，列出计算

$$2\times 32000\times 5120 + 40\times (4\times 5120\times 5120 + 3\times 5120\times 13824 + 4\times 5120) = 13,016,268,800 \approx 13B$$

如果是 (b)float16，则占用空间/显存为 13 $\times $ 2 = 26G


如果是 GQA，self-attn 参数量公式为



### KV cache 显存计算公式

$4\times b\times l\times num\_heads\times embed\_size\_per\_head \times (s+n)$

参数说明

- b: 句子条数
- l：层数
- num_heads：隐层大小 (num_key_value_heads)
- embed_size_per_head：每个头的大小 (hidden_size / num_attention_heads)
- s：输入长度
- n：输出长度
- 4：k cache+v cache，均为 float16，所以是(1+1)*2

如果只有1条句子，输入+输出 token 长度由 512 -> 1024，则会增加 $4\times 512 \times l\times h$：

Llama3-8B：$4\times 512\times 1024\times 32 / 1024 / 1024 = 64 $ M

Llama3-70B：$4\times 512\times 1024\times 80 / 1024 / 1024 = 160$ M

简单来说，每增加一个 token

8B 就会增加 0.125 M 的显存

70B 就会增加 0.3125 M 的显存

### 计算量

Embedding 可以视作一个哈希表，没有计算量

- LM Head：$2 \times b \times s\times hidden\_size\times V$
- Self-attention：
    - q_proj: $2 \times b \times s \times hidden\_size^2$
    - k_proj和v_proj: 
        - MHA: $2\times(2 \times b \times s \times hidden\_size^2)$
        - GQA: $2\times(2 \times b \times s \times hidden\_size^2 / (num\_attention\_heads \times  num\_key\_value\_heads)^2 )$
    - attn_weights:	$2 \times b \times s^2 \times hidden\_size$
    - attn_output: $2 \times b \times s^2 \times hidden\_size$
    - o_proj: $2 \times b \times s \times hidden\_size^2$
- FFN:
    - gate_proj 和 up_proj: $2\times (b \times s\times hidden\_size)\times (hidden\_size\times intermediate\_size)$
    - down_proj: $(b \times s\times intermediate\_size)\times (intermediate\_size\times hidden\_size)$

- b: 句子条数
- s：输入长度

还是以 [llama2-13B](https://huggingface.co/meta-llama/Llama-2-13b-hf/blob/main/config.json) 为例，关键参数

```json
  "hidden_size": 5120,
  "intermediate_size": 13824,
  "mlp_bias": false,
  "num_attention_heads": 40,
  "num_hidden_layers": 40,
  "num_key_value_heads": 40,
  "vocab_size": 32000,
```

假设 b=1，s=1，则整体计算量有

- LM Head: $2\times 5120\times 32000$
- self-attention:
    - $2\times 5120\times 5120$
    - $4\times 5120\times 5120$
    - $2\times 5120$
    - $2\times 5120$
    - $2\times 5120\times 5120$
- MLP:
    - $2\times 5120\times 5120\times 13824$
    - $2\times 13824\times 13824\times 5120$

模型有 40 层，所以

$2\times 5120\times 32000+ 40*(8\times 5120\times 5120+4\times 5120+ 2\times 5120\times 5120 \times 13824+2\times 13824\times 13824\times 5120) = 1.0727553e+14$

参考资料：

https://blog.csdn.net/wxc971231/article/details/135434478