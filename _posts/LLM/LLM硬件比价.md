---
title: LLM 不同硬件推理速度对比
date: 2024-07-21 20:02:00
tags: [NLP, LLM, Raspberry]
categories: [Note]
mathjax: true
---

对比不同硬件下，LLM 的推理速度

<!-- more -->

生成阶段，20 Token/s 认为是一个比较可以接受的速度，即 20 / 1000 = 0.02 s = 20 ms

| 设备 | 模型尺寸 | 速度 | 来源 | 记录时间 | 备注 |
| --- | --- | --- | --- | --- | --- |
| 1 x RasPi 5 8 GB | Llama 3 8B （Q4） | 564.31 ms, 1.77 t/s
I: 556.67 ms, T: 6.17 ms | https://github.com/b4rtaz/distributed-llama/blob/main/README.md | 2024.07.21 |  |
| 4 x RasPi 5 8 GB | Llama 3 8B （Q4） | 331.47 ms, 3.01 t/s
I: 267.62 ms, T: 62.34 ms |  |  |  |
| 8 x RasPi 4B 8 GB | Llama 2 70B（Q4） | 4842.81 ms
I: 2121.94 ms, T: 2719.62 ms |  |  |  |
| c3d-highcpu-30  | Llama 2 7B （Q4） | 101.81 ms
I: 101.06 ms, T: 0.19 ms | https://github.com/b4rtaz/distributed-llama/discussions/9 |  |  (30 vCPU, 15 core, 59 GB memory) europe-west1, AMD Genoa |
| c3d-highcpu-30 *4 | Llama 2 7B（Q4） | 53.69 ms 
I: 40.25 ms, T: 12.81 ms |  |  |  |
| c3d-highcpu-30  | Llama 2 70B（Q4） | 909.69 ms
I: 907.25 ms, T: 1.75 ms |  |  |  |
| c3d-highcpu-30 *4 | Llama 2 70B（Q4） | 293.06 ms 
I: 264.00 ms, T: 28.50 ms |  |  |  |
| M1 | Llama 7B（Q4） | 14.19 t/s | https://github.com/ggerganov/llama.cpp/discussions/4167 | 2023.11.22 | 取最慢的速度 |
| M1 Pro | Llama 7B（Q4） | 35.52 t/s |  |  |  |
| M1 Max | Llama 7B（Q4） | 54.61 t/s |  |  |  |
| M1 Ultra | Llama 7B（Q4） | 74.93 t/s |  |  |  |
| M2 | Llama 7B（Q4） | 21.7 t/s |  |  |  |
| M2 Pro | Llama 7B（Q4） | 37.87 t/s |  |  |  |
| M2 Max | Llama 7B（Q4） | 60.99 t/s |  |  |  |
| M2 Ultra | Llama 7B（Q4） | 65.95 t/s |  |  |  |
| M3 | Llama 7B（Q4） | 21.34 t/s |  |  |  |
| M3 Pro | Llama 7B（Q4） | 30.65 t/s |  |  |  |
| M3 Max | Llama 7B（Q4） | 56.58 t/s |  |  |  |
| AMD EPYC 7443P | Llama 7B（Q4） | 11.18 t/s | https://github.com/ggerganov/llama.cpp/issues/34#issuecomment-1465138574 | 2023.3.12 |  |
| Ryzen 7 3700X | Llama 7B（Q4） | 8.51 t/s | https://github.com/ggerganov/llama.cpp/issues/34#issuecomment-1465313724 | 2023.3.13 |  |
| 13900k | Llama 7B（Q4） | 14.02 t/s | https://github.com/ggerganov/llama.cpp/issues/34#issuecomment-1467067155 |  |  |
| 2x Intel Xeon Gold 5120 @ 2.20GHz | Llama 7B（Q4） | 8.68 t/s | https://github.com/ggerganov/llama.cpp/issues/34#issuecomment-1471171246 | 2023.3.16 |  |
| E5-2680v4 | Llama 7B（Q4） | 8.87 t/s | https://github.com/ggerganov/llama.cpp/issues/34#issuecomment-1517704976 | 2023.4.21 |  |
| i5 6500 | Llama 7B（Q4） | 13.82 t/s | https://github.com/ggerganov/llama.cpp/issues/34#issuecomment-1550410400 | 2023.5.17 |  |
| Hetzner Cloud Arm64 Ampere, 16 VCPU | Llama 7B（Q4） | 11.76 t/s | https://github.com/ggerganov/llama.cpp/issues/34#issuecomment-1575736794 | 2023.6.5 |  |
| 13900k | Llama 7B（Q4） | 12.65 t/s | https://github.com/ggerganov/llama.cpp/issues/34#issuecomment-1675971336 | 2023.8.12 |  |
| Snapdragon 870 / 8GB of ram | zephyr-7b (Q4) | 4.7 t/s | https://github.com/ggerganov/llama.cpp/issues/34#issuecomment-1825489115 | 2023.11.24 |  |
|  |  |  |  |  |  |
|  |  |  |  |  |  |

I ：推理每个 token花费的时间

T：通信时间