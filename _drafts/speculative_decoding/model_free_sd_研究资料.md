# Model-Free 投机解码 研究资料

## 一、基础背景：投机解码原理

### 标准投机解码（Leviathan et al., 2023; Chen et al., 2023）
- **核心思路**：小模型（draft model）先生成 γ 个候选 token，大模型（target model）一次并行验证
- **无损保证**：验证采用拒绝采样，输出分布与 target model 自回归等价
- **加速来源**：target model 的 KV cache 是内存带宽瓶颈，验证 γ 个 token 比生成 γ 个 token 快
- **关键公式**：期望接受 token 数 E[L] = (1 - α^{γ+1}) / (1 - α)，α 为平均接受率

### 为什么需要 model-free 方法
- 需要额外训练/维护一个 draft model（存储、部署、维护成本高）
- Draft model 与 target model 版本绑定（target model 更新，draft model 需重训）
- 小模型受限于参数量，接受率天花板明显
- **model-free 方法**：无需任何额外模型，直接利用 target model 自身信息生成草稿

---

## 二、方法分类

### 类别一：N-gram / 上下文查找（Lookup-Based）

#### 1. PLD（Prompt Lookup Decoding）
- **论文**：Saxena, 2023 (arXiv:2310.01558)，非正式技术报告
- **核心思路**：从 prompt 中直接查找与当前生成上下文匹配的 n-gram，作为草稿
- **适用场景**：摘要、RAG、代码补全（输出大量复制 prompt 内容的任务）
- **优点**：极简，无需额外模型或数据存储
- **缺点**：只能复用 prompt 内容，泛化性差

#### 2. ANPD（Adaptive N-gram Parallel Decoding）
- **论文**：Ou et al., 2024 (arXiv:2404.08698)
- **核心思路**：维护从已生成序列中构建的自适应 n-gram 查找表，从 1-gram 到 4-gram 按长度降序匹配
- **关键改进**：n-gram 来自已生成上下文（而非仅限 prompt），随生成过程动态更新
- **实验数据**（Multi-trajectory, DeepSeek-R1-Distill-Qwen-7B, AIME）：
  - 4 trajectories: T=45.52 (x1.71), A=1.89
  - 16 trajectories: T=47.06 (x1.77), A=1.96

#### 3. SuffixDecoding（Oliaro et al., 2024）
- **论文**：arXiv:2411.04975
- **核心思路**：用后缀自动机（suffix automaton）离线构建数据存储，在推理时高效检索匹配的历史 n-gram
- **特点**：适合 batch inference，可利用其他样本的历史序列

#### 4. Token Recycle（Luo et al., 2024）
- **论文**：arXiv:2408.08696 / arXiv:2408.01170
- **核心思路**：在 target model 推理时顺便保存 top-k token ID（保留概率信息），下次遇到相同 context 时直接复用
- **关键区别**：存储 top-k token IDs（部分概率信息），而非完整 logit 分布
- **实验数据**（STAND 论文对比，Multi-trajectory 7B AIME）：
  - 4 traj: T=61.38 (x2.30), A=2.76
  - 16 traj: T=60.86 (x2.29), A=2.77

#### 5. SAM Decoding（Hu et al., 2024）
- **论文**：arXiv:2411.10666
- **核心思路**：Speculative decoding via suffix automaton，用后缀自动机动态匹配已生成序列的重复模式
- **实验数据**（STAND 对比，7B AIME）：
  - 4 traj: T=44.35 (x1.67), A=1.81

### 类别二：Jacobi 迭代解码（Parallel Decoding）

#### 6. Lookahead Decoding（Fu et al., 2024）
- **论文**：arXiv:2402.02057，ICML 2024
- **核心思路**：
  - 维护一个"lookahead branch"：并行猜测多个位置的 token（n-step lookahead）
  - 维护一个"verification branch"：验证之前猜测是否形成 n-gram
  - 两条分支交替更新
- **本质**：Jacobi 迭代的工程实现，将猜测-验证解耦为两个并行 branch
- **加速效果**：
  - MT-bench: ~1.8x，代码生成（多 GPU）: ~4x
  - 无需训练，无需数据存储

#### 7. Jacobi Decoding（基础理论）
- **论文**：Song et al., 2021 / Santilli et al., 2023 (arXiv:2305.10427)
- **核心思路**：将自回归解码视为联立方程组求解，用 Jacobi 迭代并行更新所有位置
- **问题**：随机采样下收敛性不保证，需要额外处理

### 类别三：多头/自草稿（Self-Drafting with Extra Heads）

#### 8. Medusa（Cai et al., 2024）
- **论文**：arXiv:2401.10774，ICML 2024
- **核心思路**：在 LLM 最后一层附加多个额外解码头，每个头预测未来第 k 步的 token
  - Medusa-1：冻结主干，只训练额外头
  - Medusa-2：联合微调
- **树注意力**：多个头的预测组合成候选序列树，并行验证
- **加速效果**：Vicuna-7B 约 2.2-2.8x

#### 9. EAGLE-2 作为对比基准（model-based）
- **论文**：arXiv:2406.16858
- 在 STAND 实验中作为唯一 model-based baseline
- **实验数据**（STAND 对比，7B AIME）：
  - 4 traj: T=29.91 (x1.12), A=2.21（注意：长上下文下接受长度下降）

### 类别四：基于检索的高级方法（Retrieval-Based）

#### 10. REST（He et al., 2023）
- **论文**：arXiv:2311.08252，NAACL 2024
- **核心思路**：维护离线数据存储（datastore），推理时检索相似上下文对应的历史续写作为草稿
- **加速效果**：7B ~1.62x，13B ~2.36x（HumanEval/MT-bench）

#### 11. DOUBLE（Shen et al., 2026）
- **论文**：arXiv:2601.05524（2026年4月）
- **核心创新**：突破 PSD 理论加速上限 C
- **两个关键机制**：
  1. **迭代检索起草**：draft model 执行 γ 次检索迭代，每次更新上下文，生成 > C 个 token，突破速度上限
  2. **Target-Guided Verification**：target model 做单步检索，生成 multi-token guidance，减少 mid-sequence rejection
- **理论证明**：PSD 的理论加速上限 S_PSD ≤ C（speed ratio），DOUBLE 通过多轮检索突破此限
- **实验数据**：
  - LLaMA3.3-70B: **5.3×**（超越 EAGLE-3 训练方法）
  - Qwen3-32B: **2.8×**
  - 完全无需训练、无损

---

## 三、STAND 论文核心（主要分析对象）

### 论文信息
- **标题**：Accelerated Test-Time Scaling with Model-Free Speculative Sampling
- **arXiv**：2506.04708v2，2025年11月
- **机构**：KAIST, Amazon AGI, AirSignal
- **作者**：Woomin Song, Saket Dingliwal, Sai Muralidhar Jayanthi 等

### 核心动机
**推理模型（LRM）的 Token 重复性**：
- 在 AIME-2024 上用 DeepSeek-R1-Distill-Qwen-7B 分析：
  - 2-gram 重叠率：16 条轨迹时高达 **~97%**
  - 4-gram 重叠率：16 条轨迹时约 **~80%**
  - 即使只有 2 条轨迹，bigram 重叠率超过 **90%**
- **结论**：LRM 推理轨迹间存在高度冗余，可直接复用历史 n-gram 作为草稿

**随机 vs 确定性起草**：
- LRM 使用采样（temperature > 0）生成多样化轨迹，而现有 model-free 方法（PLD、ANPD、SAM、Saxena 2023）均为确定性起草（greedy n-gram 查找）
- 实验证明随机起草比确定性起草的接受率高 **5-8%**（AIME/GPQA/LCB 三个任务）

### STAND 三大创新

**1. Logit-based N-gram Module（基于 Logit 的 N-gram 模块）**
- 传统方法存储 token ID → 强制确定性起草
- STAND 存储 **logit 分布**（top-10 概率最高的 token 及其概率）
- 遇到相同 n-gram 时，加权平均合并分布（第 k 次出现权重 k/(k+1)，新分布权重 1/(k+1)）
- 内存效率：仅保留 top-10 token，常数内存开销

**2. Gumbel-Top-K 并行采样**
- 问题：从存储的 logit 分布中采样 k 个不重复 token（用于 tree node 扩展），传统方式需顺序采样
- 解决：Gumbel-Top-K trick（Kool et al., 2019）：对 log-prob 加 Gumbel 噪声，取 top-k，等价于无放回采样
- 优化：预计算并缓存 Gumbel 噪声，耗尽后刷新，消除实时采样开销

**3. 数据驱动的草稿树优化（Data-Driven Draft Tree Optimization）**
- 背景：tree-based SD 需要选择树结构（哪些节点展开、展开几个子节点）
- 传统方式：启发式规则（如 Token Recycle 的固定树）
- STAND 方法：
  1. 初始化一个大树（625 节点，深度 20）
  2. 在 30 条真实数据上执行 SD，追踪每个节点的实际接受率
  3. 选取接受率最高的 80 个节点重组为紧凑树
- **结果**：优化树比启发式树在 AIME 和 GPQA（OOD）上均更好，且具有泛化性
- **树结构特征**：STAND 优化树最深达 13 层（vs Token Recycle 的 7 层），尾部有长确定性路径

### 实验结果

**Multi-trajectory 解码（主要评估场景，A100 GPU）**

DeepSeek-R1-Distill-Qwen-7B：

| 方法 | 4 traj AIME T(x) | 8 traj Avg T | 16 traj Avg T | Avg A (16 traj) |
|------|-----------------|--------------|----------------|-----------------|
| Plain | 26.63 (1x) | 28.57 | 28.57 | - |
| Eagle-2 | 29.91 (x1.12) | 29.74 (x1.04) | 29.74 (x1.04) | 2.11 |
| PLD | 43.93 (x1.65) | 47.69 (x1.67) | 48.70 (x1.70) | 3.33（注：A此处是avg acc length） |
| ANPD | 45.52 (x1.71) | 51.08 (x1.79) | 52.04 (x1.82) | 2.01 |
| SAM | 44.35 (x1.67) | 49.45 (x1.73) | 51.36 (x1.80) | 1.98 |
| Recycle | 61.38 (x2.30) | 64.27 (x2.27) | 64.48 (x2.26) | 2.75 |
| SAM+Recycle | 62.20 (x2.24) | 64.68 (x2.26) | 64.62 (x2.26) | 2.69 |
| **STAND** | **64.99 (x2.44)** | **75.24 (x2.63)** | **78.15 (x2.74)** | **3.67** |

**关键洞察**：
- Token Recycle 性能不随轨迹数增加而改善（lookup 表 replace 策略无法聚合历史信息）
- STAND 随轨迹数增加优势扩大（4 traj 领先 Recycle 6%，16 traj 领先 21%）

**Single-trajectory 解码**（7B）：
- STAND: avg T=67.86 (x2.38), A=3.04
- 比 Recycle 高出约 7% throughput

**Batch Decoding**（batch=4, 7B）：
- STAND: avg T=128.10 (x1.42), A=2.63
- Recycle: avg T=92.82 (x1.03)（batch 下 Recycle 退化！）

**Test-time Tree Search (DVTS)**（7B）：
- STAND: avg T=83.51 (x2.54), A=3.59
- Recycle: avg T=70.54 (x2.14)

**消融实验**：
- Deterministic → Stochastic：acc length 2.94 → 3.24（+10.2%）
- Stochastic → +Gumbel-Top-K：acc length 3.24 → 3.30，但 throughput 大幅提升（+6.5%）
- Heuristic tree → Optimized tree：AIME throughput 59.96 → 64.99，GPQA (OOD) 77.32 → 83.47

---

## 四、关键对比维度

### Training-Free vs Training-Required
| 类型 | 代表方法 | 特点 |
|------|---------|------|
| Training-free（本文重点） | PLD, ANPD, Lookahead, REST, STAND, Token Recycle, SAM, DOUBLE | 即插即用，适应任意 LLM |
| Training-required（model-based） | EAGLE, Medusa, Hydra | 接受率更高，但需绑定模型 |

### N-gram 方法进化脉络
```
PLD (2023) → 只查 prompt
  ↓
ANPD (2024) → 查已生成上下文，动态 n-gram 表
  ↓
Token Recycle (2024) → 存 top-k token ID（确定性）
  ↓
SAM Decoding (2024) → 后缀自动机，更高效匹配
  ↓
STAND (2025) → 存 logit 分布（随机起草）+ 数据驱动树优化
  ↓
DOUBLE (2026) → 双侧检索（draft+target），突破理论上限
```

### 为什么 STAND 在多轨迹场景下优势更大
- LRM 生成的多条推理轨迹共享大量 n-gram（97% bigram overlap）
- STAND 的 logit-based N-gram 聚合：同一 n-gram 多次出现时合并分布，越来越准确
- Token Recycle replace 策略：新出现的 n-gram 直接覆盖旧值，丢失历史信息
- **规模化效应**：轨迹越多，STAND 的草稿概率分布越精确，接受率越高

---

## 五、与 Test-Time Scaling 的关系

2025 年的重要背景：LRM（Large Reasoning Model）通过增加推理时计算（test-time compute）提升准确率
- Best-of-N：生成 N 条独立轨迹取最好的
- DVTS（Diverse Verifier Tree Search）：树搜索
- 以上都需要生成**多条**长推理链 → 与 model-free SD 结合天然契合

STAND 的定位：专为 LRM 的多轨迹推理设计的 model-free SD 方法
- 精确利用推理轨迹间 n-gram 冗余
- 随轨迹数增多，性能单调提升
- 不改变输出分布（lossless）

---

## 六、论文引用的重要参考文献

### 原始投机解码
- Leviathan et al., 2023 (ICML): Fast inference via speculative decoding
- Chen et al., 2023a: Accelerating large language model decoding with speculative sampling

### Model-free 方法
- Saxena, 2023: Prompt lookup decoding (PLD)
- Ou et al., 2024 (arXiv:2404.08698): ANPD
- He et al., 2023 (arXiv:2311.08252): REST
- Luo et al., 2024 (arXiv:2408.08696): Token Recycling
- Hu et al., 2024 (arXiv:2411.10666): SAM Decoding
- Oliaro et al., 2024 (arXiv:2411.04975): SuffixDecoding
- Fu et al., 2024 (arXiv:2402.02057): Lookahead Decoding

### Model-based 方法（对比）
- Cai et al., 2024 (arXiv:2401.10774): Medusa
- Li et al., 2024c/d (arXiv:2401.15077, 2406.16858): EAGLE, EAGLE-2
- Li et al., 2024 (EAGLE-3): 多层特征融合

### 2026年最新方法
- Shen et al., 2026 (arXiv:2601.05524): DOUBLE，突破 PSD 理论上限

### Test-time Scaling
- Snell et al., 2024: Scaling LLM test-time compute optimally
- Wang et al., 2022: Self-consistency
- Beeching et al., 2024: DVTS (Diverse Verifier Tree Search)
