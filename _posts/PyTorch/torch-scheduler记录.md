---
title: torch中的scheduler
date: 2021-02-03 11:18:01
tags: [PyTorch, scheduler]
categories: [Note]
---

PyTorch学习率调整过程中版本问题引发的不同结果解析

<!-- more -->

代码

```python
import torch

optimizer1 = torch.optim.SGD([torch.randn(1, requires_grad=True)], lr=1e-3)
exp_lr_scheduler1 = torch.optim.lr_scheduler.MultiStepLR(optimizer1,
                                             milestones=[5, 10], gamma=0.1)

optimizer2 = torch.optim.SGD([torch.randn(1, requires_grad=True)], lr=1e-3)
exp_lr_scheduler2 = torch.optim.lr_scheduler.MultiStepLR(optimizer2,
                                             milestones=[5, 10], gamma=0.1)

for epoch in range(1, 15):
    exp_lr_scheduler1.step()
    exp_lr_scheduler2.step(epoch)
    print('Epoch {}, lr1 {}, lr2 {}'.format(epoch,
        optimizer1.param_groups[0]['lr'],
        optimizer2.param_groups[0]['lr']))
```

当torch版本为1.2.0时，输出如下：

```python
Epoch 1, lr1 0.001, lr2 0.001
Epoch 2, lr1 0.001, lr2 0.001
Epoch 3, lr1 0.001, lr2 0.001
Epoch 4, lr1 0.001, lr2 0.001
Epoch 5, lr1 0.0001, lr2 0.0001
Epoch 6, lr1 0.0001, lr2 0.0001
Epoch 7, lr1 0.0001, lr2 0.0001
Epoch 8, lr1 0.0001, lr2 0.0001
Epoch 9, lr1 0.0001, lr2 0.0001
Epoch 10, lr1 1.0000000000000003e-05, lr2 1.0000000000000003e-05
Epoch 11, lr1 1.0000000000000003e-05, lr2 1.0000000000000003e-05
Epoch 12, lr1 1.0000000000000003e-05, lr2 1.0000000000000003e-05
Epoch 13, lr1 1.0000000000000003e-05, lr2 1.0000000000000003e-05
Epoch 14, lr1 1.0000000000000003e-05, lr2 1.0000000000000003e-05
```

当torch版本为1.4.0时，输出如下：

```python
Epoch 1, lr1 0.001, lr2 1.0000000000000003e-05
Epoch 2, lr1 0.001, lr2 1.0000000000000003e-05
Epoch 3, lr1 0.001, lr2 1.0000000000000003e-05
Epoch 4, lr1 0.001, lr2 1.0000000000000003e-05
Epoch 5, lr1 0.0001, lr2 1.0000000000000003e-05
Epoch 6, lr1 0.0001, lr2 1.0000000000000003e-05
Epoch 7, lr1 0.0001, lr2 1.0000000000000003e-05
Epoch 8, lr1 0.0001, lr2 1.0000000000000003e-05
Epoch 9, lr1 0.0001, lr2 1.0000000000000003e-05
Epoch 10, lr1 1e-05, lr2 1.0000000000000003e-05
Epoch 11, lr1 1e-05, lr2 1.0000000000000003e-05
Epoch 12, lr1 1e-05, lr2 1.0000000000000003e-05
Epoch 13, lr1 1e-05, lr2 1.0000000000000003e-05
Epoch 14, lr1 1e-05, lr2 1.0000000000000003e-05
```

小结：`scheduler.step()`中的 `epoch`参数由于版本问题会带来不同的作用效果。1.2.0版本是判断epoch是否在某个区间内，而1.4.0版本是会直接调整学习到达最后一个区间。
