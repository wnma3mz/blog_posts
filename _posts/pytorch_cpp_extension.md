---
title: 用 Cpp 写 PyTorch 的插件
date: 2023/12/03 19:42:25
tags: [PyTorch, Cpp, CUDA]
categories: [DeepLearning]
mathjax: true
---

从零开始，用 Cpp 写 PyTorch 的插件，包括 CPU 和 GPU 的版本。

<!-- more -->

## 为什么

一般来说，在原生功能不能满足需求的时候，插件可以作为补充。比如，PyTorch 的 `torch.nn.functional` 中没有 `softmax` 函数，但是 `torch.nn` 中有，所以可以用 `torch.nn.functional.softmax` 来代替。但是，如果要用 `softmax` 的导数，就需要用到 `softmax` 的原始定义，这个时候就需要自己写插件了。

如果是比较简单的需求，则可以直接用 Python 完成。然而，当对性能要求较高时，往往会使用 Cpp 来写插件，最后甚至会优化为 CUDA 代码。

## 怎么写

从例子出发，一步步来写。假设要实现一个最简单的 Attention 模块，输入为 $q,k,v \in \mathbb{R}^{M \times N}$，输出为 $out \in \mathbb{R}^{M \times N}$（不考虑 Batch Size 以及 Head 数量的情况）。Attention 模块的计算公式为：

$$
out = \text{softmax}(qk^T)v
$$

只实现 `forward` 函数，不实现 `backward` 函数。

### CPU 版本

类似于写 Python 的库，创建一个文件夹，目录结构如下所示：

```bash
attention
├── attention.cpp
├── setup.py
└── __init__.py
```
#### Python 部分

核心代码在 `attention.cpp` 中 ，首先在 `setup.py` 中添加如下代码：

```python
from setuptools import setup
from torch.utils.cpp_extension import BuildExtension, CppExtension

setup(
    name='attention',
    ext_modules=[
        CppExtension('attention', ['attention.cpp']),
    ],
    cmdclass={
        'build_ext': BuildExtension
    })
```

这样，就可以用 `python setup.py install` 来安装插件了。或者用`pip install -e attention` 以便于快速调试

在 `__init__.py` 中导入 `attention` 模块，以在 Python 中调用 `forward` 函数，直接计算 attention。

```python
from .attention import forward
```

#### Cpp 部分

接下来，我们需要在 `attention.cpp` 中实现 `forward` 函数，为进行区分，这里使用这个函数名称`attention_forward` ，这个函数的输入是 q、k、v 三个`Tensor`，输出是 `torch::Tensor`。而具体的计算步骤可以拆解为三个步骤：

1. 矩阵的乘法
2. softmax
3. 矩阵的乘法

使用`PYBIND11_MODULE`把`attention_forward`函数暴露出去，绑定到`forward`上，这样就能用`forward`函数来调用`attention_forward`。整合后的完整代码如下，

```cpp
#include <torch/extension.h>
#include <vector>

// 参数：queries(Q)，keys(K)，values(V)
// 返回：方便起见，返回一个 vector，实际上只有一个元素，便于后续扩展
std::vector<torch::Tensor> attention_forward(
    torch::Tensor& q,
    torch::Tensor& k,
    torch::Tensor& v) {

    torch::Tensor scores = torch::matmul(q, k.transpose(0, 1));
    scores = torch::softmax(scores, 1);
    torch::Tensor attention = torch::matmul(scores, v);
    return {attention};
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("forward", &attention_forward, "Attention forward (CPU)");
}
```

#### 测试

用 Python 版本的实现来测试 Cpp 版本的实现是否正确，测试代码如下，

```python
import torch

import attention
import timeit

torch.manual_seed(42)

def py_attention(q, k, v):
    return torch.softmax(q @ k.T, dim=1) @ v

def check_forward(q, k, v):
    baseline_values = py_attention(q, k, v)
    cpp_values = attention.forward(q, k, v)[-1]

    print("base o", baseline_values)
    print("cpp  o", cpp_values)
    print(torch.all(torch.isclose(baseline_values, cpp_values)))

def compare_time(q, k, v, loop=100):
    print("py", timeit.timeit(lambda: py_attention(q, k, v), number=loop))
    print("cpp", timeit.timeit(lambda: attention.forward(q, k, v), number=loop))

if __name__ == "__main__":
    m, n = 2, 4
    device = "cuda" if torch.cuda.is_available() else "cpu"
    q, k, v = torch.rand(size=(m, n), device=device), torch.rand(size=(m, n), device=device), torch.rand(size=(m, n), device=device)
    print("q", q)
    print("k", k)
    print("v", v)
    print("="*20)
    check_forward(q, k, v)
    compare_time(q, k, v)
```

测试通过后，可以再使用 `compare_time` 函数对比一下二者的速度。理论上，二者的速度是相差无几的。因为均用的是 PyTorch 的矩阵乘法和 softmax 函数。

但是，如果需要进行更进一步的优化技巧，那么就需要自己实现矩阵乘法和 softmax 函数了。这里，我们只实现最简单的矩阵乘法和 softmax 函数，然后再对比一下二者的速度。

#### 矩阵乘法

```cpp
torch::Tensor my_matmul(const torch::Tensor &a, const torch::Tensor &b) {
    TORCH_CHECK(a.dim() == 2 && b.dim() == 2, "Input tensors must be 2-dimensional");
    TORCH_CHECK(a.size(1) == b.size(0), "Dimensions mismatch");

    auto m = a.size(0);
    auto n = b.size(1);
    auto p = a.size(1);

    torch::Tensor result = torch::zeros({m, n}, torch::dtype(torch::kFloat32));
    for (int i = 0; i < m; i++) {
        for (int j = 0; j < n; j++) {
            float sum = 0.0;
            for (int k = 0; k < p; k++) {
                sum += a[i][k].item<float>() * b[k][j].item<float>();
            }
            result[i][j] = sum;
        }
    }
    return result;
}
```

#### softmax

由于`softmax`函数比较特殊，后续会结合算子融合一起优化，所以简单的对其进行展开。用`torch::exp`和`torch::sum`实现了一遍，为了方便也可以直接使用`torch::softmax(scores, 1)`

```cpp
torch::Tensor my_softmax(const torch::Tensor& scores) {
    torch::Tensor exponents = torch::exp(scores);
    torch::Tensor sum = torch::sum(exponents, 1, true);
    return exponents / sum;
}
```

但把这两个函数替换到`attention_forward`函数中后，再次运行`compare_time`函数，发现手写的 Cpp 版本的实现要比 Python 版本的实现慢。为什么？因为，当前只是简单的实现了矩阵乘法和 softmax，而 PyTorch 中的矩阵乘法和 softmax 都是经过优化的，所以速度会更快。

另外，使用原生的矩阵乘法和 softmax 函数，可以在 GPU 上运行，而手写的矩阵乘法和 softmax 函数，只能在 CPU 上运行。因此，接下来将其改造为 GPU 版本，然后再进行优化。

### GPU 版本

#### Python 部分

在 `setup.py` 中更改为如下代码，把 `CppExtension` 改为 `CUDAExtension`

```python
from setuptools import setup
from torch.utils.cpp_extension import BuildExtension, CUDAExtension 

setup(
    name='attention',
    ext_modules=[
        CUDAExtension('attention', [
            'attention.cpp',
            'attention_kernel.cu',
        ])        
    ],
    cmdclass={
        'build_ext': BuildExtension
    })
```

#### Cpp 部分

为了兼容之前的代码，这里将之前的`attention_forward`更新为`attention_cpu_forward`，同时加了一个类型判断，如果输入的`Tensor`不在同一个设备上，则抛出异常。而对于`attention_cuda_forward`的实现需要在`attention_kernel.cu`中实现。注意：这里需要提前定义好`attention_cuda_forward`函数，否则会报错。


```cpp
#include <torch/extension.h>

#include <vector>

std::vector<torch::Tensor> attention_cuda_forward(
    torch::Tensor q,
    torch::Tensor k,
    torch::Tensor v);

torch::Tensor my_matmul(const torch::Tensor &a, const torch::Tensor &b) {
    TORCH_CHECK(a.dim() == 2 && b.dim() == 2, "Input tensors must be 2-dimensional");
    TORCH_CHECK(a.size(1) == b.size(0), "Dimensions mismatch");

    auto m = a.size(0);
    auto n = b.size(1);
    auto p = a.size(1);

    torch::Tensor result = torch::zeros({m, n}, torch::dtype(torch::kFloat32));
    for (int i = 0; i < m; i++) {
        for (int j = 0; j < n; j++) {
            float sum = 0.0;
            for (int k = 0; k < p; k++) {
                sum += a[i][k].item<float>() * b[k][j].item<float>();
            }
            result[i][j] = sum;
        }
    }
    return result;
}

std::vector<torch::Tensor> attention_cpu_forward(
    torch::Tensor q,
    torch::Tensor k,
    torch::Tensor v) {
    torch::Tensor scores = my_matmul(q, k);
    torch::Tensor attention = my_matmul(torch::softmax(scores, 1), v);
    return {scores, attention};
}

// 参数：queries(Q)，keys(K)，values(V)
std::vector<torch::Tensor> attention_forward(
    torch::Tensor &q,
    torch::Tensor &k,
    torch::Tensor &v) {
    if (!(q.device().type() == k.device().type() && q.device().type() == v.device().type())) {
        throw std::runtime_error("Input tensors q, k, and v must be on the same device");
    }

    if (q.is_cuda()) {
        return attention_cuda_forward(q, k.transpose(0, 1), v);
    } else {
        return attention_cpu_forward(q, k.transpose(0, 1), v);
    }
}


PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("forward", &attention_forward, "Attention forward (CUDA)");
}
```

#### Cu 部分

在同级目录下，创建`attention_kernel.cu`文件。首先实现`attention_cuda_forward`函数的主要逻辑。其中，主要对矩阵乘法进行了优化，使用了 CUDA 的并行计算。然后，使用`AT_DISPATCH_FLOATING_TYPES`宏来实现对不同类型的支持，这样就可以支持`float`和`double`类型了。

对于`matrix_multiply`函数，与一般写法不同的是，需要提前创建好输出的`Tensor`，然后再传入到 CUDA 的函数中。并且需要创建好`blocks`和`threads`，然后再调用 CUDA 的函数。这里指定了每个 CUDA 是有 16 x 16 线程的块，而这些块是可以并行计算的，所以能够加速计算。可参考 [An Even Easier Introduction to CUDA](https://devblogs.nvidia.com/even-easier-introduction-cuda)

而在传递参数的时候，需要使用`packed_accessor`。这里的`packed_accessor`的第一个参数是`Tensor`的类型，第二个参数是`Tensor`的维度，第三个参数是`Tensor`的类型，第四个参数是`Tensor`的维度。这里的`packed_accessor`的第三个参数和第四个参数，是为了支持 CUDA 的。

接下来就是实现`matrix_multiply_kernel`。矩阵的乘法中，如果要计算输出矩阵的第一个值，则需要用到输入矩阵的第一行和第一列。因此，这里需要根据`block`和`thread`的索引，来计算出对应的行和列。然后，就是普通的矩阵乘法的实现了。原来的矩阵乘法的实现是：

```python
out = [[0 for _ in range(n)] for _ in range(m)]
for i in range(m):
    for j in range(n):
        for k in range(p):
            out[i][j] += input1[i][k] * input2[k][j]
```

相当于把外面两个循环分别交给了 CUDA 的`block`和`thread`来计算。这样，就可以实现并行计算了。

```cpp
#include <torch/extension.h>

#include <cuda.h>
#include <cuda_runtime.h>

#include <vector>

// Matrix multiply kernel
template <typename scalar_t>
__global__ void matrix_multiply_kernel(const torch::PackedTensorAccessor<scalar_t,2,torch::RestrictPtrTraits,size_t> input1,
                                       const torch::PackedTensorAccessor<scalar_t,2,torch::RestrictPtrTraits,size_t> input2,
                                       torch::PackedTensorAccessor<scalar_t,2,torch::RestrictPtrTraits,size_t> output) {
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;

    if (row < input1.size(0) && col < input2.size(1)) {
        scalar_t value = 0.0;
        for (int k = 0; k < input1.size(1); ++k) {
            value += input1[row][k] * input2[k][col];
        }
        output[row][col] = value;
    }
}

torch::Tensor matrix_multiply(torch::Tensor input1, torch::Tensor input2) {
    int rows1 = input1.size(0);
    int cols1 = input1.size(1);
    int cols2 = input2.size(1);

    auto options = torch::TensorOptions().device(input1.device());
    torch::Tensor output = torch::zeros({rows1, cols2}, options);

    const dim3 threads(16, 16);
    const dim3 blocks((cols2 + threads.x - 1) / threads.x,
                      (rows1 + threads.y - 1) / threads.y);

    AT_DISPATCH_FLOATING_TYPES(input1.scalar_type(), "matrix_multiply_kernel", ([&] {
        matrix_multiply_kernel<<<blocks, threads>>>(
            input1.packed_accessor<scalar_t,2,torch::RestrictPtrTraits,size_t>(),
            input2.packed_accessor<scalar_t,2,torch::RestrictPtrTraits,size_t>(),
            output.packed_accessor<scalar_t,2,torch::RestrictPtrTraits,size_t>());
    }));

    return output;
}


std::vector<torch::Tensor> attention_cuda_forward(
    torch::Tensor q,
    torch::Tensor k,
    torch::Tensor v) {

    torch::Tensor scores = matrix_multiply(q, k);
    torch::Tensor attention = matrix_multiply(torch::softmax(scores, 1), v);
    return {scores, attention};
}

```

测试代码不变，依旧可以用上面的来进行检验。至此，最简单的实验就完成了。接下来，就是对其进行优化了。

## 矩阵乘法的优化



## 算子融合



### 参考


[1] [https://pytorch.org/tutorials/advanced/cpp_extension.html](https://pytorch.org/tutorials/advanced/cpp_extension.html)。