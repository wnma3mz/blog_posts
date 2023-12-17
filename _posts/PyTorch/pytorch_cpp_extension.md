---
title: 用 Cpp 写 PyTorch 的插件
date: 2023/12/03 19:42:25
tags: [PyTorch, Cpp, CUDA]
categories: [DeepLearning]
mathjax: true
---
从零开始，用 Cpp 写 PyTorch 的插件，包括 CPU 和 GPU 的版本。

[代码](https://github.com/wnma3mz/pytorch_cuda_extension)

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

这样，就可以用 `python setup.py install` 来安装插件了。或者用 `pip install -e attention` 以便于快速调试

在 `__init__.py` 中导入 `attention` 模块，以在 Python 中调用 `forward` 函数，直接计算 attention。

```python
from .attention import forward
```

#### Cpp 部分

接下来，我们需要在 `attention.cpp` 中实现 `forward` 函数，为进行区分，这里使用这个函数名称 `attention_forward` ，这个函数的输入是 q、k、v 三个 `Tensor`，输出是 `torch::Tensor`。而具体的计算步骤可以拆解为三个步骤：

1. 矩阵的乘法
2. softmax
3. 矩阵的乘法

使用 `PYBIND11_MODULE`把 `attention_forward`函数暴露出去，绑定到 `forward`上，这样就能用 `forward`函数来调用 `attention_forward`。整合后的完整代码如下，

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

由于 `softmax`函数比较特殊，后续会结合算子融合一起优化，所以简单的对其进行展开。用 `torch::exp`和 `torch::sum`实现了一遍，为了方便也可以直接使用 `torch::softmax(scores, 1)`

```cpp
torch::Tensor my_softmax(const torch::Tensor& scores) {
    torch::Tensor exponents = torch::exp(scores);
    torch::Tensor sum = torch::sum(exponents, 1, true);
    return exponents / sum;
}
```

但把这两个函数替换到 `attention_forward`函数中后，再次运行 `compare_time`函数，发现手写的 Cpp 版本的实现要比 Python 版本的实现慢。为什么？因为，当前只是简单的实现了矩阵乘法和 softmax，而 PyTorch 中的矩阵乘法和 softmax 都是经过优化的，所以速度会更快。

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

为了兼容之前的代码，这里将之前的 `attention_forward`更新为 `attention_cpu_forward`，同时加了一个类型判断，如果输入的 `Tensor`不在同一个设备上，则抛出异常。而对于 `attention_cuda_forward`的实现需要在 `attention_kernel.cu`中实现。注意：这里需要提前定义好 `attention_cuda_forward`函数，否则会报错。

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

在同级目录下，创建 `attention_kernel.cu`文件。首先实现 `attention_cuda_forward`函数的主要逻辑。其中，主要对矩阵乘法进行了优化，使用了 CUDA 的并行计算。然后，使用 `AT_DISPATCH_FLOATING_TYPES`宏来实现对不同类型的支持，这样就可以支持 `float`和 `double`类型了。

对于 `matrix_multiply`函数，与一般写法不同的是，需要提前创建好输出的 `Tensor`，然后再传入到 CUDA 的函数中。并且需要创建好 `blocks`和 `threads`，然后再调用 CUDA 的函数。这里指定了每个 CUDA 是有 16 x 16 线程的块，而这些块是可以并行计算的，所以能够加速计算。可参考 [An Even Easier Introduction to CUDA](https://devblogs.nvidia.com/even-easier-introduction-cuda)

而在传递参数的时候，需要使用 `packed_accessor`。这里的 `packed_accessor`的第一个参数是 `Tensor`的类型，第二个参数是 `Tensor`的维度，第三个参数是 `Tensor`的类型，第四个参数是 `Tensor`的维度。这里的 `packed_accessor`的第三个参数和第四个参数，是为了支持 CUDA 的。

接下来就是实现 `matrix_multiply_kernel`。矩阵的乘法中，如果要计算输出矩阵的第一个值，则需要用到输入矩阵的第一行和第一列。因此，这里需要根据 `block`和 `thread`的索引，来计算出对应的行和列。然后，就是普通的矩阵乘法的实现了。原来的矩阵乘法的实现是：

```python
out = [[0 for _ in range(n)] for _ in range(m)]
for i in range(m):
    for j in range(n):
        for k in range(p):
            out[i][j] += input1[i][k] * input2[k][j]
```

相当于把外面两个循环分别交给了 CUDA 的 `block`和 `thread`来计算。这样，就可以实现并行计算了。

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

### Matmul

重新回到矩阵乘法上，假设有两个矩阵 A1 和 A2，形状分别为 $M \times N$ 和 $N \times K$，则矩阵乘法的计算公式为：

```python
import torch

M, N, K = 4, 2, 4

A1 = torch.rand(size=(M, N))
A2 = torch.rand(size=(N, K))

output = torch.zeros(size=(M, K))
for i in range(M):
    for j in range(K):
        sum_ = 0
        for k in range(N):
            sum_ += A1[i][k] * A2[k][j]
        output[i][j] = sum_
```

一种朴素的优化手段是把最后一个循环并行计算。

```python
for i in range(M):
    for j in range(K):
        output[i][j] = sum(map(lambda x: x[0] * x[1], zip(A1[i], A2[:, j])))
```

利用多线程/进程（下文统称为job）进行并行计算可以提高程序的计算速度，但这样需要每个job都能访问到 A1 和 A2 的数据，所以这就引入了全局内存和共享内存的概念。

> - 全局内存（Global Memory）：全局内存是一种在计算机程序中可被所有线程或进程访问的内存空间。它通常用于存储全局变量、静态变量以及动态分配的内存等。全局内存的特点是可以在整个程序执行过程中进行读写操作，但它的访问速度相对较慢。
> - 共享内存（Shared Memory）：共享内存是一种特殊的内存区域，被多个线程或进程同时访问和共享。通过将数据存储在共享内存中，不同的线程或进程可以直接读取和写入这些数据，而无需使用其他的通信机制。共享内存的特点是高效的数据共享和访问速度，因为不需要进行复制或传输数据。

CUDA 中依旧存在类似的概念

> - 全局内存（Global Memory）：在 CUDA 中，全局内存是一个设备（GPU）上可见的主机（CPU）内存空间。它可以由所有的线程块和线程访问，用于存储全局变量和动态分配的内存等。全局内存的读写操作相对较慢，因为涉及主机与设备之间的数据传输。
> - 共享内存（Shared Memory）：在 CUDA 中，共享内存是位于每个线程块中的一块高速缓存内存。它被同一个线程块内的线程共享，并且比全局内存具有更快的读写速度。共享内存通常用于优化算法的性能，通过在线程块内部共享数据来减少全局内存的访问。

简而言之，全局内存可以很方便的存各种东西，但是速度慢；共享内存是一个好东西，速度块，但通常大小受限。所以，分别有两种优化：

- 全局内存中，考虑如何加速访问
- 共享内存中，考虑如何减小占用空间

这就引入了 Tiled matmul 算法。

### Tiled Matmul

- 对于加速访问，在无法控制硬件的前提下，只能通过并行的方式同时读取数据。
- 对于减小占用空间，可以通过拆分矩阵，把大矩阵拆分成若干小矩阵，然后再进行计算。

重新思考矩阵 `output`的计算过程，每个元素的计算其实是独立的，其本质可以拆成若干独立的小块，如下图所示：

![](https://raw.githubusercontent.com/wnma3mz/blog_posts/master/imgs/pytorch_cuda_ext/setup-2d0f22ecd2e9b7c84af56792d14ba18a.gif?raw=true)
From https://penny-xu.github.io/blog/tiled-matrix-multiplication

由于矩阵 `output`每个元素是完全独立的，可以将其拆成若干个小矩阵来计算。如上图所示，把矩阵 `output` 拆成了 4 个小矩阵。对应的代码如下所示：

![](https://raw.githubusercontent.com/wnma3mz/blog_posts/master/imgs/pytorch_cuda_ext/block_matrix.png)

```python
import torch

M, N, K = 4, 2, 4

A1 = torch.rand(size=(M, N))
A2 = torch.rand(size=(N, K))

block_size = 2 # 方便起见，这里设置为 N,M,K 的公约数。同时也拆成了 2x2 的 block
block_M, block_N, block_K = M // block_size, N // block_size, K // block_size

def matmul(sub_A1, sub_A2):
    output = torch.zeros(size=(sub_A1.shape[0], sub_A2.shape[1]))
    for i in range(sub_A1.shape[0]):
        for j in range(sub_A2.shape[1]):
            for k in range(sub_A2.shape[0]):
                output[i][j] += sub_A1[i][k] * sub_A2[k][j]
    return output

output11 = matmul(A1[:block_M, :], A2[:, :block_K])
output12 = matmul(A1[:block_M, :], A2[:, block_K:])
output21 = matmul(A1[block_M:, :], A2[:, :block_K])
output22 = matmul(A1[block_M:, :], A2[:, block_K:])
output = torch.cat([torch.cat([output11, output12], dim=1), torch.cat([output21, output22], dim=1)], dim=0)
print(output)
print(A1 @ A2)
assert torch.allclose(output, A1 @ A2)
```

对于，左上角矩阵 `output11`，实际上是由 `block_size`个矩阵乘法，再求和得到的 `output11 = matmul(A1[:block_M, :block_N], A2[:block_N, :block_K]) + matmul(A1[:block_M, block_N:], A2[block_N:, :block_K])`。再把它扩展的灵活一点

1. 不局限于只能扩展为 2 $\times$ 2 矩阵
2. block_size 可以针对 M, N, K 进行调整

```python
import torch

M, N, K = 4, 6, 8

A1 = torch.rand(size=(M, N))
A2 = torch.rand(size=(N, K))

block_size_M, block_size_N, block_size_K = 2, 3, 4
block_M, block_N, block_K = M // block_size_M, N // block_size_N, K // block_size_K

def block_matmul(sub_A1, sub_A2):
    output = torch.zeros(size=(sub_A1.shape[0], sub_A2.shape[1]))
    for i in range(sub_A1.shape[0]):
        for j in range(sub_A2.shape[1]):
            for k in range(sub_A2.shape[0]):
                output[i][j] += sub_A1[i][k] * sub_A2[k][j]
    return output

def matmul(A1, A2):
    output = torch.zeros(size=(A1.shape[0], A2.shape[1]))
    for i in range(0, A1.shape[0], block_M):
        start_i, end_i = i, i + block_M
        for j in range(0, A2.shape[1], block_N):
            start_j, end_j = j, j + block_N
            for k in range(0, A2.shape[0], block_K):
                start_k, end_k = k, k + block_K
                # 计算每个 block 的矩阵乘法
                sub_A1 = A1[start_i:end_i, start_k:end_k]
                sub_A2 = A2[start_k:end_k, start_j:end_j]
                # 把每个 block 的结果放到对应的位置
                output[start_i:end_i, start_j:end_j] += block_matmul(sub_A1, sub_A2)
    return output
print(matmul(A1, A2))
print(A1 @ A2)
assert torch.allclose(matmul(A1, A2), A1 @ A2)
```

![](https://raw.githubusercontent.com/wnma3mz/blog_posts/master/imgs/pytorch_cuda_ext/tmm-59dd890f48435e692c47919d0df4a5e6.gif)
From https://penny-xu.github.io/blog/tiled-matrix-multiplication

再次回顾这样做的目的，[原作者](https://penny-xu.github.io/blog/tiled-matrix-multiplication)是这么说的：

> - With or without tiling, the same number of accesses into global memory occur. The difference is that, without tiling, each thread must sequentially (one after the other) access global memory 8 times.
> - With tiling, we can parallelize the access to global memory so that each thread only sequentially accesses global memory 4 times.
> - To summarize, the point is not to reduce the number of multiplications or even the total number of global memory accesses, but rather to reduce the number of sequential global memory accesses per thread. In other words, we better share the heavy load of memory access across threads.

简而言之，作者的观点是通过拆成 block 的形式，能够并行的读取全局内存数据。个人观点，在某些情况下，拆分后的 block 可以恰好把数据放到共享内存中，以加速计算？

最后，对于 `block size`大小的选择：

- 越大的 `block size`表示拆分后的矩阵个数变少，这样访问全局内存的次数会更多。但是，每个矩阵比较大，这样线程的并行度会更高。即，IO变小，计算变快。
- 越小的 `block size`表示拆分后的矩阵个数变多，这样访问全局内存的次数会更少。但是，每个矩阵比较小，这样线程的并行度会更低。即，IO变大，计算变慢。

## 算子融合

算子融合是将多个计算操作合并为一个计算操作，以减少计算量和内存访问次数，从而提高计算效率。比如，矩阵乘法和 softmax 的融合，可以减少一次内存访问。

### softmax 的改造

softmax 的计算公式为：

$$
\text{softmax}(x_i) = \frac{e^{x_i}}{\sum_{j=1}^n e^{x_j}}
$$

而通常为了数值稳定性（避免溢出），会先计算最大值，再减去最大值，最后再计算 softmax。即：

$$
\text{softmax}(x_i) = \frac{e^{x_i - \max(x)}}{\sum_{j=1}^n e^{x_j - \max(x)}}
$$

在神经网络中，其值通常为一个矩阵（不只是一个维度），所以需要对每一行进行 softmax。示例代码如下：

```python
import torch

X = torch.rand(4, 4)

def my_softmax(X, dim=1):
    X -= torch.max(X, dim=dim, keepdim=True)[0]
    return torch.exp(X) / torch.sum(torch.exp(X), dim=dim, keepdim=True)

print(torch.softmax(X, dim=1))
print(my_softmax(X, dim=1))
assert torch.allclose(torch.softmax(X, dim=1), my_softmax(X, dim=1))
```

但这样跟矩阵乘法进行算子融合是没有优势的，两个函数还是独立计算的。算子融合是需要糅合两种计算操作。所以，需要把 softmax 函数放到矩阵乘法中进行计算。即将计算过程变成一个可迭代的过程，换而言之，随着元素的增加不断更新 softmax 的结果。

[Online normalizer calculation for softmax](https://arxiv.org/pdf/1805.02867.pdf) 提供了这么一种方式。简单起见，从一个元素开始介绍。现在有一个列表，里面只有一个元素 `[1]`，其 softmax 计算过程为：

1. exp: 计算 $e^{1}$，得到 $[e^{1}]$
2. max: 计算 `max(1)`，得到 `1`
3. -max: 计算 $e^{1 - 1} $，得到 $[e^{1 - 1} ]$
4. sum: 计算 $e^{1- 1} $ 的和，得到 $e^{1- 1} $
5. softmax: 计算 $[e^{1- 1} ] / (e^{1- 1} )$，得到 $[1]$

现在增加一个元素，观察有哪些变化。假设增加的元素为 2，现在有一个列表，里面有两个元素 `[1, 2]`，新增的 softmax 计算过程为：

1. exp: 计算 $e^{2}$，得到 $e^{2}$
2. max: 与原来的 `max(1)=1` 比较，得到 `max(1, 2)`，得到 `2`
3. -max：所有元素更新一遍，均减去最新的 `max(1, 2)=2`，得到 $[e^{1- 2} , e^{2- 2} ]$
4. sum: 重新计算一遍结果
5. softmax: 计算 $[e^{1- 2}, e^{2 - 2}] / (e^{1 - 2} + e^{2 - 2})$

对比发现，第 3 步会导致重新第 4 步计算一遍求和结果，但这个求和结果在第 5 步中作为分母是可以灵活调整的。原来是$e^{1-1}$，现在更新为$e^{1-2}+e^{2-2}$。假设原来的最大值为 `old_max`，新的最大值为 `new_max`，原来的元素为 `old_v`，则原来元素的 `exp`结果可以更新为

$$
e^{old\_v-old\_max} \rightarrow e^{old\_v-old\_max+old\_max-new\_max} = e^{old\_v-old\_max} \times e^{old\_max-new\_max}
$$

而对于新加的元素，则可以在求和部分直接加上$e^{2-2}$，所以，代码如下

```python
import numpy as np

nums = [1, 2]
sum_, max_v = 0., 0.
norm_v = 0.
for i, num in enumerate(nums):
    old_max_v = max_v
    max_v = max(max_v, num)
    norm_v = norm_v*np.exp(old_max_v - max_v) + np.exp(num - max_v)
softmax_nums = [np.exp(num-max_v)/ norm_v for num in nums] 
```

对于两个元素这个公式是适用的，那么对于多个元素呢？答案自然也是同样适用的

$$
\begin{aligned}
e^{v1-old\_max}+e^{v2-old\_max} &\rightarrow e^{v1-new\_max}+e^{v2-new\_max} \\ &= e^{v1-old\_max+old\_max-new\_max}+e^{v2-old\_max+old\_max-new\_max} \\  \\ &= e^{v1-old\_max} \times e^{old\_max-new\_max} + e^{v2-old\_max} \times e^{old\_max-new\_max} \\ &= (e^{v1-old\_max} - e^{v2-old\_max}) \times e^{old\_max-new\_max}
\end{aligned}
$$

所以，将其更新为二维矩阵的形式，就有

```python
import torch
import numpy as np

X = torch.rand(4, 4)

def online_softmax(X):
    value = torch.zeros_like(X)
    for row in range(X.shape[0]):
        row_max = 0.0
        normalizer_term = 0.0
        for col in range(X.shape[1]):
            val = X[row, col]
            old_row_max = row_max
            row_max = max(old_row_max, val)
            normalizer_term = normalizer_term * np.exp(old_row_max - row_max) + np.exp(val - row_max)
        value[row, :] = torch.exp(X[row, :] - row_max) / normalizer_term
    return value

print(torch.softmax(X, dim=1))
print(online_softmax(X))
assert torch.allclose(torch.softmax(X, dim=1), online_softmax(X))
```

### softmax + 矩阵乘法

一般的融合操作是需要在矩阵乘法后，计算结果的 softmax。即

```python
import torch

M, N, K = 4, 2, 4
A1, A2 = torch.rand(size=(M, N)), torch.rand(size=(N, K))
output = torch.softmax(A1 @ A2, dim=1)
```

在结合上面的内容后，将其改造为

```python
import torch
import numpy as np

M, N, K = 4, 2, 4
A1, A2 = torch.rand(size=(M, N)), torch.rand(size=(N, K))

def matmul_softmax(A1, A2):
    output = torch.zeros(size=(A1.shape[0], A2.shape[1]))
    for i in range(A1.shape[0]):
        row_max = 0.0
        normalizer_term = 0.0  
        for j in range(A2.shape[1]):
            val = output[i, j] = sum(map(lambda x: x[0] * x[1], zip(A1[i], A2[:, j])))
          
            old_row_max = row_max
            row_max = max(old_row_max, val)
            normalizer_term = normalizer_term * np.exp(old_row_max - row_max) + np.exp(val - row_max)
        output[i, :] = torch.exp(output[i, :] - row_max) / normalizer_term
    return output

print(torch.softmax(A1 @ A2, dim=1))
print(matmul_softmax(A1, A2))
assert torch.allclose(torch.softmax(A1 @ A2, dim=1), matmul_softmax(A1, A2))
```

更进一步，结合 Tiled matmul，将其改造为：

```python
import torch
import numpy as np

M, N, K = 4, 6, 8
A1, A2 = torch.rand(size=(M, N)), torch.rand(size=(N, K))

def block_matmul(sub_A1, sub_A2):
    output = torch.zeros(size=(sub_A1.shape[0], sub_A2.shape[1]))
    for i in range(sub_A1.shape[0]):
        for j in range(sub_A2.shape[1]):
            for k in range(sub_A2.shape[0]):
                output[i][j] += sub_A1[i][k] * sub_A2[k][j]            
    return output


def tiled_matmul_softmax(A1, A2):
    block_size_M, block_size_N, block_size_K = 2, 3, 4
    block_M, block_N, block_K = M // block_size_M, N // block_size_N, K // block_size_K

    output = torch.zeros(size=(A1.shape[0], A2.shape[1]))
    for i in range(0, A1.shape[0], block_M):
        start_i, end_i = i, i + block_M
        row_max = torch.tensor([[0. for _ in range(block_N)] for _ in range(block_M)])
        old_row_max = torch.tensor([[0. for _ in range(block_N)] for _ in range(block_M)])
        normalizer_term = torch.tensor([[0. for _ in range(block_N)] for _ in range(block_M)])

        for j in range(0, A2.shape[1], block_N):
            start_j, end_j = j, j + block_N
            for k in range(0, A2.shape[0], block_K):
                start_k, end_k = k, k + block_K
                sub_A1 = A1[start_i:end_i, start_k:end_k]
                sub_A2 = A2[start_k:end_k, start_j:end_j]
                output[start_i:end_i, start_j:end_j] += block_matmul(sub_A1, sub_A2)

            # 这里算完了每个block的结果，所以需要将其拆分成每个block，然后再计算softmax
            for ii, row in enumerate(range(start_i, end_i)):            
                for jj, col in enumerate(range(start_j, end_j)):
                    val = output[row][col]
                    old_row_max[ii][jj] = row_max[ii][jj]
                    row_max[ii][jj] = max(old_row_max[ii][jj], val)
                    normalizer_term[ii][jj] = normalizer_term[ii][jj] * np.exp(old_row_max[ii][jj] - row_max[ii][jj]) + np.exp(val - row_max[ii][jj])

        for ii, row in enumerate(range(start_i, end_i)):
            row_max_v, _ = torch.max(row_max, dim=1)
            # 重算 sum, 代入公式 old_v*exp(old_max - new_max)
            sum_ = torch.sum(normalizer_term[ii] * torch.exp(row_max[ii] - row_max_v[ii]))
            output[row, :] = torch.exp(output[row, :] - row_max_v[ii]) / sum_
    return output


print(torch.softmax(A1 @ A2, dim=1))
print(tiled_matmul_softmax(A1, A2))
assert torch.allclose(torch.softmax(A1 @ A2, dim=1), tiled_matmul_softmax(A1, A2))
```

由于这里是每次计算出来是一个 block ，所以要把 block 拆分出每个元素，计算 block 中每行每列的最大值 row_max 以及分母 normalizer_term。

最后在计算 softmax 时，要计算所有 block 的最大值 `torch.max(row_max, dim=1)`，还需要计算分母，这里需要考虑所有的 block，可以类比于合并计算 `[1, 2]`, `[3, 4]`两个列表的normalizer_term 。已知对应的 normalizer_term $e^{1-2}+e^{2-2}$ 和 $e^{3-4}+e^{3-4}$，合并后的结果应当是 $e^{1-4}+e^{2-4}+e^{3-4}+e^{3-4}$。将其公式化写作：

$$
\begin{aligned}
\sum_{i=1}^{m} e^{x_i-max(x)} + \sum_{i=1}^{n} e^{y_i-max(y)} & \rightarrow \sum_{i=1}^{m} e^{x_i-max(x,y)} + \sum_{i=1}^{n} e^{y_i-max(x,y)} \\ &= \sum_{i=1}^{m} e^{x_i-max(x)}e^{max(x)-max(x,y)} + \sum_{i=1}^{n} e^{y_i-max(x,y)}e^{max(y)-max(x,y)}
\end{aligned}
$$

假设 `max(x,y)=max(x)`，那么对应项将乘 1，这并不会对结果有任何影响。对应的代码为：

```python
sum_ = torch.sum(normalizer_term[ii] * torch.exp(row_max[ii] - row_max_v[ii]))
```

至此，完成了矩阵乘法和 softmax 的融合。接下来，会实现 cuda 版本以对比性能。

### Cuda 实现

首先，接着之前矩阵乘法的cuda实现，

```cpp
    if (row < input1.size(0) && col < input2.size(1)) {
        scalar_t value = 0.0;
        for (int k = 0; k < input1.size(1); ++k) {
            value += input1[row][k] * input2[k][col];
        }
        output[row][col] = value
    }
```

在循环之后，已经计算完了输出矩阵中的一项。需要继续算每行的 `row_max`和 `normalizer_term`。但由于这里是 cuda 中的某个 block，所以需要借助共享内存来通信每行的结果。

```cpp
        // 使用共享内存，计算每个 row 的最大值
        __shared__ scalar_t row_max[16][16];
        __shared__ scalar_t normalizer_term[16][16];
        row_max[threadIdx.y][threadIdx.x] = value; // 先把计算结果放到 row_max 中，以便于比较大小
        __syncthreads(); // 这行代码是为了保证每个线程都已经计算完了，才能进行下一步的操作
```

计算过程分为三个步骤：1. 找到每行的最大值

```cpp
        for (int i = blockDim.x / 2; i > 0; i /= 2) {
            if (threadIdx.x < i) {
                row_max[threadIdx.y][threadIdx.x] = max(row_max[threadIdx.y][threadIdx.x], row_max[threadIdx.y][threadIdx.x + i]);
            }
            __syncthreads();
        }
```

2. 计算每行的 softmax 的分母每项组成

```cpp
        normalizer_term[threadIdx.y][threadIdx.x] = exp(value - row_max[threadIdx.y][0]);
        __syncthreads();
```

3. 计算每行的 softmax 的每项之后

```cpp
        for (int i = blockDim.x / 2; i > 0; i /= 2) {
            if (threadIdx.x < i) {
                normalizer_term[threadIdx.y][threadIdx.x] += normalizer_term[threadIdx.y][threadIdx.x + i];
            }
            __syncthreads();
        }
        // 最后将其更新到输出矩阵中
        output[row][col] = exp(value - row_max[threadIdx.y][0]) / normalizer_term[threadIdx.y][0];
```

完整代码如下（这里没有实现完整的attention，仅仅是一个矩阵乘法 + softmax 计算）：

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

        // 使用共享内存，计算每个 row 的最大值
        __shared__ scalar_t row_max[16][16];
        __shared__ scalar_t normalizer_term[16][16];
        row_max[threadIdx.y][threadIdx.x] = value;
        __syncthreads();

        for (int i = blockDim.x / 2; i > 0; i /= 2) {
            if (threadIdx.x < i) {
                row_max[threadIdx.y][threadIdx.x] = max(row_max[threadIdx.y][threadIdx.x], row_max[threadIdx.y][threadIdx.x + i]);
            }
            __syncthreads();
        }
        // 计算每个 row 的 softmax 的分母
        normalizer_term[threadIdx.y][threadIdx.x] = exp(value - row_max[threadIdx.y][0]);

        __syncthreads();
        // 计算每个 row  normalizer_term之和
        for (int i = blockDim.x / 2; i > 0; i /= 2) {
            if (threadIdx.x < i) {
                normalizer_term[threadIdx.y][threadIdx.x] += normalizer_term[threadIdx.y][threadIdx.x + i];
            }
            __syncthreads();
        }

        // 计算每个 row 的 softmax
        output[row][col] = exp(value - row_max[threadIdx.y][0]) / normalizer_term[threadIdx.y][0];
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


// 这里偷懒没有换名字
std::vector<torch::Tensor> attention_cuda_forward(
    torch::Tensor q,
    torch::Tensor k,
    torch::Tensor v) {

    torch::Tensor scores = matrix_multiply(q, k);
    return {scores};
}
```

测试代码如下

```python
import torch

import mulsoftmax
import timeit

seed = 42
torch.manual_seed(seed)
torch.cuda.manual_seed(seed)

def my_py_softmax(x, dim):
    e = torch.exp(x)
    s = torch.sum(e, dim=dim, keepdim=True)
    return e / s

def py_mulsoft(q, k, v):
    # print(q@k.T)
    return torch.softmax(q @ k.T, dim=1)

def check_forward(q, k, v):
    baseline_values = py_mulsoft(q, k, v)
    cpp_values = mulsoftmax.forward(q, k, v)[0]

    print("base o", baseline_values)
    print("cpp  o", cpp_values)
    print(torch.all(torch.isclose(baseline_values, cpp_values)))

def compare_time(loop=100):
    q, k, v = torch.rand(size=(m, n), device=device), torch.rand(size=(m, n), device=device), torch.rand(size=(m, n), device=device)
    print("py", timeit.timeit(lambda: py_mulsoft(q, k, v), number=loop))
    print("cpp", timeit.timeit(lambda: mulsoftmax.forward(q, k, v)[0], number=loop))

if __name__ == "__main__":
    m, n = 16, 40
    device = "cuda"
    q, k, v = torch.rand(size=(m, n), device=device), torch.rand(size=(m, n), device=device), torch.rand(size=(m, n), device=device)
    # 先检查结果是否正确
    check_forward(q, k, v)
    q, k, v = torch.rand(size=(m, n)), torch.rand(size=(m, n)), torch.rand(size=(m, n))
    # 循环1w次，对比性能差距
    compare_time(10000)
```

输出结果如下：

|   | py     | cuda   |
| - | ------ | ------ |
| 0 | 0.5136 | 0.2909 |
| 1 | 0.6143 | 0.3322 |
| 2 | 0.7300 | 0.3608 |


### 其他实现

#### 二维矩阵转一维矩阵，实现矩阵乘法

在某些情况下，为了降低空间复杂度，会把二维矩阵展开为一维矩阵，再进行矩阵乘法。这里在这个基础上，再实现了 softmax 的计算。其本质上是把二维矩阵转为一维矩阵，再进行矩阵乘法。原来的`input1[row][k]` -> `input1[row * K + k]`，`input2[k][col]` -> `input2[k * N + col]`。但这里暂时不好实现 softmax 融合，因为 softmax 需要计算每行的最大值，这里的一维矩阵无法直接计算每行的最大值。所以这里偷懒先使用判断的方法，如果是最后一列，则计算 softmax。因此，导致算的速度会比较慢。

```cpp
#include <torch/extension.h>
#include <cuda.h>
#include <cuda_runtime.h>

#include <vector>

#DEFINE BLOCK_SIZE 256;

template <typename scalar_t>
__global__ void matrix_multiply_vector_kernel(const torch::PackedTensorAccessor<scalar_t,1,torch::RestrictPtrTraits,size_t> input1,
                                       const torch::PackedTensorAccessor<scalar_t,1,torch::RestrictPtrTraits,size_t> input2,
                                       torch::PackedTensorAccessor<scalar_t,1,torch::RestrictPtrTraits,size_t> output,
                                       const int M, const int N, const int K
                                       ) {
    int index = blockIdx.x * blockDim.x + threadIdx.x;
    int row = index / N;
    int col = index % N;

    if (row < M && col < N) {
        float value = 0.0;
        for (int k = 0; k < K; ++k) {
            value += input1[row * K + k] * input2[k * N + col];
        }
        output[row * N + col] = value;
        if (col == N - 1) {
            float row_max = 0.0;
            float normalizer_term = 0.0;        
            float old_row_max = 0.0;
            for (int i = 0; i < N; ++i) {
                old_row_max = row_max;
                row_max = max(row_max, output[row * N + i]);
                normalizer_term = normalizer_term * exp(old_row_max - row_max) + exp(output[row * N + i] - row_max);
            }
            for (int i = 0; i < N; ++i) {
                output[row * N + i] = exp(output[row * N + i] - row_max) / normalizer_term;
            }
        }
    }
}

std::vector<torch::Tensor> matmul_vector(torch::Tensor input1, torch::Tensor input2) {
    int M = input1.size(0);
    int K = input1.size(1);
    int N = input2.size(1);

    auto options = torch::TensorOptions().device(input1.device());
    
    const dim3 threads(BLOCK_SIZE);
    const dim3 blocks((M * N + threads.x - 1) / threads.x);

    // Reshape input tensors to vectors
    auto input1_vector = input1.reshape({-1});
    auto input2_vector = input2.reshape({-1});    
    torch::Tensor output_vector = torch::zeros({M * N}, options);

    AT_DISPATCH_FLOATING_TYPES(input1_vector.scalar_type(), "matrix_multiply_vector_kernel", ([&] {
        matrix_multiply_vector_kernel<<<blocks, threads>>>(
            input1_vector.packed_accessor<scalar_t,1,torch::RestrictPtrTraits,size_t>(),
            input2_vector.packed_accessor<scalar_t,1,torch::RestrictPtrTraits,size_t>(),
            output_vector.packed_accessor<scalar_t,1,torch::RestrictPtrTraits,size_t>(),
            M, N, K
        );
    }));
    return {output_vector.reshape({M, N}), output_vector.reshape({M, N})};
}

std::vector<torch::Tensor> attention_cuda_forward(
    torch::Tensor q,
    torch::Tensor k,
    torch::Tensor v) {

    return matmul_vector(q, k);
}
```

输出结果如下：

|   | py     | cuda   |
| - | ------ | ------ |
| 0 | 0.4239 | 0.4907 |
| 1 | 0.5069 | 0.4388 |
| 2 | 0.5462 | 0.5799 |

尽管用了一种比较笨的方法，但是速度实际上与原生的 pytorch 相差无几。

#### 单独实现 softmax

相较于非算子融合的写法，这里实现了一维矩阵的 softmax。相当于在计算矩阵乘法后再算 softmax。比较朴素的写法。。。

```cpp
template <typename scalar_t>
__global__ void softmax_kernel(torch::PackedTensorAccessor<scalar_t,1,torch::RestrictPtrTraits,size_t> output,
                                 const int M, const int N) {
    int index = blockIdx.x * blockDim.x + threadIdx.x;
    int row = index / N;

    if (row < M) {
        float row_max = 0.0;
        float normalizer_term = 0.0;        
        float old_row_max = 0.0;

        for (int i = 0; i < N; ++i) {
            old_row_max = row_max;
            row_max = max(row_max, output[row * N + i]);
            normalizer_term = normalizer_term * exp(old_row_max - row_max) + exp(output[row * N + i] - row_max);
        }

        for (int i = 0; i < N; ++i) {
            output[row * N + i] = exp(output[row * N + i] - row_max) / normalizer_term;
        }
    }
}

torch::Tensor matrix_softmax_vector_softmax(torch::Tensor input1, torch::Tensor input2) {
    int M = input1.size(0);
    int K = input1.size(1);
    int N = input2.size(1);

    auto options = torch::TensorOptions().device(input1.device());
    
    const dim3 threads(BLOCK_SIZE_VECTOR);
    const dim3 blocks((M * N + threads.x - 1) / threads.x);

    // Reshape input tensors to vectors
    auto input1_vector = input1.reshape({-1});
    auto input2_vector = input2.reshape({-1});    
    torch::Tensor output_vector = torch::zeros({M * N}, options);

    // 普通一维的矩阵乘法
    AT_DISPATCH_FLOATING_TYPES(input1_vector.scalar_type(), "matrix_multiply_vector_softmax_kernel", ([&] {
        matrix_multiply_vector_softmax_kernel<<<blocks, threads>>>(
            input1_vector.packed_accessor<scalar_t,1,torch::RestrictPtrTraits,size_t>(),
            input2_vector.packed_accessor<scalar_t,1,torch::RestrictPtrTraits,size_t>(),
            output_vector.packed_accessor<scalar_t,1,torch::RestrictPtrTraits,size_t>(),
            M, N, K
        );
    }));

    cudaDeviceSynchronize();

    AT_DISPATCH_FLOATING_TYPES(output_vector.scalar_type(), "softmax_kernel", ([&] {
        softmax_kernel<<<blocks, threads>>>(
            output_vector.packed_accessor<scalar_t,1,torch::RestrictPtrTraits,size_t>(),
            M, N
        );
    }));
    return output_vector.reshape({M, N});

}
```

输出结果如下：

|   | py     | cuda   |
| - | ------ | ------ |
| 0 | 0.4239 | 1.1874 |
| 1 | 0.5069 | 1.1101 |
| 2 | 0.5462 | 1.3238 |

对比发现，这里的cuda还是会比pytorch的慢一些，这是因为没有用自带的softmax，且没有使用算子融合。所以就算速度会更慢。

## 参考

[1] [https://pytorch.org/tutorials/advanced/cpp_extension.html](https://pytorch.org/tutorials/advanced/cpp_extension.html)。

[2] [Tiled matmul](https://github.com/ELS-RD/kernl/blob/main/tutorial/1%20-%20tiled%20matmul.ipynb)

[3] [Online softmax](https://github.com/ELS-RD/kernl/blob/main/tutorial/3%20-%20online%20softmax.ipynb)

[4] [Online normalizer calculation for softmax](https://arxiv.org/pdf/1805.02867.pdf)
