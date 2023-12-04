---
title: CUDA编程二次入门
date: 2018-05-18 22:47:33
tags: [GPU]
categories: [CUDA]
---
本文主要介绍了基本的CUDA程序框架及代码解析，包括如何申请设备指针内存、将数据拷贝到设备上、调用核函数进行计算、线程同步以及将结果拷贝回主机内等。同时也介绍了一些常见问题及解决方案，比如CUDA程序黑屏之后恢复的问题。接 {% post_link CUDA/记被CUDA折腾死去活来的那十天 %} 。阅读本文前需要有一部分C语言基础。

<!-- more -->

## 原始代码

在VS2015中，新建一个CUDA项目，里面会有一个 `kernel.cu`的文件。里面的代码就是官方给出的实例代码。如下所示。

```c
#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include <stdio.h>

cudaError_t addWithCuda(int *c, const int *a, const int *b, size_t size);

__global__ void addKernel(int *c, const int *a, const int *b)
{
    int i = threadIdx.x;
    c[i] = a[i] + b[i];
}

int main()
{
    const int arraySize = 5;
    const int a[arraySize] = { 1, 2, 3, 4, 5 };
    const int b[arraySize] = { 10, 20, 30, 40, 50 };
    int c[arraySize] = { 0 };

    cudaError_t cudaStatus = addWithCuda(c, a, b, arraySize);
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "addWithCuda failed!");
        return 1;
    }

    printf("{1,2,3,4,5} + {10,20,30,40,50} = {%d,%d,%d,%d,%d}\n",
        c[0], c[1], c[2], c[3], c[4]);

    cudaStatus = cudaThreadExit();
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaThreadExit failed!");
        return 1;
    }

    return 0;
}

cudaError_t addWithCuda(int *c, const int *a, const int *b, size_t size)
{
    int *dev_a = 0;
    int *dev_b = 0;
    int *dev_c = 0;

    cudaError_t cudaStatus;


    cudaStatus = cudaSetDevice(0);
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaSetDevice failed!  Do you have a CUDA-capable GPU installed?");
        goto Error;
    }


    cudaStatus = cudaMalloc((void**)&dev_c, size * sizeof(int));

    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaMalloc failed!");
        goto Error;
    }

    cudaStatus = cudaMalloc((void**)&dev_a, size * sizeof(int));
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaMalloc failed!");
        goto Error;
    }

    cudaStatus = cudaMalloc((void**)&dev_b, size * sizeof(int));
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaMalloc failed!");
        goto Error;
    }



    cudaStatus = cudaMemcpy(dev_a, a, size * sizeof(int), cudaMemcpyHostToDevice);
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaMemcpy failed!");
        goto Error;
    }

    cudaStatus = cudaMemcpy(dev_b, b, size * sizeof(int), cudaMemcpyHostToDevice);
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaMemcpy failed!");
        goto Error;
    }


    addKernel<<<1, size>>>(dev_c, dev_a, dev_b);


    cudaStatus = cudaThreadSynchronize();
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaThreadSynchronize returned error code %d after launching addKernel!\n", cudaStatus);
        goto Error;
    }


    cudaStatus = cudaMemcpy(c, dev_c, size * sizeof(int), cudaMemcpyDeviceToHost);
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaMemcpy failed!");
        goto Error;
    }

Error:
    cudaFree(dev_c);
    cudaFree(dev_a);
    cudaFree(dev_b);

    return cudaStatus;
}
```

## 拆解原始代码

下面对上面的代码逐一进行拆解进行说明。最原始最简单的代码应该是下面这样。在配置正确的情况下，下面的代码应该也是可以正确运行并输出的。

设备指的是GPU设备

```c
// 引入相关模块
#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include <stdio.h>

// 调用核函数，利用gpu进行计算
__global__ void addKernel(int *c, const int *a, const int *b) {
    // 获取当前线程id
    int i = threadIdx.x;
    // 在同一个线程id下，a数组与b数组元素对应相加
    c[i] = a[i] + b[i];
}

int main() {
    // 一般c语言的语法，在这里不做介绍
    const int arraySize = 5;
    const int a[arraySize] = { 1, 2, 3, 4, 5 };
    const int b[arraySize] = { 10, 20, 30, 40, 50 };
    int c[arraySize] = { 0 };


    // 申请设备指针，每个变量对应一个设备指针。所以这里有三个
    int *dev_a = 0;
    int *dev_b = 0;
    int *dev_c = 0;

    // 选择GPU设备，如果只有一个就只需选择0
    cudaSetDevice(0);

    // 申请设备指针的内存
    cudaMalloc((void**)&dev_c, size * sizeof(int));
    cudaMalloc((void**)&dev_a, size * sizeof(int));
    cudaMalloc((void**)&dev_b, size * sizeof(int));

    // 将主机上的a、b数组的数据拷贝到设备上。
    cudaMemcpy(dev_a, a, size * sizeof(int), cudaMemcpyHostToDevice);
    cudaMemcpy(dev_b, b, size * sizeof(int), cudaMemcpyHostToDevice);

    // 调用核函数，注意这里调用函数的语法与一般的不同。<<<1, size>>>指的是分配一个线程组，里面有size个线程
    addKernel<<<1, size>>>(dev_c, dev_a, dev_b);

    // 进行线程同步
    cudaThreadSynchronize();

    // 将结果拷贝回主机里的c变量中
    cudaMemcpy(c, dev_c, size * sizeof(int), cudaMemcpyDeviceToHost);

    // 释放设备指针内存
    cudaFree(dev_c);
    cudaFree(dev_a);
    cudaFree(dev_b);

    // 输出结果
    printf("{1,2,3,4,5} + {10,20,30,40,50} = {%d,%d,%d,%d,%d}\n", c[0], c[1], c[2], c[3], c[4]);

    // 退出线程
    cudaThreadExit();

    return 0;
}
```

`kernel.cu`里面的代码是基于上面这个代码进行拓展。

1. 每一次操作之后，进行判断状态 `cudaStatus`，如果状态异常就退出，这是一种比较好的编程习惯，当然带来的影响就是可读性略微降低了。
2. 封装了一个函数，将操作主机的代码与操作设备的代码（比如申请设备指针内存与释放设备指针等）。

## 拓展代码

当然，我们还可以基于此再次进行拓展。下面介绍几个比较常用的拓展代码。

待补充。。。。。

## 补充说明

如果在运行cuda代码的时候，显示屏突然黑屏，之后又恢复正常，是因为程序时间运行过长，触发了TDR事件，导致黑屏，显卡运算中断。解决方案如下：

1. 开始菜单中找到 `Nsight Monitor`并打开。（Win10可以直接搜索）
2. 在任务栏中打开 `Nsight Monitor`，单击右下角的 `Nsight Monitor options`。
3. 设置 `General`的 `Microsoft Display Driver`中 `WDDM TDR Delay`中的值，把 `2`(默认一般是2)，调大，比如200。
4. 保存后退出，重启电脑即可。

详见[解决CUDA程序的黑屏恢复问题](http://blog.163.com/yuhua_kui/blog/static/9679964420146183211348/)
