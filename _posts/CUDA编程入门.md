---
title: CUDA编程入门
date: 2018-05-13 18:57:22
tags: [C, CUDA, 并行, GPU]
categories: [CUDA]
---
CUDA编程入门，接[《记被CUDA折腾死去活来的那十天》](https://wnma3mz.github.io/hexo_blog/2018/05/05/%E8%AE%B0%E8%A2%ABCUDA%E6%8A%98%E8%85%BE%E6%AD%BB%E5%8E%BB%E6%B4%BB%E6%9D%A5%E7%9A%84%E9%82%A3%E5%8D%81%E5%A4%A9/)

<!-- more -->

参考文章：[CUDA从入门到精通](https://blog.csdn.net/augusdi/article/details/12833235)

## 注意事项

远程桌面登陆服务器时，远程终端不能运行CUDA程序。原因是因为远程桌面登陆调用的显卡资源是本地的显卡，运行程序的时候没有远程的显卡，自然会报错。所以可以用远程终端进行登陆，或者远程服务器配置两块显卡，一块用于显示一块用于计算。



话不多说，直接上代码。

## 原始的Demo（线程并行）

目的就是a, b两个数组对应元素进行相加

```c
// 导入cuda相关模块
#include "cuda_runtime.h"
#include "device_launch_parameters.h"
// c中自带的模块
#include <stdio.h>

// 声明一个函数后面需要调用
cudaError_t addWithCuda(int *c, const int *a, const int *b, size_t size);

// __global__前缀表示用于控制设备，还有一个控制设备的前缀是__device__
__global__ void addKernel(int *c, const int *a, const int *b)
{
    // 这个函数是运行在GPU上的，称为核函数
    int i = threadIdx.x;
    c[i] = a[i] + b[i];
}

int main()
{
    const int arraySize = 5;
    const int a[arraySize] = { 1, 2, 3, 4, 5 };
    const int b[arraySize] = { 10, 20, 30, 40, 50 };
    int c[arraySize] = { 0 };

    // cuda_Error_t指的是cuda错误类型，取值为整数
    cudaError_t cudaStatus = addWithCuda(c, a, b, arraySize);
    // cudaDeviceProp是设备属性结构体
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "addWithCuda failed!");
        return 1;
    }

    printf("{1,2,3,4,5} + {10,20,30,40,50} = {%d,%d,%d,%d,%d}\n",
        c[0], c[1], c[2], c[3], c[4]);

    // cudaThreadExit must be called before exiting in order for profiling and
    // tracing tools such as Nsight and Visual Profiler to show complete traces.
    cudaStatus = cudaThreadExit();
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaThreadExit failed!");
        return 1;
    }

    return 0;
}

// 用于使用CUDA并行添加向量的辅助函数
cudaError_t addWithCuda(int *c, const int *a, const int *b, size_t size)
{
    // GPU设备端的数据指针
    int *dev_a = 0;
    int *dev_b = 0;
    int *dev_c = 0;
    // GPU状态指示
    cudaError_t cudaStatus;

    // 当电脑上有多个显卡支持CUDA时，选择某个GPU来进行运算。0表示设备号，还有1,2,3……
    cudaStatus = cudaSetDevice(0);
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaSetDevice failed!  Do you have a CUDA-capable GPU installed?");
        goto Error;
    }

    // 分配GPU设备端的内存。三个变量，所以需要分配三次。
    cudaStatus = cudaMalloc((void**)&dev_c, size * sizeof(int));
    // 申请失败就报错
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


    // 拷贝数据到GPU中，已有两个变量，所以需要拷贝两次
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

    // 运行核函数。关键代码
    addKernel<<<1, size>>>(dev_c, dev_a, dev_b);

    // 同步线程
    cudaStatus = cudaThreadSynchronize();
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaThreadSynchronize returned error code %d after launching addKernel!\n", cudaStatus);
        goto Error;
    }

    // 将结果拷贝回主机
    cudaStatus = cudaMemcpy(c, dev_c, size * sizeof(int), cudaMemcpyDeviceToHost);
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaMemcpy failed!");
        goto Error;
    }

Error:
    // 释放GPU设备端内存
    cudaFree(dev_c);
    cudaFree(dev_a);
    cudaFree(dev_b);

    return cudaStatus;
}
```

以上是线程并行，CUDA官方初始给的例子。

启动核函数的调用过程，即`addKernel`函数。

<<<>>>表示运行时配置符号，里面1表示只分配一个线程组（又称线程块、Block），size表示每个线程组有size个线程（Thread）。本程序中size根据前面传递参数个数应该为5，所以运行的时候，核函数在5个GPU线程单元上分别运行了一次，总共运行了5次。这5个线程是如何知道自己“身份”的？是靠`threadIdx`这个内置变量，它是个dim3类型变量，接受<<<>>>中第二个参数，它包含x,y,z 3维坐标，而我们传入的参数只有一维，所以只有x值是有效的。通过核函数中`int i = threadIdx.x;`这一句，每个线程可以获得自身的id号，从而找到自己的任务去执行。

## 块并行

```c
__global__ void addKernel(int *c, const int *a, const int *b)
{
    int i = threadIdx.x;
    c[i] = a[i] + b[i];
}
// 改为
__global__ void addKernel(int *c, const int *a, const int *b)
{
    int i = blockIdx.x;
	c[i] = a[i] + b[i];
}

addKernel<<<1, size>>>(dev_c, dev_a, dev_b);
// 改为
addKernel<<<size,1 >>>(dev_c, dev_a, dev_b);
```

这里与线程并行的区别在于，由`threadIdx`改为了`blockIdx`。`<<<1, size>>>`改为了`<<<size, 1>>>`。

线程并行是细粒度并行，调度效率高；块并行是粗粒度并行，每次调度都要重新分配资源，有时资源只有一份，那么所有线程块都只能排成一队，串行执行。

有时采用分治法，将一个大问题分解为几个小规模问题，将这些小规模问题分别用一个线程块实现，线程块内可以采用细粒度的线程并行，而块之间为粗粒度并行，这样可以充分利用硬件资源，降低线程并行的计算复杂度。适当分解，降低规模，在一些矩阵乘法、向量内积计算应用中可以得到充分的展示。

![img](https://img-blog.csdn.net/20130723220559500?watermark/2/text/aHR0cDovL2Jsb2cuY3Nkbi5uZXQva2trNTg0NTIw/font/5a6L5L2T/fontsize/400/fill/I0JBQkFCMA==/dissolve/70/gravity/Center)

多个线程块组成一个线程格(Grid)。一维线程->二维线程块->三维线程格。

## 流并行

```c
addKernel<<<size, 1>>>(dev_c, dev_a, dev_b);
// 改为
cudaStream_t stream[5];
for(int i = 0;i < 5;i++)  {
    // 创建流
    cudaStreamCreate(&stream[i]);
}
for(int i = 0;i < 5;i++)  {
	// 执行流。线程块数(1)，每个线程中线程数(1)，每个block用到的共享内存大小(0)，流对象(当前核函数在哪个流上运行)
    addKernel<<<1, 1, 0, stream[i]>>>(dev_c + i, dev_a + i, dev_b + i);
}
// 控制整个设备的所有流同步
cudaDeviceSynchronize();

Error:
	cudaFree(dev_c);
    cudaFree(dev_a);
    cudaFree(dev_b);
// 改为
Error:
	for(int i = 0;i < 5;i++)  {
        //销毁流
        cudaStreamDestroy(stream[i]);
    }
	cudaFree(dev_c);
    cudaFree(dev_a);
    cudaFree(dev_b);
```

利用多个网格（线程格）进行流并行。这样实现的并行可以达到比块并行更粗的细粒度。

## 线程通信

这次目的分三个线程做三个任务，1-5数字进行求和，1-5每个数字求平方和，1-5累乘

```c
// 更改addKernel函数内容如下
__global__ void addKernel(int *c, const int *a) {
    // 线程块数：1，块号为0
    // 线程数：5，线程号0~4
    // 共享存储器大小：5个int型变量大小（5 * sizeof(int）)

    int i = threadIdx.x;
    // 将数据一次读入到SM（Shared Memory）内部
    extern __shared__ int smem[];
    smem[i] = a[i];

    // 0号线程做平方和
    if(i == 0) {
        c[0] = 0;
        for(int d = 0; d < 5; d++) {
            c[0] += smem[d] * smem[d];
        }
    }
    // 1号线程做累加
    if(i == 1) {
        c[1] = 0;
        for(int d = 0; d < 5; d++) {
            c[1] += smem[d];
        }
    }
    // 2号线程做累乘
    if(i == 2) {
        c[2] = 1;
        for (int d = 0; d < 5; d++) {
            c[2] *= smem[d];
        }
    }
}


// 更改main函数的内容
int main() {
 	// 跟普通的C语言代码编写差不多
    const int arraySize = 5;
    const int a[arraySize] = { 1, 2, 3, 4, 5 };
    int c[arraySize] = { 0 };

    cudaError_t cudaStatus = addWithCuda(c, a, arraySize);
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "addWithCuda failed!");
        return 1;
    }
    printf("\t1+2+3+4+5 = %d\n\t1^2+2^2+3^2+4^2+5^2 = %d\n\t1*2*3*4*5 = %d\n\n\n\n\n\n", c[1], c[0], c[2]);

    cudaStatus = cudaThreadExit();
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaThreadExit failed!");
        return 1;
    }
    return 0;
}

// 更改addWithCuda内容
cudaError_t addWithCuda(int *c, const int *a,  size_t size)
{
    // 去除c变量相关内容，如申请内存、GPU设备端指针。其余内容差不多
    int *dev_a = 0;
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
    if (cudaStatus != cudaSuccess){
        fprintf(stderr, "cudaMalloc failed!");
        goto Error;
    }

    cudaStatus = cudaMemcpy(dev_a, a, size * sizeof(int), cudaMemcpyHostToDevice);
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaMemcpy failed!");
        goto Error;
    }

    // 第三个参数为共享内存大小（字节数）
    addKernel<<<1, size, size * sizeof(int), 0>>>(dev_c, dev_a);

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
    return cudaStatus;
}
```



## 性能剖析

测试对比线程并行和块并行的速度

```c
#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include <stdio.h>

cudaError_t addWithCuda(int *c, const int *a, const int *b, size_t size);

__global__ void addKernel_blk(int *c, const int *a, const int *b) {
    int i = blockIdx.x;
    c[i] = a[i]+ b[i];
}
__global__ void addKernel_thd(int *c, const int *a, const int *b) {
    int i = threadIdx.x;
    c[i] = a[i]+ b[i];
}
int main() {
    const int arraySize = 1024;
    int a[arraySize] = {0};
    int b[arraySize] = {0};
    for(int i = 0;i < arraySize;i++) {
        a[i] = i;
        b[i] = arraySize-i;
    }
    int c[arraySize] = {0};

    cudaError_t cudaStatus;
    int num = 0;
    // 设备属性结构体
    cudaDeviceProp prop;
    // 获取设备总数
    cudaStatus = cudaGetDeviceCount(&num);
    for(int i = 0;i<num;i++) {
        // 获取设备属性
        cudaGetDeviceProperties(&prop,i);
    }
    cudaStatus = addWithCuda(c, a, b, arraySize);
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "addWithCuda failed!");
        return 1;
    }

    cudaStatus = cudaThreadExit();
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaThreadExit failed!");
        return 1;
    }
    for(int i = 0;i<arraySize;i++) {
        if(c[i] != (a[i]+b[i])) {
            printf("Error in %d\n",i);
        }
    }
    return 0;
}

cudaError_t addWithCuda(int *c, const int *a, const int *b, size_t size) {
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
    if (cudaStatus != cudaSuccess)
    {
        fprintf(stderr, "cudaMalloc failed!");
        goto Error;
    }
    // Copy input vectors from host memory to GPU buffers.
    cudaStatus = cudaMemcpy(dev_a, a, size * sizeof(int), cudaMemcpyHostToDevice);
    if (cudaStatus != cudaSuccess)
    {
        fprintf(stderr, "cudaMemcpy failed!");
        goto Error;
    }
    cudaStatus = cudaMemcpy(dev_b, b, size * sizeof(int), cudaMemcpyHostToDevice);
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaMemcpy failed!");
        goto Error;
    }
    // 计算核函数执行时间
    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);
    cudaEventRecord(start, 0);
    // 由于数据量小，执行1000次扩大执行时间。blk是块并行，thd是线程并行
    for(int i = 0;i < 1000;i++) {
//      addKernel_blk<<<size,1>>>(dev_c, dev_a, dev_b);
        addKernel_thd<<<1,size>>>(dev_c, dev_a, dev_b);
    }
    cudaEventRecord(stop,0);
    cudaEventSynchronize(stop);

    // stop-start，执行总时间
    float tm;
    cudaEventElapsedTime(&tm, start, stop);
    printf("GPU Elapsed time:%.6f ms.\n", tm);


    // cudaThreadSynchronize waits for the kernel to finish, and returns
    // any errors encountered during the launch.
    cudaStatus = cudaThreadSynchronize();
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaThreadSynchronize returned error code %d after launching addKernel!\n", cudaStatus);
        goto Error;
    }
    // Copy output vector from GPU buffer to host memory.
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

