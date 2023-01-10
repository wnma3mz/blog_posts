---
title: 记被CUDA折腾死去活来的那十天
date: 2018-05-05 23:11:25
tags: [CUDA, GPU, 显卡]
categories: [CUDA]
---

谨以此文章纪念大二的一次折腾CUDA环境搭建经历。

<!-- more -->


为了提高矩阵运算的速度和不浪费实验室的GPU，所以开始折腾CUDA环境，入门GPU Coder。感谢@小岳岳 全程的帮助，学习了很多硬件知识；感谢@浪浪 学长的指导，少走了一些弯路。本文会尽量从零开始介绍硬件知识过渡到系统和软件环境的搭建。

贴上学长的博客[GPU Coder初体验](http://amourll.cn/2018/03/27/gpu-coder/)

## 基础介绍

首先祭出显卡天梯图，时间截止于博客编辑完的日期（2018年4月）。

![](https://raw.githubusercontent.com/wnma3mz/blog_posts/master/imgs/cuda/5af851eca8de0.jpg)

关于显卡的一些基本知识可以看知乎的这个贴：[知识扫盲I给大众的电脑显卡选购指南](https://zhuanlan.zhihu.com/p/33743591)

这里再简单补充一些基础知识，GPU（Graphic   Process  Unit ）其实是显卡的核心部件，分为AMD和Nvidia两种。一般情况下，做科学计算或者深度学习都是用Nvidia显卡，当然用AMD的也有在这里我用的是Nvidia所以就不介绍如何搭建AMD的编程环境了。

### 关于GPU编程

所谓GPU编程，我的理解是用代码操作GPU进行运算。通常来说，我们写代码用软件都是调用CPU来工作的，显卡在常规的认知中（某某年以前）都是用来作为图像渲染、玩游戏的。随着时代的发展，关于图像处理的数据越来越多，交给CPU来计算就显得过慢，所以人们开始用起了GPU，GPU原本发明的用途就是为了处理图像。所以如果数据是关于图像处理的，交给GPU来计算远远好于CPU的。

### 关于并行计算

GPU并行：CUDA，OpenCL（都比较麻烦，建议CUDA）
CPU并行：MPI，OpenMP。MPI比较麻烦，OpenMP容易实现

OpenAcc：OpenMP和CUDA的混合

这里使用的方案是CUDA（读作：库达）。

### 实验室的显卡介绍

Tesla C2050、两块Titans X、一块2010/2011年的AMD卡（具体忘了，反正只用来做图像显示）

配置情况：特斯拉+AMD，AMD用做显示，特斯拉用来计算。至于为什么不上泰坦，是因为要预备留着后面买的新服务器，这台是之前（2010/2011）的服务器。简单说说特斯拉显卡，特斯拉显示是英伟达专门生产用作科学计算的，一般情况下特斯拉显卡的价格和计算性能都可以碾压其他类型的显卡，而且支持双精度。不过在这的话，泰坦还是可以吊打这块特斯拉的，唯一的缺点大概泰坦是不支持双精度吧。

## 正文

### Ubuntu + Tesla + AMD

#### 结果

安装失败

#### 遇到问题及解决方案

1. 特斯拉不能作为图像显示。解决方案：换AMD显卡做显示。。。
2. Ubuntu16.04之后不支持AMD显卡驱动。解决方案：老版本的Ubuntu或者搜索关键词`amdgpu-pro`、`Ubuntu`
3. 安装CUDA是否还需要安装英伟达的驱动。Windows下是不需要的，Ubuntu不知道，很多博客中都有安装英伟达驱动这一步骤。
4. 安装完英伟达驱动之后，出现循环登陆或者不能显示图形界面的情况。解决方案：卸载英伟达驱动，重启。
5. 英伟达（特斯拉）和AMD双显卡的问题。这种情况网上搜出来的结果比较少，大多都是核显或集显+英伟达显卡。所以在尝试了N次之后，放弃了。

#### 补充

当时不知道装CUDA是否需要安装英伟达驱动，所以就安装了。但是安装完之后重启，就不能显示图形界面，原因如问题一。很矛盾的点就在于，如果只安装CUDA不装英伟达驱动，安装CUDA是失败的。综上，不懂如何在AMD的基础上用特斯拉搭建环境。（最后可能怀疑是显卡本身出了问题）

[CUDA下载地址](https://developer.nvidia.com/cuda-downloads)，这里选择好系统环境，强烈推荐下载`runfile`，用它进行安装。不推荐安装最新版本。CUDA官网虽声明说CUDA向下兼容，但根据网上反馈和自己的实验来看，CUDA并没有做到。关于主板BIOS设置显卡的问题，实验室这块并不支持。。。。凉凉

#### 一些实用命令

```bash
# 在不能进入图形界面的时候，按住Ctrl+Alt+F1/F2/F3进入终端界面，输入下面的命令卸载英伟达驱动。卸载完重启即可
$ sudo apt-get remove --purge nvidia-*
$ reboot

# 查看显卡信息
$ lspci | grep -i vga
# 查看GPU型号
$ lspci | grep -i nvidia
# 查看NVIDIA驱动版本
$ sudo dpkg --list | grep nvidia-*
```

可以参考的一些博客如下：

[Ubuntu16.04安装NVIDIA显卡驱动和CUDA时的一些坑与解决方案](https://blog.csdn.net/chaihuimin/article/details/71006654?locationNum=2&fps=1)

[ubuntu16.04 安装NVIDIA和CUDA8.0](http://www.cnblogs.com/sp-li/p/7680526.html)

[Ubuntu安装和NVIDIA驱动和安装](https://blog.csdn.net/wonengguwozai/article/details/52664597)

[主板设置独显/集显](https://jingyan.baidu.com/article/d8072ac468a91dec95cefdf0.html)

以上，来来回回重装过无数遍系统，耗时5-6天。其中，大部分耗费的时间在于下载和安装，当然还有相当一部分在重装系统的时间上，而且一开始装的是最新的17.10，之后又回退到了16.04，做了几次系统盘重装了无数次系统，并且还只能在课余时间（基本是晚上，和岳岳在实验室折腾）。可以说是相当惨了，后续已经基本熟练到，出了问题直接手敲命令重新来一遍，不需要查资料（熟悉到令人心疼）。

#### 总结经验教训

1. 踩坑之前，先去请教已经踩过坑的朋友或者去官方论坛查看安装记录或者搜索相关博客。已经两年了，我的习惯还没有改过来QAQ，喜欢自己一遍又一遍踩坑之后再搜索
2. 提前准备一些工具，比如如果是大型工具，就可能会弄坏（“脏”）系统，备一个系统盘。还有一些能提前获取的压缩包或者安装包。
3. 踩坑的时候要脚踏实地，不要浮躁冲动。明明不需要重装系统，浪费时间，就不要直接动手格盘重装

### Windows10 + Tesla + AMD

按照小岳岳的意思，Windows装这种双显卡驱动啥的会比Linux方便很多，所以改战Windows，重装的系统版本是专业版。

#### 结果

失败

#### 遇到的问题

AMD驱动能够正常显示图形并且可以正常安装驱动，但是那块特斯拉显卡不行。系统能识别出来有两块显卡，但是在安装完英伟达驱动之后，并不能设置特斯拉。且两块显卡，AMD显示正常，特斯拉有个黄色的感叹号。所以是否怀疑是否是显卡本身有了问题（毕竟这么久了QAQ）

#### 补充

特斯拉显卡本身不是安装一般的英伟达驱动的，而是需要安装`TCC`驱动程序，[官网介绍](http://www.nvidia.cn/object/software-for-tesla-products-cn.html)。这里一开始我是不知道的，所以可能是这里出了问题。但是很矛盾的点在于，CUDA在Windows可以直接安装驱动程序。我也试过直接安装CUDA也不行。浪浪的意思是，特斯拉只能在Server版本上运行，我和小岳岳对这一点持怀疑态度。

关于完整的安装过程就不在这个失败部分介绍了，下一节完整的说明Windows下安装过程及注意事项。此部分耗时2-3天。

### Windows10 + Titan X

由于以上种种原因，并且已经快折腾不下去了，所以跟教授直接申请换泰坦。嗯，所以这一波一天半大概就成功了，只失败了一次。泰坦是支持图形显示的，为了减少问题所以也将AMD换下了（第一次没卸，出了一点问题，可能跟AMD有关）。

#### 软件版本

Windows10专业版+VS2013专业版+CUDA7.5

这里需要注意的是，在前文中提到CUDA不向下兼容，所以推荐不安装最新的，保险起见就选了较旧的版本。关于VS版本选择问题，这里CUDA7.5网上最多的搭配是VS2013，但是这里需要说明VS2013必须要是专业版，否则不支持CUDA。在VS2015之后的版本就没有这个问题。还有就是顺序问题，**一定要先安装VS再安装CUDA**，不需要安装英伟达驱动。

#### 总体流程

1. 安装Windows系统，激活系统，最好不进行安全更新。卸载一些不必要的软件防止占用空间大小（比如说我只有一块128G的固态，惨！）

2. 安装VS2013专业版

3. 安装CUDA7.5，建议local版。如果这里安装失败可以进入**服务**，找到**Windows Installer**，手动启动它即可。还有一些其他问题，在文末贴了链接。

4. 写CUDA的环境配置。

   ```bash
   # cuda7.5安装完成之后在系统环境变量中自动配置了两个系统变量
   CUDA_PATH：C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v7.5
   CUDA_PATH_V7_5：C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v7.5
   # 但是为了之后的vs2013的配置做准备我们需要在配置五个系统变量，请根据具体的安装路径进行设置CUDA_SDK_PATH, 这里我用的是默认设置
   CUDA_BIN_PATH：%CUDA_PATH%\bin
   CUDA_LIB_PATH：%CUDA_PATH%\lib\Win32
   CUDA_SDK_BIN：%CUDA_SDK_PATH%\bin\Win64
   CUDA_SDK_LIB：%CUDA_SDK_PATH%\common\lib\x64
   CUDA_SDK_PATH：C:\ProgramData\NVIDIA Corporation\CUDA Samples\v7.5
   # 在系统环境变量path后添加如下内容
   %CUDA_LIB_PATH%
   %CUDA_BIN_PATH%
   %CUDA_SDK_LIB_PATH%
   %CUDA_SDK_BIN_PATH%
   ```

5. 进行测试

#### 参考文章

[win10+vs2013+cuda7.5环境搭建](https://blog.csdn.net/u011821462/article/details/50145221)

[VS2013下CUDA 7.5安装](https://blog.csdn.net/u012033124/article/details/52169823)

CUDA安装的一些错误可以参考[这篇文章](https://blog.csdn.net/xuxiatian/article/details/50577960)，但是在某些情况下，可能系统环境已经被污染了，最好的方法还是重装系统。

耗时1-2天，总归是很愉快的结束了整个环境搭建。在某些情况下，有些环境的搭建不一定需要（最好不要）用最新的版本，可能会有一些bug，而且之前踩坑的人比较少可借鉴的东西较少。总体来说，抛开之前很奇怪的硬件配置之外，安装过程还是并不难的。接下来，便开始了我的CUDA编程之旅。

人生路漫漫，谨以此文章纪念上大学来折腾环境最坑爹的一次记录。
