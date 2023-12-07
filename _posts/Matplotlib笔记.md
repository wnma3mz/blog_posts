---
title: Matplotlib笔记
date: 2017-08-26 09:46:23
tags: [Python, Matplotlib]
categories: [Visualization]
---

解决Python中文乱码问题的完整方案及Matplotlib绘图基础操作

<!-- more -->

# 中文乱码解决方案

支持所有平台的解决方案

## 定位默认字体位置

```bash
# 首先进入python的交互环境
>>>python

>from matplotlib.font_manager import findfont, FontProperties
# 当前使用的默认字体
>findfont(FontProperties(family=FontProperties().get_family()))
# 在此记住返回值
```

## 下载支持中文的字体

在这里推荐微软雅黑，msyh.ttf。

[推荐链接](http://www.monmonkey.com/sonota/font1/getfont.html)

## 偷梁换柱

在刚刚查询的那个文件夹中，将原来的默认字体重命名，之后将我们下载msyh.ttf替换进去，并重命名为那个默认字体之前的名字

## 大功告成

当然为了保险起见，还可以操作下面一步。到现在为止，可以测试看看是否成功输出中文。

找到matplotlib的安装目录，一般是在python的安装目录下的\Lib\site-packages\matplotlib\mpl-data,之后修改matplotlibrc文件

```bash
# 将下面这两行冒号后面的字符替换
...
#font.sans-serif     : nothing
...
#verbose.level  : debug
```

# matplotlib.pyplot 基础操作

```python
import matplotlib.pyplot as plt

x_list = []
y_list = []

# 设定画布
plt.figure(figsize=(8, 6), dpi=80)

# 画y_list的折线图，格式为'ro-'，颜色为blue
plt.plot(y_list, 'ro-', color='blue')

# 以x_list为x轴，y_list为y轴画折线图，前提x_list内容全为整型数字
plt.plot(x_list, y_list)

# 以x_list为x轴，y_list为y轴，x_list为字符串
x = range(len(x_list))
plt.plot(x, y_list)

# 设定横坐标, 跟着上面的字符串x_list的设定，每个字符串旋转角度为45，字体大小为20
plt.xticks(x, x_list, rotation=45, fontsize=20)

# 设定标题，注如果需要设定中文需要更改matplotlib的字体配置
plt.title('title')

# 保存图片名为plt
plt.savefig("plt.png")

# 运行完打开图片
plt.show()

# 注：如果需要保存图片，切记需要在打开图片前执行命令，否则保存图片为空白
```

# 搬运工系列

[5种快速易用的Python Matplotlib数据可视化方法](https://juejin.im/post/5a9e14726fb9a028b86d87c9)

[Matplotlib 教程](http://www.labri.fr/perso/nrougier/teaching/matplotlib/)

[Matplotlib 教程（中文翻译版本）](https://liam0205.me/2014/09/11/matplotlib-tutorial-zh-cn/)
