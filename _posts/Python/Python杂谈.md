---
title: Python杂谈
date: 2018-03-14 15:24:02
tags: [GIL, 元类]
categories: [Python]
---
理解Python中的GIL(Global Interpreter Lock)，以及如何使用多线程和多进程，以及弱引用。

<!-- more -->

## GIL

建议阅读文章

[python 线程，GIL 和 ctypes](http://zhuoqiang.me/python-thread-gil-and-ctypes.html)

[谈谈python的GIL、多线程、多进程](https://zhuanlan.zhihu.com/p/20953544)

GIL, `Global Interpreter Lock`，全局解释器锁

产生原因: 防止多线程并发执行机器码的Mutex。

好处: 不需要考虑线程安全问题，在单线程的情况下更快(相对而言)，和C库结合时更加方便

形象的比方:

> 我们把整个进程空间看做一个车间，把线程看成是多条不相交的流水线，把线程控制流中的字节码看作是流水线上待处理的物品。Python 解释器是工人，整个车间仅此一名。操作系统是一只上帝之手，会随时把工人从一条流水线调到另一条——这种“随时”是不由分说的，即不管处理完当前物品与否。

> 若没有 GIL。假设工人正在流水线 A 处理 A1 物品，根据 A1 的需要将房间温度（一个全局对象）调到了 20 度。这时上帝之手发动了，工人被调到流水线 B 处理 B1 物品，根据 B1 的需要又将房间温度调到了 50 度。这时上帝之手又发动了，工人又调回 A 继续处理 A1。但此时 A1 暴露在了 50 度的环境中，安全问题就此产生了。

> 而 GIL 相当于一条锁链，一旦工人开始处理某条流水线上的物品，GIL 便会将工人和该流水线锁在一起。而被锁住的工人只会处理该流水线上的物品。就算突然被调到另一条流水线，他也不会干活，而是干等至重新调回原来的流水线。这样每个物品在被处理的过程中便总是能保证全局环境不会突变。

在解释器解释python代码时，首先会获得这把锁，需要I/O操作就是释放这把锁。如果是纯计算程序，没有I/O操作，解释器会每隔100次操作释放一次锁（Python3.2之后使用固定时间(5ms)来释放）。

GIL只会影响严重依赖CPU的程序（计算型）, 所以运行Python的时候，同一时间只能执行一个线程(单核CPU下的多线程都只是并发)。运行Python程序就只能占用一个核的CPU资源，哪怕使用了 `threading`多线程也是一样。

CPU密集性代码(循环)：不适用多线程。不断触发GIL，线程来回切换，消耗资源

IO密集型代码(文件处理，网路爬虫)：多线程能够提高效率。在一个线程等待的同时，切换到另一个线程，不浪费CPU资源从而提升效率。

多核多线程比单核多线程更差，单核下多线程每次释放GIL，唤醒的线程都能获取GIL锁，可以无缝执行。但是多核下，一个CPU释放完GIL之后，所有CPU的线程又开始竞争，可能会被原来的CPU获取，导致其他CPU线程唤醒之后必须等待到下一次释放。这样造成的结果称为线程颠簸(thrashing), 效率会更低下。

以上，如果Python需要利用多核CPU，就使用多进程(`multiprocessing`)，各自拥有独立的GIL，互不干扰，从而达到并行执行。还有一种方案就是使用C进行拓展，手动释放GIL，这里不做介绍。

参考代码

```python
#! /usr/bin/python
# -*- coding: utf-8 -*-   
from time import time
from threading import Thread
from multiprocessing import Process
def spawn_n_threads(n, target):
    threads = []

    for _ in range(n):
        thread = Thread(target=target)
        thread.start()
        threads.append(thread)

    for thread in threads:
        thread.join()

def spawn_n_processes(n, target):
    threads = []

    for _ in range(n):
        thread = Process(target=target)
        thread.start()
        threads.append(thread)

    for thread in threads:
        thread.join()

def test(target, number=10, spawner=spawn_n_threads):
    for n in range(1, 5):
        start_time = time()
        for _ in range(number):
            spawner(n, target)
        end_time = time()
        print("Time elapsed with {} branch(es): {:.6f} sec(s)".format(n, end_time - start_time))
 
def fib():
    a = b = 1
    for _ in range(100000):
        a, b = b, a + b
if __name__ == "__main__":
    print("threads:")
    test(fib)
    print("processes:")
    test(fib, spawner=spawn_n_processes)
```

P.S.

- 并行：同时处理两个及以上的事件在同一时刻发生
- 并发：同时处理两个及以上的事件在同一事件间隔内发生

## 弱引用

推荐阅读

[Python弱引用的使用与注意事项](http://blog.soliloquize.org/2016/01/21/Python%E5%BC%B1%E5%BC%95%E7%94%A8%E7%9A%84%E4%BD%BF%E7%94%A8%E4%B8%8E%E6%B3%A8%E6%84%8F%E4%BA%8B%E9%A1%B9/)

弱引用与强引用相对，是指不能确保其引用的对象不会被垃圾回收器回收的引用。若一个对象若只被弱引用所引用，则被认为是不可访问（或弱可访问）的，病因此可能在任何时刻被回收。主要作用就是减少循环引用，减少内存中不必要的对象存在的数量。

Python用 `weakref`模块创建对象的弱引用，当引用计数为0或只存在对象的弱引用时将回收这个对象。

内存管理时使用，cache编程，垃圾回收(GC)

## while 1与while True

[Python天坑系列（一）：while 1比while True更快？](http://www.pythoner.com/356.html)

在python2中while 1会比while True更快

```python
#! /usr/bin/python
# -*- coding: utf-8 -*-

import timeit

def while_one():
    i = 0
    while 1:
        i += 1
        if i == 10000000:
            break

def while_true():
    i = 0
    while True:
        i += 1
        if i == 10000000:
            break

if __name__ == "__main__":
    w1_time, wt_time = 0, 0
    for _ in range(10):
        w1 = timeit.timeit(while_one, "from __main__ import while_one", number=3)
        w1_time += w1
    for _ in range(10):
        wt = timeit.timeit(while_true, "from __main__ import while_true", number=3)
        wt_time += wt
    print("while one: %s\nwhile_true: %s" % (w1_time, wt_time))
```

```bash
# python2的输出
while one: 9.27971386909
while_true: 13.5762348175
# python3的输出
while one: 14.19957690299998
while_true: 13.785004297995329
```

原因：python2中True/False不是关键字，python3添加了True/False做为关键字。每次在循环的时候，True/False需要被检查，所以while 1会更快

```python
Python 2.7.14 (default, Sep 23 2017, 22:06:14)
[GCC 7.2.0] on linux2
Type "help", "copyright", "credits" or "license" for more information.
>>> import keyword
>>> keyword.kwlist
['and', 'as', 'assert', 'break', 'class', 'continue', 'def', 'del', 'elif', 'else', 'except', 'exec', 'finally', 'for', 'from', 'global', 'if', 'import', 'in', 'is', 'lambda', 'not', 'or', 'pass', 'print', 'raise', 'return', 'try', 'while', 'with', 'yield']

Python 3.6.3 (default, Oct  3 2017, 21:45:48)
[GCC 7.2.0] on linux
Type "help", "copyright", "credits" or "license" for more information.
>>> import keyword
>>> keyword.kwlist
['False', 'None', 'True', 'and', 'as', 'assert', 'break', 'class', 'continue', 'def', 'del', 'elif', 'else', 'except', 'finally', 'for', 'from', 'global', 'if', 'import', 'in', 'is', 'lambda', 'nonlocal', 'not', 'or', 'pass', 'raise', 'return', 'try', 'while', 'with', 'yield']
```

## if x == True 与 if x

```python
#! /usr/bin/python
# -*- coding: utf-8 -*-

import timeit

def if_x_eq_true():
    x = True
    if x == True:
        pass

def if_x():
    x = True
    if x:
        pass

if __name__ == "__main__":
    if1 = timeit.timeit(if_x_eq_true, "from __main__ import if_x_eq_true", number = 1000000)
    if2 = timeit.timeit(if_x, "from __main__ import if_x", number = 1000000)

    print("if_x_eq_true: %s\nif_x: %s" % (if1, if2))
```

```bash
# python2
if_x_eq_true: 0.118808984756
if_x: 0.0920550823212
# python3
if_x_eq_true: 0.0951261969985353
if_x: 0.07899275699855934
```

不管python2还是python3, `if x:`从各方面来说都会优于 `if x == True`。后者会比前者多出一个检查True值和比较的操作。

## type与object

[Python 的 type 和 object 之间是怎么一种关系？](https://www.zhihu.com/question/38791962)

## 深刻理解Python中的元类(metaclass)

[深刻理解Python中的元类(metaclass)](http://blog.jobbole.com/21351/)
