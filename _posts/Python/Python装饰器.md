---
title: Python装饰器
date: 2018-01-30 10:41:52
tags: [Decorators]
categories: [Python]
---
这篇文章主要介绍了Python的装饰器是什么和一些Python内置装饰器的使用

<!-- more -->

参考文章:

[如何理解Python装饰器？](https://www.zhihu.com/question/26930016)

[详解Python的装饰器](https://www.cnblogs.com/cicaday/p/python-decorator.html)

## 装饰器是什么

python的装饰器本身就是一个函数，使用它可以在某些场景下简化python代码，增加一些额外功能。我的理解是，python装饰器可以帮助同时多个函数做一样的事情（比如打印日志），而避免了每个函数都需要额外加上一样的代码。

举个例子，假设我们有两个函数

```python

def func_one():
    print("this is func_one")

def func_two():
    print("this is func_two")

if __name__ == "__main__":
    func_one()
    func_two()
```

以上一段代码片段很容易理解，运行结果就是两行输出。那么问题来了，如果这个时候我需要在每个函数里面加上一个输出，比如`print("this is func")`。很简单，我们想到了第一种解决方案就是同时在两个函数里面加上这么一行。

但是如果我们需要在很多个函数在上很多个这种重复代码，我们这么做就很浪费时间了，而且容易出错。这个时候我们想到了第二种解决方案。定义一个新的函数，在每次使用函数之前，先调用新函数。

```python

def func_one():
    print("this is func_one")

def func_two():
    print("this is func_two")

def func_print(func):
    print("this is func")
    func()

if __name__ == "__main__":
    func_print(func_one)
    func_print(func_two)

```

感觉不错，但是如果这样无疑改变了原来的逻辑结构，且每次需要传递函数给`func_print`。下面就轮到我们的主角登场了————装饰器。

```python

def func_print(func):

    def wrapper(*args, **kwargs):
        print("this is func")
        return func(*args, **kwargs)
    return wrapper

def func_one():
    print("this is func_one")

def func_two():
    print("this is func_two")

if __name__ == "__main__":
    func_one = func_print(func_one)
    func_two = func_print(func_two)
    func_one()
    func_two()
```

这样也能达到我们的想要的效果，但是比之前的实现方式更难看懂。而且调用起来似乎更复杂了。

第一个问题：更难看懂

可能是对`*args`、`**kwargs`的参数理解不是很好。简单解释一下就是说,`*args`可以理解为一个`list`，接收多个参数。`**kwargs`可以理解为一个`dict`，接收多个`key`和`value`。接收的参数数量不定，可以没有，也可以很多个。再具体一点可以百度看看。

第二个问题：调用更复杂

其实上面的例子其实调用装饰器的一种方式，`python`有`@`的这个语法糖，所以一般情况下，我们是像下面一样调用的。

```python

def func_print(func):

    def wrapper(*args, **kwargs):
        print("this is func")
        return func(*args, **kwargs)
    return wrapper

@func_print
def func_one():
    print("this is func_one")

@func_print
def func_two():
    print("this is func_two")

if __name__ == "__main__":
    func_one()
    func_two()
```

好的，简单的装饰器调用基本就是这样，难度进阶，如何使用带参数的装饰器。换句话就说，如果我们需要根据每个函数自定义一些东西，如何传递参数给装饰器。

```python

def func_print(func_name):

    def decorator(func):

        def wrapper(*args, **kwargs):
            print("this is %s" % func_name)
            return func(*args, **kwargs)
        return wrapper

    return decorator


@func_print("one")
def func_one():
    print("this is func_one")


@func_print(func_name="two")
def func_two():
    print("this is func_two")


if __name__ == "__main__":
    func_one()
    func_two()


```

观察`func_print`部分代码，与之前的不同在于多加了一个`decorator`函数封装了`wrapper`函数。我的理解是这样的：`func_print`作用是装饰器的名字和接收外部的参数，`decorator`接收外部的函数，`wrapper`返回结果给函数。`decorator`翻译为中文是**装饰**。`wrapper`翻译过来是**包装**。`decorator`装饰函数，`wrapper`包装函数。

当然，`decorator`和`wrapper`这两个函数名字可以自定义，在这里为了好理解所以用了这两个名字

## 内置装饰器

### `@property`

如果要了解使用这个装饰器，就需要知道在不用装饰器的情况下，写一个属性。这里使用python自带的一个例子。

```python
class C(object):
    def getx(self): return self._x
    def setx(self, value): self._x = value
    def delx(self): del self._x
    x = property(getx, setx, delx, "I'm the 'x' property.")

```

上面就是定义一个python属性的标准写法，下面使用`@property`来定义属性

```python

class C(object):
    @property
    def x(self):
        "I am the 'x' property."
        return self._x
    @x.setter
    def x(self, value):
        self._x = value
    @x.deleter
    def x(self):
        del self._x
```

这样有什么好处呢，原来的定义，属性访问并没有做限制。使用`@property`可以使`x`这个属性更加安全，从获取、设置必须与属性名一致。

Emmmm.....我的理解就是加上`@property`的话，定义属性会更加安全更加优雅。毕竟是面向对象了，天知道你的用户会对你的属性做一些什么奇奇怪怪的事情。


