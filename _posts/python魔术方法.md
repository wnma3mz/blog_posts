---
title: Python魔术方法
date: 2018-03-14 15:24:02
tags: [Python, magic methods]
categories: [Python知识点]
---
这篇文章主要介绍了Python的魔术方法

<!-- more -->


参考文章:

[Python 魔术方法指南](http://pycoders-weekly-chinese.readthedocs.io/en/latest/issue6/a-guide-to-pythons-magic-methods.html)

魔术方法，顾名思义是一种可以给对象(类)增加魔法的特殊方法，它们的表示方法一般是用双下划线包围(如`__init__`)


## 对象实例化

```python
from os.path import join

class FileObject:
    '''给文件对象进行包装从而确认在删除时文件流关闭'''

    def __init__(self, filepath='~', filename='sample.txt'):
        #读写模式打开一个文件
        self.file = open(join(filepath, filename), 'r+')

    def __del__(self):
        self.file.close()
        del self.file
```

- `__new__(cls, [...])`: 在一个对象实例化调用的第一个方法，第一个参数是类，剩下的参数传递给`__init__`

- `__init__(self, [...])`: 类的初始化方法。当构造函数被调用的时候的任何参数够会传给它

- `__del__(self)`: `__new__`和`__init__`为对象的构造器，`__del__`为对象的析构器。当一个对象进行过垃圾回收时候的行为就是这个方法。当一个对象在删除的时候需要更多的清洁工作，就需要使用这个方法

## 比较

```python
class Word(str):
'''存储单词的类，定义比较单词的几种方法'''

    def __new__(cls, word):
        # 注意我们必须要用到__new__方法，因为str是不可变类型
        # 所以我们必须在创建的时候将它初始化
        if ' ' in word:
            print("Value contains spaces. Truncating to first space.")
            word = word[:word.index(' ')] #单词是第一个空格之前的所有字符
        return str.__new__(cls, word)

    def __gt__(self, other):
        return len(self) > len(other)
    def __lt__(self, other):
        return len(self) < len(other)
    def __ge__(self, other):
        return len(self) >= len(other)
    def __le__(self, other):
        return len(self) <= len(other)
```

- `__cmp__(self, other)`: 用于比较的魔术方法，实现了所有的比较符号。如果`self` < `other`返回一个负数；如果`self` == `other`返回０；如果`self` > `other`返回正数。一般情况下是分别定义每一个比较符号。如果想实现所有的比较符号，就需要调用这个方法。

- `__eq__(self, other)`: 等号
- `__ne__(self, other)`:　不等号
- `__lt__(self, other)`:　小于
- `__gt__(self, other)`:　大于
- `__le__(self, other)`:　小于等于
- `__ge__(self, other)`:　大于等于


## 一元操作符和函数

- `__pos__(self)`：实现正号的特性
- `__neg__(self)`：实现负号的特性
- `__abs__(self)`：abs, 绝对值
- `__invert__(self)`：实现~的特性

## 普通算数操作符

- `__add__(self, other)`: 实现加法
- `__sub__(self, other)`: 实现减法
- `__mul__(self, other)`: 实现乘法
- `__floordiv__(self, other)`: 实现整除法
- `__div__(self, other)`: 实现除法
- `__mod__(self, other)`: 实现取模
- `__pow__(self, other)`: 实现指数
- `__lshift__(self, other)`: 实现<<
- `__rshift__(self, other)`: 实现>>
- `__and__(self, other)`: 实现&
- `__or__(self, other)`: 实现|
- `__xor__(self, other)`: 实现^

## 其他

```python
def __setattr__(self, name, value):
    self.name = value
    #每当属性被赋值的时候， ``__setattr__()`` 会被调用，这样就造成了递归调用。
    #这意味这会调用 ``self.__setattr__('name', value)`` ，每次方法会调用自己。这样会造成程序崩溃。

def __setattr__(self, name, value):
    self.__dict__[name] = value  #给类中的属性名分配值
    #定制特有属性
```

```python
class AccessCounter:
    '''一个包含计数器的控制权限的类每当值被改变时计数器会加一'''

    def __init__(self, val):
        super(AccessCounter, self).__setattr__('counter', 0)
        super(AccessCounter, self).__setattr__('value', val)

    def __setattr__(self, name, value):
        if name == 'value':
            super(AccessCounter, self).__setattr__('counter', self.counter + 1)
    #如果你不想让其他属性被访问的话，那么可以抛出 AttributeError(name) 异常
        super(AccessCounter, self).__setattr__(name, value)

    def __delattr__(self, name):
        if name == 'value':
            super(AccessCounter, self).__setattr__('counter', self.counter + 1)
        super(AccessCounter, self).__delattr__(name)]
```

- `__str__(self)`: 返回值，人类可读

- `__repr__(self)`: 返回值，机器可读

- `__unicode__(self)`: 返回unicode()

- `__hash__(self)`: hash()返回一个整形

- `__nonzero__(self)`: bool()返回的值

- `__getattr__(self, name)`: 当用户试图获取一个不存在（或不建议的属性）的属性时，给出一些警告

- `__setattr__(self, name, value)`: 无论属性是否存在，都允许你定义对属性的赋值行为

- `__delattr__(self, name)`： 删除一个属性

## 封装附加魔术方法

```python
class FunctionalList:
'''一个封装了一些附加魔术方法比如 head, tail, init, last, drop, 和take的列表类。
'''

    def __init__(self, values=None):
        if values is None:
            self.values = []
        else:
            self.values = values

    def __len__(self):
        return len(self.values)

    def __getitem__(self, key):
        #如果键的类型或者值无效，列表值将会抛出错误
        return self.values[key]

    def __setitem__(self, key, value):
        self.values[key] = value

    def __delitem__(self, key):
        del self.values[key]

    def __iter__(self):
        return iter(self.values)

    def __reversed__(self):
        return reversed(self.values)

    def append(self, value):
        self.values.append(value)
    def head(self):
        return self.values[0]
    def tail(self):
        return self.values[1:]
    def init(self):
        #返回一直到末尾的所有元素
        return self.values[:-1]
    def last(self):
        #返回末尾元素
        return self.values[-1]
    def drop(self, n):
        #返回除前n个外的所有元素
        return self.values[n:]
    def take(self, n):
        #返回前n个元素
        return self.values[:n]
```

## 反射

- `__instancecheck__(self, instance)`: 检查一个实例是否为定义类的实例
- `__subclasscheck__(self, subclass)`: 检查一个类是否为定义类的子类

## 更改状态

- `__call__(self, [...])`: 经常改变状态的时候，调用这个方法是一种直接和优雅的方法

```python
class Entity:
'''调用实体来改变实体的位置。'''

    def __init__(self, size, x, y):
        self.x, self.y = x, y
        self.size = size

    def __call__(self, x, y):
        '''改变实体的位置'''
        self.x, self.y = x, y
```

## 会话管理

使用with语句来调用

```python
class Closer:
'''通过with语句和一个close方法来关闭一个对象的会话管理器'''

    def __init__(self, obj):
        self.obj = obj

    def __enter__(self):
        return self.obj # bound to target

    def __exit__(self, exception_type, exception_val, trace):
        try:
            self.obj.close()
        except AttributeError: # obj isn't closable
            print('Not closable.')
        return True # exception handled successfully
```



## `__slots__`

限制class的属性，但不限制class的子类的属性，除非子类也使用`__slots__`

例子：

使用`__slots__`会比不使用，减少40%~50%的内存

```python
# 不使用__slots__
class MyClass(object):
  def __init__(self, name, identifier):
      self.name = name
      self.identifier = identifier
      self.set_up()

# 使用__slots__
class MyClass(object):
  __slots__ = ['name', 'identifier']
  def __init__(self, name, identifier):
      self.name = name
      self.identifier = identifier
      self.set_up()
```
