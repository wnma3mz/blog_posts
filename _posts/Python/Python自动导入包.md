---
title: Python 自动导入包
date: 2023/12/03 00:57:00
tags: [importlib]
categories: [Python]
---

在一个文件夹中，有很多个 Python 文件，每个文件都有若干重名函数，需要一一导入。这个时候，可以用 Python 的自动导入包来实现。

<!-- more -->

## 需求描述

假设有一个文件夹

```bash
func/
├── __init__.py
├── func1.py
├── func2.py
├── func3.py
```

每个文件都有一个重名函数，比如

```python
# func1.py
def foo():
    print("foo")
```

这不仅导致 `import` 代码不够简洁，且在新增文件时，还得在 `__init__.py` 中添加 `import` 语句。甚至在调用处也得增加代码来处理。

## 解决方案

Python 的自动导入包可以解决这个问题。

```python
import os
import glob
import importlib

# 获取当前文件夹下所有的 .py 文件
files = glob.glob(os.path.join(os.path.dirname(__file__), "*.py"))
func_dict = {}
for file_ in files:
    func_name = os.path.basename(file_)[:-3]
    load_module = importlib.import_module(f"func.{func_name}")
    globals()[load_module.__name__] = getattr(load_module, "foo")

    # 将其放到一个字典中，以按需处理，
    func_dict[func_name] = globals()[load_module.__name__]
```

这样，不仅不需要在 `func/__init__.py` 中添加 `import` 语句，也不需要在调用处添加额外的代码。甚至，还要根据需求，从字典中调用对应函数。当然，这需要提前进行约定，比如全部放到`func`文件夹下，且都用`foo`这个名字作为函数名。