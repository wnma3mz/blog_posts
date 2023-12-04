---
title: Python3的Tools-scripts整理
date: 2020-06-14 21:12:39
tags: [Scripts]
categories: [Python]
---

Python3的Tools-scripts整理

<!-- more -->

文件路径为`Anaconda3\Tools\scripts`

### byext.py

分析文件/文件夹

```python
# py文件
ext  bytes  files  lines  words
.py   2655      1     91    265
TOTAL   2655      1     91    265
ext  bytes  files  lines  words

# 二进制文件
   ext binary  bytes  files
<none>      1  10382      1
 TOTAL      1  10382      1
   ext binary  bytes  files

# 文件夹
  ext binary  bytes   dirs  files  lines  words
 .png      2   2823             2
  .py         10471             5    224    647
 .txt         17404             3    554    960
<dir>                    1
TOTAL      2  30698      1     10    778   1607
  ext binary  bytes   dirs  files  lines  words
```

|输出|主要语法|
| ----- | --------------------------------------- |
| ext   | `os.path.normcase(os.path.splitext[1])` |
| files | `os.path.isfile` |
| bytes | 以二进制(`rb`)读文件，`len`的输出       |
| lines | `str(data, 'latin-1').splitlines()`     |
| words | `len(data.split())`的输出               |
|binary|`if *b*'\0' in data`|
|dirs|os.path.isdir|
|lnk|os.path.islink|

### checkpip.py

```python
The latest version of setuptools on PyPI is 46.2.0, but ensurepip has 40.8.0
The latest version of pip on PyPI is 20.1, but ensurepip has 19.0.3
```

```python
# 主要是检查版本
import ensurepip

print(ensurepip._PROJECTS)
# [('setuptools', '40.8.0'), ('pip', '19.0.3')]
```

### checkpyc.py

检查pyc文件是否最新

```python
import importlib.util

print(importlib.util.MAGIC_NUMBER)
# b'B\r\r\n'
```

### cleanfuture.py

修改py文件的future语法

```python
import sys

def errprint(*args):
    strings = map(str, args)
    msg = ' '.join(strings)
    if msg[-1:] != '\n':
        msg += '\n'
    sys.stderr.write(msg)
    
errprint("Usage:", __doc__)
```

### crlf.py

文件中CRLF用LF替换

### lfcr.py

文件中LF用CRLF替换

迭代：`itertools.count()`

### reindent.py

py文件改为4空格缩进

### untabify.py

同上

### texi2html.py

texi to html



`import keyword`

`import webbrowser `
