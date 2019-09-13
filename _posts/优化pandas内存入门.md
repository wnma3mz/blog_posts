---
title: 优化pandas内存入门
date: 2018-03-17 16:11:12
tags: [Python, pandas]
categories: [数据]
---
在平常是用`pandas`的时候，虽然用的很愉快，但遇到数据量很大的时候总是很慢，这篇文章主要介绍了一些简单优化`dataframe`使用内存的方法，大大提高`pandas`使用的效率。

<!-- more -->

## 简单的概念

```python
# coding: utf-8
import pandas as pd

# 读取数据
df = pd.read_csv("test.csv")

# 得到精确的内存信息
df.info(memory_usage='deep')

# 说明：之后都用df来表示读取到的dataframe
```

pandas中每一个数据类型都有一个专门的类来处理。

- `ObjectBlock`： 字符串列的块
- `FloatBlock`： 浮点数列的块
- `Numpy ndarray`：整型和浮点数值的块（非常快，用C数组构建的）


```python
for dtype in ['float', 'int', 'object']:
    # 选中对应的dtype列
    selected_dtype = df.select_dtypes(include=[dtype])
    # 查看内存使用量的平均值
    mean_usage_b = selected_dtype.memory_usage(deep=True).mean()
    # 获取到的数据单位为K, 这里转换一下
    mean_usage_mb = mean_usage_b / 1024 ** 2
    print("Average memory usage for {} columns:{:03.2f} MB".format(dtype, mean_usage_mb))
```

这里我们可以发现`object`类占用内存最大

### 子类型(subtype)

| memory usage | float   | int   | uint   | datetime   | bool | object |
| ------------ | ------- | ----- | ------ | ---------- | ---- | ------ |
| 1 bytes      |         | int8  | uint8  |            | bool |        |
| 2 bytes      | float16 | int16 | uint16 |            |      |        |
| 4 bytes      | float32 | int32 | uint32 |            |      |        |
| 8 bytes      | float64 | int64 | uint64 | datetime64 |      |        |
| variable     |         |       |        |            |      | object |

一个`int8`类型值使用1个字节的存储空间，可以表示256(2^8)个二进制数，即-128到127的所有整数值。根据表可以得出，对于每个列，应该尽可能"抠"一点， 物尽其用。这里的`uint`表示无符号整型，可以更有效地处理正整数的列。

## 优化

### 数值型数据

这里可以使用`to_numeric()`对数值类型进行`downcast`（向下转型）的操作。

```python
# 选中整数型的列
df_int_columns = df.select_dtypes(include=['int'])
# 向下转型
df_int_columns.apply(pd.to_numeric, downcast='unsigned')

# 选中浮点型数据
df_float_columns = df.select_dtypes(include=['float'])
# 向下转型
df_float_columns.apply(pd.to_numeric, downcast='float')
```

这里能够有效的利用内存，但是缺点很明显的就是只能用于数值型数据，而且优化的空间有限。

### `object`型数据

`object`类型实际上使用的是Python字符串对应的值，由于Python解释性语言的特性，字符串存储方式很碎片化（消耗更多内存，访问速度更慢）。`object`列中的每个元素都是一个指针。`object`类型的数据内存使用情况是可变的。如果每个字符串是单独存储的，那么实际上字符串占用内存是很大。

```python
In [1]: from sys import getsizeof
# 字符串本身内存使用情况
In [2]: s1 = "hello world!"
In [3]: s2 = "hello pandas!"
In [4]: getsizeof(s1), getsizeof(s2)
Out[4]: (61, 62)

# 字符串(object)在pandas中内存使用情况
In [9]: import pandas as pd
In [10]: ser = pd.Series([s1, s2])
In [11]: ser.apply(getsizeof)
Out[11]:
0    61
1    62
dtype: int64
```

这里通过观察看到，在这里pandas使用了int64来存储，实际占用大小与字符串本身是一样的。这里可以使用`Categoricals`来优化`object`类型。`Categoricals`的工作原理我理解为，某个`object`类有有限的分类情况(比如只有`red`、`yellow`、`blue`、`black`等颜色相关)，那么`Categoricals`将这些分类`object`对象转换为`int`子类型（对应上面的0, 1, 2, 3）

```python
# 选中object类型
df_obj_columns = df.select_dtypes(include=['object'])
# 查看object类型列的相关信息
df_obj_columns.describe()
```

如果`unique`(不同值)的数量很少，那么就可以使用这种优化方案。

```python
# 假设某一列符合优化条件，为df_obj_less，使类型转换为category
df_category_column = df_obj_less.astype('category')
# 转换之后，主观观察数据没有什么区别，只是类型改变了
# 观察转换之后实际上的数据
df_category_column.cat.codes
# 变成了由0,1,2等数字构成的int8类型的数据
```

这里提升的空间远远超过第一步优化的空间，具体要结合数据来检验（最好貌似可以减少98%的使用量）。这里有个很大的缺点就是无法进行数值计算，即没有办法使用`pd.Series.min()`、`pd.Series.max()`等与数值相关操作。

还有一个问题就是在有多少个`unique`(不同值)的情况下才使用这种方法。首先，毫无疑问的是如果需要计算操作的列是不能使用的。第二是如果`unique`的比例小于50%(个人觉得比例应该更小)就可以使用这种情况，如果过多的话，转换之后消耗的内存会更多（不仅需要存储string还有int）。

附上筛选`unique`比例小于50%的代码

```python
# 提取unique少于50%的object列
df_obj_columns = df.select_dtypes(include=['object'])
for obj_col in df_obj_columns:
    unique_values = len(df_obj_columns[obj_col].unique())
    total_values = len(df_obj_columns[obj_col])
    if unique_values / total_values < 0.5:
        obj_col.astype('category')

# 建议提前copy一份数据，再做上面的操作
```

### `datetime`类型

这里其实不能叫做优化，因为这里是将数值型数据转换为`datetime`类型，虽然提高了内存的使用，但是转换为`datetime`类型的数据更容易进行分析（时间序列分析）。当然如果有`datetime`类型不在分析的范围内，自然可以无视。

```python
# 假设df_num_col为需要转换的列, 这里format格式看情况更改
df_num_col = pd.to_datetime(df_num_col, format="%Y%m%d")
```

## 总结

写了这么多，如果每次等加载完数据再做优化操作，感觉有些鸡肋。但是其实可以使用`read_csv`等读取函数的几个参数来帮助我们在读取的时候就完成优化步骤。

```python
# 首先，需要整理出每一列最终的数据类型，组成一个dict
dtypes_lst = [col.name for name in df.dtypes]
column_types = dict(zip(dtypes.index, dtypes_lst))

# 重新读取数据, dtype指定数据格式，parse_dates指定列转换格式为datetime,infer_datetime_format尝试解析字符串形式的datatime格式（在一些情况下可以加快解析速度5-10倍）
new_df = pd.read_csv('test.csv', dtpye=column_types, parse_dates=['date'], infer_datetime_format=True)
```



