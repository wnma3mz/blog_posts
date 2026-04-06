---
title: Bash 操作查询
date: 2025-09-06 17:33:52
tags: [Bash, Shell]
categories: [Reference]
---

Bash 操作查询

<!-- more -->

### 如何统计当前目录下的文件数量？

要统计**只包含当前目录**的文件数量（不包括子目录），可以使用 `ls`、`grep` 和 `wc` 这几个命令的组合。

```bash
ls -l | grep "^-" | wc -l
```

  * `ls -l`：以长格式列出当前目录下的所有文件和目录。
  * `grep "^-"`：过滤 `ls -l` 的输出，只保留以 `-` 开头的行。在长格式列表中，`-` 代表这是一个文件。
  * `wc -l`：统计过滤后的行数，即文件的总数。

如果你想**同时统计子目录**中的所有文件，`find` 命令会更有效：

```bash
find . -type f | wc -l
```

  * `find .`：从当前目录（`.`）开始查找。
  * `-type f`：指定查找类型为文件（`f`）。
  * `wc -l`：统计找到的文件总数。

-----

### 如何只显示目录列表中的第一个文件？

可以使用 `ls` 和 `head` 命令配合来完成。

使用 `ls -1` 选项，它会让每个文件单独占一行。然后通过管道传递给 `head -n 1`，就可以只获取第一行。

```bash
ls -1 | head -n 1
```

另一种方法是使用 `ls -m` 选项，它会用逗号分隔的方式列出文件。同样，结合 `head` 也能达到目的。

```bash
ls -m | head -n 1
```

-----

### 如何将一个 JSONL 文件中的每一行复制多遍？

使用 `awk` 命令可以轻松实现这个功能。下面的例子将 `your_input.jsonl` 文件中的每一行都复制 5 遍，然后输出到 `your_output.jsonl` 文件中。

```bash
awk '{for (i=1; i<=5; i++) print}' your_input.jsonl > your_output.jsonl
```

-----

### 如何搜索文件中的特定文本，并显示文件名和行号？

`grep` 命令是完成这项任务的最佳工具。

要在当前目录下所有文件中搜索“Hello World\!”，并显示文件名和对应的行号，可以使用 `-H`（显示文件名）和 `-n`（显示行号）这两个参数。

```bash
grep -Hn "Hello World!" *
```

如果你想搜索包含“Hello World\!”但**不包含**“XXX”的行，可以将结果通过管道传递给另一个 `grep`，并使用 `-v`（反向匹配）参数。

```bash
grep -Hn "Hello World!" * | grep -v "XXX"
```

-----

### 如何批量结束所有与某个程序相关的进程？

如果要结束所有正在运行的 Python 脚本，可以结合 `ps`、`grep`、`awk` 和 `xargs` 这几个命令。

```bash
ps -ef | grep python | grep -v grep | awk '{print $2}' | xargs kill -9
```

  * `ps -ef`：列出所有正在运行的进程。
  * `grep python`：从中筛选出包含“python”关键字的进程。
  * `grep -v grep`：这一步非常关键，它用于排除 `grep` 命令本身，避免其被误杀。
  * `awk '{print $2}'`：打印出进程列表中的第二列，即进程 ID（PID）。
  * `xargs kill -9`：将 `awk` 输出的 PID 作为参数传递给 `kill -9` 命令，强制终止这些进程。


### 查找 a.txt 中每行出现「key」的次数，必须要等于 2 次

```bash
awk '/key/{count++} END {print count}' a.txt | grep -c '^2$'
```

* awk '/key/{count++} END {print count}' a.txt: 这个命令使用 awk 来统计每行中“key”出现的次数。
    * /key/: 这是一个正则表达式，匹配包含“key”的行。
    * count++: 对于匹配到的每一行，计数器 count 加 1。
    * END {print count}: 在处理完所有行后，打印每行“key”出现的总次数。 注意，这里是每行出现的总次数，而不是所有行加起来的总次数。
* grep -c '^2$': 这个命令使用 grep 过滤 awk 的输出，并统计匹配的行数。
    * `'^2$'`: 这是一个正则表达式。 `^` 表示行首，2 表示数字 2，$ 表示行尾。 因此，这个正则匹配只包含数字 2 的行。 因为awk输出的是每行“key”出现的次数，所以这里匹配2表示出现了两次的行。
    * -c: grep 的 -c 选项表示统计匹配到的行数。
