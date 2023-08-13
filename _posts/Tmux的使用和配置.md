---
title: Tmux 的使用和配置
date: 2023/08/13 17:11:22
tags: [笔记]
categories: []
mathjax: false
katex: false
---

Tmux 的常用操作

<!-- more -->

以下内容由Claude生成

# Tmux 的使用和配置

## Tmux简介

Tmux是一个终端复用软件,可以在一个终端窗口中管理多个会话和窗口。

## Tmux的基本操作

### 新建Tmux会话

```bash
tmux new -s session_name
```

### 重新连接到指定会话

```bash
tmux a -t session_name
```

### 退出当前会话

```bash
exit
```

### 列出所有会话

```bash
tmux ls
```

### 终止指定会话

```bash
tmux kill-session -t session_name
```

### 终止所有会话

```bash
tmux kill-server
```

### 切换会话窗口

```
ctrl+b then c  # 切换到下一个窗口
ctrl+b then p  # 切换到上一个窗口
ctrl+b then n  # 切换到下一个窗口
ctrl+b then number # 切换到指定窗口,number为窗口编号
```

### 创建新窗口

```
ctrl+b then c # 创建新窗口并切换到新窗口
```

### 关闭当前窗口

```
ctrl+b then & # 关闭当前窗口
```

### 切换窗口布局

```
ctrl+b then space # 切换布局,默认包括even-horizontal、even-vertical、main-horizontal、main-vertical、tiled布局
```

### 横向和纵向分割窗口

```
ctrl+b then " # 横向分割窗口
ctrl+b then % # 纵向分割窗口  
```

### 切换到指定窗口

```
ctrl+b then o # 切换到下一个分割窗口
ctrl+b then 方向键 # 切换到指定窗口
```

### 调整分割窗口大小

```
ctrl+b then alt+方向键 # 调整分割窗口大小
```

## Tmux的配置

### 启用鼠标支持

```
set -g mouse on
```

### 设置自动命名窗口

```
set-window-option -g automatic-rename on
set-option -g allow-rename off
```

### 复制模式

```
setw -g mode-keys vi # 设置vi模式
bind-key -T copy-mode-vi v send-keys -X begin-selection # 开始选择文本 
bind-key -T copy-mode-vi y send-keys -X copy-selection # 复制选择的文本
```

### 其他常用配置

```
# 鼠标支持
set -g mouse on

# 启用状态栏
set -g status on

# 设置状态栏颜色   
set -g status-style fg=white,bg=black

# 自动重命名窗口 
setw -g automatic-rename on

# 启用活动告警
setw -g monitor-activity on

# 设置命令前缀为Ctrl+a
set -g prefix C-a
unbind C-b
bind C-a send-prefix

# 复制模式
bind -T copy-mode-vi v send -X begin-selection
bind -T copy-mode-vi y send -X copy-selection
```

以上就是Tmux的常用操作和配置,可以根据自己的需要进行调整和扩展。Tmux可以大大提高我们的工作效率,推荐经常使用终端的同学都去学习和使用。