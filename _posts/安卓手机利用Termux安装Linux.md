---
title: 安卓手机利用Termux安装Linux
date: 2021-07-21 21:36:28
tags: [linux, 安卓]
categories: [笔记]
---
在手机上利用Termux安装Linux，免root

<!-- more -->

### 安装

不要从Google Play上安装，正确的安装包地址[f-droid](https://f-droid.org/packages/com.termux/)

下载完成后，正常安装，但需要额外的给存储空间权限。

虽然说可以免root安装，但没有root权限，有些操作不能很好的完成，如`top`

### 连接

在手机上配置太麻烦，所以可以使用同一局域网下的电脑进行配置

```bash
# 手机上
# 安装ssh服务
pkg install openssh -y
# 开启ssh服务器
sshd
# 配置密码
passwd
# 输入想要配置的密码

# 电脑终端, ip填写对应的手机ip，8022是termux默认的ssh端口
ssh root@[ip] -p8022
# 输入刚刚的密码
```

至此，就能在电脑端进行配置

### 配置

```bash
# 安装第三方库
pkg install proot git python wget -y

# 一键安装项目
git clone https://github.com/sqlsec/termux-install-linux.git
cd termux-install-linux
python termux-linux-install.py
# 选择Ubuntu
cd ~/Termux-Linux/Ubuntu
./start-ubuntu.sh

# 或者，ubuntu版本，最新支持20.04
git clone https://github.com/MFDGaming/ubuntu-in-termux
cd ubuntu-in-termux
chmod +x ubuntu.sh
./ubuntu.sh -y
./startubuntu.sh
```

这样安装的Linux是与Termux共享文件目录的，所以可以直接将文件拷贝在`ubuntu-fs/root`文件夹中，启动`./startubuntu.sh`后，就能直接看到文件。

注：若进入系统后，无法更新软件源，可能是dns存在问题，可以在termux中更改dns解析（因为系统中不好编辑）。

```bash
vim ubuntu-fs/etc/resolv.conf

`
nameserver 114.114.114.114
nameserver 8.8.8.8
`
# 若依旧不行，可尝试其他dns，如1.1.1.1等
```



### 其他

```bash
pkg install vim tmux -y

# frp项目地址，手机请注意下载arm版本，一般为arm64，不能直接在Linux系统中配置，必须得在Termux中操作
https://github.com/fatedier/frp/releases

# 查看手机温度，需要除以1000
cat /sys/class/thermal/thermal_zone0/temp

# 查看手机电池容量，需root
cat /sys/class/power_supply/battery/capacity

# 配置termux的启动操作
vim ~/.bashrc

`
sshd # 开启ssh服务，便于连接
# 配置frp相关操作，便于直接远程连接
# 配置进入Linux系统操作
`
```

