---
title: ssh笔记（端口转发）
date: 2018-02-11 19:40:56
tags: [ssh, linux]
mathjax: true
categories: [奇技淫巧]
---

本篇文章只做一个神经网络入门知识的梳理和个人的理解。

<!-- more -->


说明：localhost表示本地主机，`localuser`表示本地用户（内网用户），user表示远程服务器用户名，host表示远程服务器ip.

注意：内网电脑也必须要开启ssh服务，即安装openssh-server服务

```bash
# 利用ssh连接远程服务器，-p 22表示22端口连接，这里不加也行，ssh默认是22端口连接
ssh -p 22 user@remotehost
```



```bash
#　本地端口转发
ssh -L 2222:localhost:22 user@remotehost
# 在本地（localhost）执行这个命令，输入密码。首先会连接到远程服务器上
# 接下来，在本地（localhost）另开一个终端，执行下面命令，也可以达到连接远程服务器的作用
# user同上面命令的user，localhost是本地（照抄即可）
ssh -p 2222 user@localhost


# 远程端口转发
ssh -R 2222:localhost:22 user@remotehost
# 在本地（localhost）执行这个命令，输入密码。首先会连接到远程服务器上
# 接下来，在远程服务器上执行下面命令，可以连接到本地（localhost）
# localuser表示的是localhost的用户名，localhost表示本地（远程服务器的本地，照抄即可）
ssh -p 2222 localuser@localhost


# 动态端口转发
ssh -D 2222 user@remotehost
# 在本地（localhost）执行这个命令，输入密码。首先会连接到远程服务器上
# 接下来，不关闭终端，更改本地浏览器的代理服务器设置，流量便通过代理服务器进行转发
```

ssh部分参数介绍

可以用在上面的命令中

```bash
-f 后台认证用户密码，即不用登录到远程主机。通常与-N连用
-C 压缩数据传输
-N　不执行脚本或者命令。通常与-f连用
-g 允许远程主机连接到建立的转发端口。
-q 静默模式，使大多数警告和诊断消息被压制。（如果有警告信息，不进行输出）
```

举个栗子

```bash
# 本地端口转发。命令执行后，不登录到远程服务器，压缩数据传输，也不输出警告信息
ssh -gfCNL 2222:localhost:22 user@remotehost -q

# 如果将命令放到了后台，关掉进程需要使用kill命令
```

如果网络不稳定可以考虑使用`autossh`。这个需要额外下载，使用的时候将`ssh`替换为`autossh`即可。它的工作原理简单来说，就是有个超时机制，如果中断，便重新连接。





关于用途

1. 本地端口转发。假设有两台服务器A和B，B需要访问A上面的一个应用（网站），但是这个应用只能在A上面使用。在B上使用本地端口转发，将某个端口转发给A应用的端口。`ssh -L 2333:<A>:80 localuser@localhost`。`<A>`表示的是A的ip，此时访问`localhost:2333`就能达到直接访问`ip:80`的效果。
2. 本地端口转发高级版（多机版）。同上面的假设，但是现在有三台或者三台以上的服务器A、B、C……这个时候，使用`-g`参数，运行远程主机连接转发端口。这样便可以达到C访问B再访问A的目的。注意：确保ABC三台服务器网络连接是安全的，请谨慎。
3. 远程端口转发。假设有一台服务器（公网）A，两台内网计算机B、C。换句话说，B可以访问A，C可以访问A，但是B、C之间也不能相互访问。假设现在让B可以访问C。在C上使用远程端口转发到A上，B连接A，再使用`ssh`命令即可连接。
4. 动态端口转发。在本地使用这个命令连接到服务器之后，便可以使用服务器的SOCK5代理来上网（具体就不可说了）。需要在浏览器或者系统上进行下面的设置。

注意事项：

1. 端口转发是通过ssh连接建立的，所以关闭了端口，端口转发也会关闭
2. 选择远程端口号的时候，一般是无权绑定`1-1023`端口的，只能使用管理员权限才能绑定。一般是使用` 1024-65535`之间的一个端口





内网转发内网流量，A，C都在局域网下，二者不可相互连接。B是公网服务器，A，C皆可连接B，但B不可以连接A，C。现在需求是A借助C的流量上网。

```bash
# A机器执行。user和remotehost是B机器的用户名和ip
ssh -L localhost:3000:localhost:2000 user@remotehost
# C机器执行。user和remotehost是B机器的用户名和ip
ssh -R 2222:localhost:22 user@remotehost
# B机器执行。user是C机器的用户名
ssh -D 2000 -p 2222 user@localhost

# A机器上代理：SOCKS5://127.0.0.1:3000
```



win10开启ssh服务器端，1. 应用和功能：下载openssh服务端；2. 服务：开启openssh server服务；3. 连接用户名win10系统盘的用户文件下的文件名，比如我的是Administrator（即和公用文件夹并列的文件夹）。



参考文章

[SSH原理与运用（二）：远程操作与端口转发](http://www.ruanyifeng.com/blog/2011/12/ssh_port_forwarding.html)

[实战 SSH 端口转发](https://www.ibm.com/developerworks/cn/linux/l-cn-sshforward/)

[SSH隧道与端口转发及内网穿透](http://blog.creke.net/722.html)

[利用反向ssh从外网访问内网主机](https://blog.mythsman.com/2017/01/14/1/)

[SSH隧道翻墙的原理和实现](http://www.pchou.info/linux/2015/11/01/ssh-tunnel.html)

[玩转SSH端口转发](https://blog.fundebug.com/2017/04/24/ssh-port-forwarding/)

[openssh的三种tcp端口转发参数](https://www.ibm.com/developerworks/community/blogs/5144904d-5d75-45ed-9d2b-cf1754ee936a/entry/20160911?lang=en)

