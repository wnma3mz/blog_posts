---
title: ssh笔记（端口转发）
date: 2018-02-11 19:40:56
tags: [ssh, linux]
mathjax: true
categories: [奇技淫巧]
---

ssh端口转发

<!-- more -->


说明：localhost表示本地主机，`localuser`表示本地用户（内网用户），user表示远程服务器用户名，host表示远程服务器ip.

注意：内网电脑也必须要开启ssh服务，即安装openssh-server服务

```bash
# 利用ssh连接远程服务器，-p 22表示22端口连接，这里不加也行，ssh默认是22端口连接
ssh -p 22 user@remotehost
```

## 基本操作

```bash
#　本地端口转发，将本地的2222端口转发至本机的22端口
ssh -L 2222:localhost:22 user@remotehost
# 在本地（localhost）执行这个命令，输入密码。首先会连接到远程服务器上
# 接下来，在本地（localhost）另开一个终端，执行下面命令，也可以达到连接远程服务器的作用
# user同上面命令的user，localhost是本地（照抄即可）
ssh -p 2222 user@localhost


# 远程端口转发，将本地的22端口转发至远程的2222端口
ssh -R 2222:localhost:22 user@remotehost
# 在本地（localhost）执行这个命令，输入密码。首先会连接到远程服务器上
# 接下来，在远程服务器上执行下面命令，可以连接到本地（localhost）
# localuser表示的是localhost的用户名，localhost表示本地（远程服务器的本地，照抄即可）
ssh -p 2222 localuser@localhost


# 动态端口转发，本机的2222端口绑定远程的某个端口(socks5流量)
ssh -D 2222 user@remotehost
# 在本地（localhost）执行这个命令，输入密码。首先会连接到远程服务器上
# 接下来，不关闭终端，更改本地浏览器的代理服务器设置，流量便通过代理服务器进行转发
```

## 参数详解

```bash
-L [bind_address:]port:host:hostport
```

Specifies that connections to the given TCP port or Unix socket on the local (client) host are to be forwarded to the given host and port, or Unix socket, on the remote side.  This works by allocating a socket to listen to either a TCP port on the local side, optionally bound to the specified bind_address, or to a Unix socket.  Whenever a connection is made to the local port or socket, the connection is forwarded over the secure channel, and a connection is made to either host port hostport, or the Unix socket remote_socket, from the remote machine.

Port forwardings can also be specified in the configuration file. Only the superuser can forward privileged ports.  IPv6 addresses can be specified by enclosing the address in square brackets.

By default, the local port is bound in accordance with the GatewayPorts setting.  However, an explicit bind_address may be used to bind the connection to a specific address.  The bind_address of “localhost” indicates that the listening port be bound for local use only, while an empty address or ‘*’ indicates that the port should be available from all interfaces.

将本地主机转发到远程主机和端口。也可以直接在配置文件中指定。默认本地端口是根据GatewayPorts设置绑定的，可以直接使用bind_address将连接绑定到指定地址。如果bind_address为localhost表示监听端口只在本地使用，如果不填或*表示应用所有端口

```bash
-R [bind_address:]port:host:hostport
```

Specifies that connections to the given TCP port or Unix socket on the remote (server) host are to be forwarded to the given host and port, or Unix socket, on the local side.  This works by allocating a socket to listen to either a TCP port or to a Unix socket on the remote side. Whenever a connection is made to this port or Unix socket, the connection is forwarded over the secure channel, and a connection is made to either host port hostport, or local_socket, from the local machine.

Port forwardings can also be specified in the configuration file. Privileged ports can be forwarded only when logging in as root on the remote machine. IPv6 addresses can be specified by enclosing the address in square brackets.

By default, TCP listening sockets on the server will be bound to the loopback interface only.  This may be overridden by specifying a bind_address.  An empty bind_address, or the address ‘*’, indicates that the remote socket should listen on all interfaces.  Specifying a remote bind_address will only succeed if the server's GatewayPorts option is enabled (see sshd_config(5)).

If the port argument is ‘0’, the listen port will be dynamically allocated on the server and reported to the client at run time.  When used together with -O forward the allocated port will be printed to the standard output.

基本同-L参数命令，如果port为0，监听端口将会动态进行分配。

```bash
-D [bind_address:]port
```

Specifies a local “dynamic” application-level port forwarding.  This works by allocating a socket to listen to port on the local side, optionally bound to the specified bind_address.  Whenever a connection is made to this port, the connection is forwarded over the secure channel, and the application protocol is then used to determine where to connect to from the remote machine.  Currently the SOCKS4 and SOCKS5 protocols are supported, and ssh will act as a SOCKS server.  Only root can forward privileged ports.  Dynamic port forwardings can also be specified in the configuration file.

IPv6 addresses can be specified by enclosing the address in square brackets.  Only the superuser can forward privileged ports. By default, the local port is bound in accordance with the GatewayPorts setting. However, an explicit bind_address may be used to bind the connection to a specific address. The bind_address of “localhost” indicates that the listening port be bound for local use only, while an empty address or ‘*’ indicates that the port should be available from all interfaces.

指定本地动态端口转发。通过分配一个socket来监听本地端口。支持SOCKS4和SOCKS5协议，相当于ssh充当SOCKS服务器。配置文件中也可以指定。bind_address为localhost表示监听端口仅限本地使用，如果为空或者*表示该端口可通过所有接口
```
ssh部分参数介绍

可以用在上面的命令中

​```bash
-f 后台认证用户密码，即不用登录到远程主机。通常与-N连用
-C 压缩数据传输
-N 不执行脚本或者命令。通常与-f连用
-g 允许远程主机连接到建立的转发端口。
-q 静默模式，使大多数警告和诊断消息被压制。（如果有警告信息，不进行输出）
```

举个栗子

```bash
# 本地端口转发。命令执行后，不登录到远程服务器，压缩数据传输，也不输出警告信息
ssh -gfCNL 2222:localhost:22 user@remotehost -q

# 如果将命令放到了后台，关掉进程需要使用kill命令
```

如果网络不稳定可以考虑使用`autossh`。这个需要额外安装，使用的时候将`ssh`替换为`autossh`即可。它的工作原理简单来说，就是有个超时机制，如果中断，便重新连接。



## 用途介绍

1. 本地端口转发。假设有两台服务器A和B，B需要访问A上面的一个应用（网站）。

   ```bash
   # 在B上使用本地端口转发，将某个端口转发给A应用的端口。
   # B机器
   ssh -L 2333:IP_A:80 localuser@localhost
   # 此时访问localhost:2333等于pi_A:80
   
   # 比如A为公网IP（11.11.11.11），在6000端口开启某项服务
   # B通过在本机使用, 本地用户名user
   ssh -L 2333:11.11.11.11:6000 user@localhost
   # 即可通过localhost:2333来代替11.11.11.11
   ```

2. 本地端口转发高级版（多机版）。同上面的假设，但是现在有三台或者三台以上的服务器A、B、C……这个时候，使用`-g`参数，运行远程主机连接转发端口。这样便可以达到C访问B再访问A的目的。注意：确保ABC三台服务器网络连接是安全的，请谨慎。

3. 远程端口转发。假设有一台服务器（公网）A，两台内网计算机B、C。

   ```bash
   # B、C均可连接A，但是其他两两之间不能相互连接。
   # 现在需要实现两两之间都能互相连接
   
   # A连接到B、C
   # B、C远程端口转发，A机器（11.11.11.11），用户名user
   # B机器
   ssh -R 2221:localhost:22 user@11.11.11.11
   # C机器
   ssh -R 2222:localhost:22 user@11.11.11.11
   
   # 这样在A机器执行如下命令就能连接B、C了
   # 连接B
   ssh userB@localhost -p2221
   # 连接C
   ssh userC@localhost -p2222
   
   # B、C之间互连
   # 其实就是通过A作为跳板来登录，即先登录A
   ssh user@11.11.11.11
   # 再登录B or C即可
   ```

4. 动态端口转发。在本地使用这个命令连接到服务器之后，便可以使用服务器的SOCK5代理来上网。需要在浏览器或者系统上进行下面的设置。

   ```bash
   # A机器（user，11.11.11.11）
   # B机器上执行
   ssh -D 1234 user@11.11.11.11
   # 通过设置SOCKS5代理，socks5://127.0.0.1:1234，就可以通过A机器的流量上网。可查询ip来检验是否成功
   ```

   

注意事项：

1. 端口转发是通过ssh连接建立的，所以关闭了端口，端口转发也会关闭
2. 选择远程端口号的时候，一般是无权绑定`1-1023`端口的，只能使用管理员权限才能绑定。一般是使用` 1024-65535`之间的一个端口

## 内网流量转发

内网转发内网流量，A，C都在局域网下，二者不可相互连接。B是公网服务器，A，C皆可连接B，但B不可以连接A，C。现在需求是A借助C的流量上网。

```bash
# A机器执行。user和remotehost是B机器的用户名和ip
ssh -L localhost:3000:localhost:2000 user@remotehost
# C机器执行。user和remotehost是B机器的用户名和ip
ssh -R 2222:localhost:22 user@remotehost
# B机器执行。user是C机器的用户名
ssh -D 2000 -p 2222 user@localhost

# 或两条命令，其实就是两条并一条
# C机器
ssh -R 2222:localhost:22 user@remotehost
# A机器
ssh -L localhost:3000:localhost:2000 user@remotehost -t ssh -D 2000 -p 2222 user@localhost

# A机器上代理：SOCKS5://127.0.0.1:3000
```

## 一键命令

```bash
# A在内网，B在公网，已通过B连接A。-R命令
# BUSER和BHOST对应正常连接B的SSH用户名和密码，AUSER为B连接A的用户名，port为-R参数的port

# 此时需要在C机器上，先连接B，再通过B连接A。C机器执行命令。
ssh -t BUSER@BHOST ssh AUSER@localhost -p port

# or （Win10上测试失败）
ssh BUSER@BHOST -J AUSER@localhost:port
```

```powershell
# 一键连接内网服务器，windows下的config配置
# jump为跳板服务器，jump_server_ip、username、port分别对应跳板机的ip、用户名和连接端口
# target_server为内网服务器，基本同上，ProxyCommand配置如下所示，需要使用ssh命令的绝对路径
Host jump
    ForwardAgent yes
    HostName jump_server_ip
    User username
    Port port

Host target_server
    ForwardAgent yes
    HostName target_server_ip
    User username
    Port port
    ProxyCommand C:\\Windows\\System32\\OpenSSH\\ssh.exe -q -W %h:%p jump
```

## 注意事项

win10开启ssh服务器端，1. 应用和功能：下载openssh服务端；2. 服务：开启openssh server服务；3. 连接用户名win10系统盘的用户文件下的文件名，比如我的是Administrator（即和公用文件夹并列的文件夹）。









参考文章

[SSH原理与运用（二）：远程操作与端口转发](http://www.ruanyifeng.com/blog/2011/12/ssh_port_forwarding.html)

[实战 SSH 端口转发](https://www.ibm.com/developerworks/cn/linux/l-cn-sshforward/)

[SSH隧道与端口转发及内网穿透](http://blog.creke.net/722.html)

[利用反向ssh从外网访问内网主机](https://blog.mythsman.com/2017/01/14/1/)

[SSH隧道翻墙的原理和实现](http://www.pchou.info/linux/2015/11/01/ssh-tunnel.html)

[玩转SSH端口转发](https://blog.fundebug.com/2017/04/24/ssh-port-forwarding/)

[openssh的三种tcp端口转发参数](https://www.ibm.com/developerworks/community/blogs/5144904d-5d75-45ed-9d2b-cf1754ee936a/entry/20160911?lang=en)

https://github.com/microsoft/vscode-remote-release/issues/230