---
title: 年度推荐软件Tailscale：连接团队设备和开发环境的利器
date: 2023/04/03 16:41:22
tags: [笔记,工具]
categories: []
mathjax: false
katex: false
---
本文介绍了 Tailscale 这款软件可以轻松架设异地组网，解决调试难题、实现流量转发和开放服务等多个需求。

<!-- more -->

[Tailscale](https://tailscale.com/)Tailscale connects your team's devices and development environments for easy access to remote resources. （摘自官网）

于笔者而言，Tailscale能够用一个软件轻松的解决三大需求。

- 异地组网：软件主打功能。就是将物理位置不同的机器通过该软件，能够使它们处于同一网络环境下，方便远程调试。
- 流量转发：A机器在A地，可以借助B机器在B地的网络进行上网。
- 开放服务：A机器自建Web服务，不仅可以在组网环境下使用，甚至官方提供了公网的链接（https），即可以不在组网环境下也能够访问。

写在开头，免费版的权益如下图所示，支持20个设备，且域名不可自定义（子域名可以），速度在100-300k/s左右。在Android和iOS设备上不支持同时开启两个VPN，即Clash无法和Tailscale同时使用。

![](https://raw.githubusercontent.com/wnma3mz/blog_posts/master/imgs/tailscale/1680514288836.png)

## 安装

首先需要在官网进行注册，值得一提的是注册方式最好选用能够在日常网络环境下能打开的账号。比如选用Github注册的话，之后所有的设备都需要使用Github登录才能组网。其目前无法在一个账号上绑定多种登录方式。

官网提供了详尽的[下载](https://tailscale.com/download)安装说明，移动端可以直接下载对应的App，个人电脑和服务器可以使用常规的命令和软件包的方式进行下载安装。故对于一般的机器不做介绍，本节主要介绍路由器、树莓派、群晖的安装方法。其他机型，可以查看这个[地址](https://pkgs.tailscale.com/stable/)。

对于Linux设备，安装步骤主要分两步：

1. 启动服务，`sudo tailscale up`
2. 加入组网，`tailscale ip -4`

对于第二步，输入完成后会有一个链接需要使用对应账号进行登录，可以将链接复制到其他设备用浏览器打开登录。在非常规机器上，首先需要确认机器的CPU架构，命令 `uname -a`。

### 群晖

群晖的[下载地址](https://pkgs.tailscale.com/stable/#spks)，需要根据系统版本以及CPU架构选择下载。下载完成后，打开群晖的**套件中心**，右上角的**手动安装**，上传安装包，按照步骤完成安装即可。

![](https://raw.githubusercontent.com/wnma3mz/blog_posts/master/imgs/tailscale/170356.png)

安装完成后，点击对应的图标，安装对应的步骤即可启动。

### 路由器

路由器的[下载地址](https://pkgs.tailscale.com/stable/#static)，根据CPU架构选择下载。并将安装包上传至路由器，根据对应的路由器安装命令进行解压安装。

### 树莓派

树莓派的[下载地址](https://pkgs.tailscale.com/stable/#raspbian-stretch)，根据型号三选一下载。解压安装。树莓派Zero的是Raspbian Stretch

## 异地组网

当设备安装完成，并登录后，该设备会自动加入组网。可以在[设备管理](https://login.tailscale.com/admin/machines)界面查看所有的设备，如下图所示。

![](https://raw.githubusercontent.com/wnma3mz/blog_posts/master/imgs/tailscale/171953.png)

第一列的设备名称可以根据自己的喜好进行更改，并且可以直接作为域名使用。假设有个机器叫做AAA且用户名为root，那么可以直接 `ssh root@AAA`进行登录。或者群晖名为AAA，那么浏览器直接打开AAA就能访问。

登录的设备默认是若干天（最长180）后需要重新登录一次，可以在最后一列三个点进行编辑，对该功能进行禁用。禁用成功后，会在第一列显示 `Expiry disabled`

![](https://raw.githubusercontent.com/wnma3mz/blog_posts/master/imgs/tailscale/172831.png)

## 流量转发

本节参考[文档](https://tailscale.com/kb/1019/subnets/)

正如开头的图显示的，该功能仅支持以一个设备作为流量出口。当有一个局域网的所有设备不方便接入组网时，可以通过接入该局域网的一个设备（如路由器），再访问该局域网的其他设备。如下图，`Subnet Router`相当于桥梁，承接了组网和本地组网。组网内的设备可以通过Router访问局域网内的设备，反之亦可。接下来在 `Router`设备上操作。

![](https://tailscale.com/kb/1019/subnets/subnets.png)

### Step 1

```bash
# 开启端口转发。有的机器可能默认开启，无需操作。
echo 'net.ipv4.ip_forward = 1' | sudo tee -a /etc/sysctl.d/99-tailscale.conf
echo 'net.ipv6.conf.all.forwarding = 1' | sudo tee -a /etc/sysctl.d/99-tailscale.conf
sudo sysctl -p /etc/sysctl.d/99-tailscale.conf
```

```bash
# 创建Subnet，192.168.x.y可以换成10.0这种。可以只写一个，x.y替换任意数字
sudo tailscale up --advertise-routes=192.168.x.y/24,192.168.x.z/24
```

### Step 2

在机器管理界面，点击对应机器的右侧三个点，再点击 `Edit route settings`。如下图所示，两个功能均开启。若仅需实现设备访问，不需要流量转发，则可以不开启第二个。

![](https://raw.githubusercontent.com/wnma3mz/blog_posts/master/imgs/tailscale/1680522033220.png)

### Step 3

自此，完成了桥接局域网和组网的功能。要实现流量转发，在Tailscale中叫做Exit Node。一方面需要在上一步开启第二个功能，此时Router就具备了作为流量出口的能力。另一方面，假设此时组网内的A设备需要借助Router的流量出口，就需要在A设备上进行操作。

在Router设备上，

```bash
# 创建Subnet，192.168.x.y可以换成10.0这种。可以只写一个，x.y替换任意数字
sudo tailscale up --advertise-exit-node --advertise-routes=192.168.x.y/24
```

一般有图形化界面的机器，可以直接在软件层面完成。如Windows，此时系统的所有流量均为走这个接口。

**Windows**
![](https://tailscale.com/kb/1103/exit-nodes/exit-node-windows-menu.png)

**Linux**

```bash
sudo tailscale up --exit-node=<exit-node-ip>
```

该功能主要的用途有以下几点

- 本节开头所提到的，通过一个Router，从而无需所有设备参与链接。
- 流量转发，如果有一个节点在海外。那么可以以此来做跳板
- 本文开头所提到的，移动设备上无法与其他VPN并存，那么此时通过这种方式，在Router设备上做分流。

## 开放服务

截至本文完成日期，该功能还在Beta阶段，并且要求版本>=1.38.3。[详细文档](https://tailscale.com/kb/1223/tailscale-funnel/)

### Step 1

如果仅需在组网内访问，则可以跳过该步骤。

在[Access controls](https://login.tailscale.com/admin/acls)控制页面加入配置，新增 `nodeAttrs`和修改 `groups`，其中 `wnma3mz@github`替换为对应账号。

![](https://raw.githubusercontent.com/wnma3mz/blog_posts/master/imgs/tailscale/1680523192944.png)
![](https://raw.githubusercontent.com/wnma3mz/blog_posts/master/imgs/tailscale/1680523204260.png)

```json
    "groups": {
         "group:can-funnel": [
         "wnma3mz@github",
         ],
     },
     "nodeAttrs": [
         {
             "target": ["group:can-funnel"],
             "attr":   ["funnel"],
         },
     ],
```

### Step 2

此时，需要在A机器上开启服务。申请[HTTPS证书](https://tailscale.com/kb/1153/enabling-https/)，下面的命令会提示后面需要加入什么域名。按照提示复制再执行即可。可能会有一些报错，需要根据不同情况进行更改。该功能可能会影响提供公网服务，但不影响组网服务。

```bash
tailscale cert
```

在开放服务前有两个命令需要进行区分，`tailscale serve`和`tailscale funnle`前者是开放服务到组网环境，后者是开放到整个互联网环境。

常用有两个技巧

- 把已经开放服务的端口转发到其他端口，或者主域名下。比如这里可以把5000端口转发到主域名下的/api路由下，即可以通过 `http://A/api`访问A机器的5000端口服务。`--bg`表示后台运行，默认是在前台启动，便于`Ctrl+C`关闭。

```bash
tailscale serve --bg --set-path /api http://127.0.0.1:5000/
```

把 4747 端口转发到 8443 端口，即可以通过 `https://A:8443`访问A机器的 4747 端口服务。

```bash
tailscale serve https:8443 / http://localhost:4747
```

- 部署静态服务

假设本地有一个`/data/index.html`文件，想要开放访问。则运行下面的命令，可以通过`http://A/index`访问A机器的`/data/index.html`文件。

```bash
tailscale serve --bg --set-path /index /data/index.html
```

如果`index.html`文件用了本机其他静态资源，同样需要进行挂载，比如挂载`/data/static`文件夹

```bash
tailscale serve --bg --set-path /static/ /data/static
```

如果要关闭服务，可以使用下面的命令，仅关闭这一条。

```bash
tailscale serve --bg --set-path /api off
```

如果希望指定端口，则可以用下面命令，默认端口为 80。

```bash
# 带端口开放
tailscale serve --bg --http 10001 --set-path /index /data/index.html
# 带端口关闭
tailscale serve --bg --http 10001 --set-path /index off
```

### Step 3

对外开放服务，下面的命令会直接把原本仅能在局域网内访问的服务提供到公网。实测下载速度200-300k/s，有些地方不能直接访问，可能要挂梯子。目前 tailscale 仅支持三个端口可以dui外开放，分别是 443、8443、10000。

```bash
tailscale funnel --bg https+insecure://localhost:443
tailscale funnel --bg https+insecure://localhost:8443
tailscale funnel --bg https+insecure://localhost:10000
```

使用方法同`serve`，换成`funnel`即可。

```bash
# 查看状态的命令
tailscale funnel status
# 关闭状态的命令
tailscale funnel 443 off
```
