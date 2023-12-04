---
title: OpenWrt路由器折腾记录
date: 2023-01-10 20:21:22
tags: [路由器, Linux, OpenWrt]
categories: [Linux]
---

在路由器上安装和配置OpenWrt。它包括安装软件包和设置防火墙、接口和各种网络连接的分步说明。

<!-- more -->

## 基本界面

![](https://raw.githubusercontent.com/wnma3mz/blog_posts/master/imgs/openwrt/1673353557758.png)

准备：

- 固件
- 待恢复的配置
- 需要安装的软件+运行的命令

配置出问题，直接重刷。同时方便起见，要及时备份当前的配置。

![](https://raw.githubusercontent.com/wnma3mz/blog_posts/master/imgs/openwrt/1673353799884.png)

## 软件包的安装

能在线安装的可以直接在线安装。直接从"系统->软件包"里面找到需要安装的软件。

但有时候在线安装会由于网络，软件源、软件版本等问题，无法自动安装。因此需要我们手动找到ipk文件进行安装。使用下面的命令进行安装。

```bash
opkg install XXX
```

官方软件下载网页：[https://downloads.openwrt.org/releases/](https://downloads.openwrt.org/releases/)

```bash
# 系统版本可以直接从管理界面的概况中得知，这里是17.01.4

# 内核版本
>>> uname -a 
Linux dw33d 4.4.194 #0 Tue Oct 17 17:46:20 2017 mips GNU/Linux
# 查看支持的架构
>>> opkg print-architecture
arch all 1
arch noarch 1
arch mips_24kc 10
```

对应我这里的第三方包就是：[https://downloads.openwrt.org/releases/17.01.4/packages/mips_24kc/packages/](https://downloads.openwrt.org/releases/17.01.4/packages/mips_24kc/packages/)

基础的库：[https://downloads.openwrt.org/releases/17.01.4/packages/mips_24kc/base](https://downloads.openwrt.org/releases/17.01.4/packages/mips_24kc/base)

luci：[https://downloads.openwrt.org/releases/17.01.4/packages/mips_24kc/luci/](https://downloads.openwrt.org/releases/17.01.4/packages/mips_24kc/luci/)

### tmux

以tmux安装为例，该软件无法直接安装。

```bash
# 1. 从package里面下载tmux的ipkg文件
# 2. 执行安装命令
>>>opkg install tmux_2.3-1_mips_24kc.ipk
# 3. 报错信息，一些依赖文件没有。所以从base里面找到对应缺失的ipk文件。并进行安装
>>>opkg install terminfo_6.0-1_mips_24kc.ipk libncurses_6.0-1_mips_24kc.ipk
# 4. 此时，再安装tmux。安装成功
>>>opkg install tmux_2.3-1_mips_24kc.ipk

# 但安装libevent2这个库时，提示当前已安装，并且版本过高，并不会自动降级。所以需要手动卸载原有的软件
>>>opkg remove libevent2 --force-depends
# 卸载失败，提示已有软件依赖该库。所以我们需要强制进行卸载，将依赖软件也进行卸载。加上--force 命令
# 之后再安装低版本的libncurses
>>>opkg install libevent2_2.0.22-1_mips_24kc.ipk

# 最后再运行tmux，成功
```

其他软件应该同理，需要一步一步找到依赖文件，并且耐心地解决各种问题。dw33d内置了一个16g的sd卡，所以我把一些重装必备的ipk文件放到里面，便于快速安装。并且刷写固件并不会影响该sd卡的内容，可以放心存储。

### Tailscale

异地组网工具，从官网下载找到mips版本，放到sd卡中，避免刷新固件有影响。相当于单文件就可以运行。

```bash
# 解压文件
>>> tar zxvf tailscale_1.34.1_mips.tgz
# 到对应的目录中，里面会有一些文件。主要是两个可执行文件tailscale和tailscaled。
# 第一次运行需要启动tailscaled服务
>>>./tailscaled --state=tailscaled.state
# 此时需要新开一个窗口，登录tailscale
>>>./tailscale login 
# 可以直接把网址复制出来，用其他设备的浏览器进行登录
```

登录完毕之后，组网就完成了。此时，可以关闭该窗口。但为了后台启动，所以可以使用tmux，或者nohup命令。

```bash
nohup ./tailscaled --state=tailscaled.state > t.log 2>&1 &
```

同时为了保证开机自启动，所以我们可以在web管理界面中"系统->启动项"中，最后的"本地启动脚本"中插入(`exit 0`之前行)

```bash
# 先切换到对应目录，再执行
cd /mnt/sda1/tailscale_1.34.1_mips && nohup ./tailscaled --state=tailscaled.state > t.log 2>&1 &
```

这样，就能保证异地管理该路由器的web界面，并且可以直接ssh登录

## 防火墙

"网络->防火墙"中的默认配置，wan口是拒绝入站数据的。这样就会导致如果该路由器跟其他设备同属于一个局域网，ssh和web界面均不能通过ip登录。具体的例子是：

- 该路由器连接了一个上级路由器A，获取到的IP是192.168.0.X。分配给下面的设备的IP是192.168.1.X
- 另一个设备接入路由器A时，获取到的IP是192.168.0.Y。此时能ping通192.168.0.X，但是无法进行ssh，也无法通过192.168.0.X访问管理界面
- 另一个设备接入该路由器时，获取到的IP是192.168.1.Y。此时能够通过ssh和web访问192.168.1.X。

需求：在同局域网下的其他设备也能访问该路由器。

做法：

- 一个简单的做法就是如下图所示，把"防火墙"中的"常规设置"里面的wan口，入站数据从**拒绝**改成**接受**。（默认是拒绝入站的）

![](https://raw.githubusercontent.com/wnma3mz/blog_posts/master/imgs/openwrt/1680509223.png)

- 另一个做法是配置"防火墙->端口转发"，把需要配置的端口转发出来。如下图所示，web界面的端口是80。所以从wan口出来的访问80端口的流量，直接转发到192.168.4.1（该路由器分配给下面的IP，不是获取的IP）

![](https://raw.githubusercontent.com/wnma3mz/blog_posts/master/imgs/openwrt/1673356610477.png)

关于流量规则这里，则是在系统内部决定是否开放某个端口。可以参考默认的一些配置进行开放，一般应该是不需要调整的。

![](https://raw.githubusercontent.com/wnma3mz/blog_posts/master/imgs/openwrt/1673356804612.png)

## 接口

"网络->接口"

![](https://raw.githubusercontent.com/wnma3mz/blog_posts/master/imgs/openwrt/1673357070459.png)

我主要会有以下四种上网方式（不会专业术语，简单描述现象）：

- 桥接光猫

  - 将路由器WAN口接入光猫LAN口，代替光猫进行拨号，需要知道宽带的账号密码。
  - 在WAN口中，协议选择**PPPoE**，输入账号密码
  - 此时该路由器就是主路由器，一般同时有ipv4和ipv6两个地址，假设ipv4的ip为192.168.1.1
  - 接入该路由器的设备，由该路由器进行分配ip，ip与路由器同网段，如192.168.1.2
- 桥接上一级路由器

  - 将路由器WAN口接入上一级路由器LAN口，无需知道账号密码
  - 在WAN口中，协议选择**DHCP客户端**
  - 在物理设置中，选择桥接接口，默认**以太网交换机: eth0**开启。勾选两个无线网络。如下图所示
  - ![](https://raw.githubusercontent.com/wnma3mz/blog_posts/master/imgs/openwrt/1673524264094.png)
  - 此时，IP同上一级路由器同网段，且如果有IPv6，则连接该路由器的设备也有IPv6
- 做NAT

  - 将路由器WAN接入上一级路由器的LAN口，相当于把该路由器做一个新的入口。
  - 在WAN中，协议选择**DHCP客户端**或**静态地址**（需要额外配置）
  - 此时，接入该路由器的设备会独立上一级路由器的设备，即网段不同。
  - 注：如果此时还需要ipv6，可以参考这篇文件进行配置。[https://www.lategege.com/?p=676](https://www.lategege.com/?p=676)
    - 简单来说，就是需要选择静态IP，并在**WAN口**和**LAN**中的"DHCP服务器"中的IPv6设置中，路由器广告服务，DHCPv6服务，NDP-代理三个均选择中继模式
    - 并且在终端中 `vim /etc/config/dhcp`，找到 `config dhcp 'wan'`对应的选项，加入 `option master '1'`这一行
- WiFi放大器

  - 无需网线
  - "网络->无线"中的搜索，找到需要放大WiFi，加入网络。看情况选择放大2.4G还是5G的WiFi

    ![](https://raw.githubusercontent.com/wnma3mz/blog_posts/master/imgs/openwrt/1673357519480.png)
  - WPA密钥就是WiFi密码，新的网络名称可以默认也可以另起一个以便区分。
  - ![](https://raw.githubusercontent.com/wnma3mz/blog_posts/master/imgs/openwrt/1673357576678.png)
  - 此时接口中会多一个接口
  - 其他同桥接
