---
title: 端口转发工具笔记（端口转发）
date: 2020-08-08 15:15:23
tags: [windows, linux]
mathjax: true
categories: [奇技淫巧]
---

端口转发

防火墙工具，端口转发

<!-- more -->

继[ssh端口转发](https://wnma3mz.github.io/hexo_blog/2018/02/11/ssh/)，根据新的需求衍生出新技巧

### 需求：

1. 继跳板机中转登录后，觉得需要中转登录还是太麻烦，所以希望能够跟普通ssh一样登录

### 工具：

1. 防火墙配置（iptables/ufw/firewall）
2. ssh



### 先验知识：

1. [ssh端口转发](https://wnma3mz.github.io/hexo_blog/2018/02/11/ssh/)
2. 防火墙配置
   1. iptables（Centos7之前自带）
   2. firewall（Centos7之后自带）
   3. ufw（Ubuntu自带）
   4. netsh（Windows自带）



### 实验：

1. 假设内网机器A，内网机器C，公网机器B。A通过将22端口转发至B的3000端口来进行跳板登录。所以C需要执行如下命令来登录A

   ```bash
   # C登录B
   ssh userB@ip_B
   # B登录A
   ssh userA@127.0.0.1 -p 3000
   ```

2. 通过防火墙本机端口转发，将本机的3000端口转发至3000端口

   ```bash
   # 两者的前提需要在/etc/sysctl.conf中，令net.ipv4.ip_forward = 1
   # 使配置生效
   sysctl -p
   
   # iptables
   iptables -t nat -A PREROUTING -p tcp -i eth0 -d 127.0.0.1 --dport 3000 -j DNAT --to 127.0.0.1:3000
   
   # firewall
   # 添加端口
   firewall-cmd --add-port=3000/tcp --permanent
   # 端口转发
   firewall-cmd --permanent --add-forward-port=port=3000:proto=tcp:toport=3000 --permanent
   
   # --permanent 永久生效
   # firewall添加后，需要重启
   firewall-cmd --reload
   
   # netsh
   netsh interface portproxy add v4tov4 listenaddress=127.0.0.1 listenport=2121 connectaddress=ip_B connectport=222
   ```

3. 以上，即可通过如下命令直接登录A

   ```bash
   ssh userA@ip_B -p 3000
   ```

