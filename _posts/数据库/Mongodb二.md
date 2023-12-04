---
title: Mongodb：
date: 2021-07-21 20:44:28
tags: [mongo, 数据库]
categories: [Database]
---
Mongodb4.2.x数据库配置的第二部分，副本集架构的配置

<!-- more -->

## 准备

基本同原文，由于是副本集配置，所以在需要配置的服务器上进行相同的配置

选择：

1. 单机上，配置不同port的mongo服务
2. 不同机器上，配置mongo服务
3. 要求同网段，所以配置的ip中不能写127.0.0.1
4. 可以 `vim /etc/hosts`，方便管理

```bash
# 下载压缩包
wget http://fastdl.mongodb.org/linux/mongodb-linux-x86_64-rhel80-4.2.15.tgz

# 解压缩
tar -zxvf mongodb-linux-x86_64*.tgz

# 重命名
mv mongodb-linux-x86_64* mongodb
# 移动到你想放置的目录下，这里我放在/opt目录下
mv mongodb /opt/

# 进行mongodb主目录
cd /opt/
# 建立存放数据文件和日志文件的目录
mkdir -p data/test/logs
mkdir -p data/test/db
# 创建配置文件，并写入如下配置
vim bin/mongodb.conf
```

更新部分，配置文件用yaml格式

```yaml
systemLog:
  destination: file
  path: "/opt/mongodb/data/test/logs/mongodb.log"
  logAppend: true
storage:
  dbPath: "/opt/mongodb/data/test/db"
  journal:
    enabled: true 
net:
  # 允许所有IP访问，也可以用逗号隔开，控制IP访问
  bindIp: 0.0.0.0 
  # 设置端口号（默认的端口号是27017，可以根据个人需求进行更改）
  port: 27017
processManagement:
  # 设置为以守护进程的方式运行，即在后台运行
  fork: true
replication:
  # 副本集名字
  replSetName: "rs0" 
```

### 启动

```bash
# 在所有机器上启动mongo，此时无账号密码
# 以配置文件的方式启动
bin/mongod --config mongodb.conf
```

### 配置副本集

在任意一台服务器A上，进行配置副本集

```bash
# 将副本集的初始化配置写入js
vim init_replica.js
```

```yaml
# 其中rs0一定要跟上面的replSetName一样
config = { _id:"rs0", members:[{_id:0,host:"ip0:port0"},{_id:1,host:"ip1:port1"}]}
rs.initiate(config) 
rs.status()
```

```bash
bin/mongod --port 27017 < init_replica.js
```

### 验证

在A服务器上进行操作，添加账号密码

```bash
# 进行mongodb的交互环境
bin/mongod --port 27017

# 进行admin数据库，创建管理员用户root，密码为password，权限是超级用户（最高）
>use admin
>db.createUser({ user: "admin", pwd: "password", roles: [{ role: "userAdminAnyDatabase", db: "admin" }] })
>db.createUser({user:"root",pwd:"password",roles:["root"]})
# 验证是否创建成功，返回1表示成功
>db.auth({"root", "password"})
# 退出交互环境
>exit

# 找到mongo的服务进程pid
ps ux | grep mongod
# 杀死进程，填写对应的pid
kill PID

# 以密码权限验证启动服务
bin/mongod --config mongodb.conf --auth
```

### OpenSSL证书文件

### 读写分离

### 主节点选举配置
