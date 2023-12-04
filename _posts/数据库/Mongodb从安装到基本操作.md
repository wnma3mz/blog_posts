---
title: Mongodb从安装到基本操作
date: 2017-11-24 17:12:28
tags: [mongo, 数据库]
categories: [Database]
mathjax: false
katex: false
---
这篇文章主要从零开始介绍如何在Centos7下安装Mongodb, 用Python3连接Mongodb, 用可视化工具Robo 3T连接Mongodb, 之后介绍了一些基本的mongo操作方法, 其中含有一些进阶操作

<!-- more -->

## 安装Mongodb(Centos7)

1. 官网上找到需要下载的版本压缩包进行下载。[官网链接]( https://fastdl.mongodb.org/linux/)

   或者在命令行下使用`wget`命令进行下载

   ```bash
   # 这里下载的是3.4.10版本
   wget https://fastdl.mongodb.org/linux/mongodb-linux-x86_64-rhel70-3.4.10.tgz
   ```

2. 进行解压缩

   ```bash
   # 解压缩
   tar -zxvf mongodb-linux-x86_64*.tgz
   # 重命名
   mv mongodb-linux-x86_64* mongodb
   # 移动到你想放置的目录下，这里我放在/opt目录下
   mv mongodb /opt/
   ```

3. 进行相应的配置，mongodb默认没有任何配置

   ```bash
   # 进行mongodb主目录
   cd /opt/
   # 建立存放数据文件和日志文件的目录
   mkdir -p data/test/logs
   mkdir -p data/test/db
   # 创建配置文件，并写入如下配置
   vim bin/mongodb.conf
     `
       # 设置数据文件的存放目录
       dbpath = /opt/mongodb/data/test/db

       # 设置日志文件的存放目录及其日志文件名
       logpath = /opt/mongodb/data/test/logs/mongodb.log

       # 设置端口号（默认的端口号是27017，可以根据个人需求进行更改）
       port = 27017

       # 设置为以守护进程的方式运行，即在后台运行
       fork = true

       # 是否不允许表扫描
       nohttpinterface = true
     `
   # 保存退出
   ```

4. 启动mongodb

  ```bash
  # 以配置文件的方式启动
  ./bin/mongod --config mongodb.conf
  ```

  报错一： `ERROR: child process failed, exited with error number 1`
  检查mongodb.conf的文件路径是否配置错误

  报错二：`ERROR: child process failed, exited with error number 100`
  很可能是没有正常关闭导致的，那么可以删除 `mongod.lock`文件，这里对应我的配置路径在`data/`里面

5. 其他

  ```bash
  # 链接命令，方便调用mongo命令
  ln -s /opt/mongodb/bin/mongo /usr/bin
  # 查看mongodb进程
  ps -aux |grep mongodb
  # 检查端口运行情况
  netstat -lanp | grep 27017
  # 终止mongodb服务，PID从ps命令获取
  kill -15 PID
  # 添加自启动命令
  vim /etc/rc.local
  # 末尾追加一行
  `/opt/mongodb/bin/mongod --config mongodb.conf`
  # 保存退出
  ```

6. 设置密码权限，默认无密码

  ```bash
  # 进行mongodb的交互环境
  ./mongo
  # 如果进入失败，请检查是否添加了软链接和是否启动了mongodb服务

  # 进行admin数据库，创建管理员用户root，密码为password，权限是超级用户（最高）
  >use admin
  >db.createUser({user:"root",pwd:"password",roles:["root"]})
  # 验证是否创建成功，返回1表示成功
  >db.auth({"root", "password"})
  # 退出交互环境
  >exit
  # 重启mongodb服务
  # 杀死mongodb进程，参照上面的方法
  # 以密码权限验证启动服务
  opt/mongodb/mongod --config mongodb.conf --auth
  ```

## 使用Python连接Mongodb

```bash
# 下载第三方包
pip3 install pymongo
```

```python
# /usr/bin/python3
from pymongo import MongoClient

# 方法一
# host主机名，27017连接端口
client = MongoClient(host, 27017)
db_auth = client.admin
# 登陆的用户名(username)和密码(password)
db_auth.authenticate(username, password)
# 连接指定数据库，数据库名为db_name
db_name = client["db_name"]

# 方法二
# 用户名(username)、密码(password)、主机名(host)、端口(port)。注：这里的password不能出现@符号，如果用@符号就需要使用方法一
client = MongoClient("mongodb://username:password@host:port")
# 连接指定数据库，数据库名为db_name
db_name = client["db_name"]

# 操作数据库，db_name["db_set"]=db_name.db_set
"""
查找数据
在db_name数据库中的db_set集合里面找到name是a_name的json
find是返回一个指针（可以使用列表的方式来读取），find_one返回一个dict
"""
db_name["db_set"].find_one({"name": "a_name"})


# 插入数据
# 插入一个json到set1集合中，a_dict是一个字典（如果set1不存在则会被自动创建）
db_name.set1.insert_one(a_dict)
# 插入多个json到set1集合中，a_dict_lst是一个列表，里面的元素是由字典组成的
db_name.set1.insert_many(a_dict_lst)

# 更新数据
# 查找到一个数据之后，对里面的某个数据进行更改，再使用save方法保存
data_one = db_name["db_set"].find_one({"name": "a_name"})
data_one['age'] += 3
db_name.db_set.save(data_one)
# 更新多条记录，找到db_set集合中满足age=20，sex=0的数据，将name改为user1
db_name.db_set.update({"name": "user1"}, {"$set":{"age": 20, "sex": 0}})

# 删除数据
id = db_name.db_set.find_one({"name": "user1"})["_id"]
# 根据 id 删除一条记录
db_name.db_set.remove(id)
# 删除集合里的所有记录
db_name.db_set.remove()
# 删除name=user1的记录
db_name.db_set.remove({"name": "user1"})
```

## 使用Robo 3t连接Mongodb

强烈推荐Robo 3t，[官网地址](https://robomongo.org/)

1. 到官网找到对应的系统版本进行下载

2. 安装

   - Windows的安装十分简单就不赘述了

   - Ubuntu的安装
     ```bash
     # 解压缩
     tar -zxzf robomongo*.tar.gz
     # 修改文件名（可选）
     mv robomongo* robo3t
     # 移动到/opt目录（可选）
     mv robo3t /opt
     # 建立软链接
     ln -s /opt/robo3t/robomongo /usr/bin
     # 启动
     robomongo
     ```
     ```bash
     # 如果报如下错误
     `
     This application failed to start because it could not find or load the Qt platform plugin "xcb" in "".

     Available platform plugins are: xcb.

     Reinstalling the application may fix this problem.
     Aborted (core dumped)
     `
     # 解决办法
     mkdir ~/robo-backup
     mv /opt/robo3t/lib/libstdc++* ~/robo-backup/
     # 之后便可以重新启动mongo试试看了
     ```

3. 使用

   1. 首先创建新的连接，如下图所示。在`Connection中，``Name`为自定义连接名，`Address`填入对应的`host`和`port`；
     ![](https://raw.githubusercontent.com/wnma3mz/blog_posts/master/imgs/mongodb/20171124171027733.png)
   2. 在`Authentication`中，勾选`Perform authentication`，填上对应的`User Name`和`Password`。填写完成之后可以点击`Test`验证是否能够连接成功，如果无误的话，可以点击`Save`保存退出。

    ![](https://raw.githubusercontent.com/wnma3mz/blog_posts/master/imgs/mongodb/20171124171111807.png)

   3. 连接Mongodb。点击`Connect`，之后就可以可视化数据库中的数据。也可以使用鼠标简单的进行创建/删除数据库，新建/修改信息，也可以使用命令查找数据。


## CURD操作（主要是查找数据）

**CRUD**

- C: Create Operations, 创建
- R: Read Operations, 读取
- U: Update Operations, 更新
- D: Delete Operations, 删除


mongo数据库至上而下是：`Database`-->`Collections`-->`Documents`。我在这里分别理解成数据库、集合、记录。

这里直接用官方给的数据来操作。假设以下数据放在了`fruits`集合中。

```json
{ "_id": "apples", "qty": 5, "score": [-1, 3], "links": [{"Uin": 124, "NickName": "123"}]}
{ "_id": "bananas", "qty": 7, "score": [1, 5]}
{ "_id": "oranges", "qty": { "in stock": 8, "ordered": 12 },  "score": [5, 5] }
{ "_id": "avocados", "qty": "fourteen" , "score": [5, 5]}
```

### 基本操作

`json`数据的嵌套查找，就是说，如果有字段包含json，可以直接用`.`进行连接。

查找`qty`中的`ordered`为12的字段

```bash
db.fruits.find({"qty.ordered": 12})
```

多重筛选，同时满足多个要求的记录

查找`qty`为5，且`_id`为"apples"的记录

```bash
db.fruits.find({"qty": 5, "_id": "apples"})
```

第一个大括号内容表示筛选条件，第二个大括号表示显示内容

筛选条件就不说明了，第二个大括号表示打印`_id`字段和`score`字段，不打印其他字段，

```bash
db.fruits.find({"qty": 5}, { "_id": 1 , "score": 1})
```

### 比较

下面至上而下分别表示，在`fruits`集合中查找字段`qty`与4的大小关系所有记录。

```bash
db.fruits.find({"qty": 4})          # 等于
db.fruits.find({"qty": {"$gt": 4}})   # 大于
db.fruits.find({"qty": {"$lt": 4}})   # 小于
db.fruits.find({"qty": {"$gte": 4}})  # 大于等于
db.fruits.find({"qty": {"$lte": 4}})  # 小于等于
db.fruits.find({"qty": {"$ne": 4}})   # 不等于
```

当然，也可以两个比较符一起操作，比如下面这样。

```bash
db.fruits.find({"qty": {"$gt": 3, "$lt": 5}})  # 大于3小于5。
```

如果对`score`这个字段使用比较，比如下面这样。

```bash
db.fruits.find({"score": {"$gt": 2, "$lt": 5}})  # 大于3小于5。
```

将会返回下面两条记录。理由是，字段里的数组里有含有大于2小于5的数字。

```json
/* 1 */
{
    "_id" : "apples",
    "qty" : 5,
    "score" : [
        -1,
        3
    ],
    "links" : [
        {
            "Uin" : 124,
            "NickName" : "123"
        }
    ]
}

/* 2 */
{
    "_id" : "bananas",
    "qty" : 7,
    "score" : [
        1,
        5
    ]
}
```

###关系

#### $in

查找`_id`字段中包含`5`或者`ObjectId("507c35dd8fada716c89d0013")`的记录

```bash
db.fruits.find({"_id": {"$in":[5, ObjectId("507c35dd8fada716c89d0013")] }})
```

#### $elemMatch

用于字段中含有数组，且数组中含有`json`数据的的查找。

查找`links`这个含有数组的字段，找出数组中`Uin`等于124的记录

```bash
db.fruits.find({"links": {"$elemMatch": {"Uin": 124}}})
```

#### $slice

切片，只打印字段中数组前n个数据

筛选出记录之后，只打印记录`score`字段的前1条数据

```bash
db.fruits.find({"qty": 5}, {"_id": 1, "score": {"$slice": 1}})
```

输出如下

```json
/* 1 */
{
    "_id" : "apples",
    "score" : [
        -1
    ]
}
```

#### $regex

使用正则表达式，更多操作请查看官方文档:  [$regex](https://docs.mongodb.com/manual/reference/operator/query/regex/index.html)

找出`_id`字段中所有以a开头的记录

```bash
db.fruits.find({"_id": {"$regex": "^a"}})
```

### 后缀

#### sort——排序

```bash
db.fruits.find().sort({"qty": 1}) # 升序
db.fruits.find().sort({"qty": -1}) # 降序
```

#### limit——限制

```bash
db.fruits.find().limit(3) # 对查询结果只输出前3个，可以加在sort后面结合使用
```

#### skip——跳过

```bash
db.fruits.find().skip(3) # 对查询结果跳过输出前3个，即不输出前3个，可以加在sort后面结合使用
```

#### count——计数

```bash
db.fruits.find().count()  # 返回查询结果总数
```

#### collation——基于特定语言进行操作

没有怎么看懂，我理解的大概意思就是说，运行根据特定语言来进行查找。感觉用的不多，这里贴上官方给的例子

```bash
db.fruits.find().collation( { locale: "en_US", strength: 1 } )
```


### 个人常用操作

例子：
```json
// test_db
{
   "_id" : ObjectId("56063f17ade2f21f36b03133"),
   "title" : "MongoDB 教程",
   "description" : "MongoDB 是一个 Nosql 数据库",
   "by" : "菜鸟教程",
   "url" : "http://www.runoob.com",
   "tags" : [
           "mongodb",
           "database",
           "NoSQL"
   ],
     "links" : [{
           "UserName" : "@a1bf7a55ad8478017b95124774687769",
           "Uin" : 12266535,
           "NickName" : "一二三",
       },
       {
           "UserName" : "@5571e120b5d8f54f93338c76f62b9c419b818c29373dca9f331c2911a203bfbe",
           "Uin" : NumberLong(3379331828),
           "NickName" : "灵灵八",
       }]
}
```

#### 返回查找结果的数目
```bash
db.test_db.find({"title": "MongoDB 教程"}).count()
```
#### 查找某个key中的数组是否某包含某内容
```bash
db.test_db.find({"tags": {"$in": ["mongodb"]}})
```
#### 使用正则进行查找
```bash
db.test_db.find({"url": {"$regex": "^http"}})
```

#### 查找某个由数组构成的key方法
```bash
db.test_db.find({"links":{"$elemMatch":{"Uin":12266535,"NickName":"一二三"}}})
```


## 附录

### 数据库权限说明表
| 名称           | 权限                                       |
| ------------ | :--------------------------------------- |
| 数据库用户角色      | read、readWrite                           |
| 数据库管理角色      | dbAdmin、dbOwner、userAdmin                |
| 集群管理角色       | clusterAdmin、clusterManager、clusterMonitor、hostManager |
| 备份恢复角色       | backup、restore                           |
| 所有数据库角色      | readAnyDatabase、readWriteAnyDatabase、userAdminAnyDatabase、dbAdminAnyDatabase |
| 超级用户角色       | root                                     |
| 提供了系统超级用户的访问 | dbOwner 、userAdmin、userAdminAnyDatabase  |
| 内部角色         | __system                                 |
