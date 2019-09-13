---
title: 在Hadoop基础上Hive的安装
date: 2017-05-07 13:03:12
tags: [Hadoop, Hive, 大数据]
categories: [环境搭建]
---
这篇文章主要介绍了Hive在Hadoop上的安装及一些基本的Hive操作

<!-- more -->


写在开头
环境
Hadoop单机
Centos7
Hadoop-2.7.3
hadoop位置：/usr/loacl/hadoop

参考文章
[Hadoop集群之Hive安装配置](http://yanliu.org/2015/08/13/Hadoop%E9%9B%86%E7%BE%A4%E4%B9%8BHive%E5%AE%89%E8%A3%85%E9%85%8D%E7%BD%AE/)

**下载Hive**
------
下载源码包
```bash
#在hadoop目录下操作
>>>cd /usr/local/hadoop

#用wget下载
>>>wget http://mirrors.cnnic.cn/apache/hive/hive-1.2.1/apache-hive-1.2.1-bin.tar.gz

#也可以在图形界面下下载之后，上传压缩包

#解压缩包
>>>tar -zxvf apache-hive-1.2.1-bin.tar.gz

#配置环境变量
>>>vim /etc/profile
#位置在之前配置的变量之后，大概12行左右，因为之前配置了jdk和hadoop变量
export HIVE_HOME=/usr/local/hadoop/apache-hive-1.2.2-bin
export PATH=$HIVE_HOME/bin:$PATH
#适当位置添加"$HIVE_HOME/bin:"
#保存退出

#使文件生效
>>>source /etc/profile
```

**安装MySQL**
-------

MariaDB数据库管理系统是MySQL的一个分支，主要由开源社区在维护，采用GPL授权许可。开发这个分支的原因之一是：甲骨文公司收购了MySQL后，有将MySQL闭源的潜在风险，因此社区采用分支的方式来避开这个风险。MariaDB的目的是完全兼容MySQL，包括API和命令行，使之能轻松成为MySQL的代替品。

在Centos7下，使用命令安装mysql，会安装成mariadb。

在这里先介绍安装MariaDB，原因如上。

### 安装MariaDB

```python
#安装
>>>yum install mariadb-server mariadb

#启动
>>>systemctl start mariadb

#进入MySQL
>>>mysql -u root -p

#命令行变成如下，可能有点不习惯
MariaDB [(none)]>


#P.S.
#相关命令
>>>systemctl start mariadb #启动MariaDB
>>>systemctl stop mariadb #停止MariaDB
>>>systemctl restart mariadb #重启MariaDB
>>>systemctl enable mariadb #设置开机启动
```

### 正式安装MySQL

```python
#下载，在这里使用的是命令行下载，也建议在图形界面下载，然后上传至服务器
>>>wget http://dev.mysql.com/get/mysql-community-release-el7-5.noarch.rpm

#使用rpm安装
>>>rpm -ivh mysql-community-release-el7-5.noarch.rpm

#使用yum安装mysql-community-server
>>>yum install mysql-community-server

#启动服务
>>>service mysqld start

#进入mysql，第一次进入无密码
>>>mysql -u root -p

#命令行变成如下
mysql>
```

### 配置MySQL

编码配置
```python
#进入配置文件，若未安装vim，建议先使用命令yum install vim安装vim
>>>vim /etc/my.cnf

#最后加上编码配置
[mysql]
default-character-set =utf8

#此处字符编码必须和/usr/share/mysql/charsets/Index.xml中一致。
#不过一般情况下使用的都是utf8
```

设置密码
```bash
#下面三种方法需要进入mysql
>>>mysql -u root -p

#方法一
mysql>insert into user(host,user,password) values('%','user_name',password("password");
#方法二
mysql>set password for user_name = password("password");
#方法三
mysql>grant all on *.* to user_name@% identified by "password";

#下面这一种方法可直接在shell下设置密码
>>>mysqladmin -u root password "password"
```
远程连接

```bash
#进入mysql
>>>mysql -u root -p

#把在所有数据库的所有表的所有权限赋值给位于所有IP地址的root用户。
mysql> grant all privileges on *.* to root@'%'identified by "password";
```

P.S.上文引号中的user_name表示数据库的用户名，password表示对应用户的密码。即这两项是由读者自行定义的。

### 常见问题及解决方案

#### Mysql创建用户失败

```bash
# mysql在my.ini的配置文件中设置了严格模式，所以我们需要进行修改

# 第一步，寻找配置文件
>>>whereis my.ini

# 第二步，根据上一步结果进行vim编辑
>>>vim /../my.ini

# 第三步，在vim里面搜索sql-mode,删除STRICT_TRANS_TABLES，保存退出即可

# 第四步，保险起见，使刚刚的配置文件立即生效
>>>source /../my.ini
```

#### 支持中文

- Centos7

    ```python
    # 编辑文件
    >>>vim /etc/my.cnf
    # 在对应[xx]下增加修改如下代码
    [client]
    port = 3306
    socket = /var/lib/mysql/mysql.sock
    default-character-set=utf8

    [mysqld]
    port = 3306
    socket = /var/lib/mysql/mysql.sock
    default-storage-engine=INNODB
    character-set-server=utf8
    collation-server=utf8_general_ci

    [mysql]
    no-auto-rehash
    default-character-set=utf8
    # 保存退出，重启服务
    # 重新登陆mysql检查是否成功，方法见下
    ```


- Ubuntu16.04

    ```python
    >>>sudo vim /etc/mysql/mysql.conf.d/mysqld.cnf
    # 在对应[xx]下增加以下内容，如果不存在[xx]自行增加
    [mysqld]
    character_set_server=utf8
    [mysql]
    default-character-set=utf8
    [mysql.server]
    default-character-set=utf8
    [mysqld_safe]
    default-character-set=utf8
    [client]
    default-character-set=utf8
    # 重启mysql
    >>>service mysql restart
    # 进入mysql查看配置参数
    >>>mysql -uroot -p
    # 查看database 的value变为utf8即可
    >show variables like '%character%';

    # p.s.如果之前创建了表，表编码不会改变
    ```



#### MySQL报错“1366 - Incorrect integer value: '' XXXXXXX' at row 1 ”


修改方法:（两种,建议第二种）

1:命令行  set names gbk；(此为设置通信编码)

2:my.cnf中查找sql-mode

```
将
sql-mode="STRICT_TRANS_TABLES,NO_AUTO_CREATE_USER,NO_ENGINE_SUBSTITUTION"，

修改为
sql-mode="NO_AUTO_CREATE_USER,NO_ENGINE_SUBSTITUTION"，
```
重启mysql后即可


#### **关于my.cnf和my.ini的说明**

my.cnf常见于Linux系统，my.ini常见与windows系统，二者都是属于mysql的配置文件。一般好像在一个系统下就是只出现一种配置文件，具体区别没有深入了解，修改配置文件，根据自己的系统进行查找修改配置文件即可


#### **Mysql交互环境自动补全**

```bash
# 修改配置文件
>>>vim /etc/my.cnf
# 在[mysql]部分添加auto-rehash
# 保存退出，重启
>>>service mysqld restart
```


**配置Hive**
------
我们之前在hadoop目录下安装了Hive，位置为/usr/local/hadoop/apache-hive-1.2.1-bin

```bash
#进入hive配置目录下
>>>cd /usr/local/hadoop/apache-hive-1.2.1-bin/conf

#修改hive-default.xml.template
#首先复制
>>>cp hive-default.xml.template hive-default.xml
#修改文件
>>>vim hive-default.xml

# 1. 第一步将<configuration></configuration>中内容删除大概是18-3908行
# vim删除命令-->:18,3908d
# 2.将下面<configuration></configuration>中内容复制进去,分别将3306喝9083前的user_name改为当前的用户名
<configuration>
    <property>
        <name>javax.jdo.option.ConnectionURL</name>
        <value>jdbc:mysql://user_name:3306/hive?createDatabaseIfNotExist=true</value>
        <description>JDBC connect string for a JDBC metastore</description>
    </property>
    <property>
        <name>javax.jdo.option.ConnectionDriverName</name>
        <value>com.mysql.jdbc.Driver</value>
        <description>Driver class name for a JDBC metastore</description>
    </property>

    <property>
        <name>javax.jdo.option.ConnectionUserName</name>
        <value>hive<value>
        <description>username to use against metastore database</description>
    </property>
    <property>
        <name>javax.jdo.option.ConnectionPassword</name>
        <value>hive</value>
        <description>password to use against metastore database</description>
    </property>

    <property>
    <name>hive.metastore.uris</name>
    <value>thrift://user_name:9083</value>
    </property>
</configuration>
```

**下载JDBC**
------
```bash
#命令行下载,也可以使用图形界面上传文件
>>> wget http://cdn.mysql.com/Downloads/Connector-J/mysql-connector-java-5.1.36.tar.gz

#将文件复制进Hive的lib目录下，原因，Hive的自带的那个版本低，可能失效
>>> cp mysql-connector-java-5.1.33-bin.jar /usr/local/hadoop/apache-hive-1.2.1-bin/lib/
```

启动Hive
```bash
>>>hive --service metastore &
>>>jps
#结果会多出一个进程

#进入hive目录
>>>cd /usr/local/hadoop/apache-hive-1.2.1-bin/bin
#启动hive，可能有点慢
>>>hive

#若出现hive的命令行即代表成功，如下
hive>
```

## 常见问题及解决方案

Logging initialized using configuration in jar:file:/home/hadoop/apache-hive-1.2.1-bin/lib/hive-common-1.2.1.jar!/hive-log4j.properties
Exception in thread "main" java.lang.RuntimeException: java.lang.RuntimeException: Unable to instantiate org.apache.hadoop.hive.ql.metadata.SessionHiveMetaStoreClient
        at org.apache.hadoop.hive.ql.session.SessionState.start(SessionState.java:522)

解决方法，直接关闭防火墙

```bash
#这里的系统为Centos7，所以使用此命令
>>>systemctl stop firewalld
```

hive metastore 启动出错解决
```bash
# 查看与hive相关进程是否启动
>>>ps -ef | grep hive
# kill相关进程，为进程号
>>>kill num
# 重新启动
>>>./hive
```

更多问题见此文章
[Hive常见问题汇总](http://blog.csdn.net/freedomboy319/article/details/44828337)

## Hive的学习笔记

### 新建表
---

```sql
-- 新建一张表，名为“test”，里面有“name”、“id”两类，分别是“string”、“int”的数据类型，以“|”隔开一列，表是作为textfile的。
hive> CREATE TABLE test (name String, id int) ROW FORMAT DELIMITED FIELDS TERMINATED BY '|' STORED AS TEXTFILE;
```

### 加载表
---

```sql
-- 从本地的/home/user/test.txt文件，将数据加载进test这个表
hive> LOAD DATA LOCAL INPATH '/home/user/test.txt' OVERWRITE INTO TABLE test;
```

### 关联表
---

```sql
-- 将两张表通过一个或多个字段关联在一起
-- 将test_a和test_b表的id字段关联在一起，再加上“name”字段，组成一个新表
hive> SELECT test_a.id, test_b.name FROM test_a, test_b JOIN test_b ON (test_a.id = test_b.id);
```

### 保存表
---

```sql
-- 由于hive下执行任务之后，并不会保存数据，所以我们使用INSERT命令来保存命令
-- 插入数据到test表中
hive> INSERT OVERWRITE TABLE test
     > ...

-- 保存表到hdfs上,保存输出结果到hdfs下的/out/目录下
hive> INSERT OVERWRITE LOCAL DIRECTORY "/out/"
    > ...
```

### 排序问题
----

 1. order by

 hive> SELECT * FROM test ORDER BY id;

 2. sort by
hive> SELECT * FROM test SORT BY id;

 3. distribute by
hive> SELECT * FROM test ORDER BY name DISTRIBUTE BY id;

 4. DISTRIBUTE BY with SORT BY
DISTRIBUTE BY和GROUP BY有点类似，DISTRIBUTE BY控制reduce如何处理数据，而SORT BY控制reduce中的数据如何排序。
注意：hive要求DISTRIBUTE BY语句出现在SORT BY语句之前。

 5. Cluster By

 cluster by 除了具有 distribute by 的功能外还兼具 sort by 的功能。
 默认升序排序，但DISTRIBUTE BY的字段和SORT BY的字段必须相同，且不能指定排序规则。

总结：

ORDER BY是全局排序，但在数据量大的情况下，花费时间会很长
SORT BY是将reduce的单个输出进行排序，不能保证全局有序
DISTRIBUTE BY可以按指定字段将数据划分到不同的reduce中
当DISTRIBUTE BY的字段和SORT BY的字段相同时，可以用CLUSTER BY来代替 DISTRIBUTE BY with SORT BY。



### hive常用函数
--------

```bash
# 以下无特殊说明，返回值皆为string
# 字符串连接
hive>concat(strA, strB, ...)
# 带分隔符的字符串连接，即连接之后字符串存在分隔符，strSEP表示分隔符
hive>concat_wd(strSEP, strA, strB, ...)

# 字符串截取
# 从整型数num0开始截取，截取到结尾为止
hive>substr(strA, num0)
# 从整型数num0开始截取，截取到num1为止
hive>substr(strA, num0, num1)

# 字符串转大写
hive>upper(str)
# 字符串转小写
hive>lower(str)

# 除去字符串中空格
hive>trim(str)
# 除去字符串左边空格
hive>ltrim(str)
# 除去字符串右边空格
hive>rtrim(str)

# 正则替换字符
# 将strA中符合strRE的部分替换为strB
hive>regexp_replace(strA, strRE, strB)

# 重复字符串
# str为字符串，num为重复次数
hive>repeat(str, num)

# 分割字符串
# 以strpat为分割符，返回字符串数组
hive>spilt(str, strpat)
```

### hive导出数据
--------

```bash
# 导出至本地，content表示本地目录
hive>insert overwrite local directory 'content'
    >select * from table_name;

# 导出至hdfs，content表示hdfs目录
hive>insert overwrite directory 'content'
    >select * from table_name;

# 导出至hive另一张表othertable，前提hive中存在这张表
hive>insert overwrite othertable
    >select * from table_name;
```


### 转换类型
----

cast(xxx as xx)
将xxx类型转换为xx类型

下表为是否可转换类型的说明
![转换操作](http://img.blog.csdn.net/20170612195815151?watermark/2/text/aHR0cDovL2Jsb2cuY3Nkbi5uZXQvd25tYTNteg==/font/5a6L5L2T/fontsize/400/fill/I0JBQkFCMA==/dissolve/70/gravity/SouthEast)
　注：由于表格比较大，这里对一些比较长的字符串进行缩写，ts是timestamp的缩写,bl是boolean的缩写,sl是smallint的缩写,dm是decimal的缩写,vc是varchar的缩写,ba是binary的缩写。


































