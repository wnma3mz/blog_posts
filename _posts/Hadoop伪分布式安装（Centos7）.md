---
title: Hadoop伪分布式安装（Centos7）
date: 2017-04-09 17:42:00
tags: [Linux, Hadoop, 大数据]
categories: [Environment]
---
这篇文章主要介绍了Hadoop的伪分布式安装和一些基础操作

<!-- more -->

![](https://raw.githubusercontent.com/wnma3mz/blog_posts/master/imgs/hadoop/wKiom1SFSNnAeaEUAADD3ZcjTjw828.jpg)

## 文章开头

环境配置：Centos7

Hadoop版本：hadoop：2.7.3

JDK版本：jdk-8u111-linux-x64.tar.gz


## 安装之前

1. 下载hadoop安装包, [下载网址](http://hadoop.apache.org/releases.html)。进入官网之后，点击对应版本的“binary”，之后点击链接下载即可。

2. jdk。通过官网下载jdk

3. 通过Xshell，上传两个安装包至/usr/local/。服务器与本地下载和上传需要lrzsz这个软件

```bash
yum install lrzsz
```

之后可手动拖动文件至服务器当前目录下，也可通过命令
“rz”来打开窗口，从而进行上传。
（下载命令“sz”）

## 解压缩及配置

在`/usr/local/`目录下进行解压
```bash
#请根据下载版本，进行解压缩
tar -zxvf jdk-8u111-linux-x64.tar.gz
tar -zxvf hadoop-2.7.3.tar.gz
```


### 配置Java环境变量（可与配置Hadoop环境变量配置一起进行）


```bash
vim /etc/profile
#第十行左右输入命令，注意版本
export JAVA_HOME=/usr/local/jdk1.8.0_111
export PATH=.:$JAVA_HOME/bin:$PATH

#保存退出
source /etc/profile
```

### 配置Hadoop环境变量

```bash
vim /etc/profile
#在刚刚JAVA_HOME之下一行，注意版本

#此处的PATH承接上面Java环境变量的配置，即是在原来基础上进行增加而不是另起一行
export HADOOP_HOME=/usr/local/hadoop-2.7.3
export PATH=.:$JAVA_HOME/bin:$HADOOP_HOME/bin:$HADOOP_HOME/sbin:$PATH
#保存退出

#更新设置
source /etc/profile
```

### 配置启动Hadoop

```bash
#进入配置文件目录
cd /usr/local/hadoop-2.7.3/etc/hadoop/
#修改配置文件
vim hadoop-env.sh
#第二十五行
export JAVA_HOME=/usr/local/jdk1.8.0_111
#保存退出

vim core-site.xml
#添加<configuration></configuration>中间的配置
<configuration>
    <!-- 指定HDFS老大（namenode）的通信地址 -->
    <property>
	    <name>fs.defaultFS</name>
	    <value>hdfs://hadoopbonc1:9000</value>
    <!-- 指定hadoop运行时产生文件的存储路径 -->
    <property>
        <name>hadoop.tmp.dir</name>
    </property>
</configuration>
#此处fs.defaultFS的vaule写本机主机名，即第一个value的hadoopbonc1换成主机名
#保存退出

vim hdfs-site.xml
#添加如上配置
<configuration>
    <!-- 设置hdfs副本数量 -->
    <property>
         <name>dfs.replication</name>
         <value>1</value>
    </property>
</configuration>
#保存退出
```

### 配置SSH免密码登陆（密码互通）

```bash
#配置前
ssh localhost
#回车后，要求输入本机登陆密码

#配置主机名
vim /etc/sysconfig/network
source /etc/sysconfig/network

#配置hosts
vim /etc/hosts
ssh-keygen

#把公钥和私钥复制到相应的节点（因为本机是单节点，所以复制到本机）
#此处的hadoopbonc1换成主机名
ssh-copy-id hadoopbonc1
```

### hdfs启动与停止


```bash
#第一次启动需要格式化，理由跟磁盘第一次使用需要格式化一样
#以后启动就不需要格式化
hdfs namenode -format
#启动hdfs

#在hadoop目录下
cd /usr/locate/hadoop-2.7.3/
#启动
./sbin/start-dfs.sh
#输入jps检查是否成功
jps
#出现四行顺序不定，分别为SecondaryNameNode、DataNode、NameNode、Jps，即表示成功
#打开浏览器输入：服务器ip:50070

#若无响应则需要开放50070端口，则需开放50070端口
firewall-cmd --zone=public --add-port=50070/tcp --permanent
firewall-cmd --reload

#hdfs的停止
#同样在hadoop目录下
./sbin/stop-dfs.sh

```

### 配置和启动YARN

```bash
#切换配置文件目录
cd /usr/local/hadoop-2.7.3/etc/hadoop/
mv mapred-site.xml.template mapred-site.xml
vim mapred-site.xml
#添加<configuration></configuration>中间的配置
<configuration>
    <!-- 通知框架MR使用YARN -->
    <property>
        <name>mapreduce.framework.name</name>
        <value>yarn</value>
    </property>
</configuration>
#保存退出

vim yarn-site.xml
#添加如上配置
<configuration>
	<!-- reducer取数据的方式是mapreduce_shuffle -->
    <property>
         <name>yarn.nodemanager.aux-services</name>
         <value>mapreduce_shuffle</value>
    </property>
</configuration>
#保存退出

#启动YARN
start-yarn.sh
#输入jps检查是否成功
jps
#出现六行顺序不定，分别为SecondaryNameNode、DataNode、NameNode、Jps、ResourceManager、NodeManager，即表示成功
#打开浏览器输入：服务器ip:8088
#若无响应，可参照上文中的开放50070端口，开放8088端口，只需将50070换为8088即可
```


## 测试


在本地新建一个文件，如在`/home/user/`下新建`words.txt`，内容如下
```bash
hello world
hello hadoop
hello csdn
hello
```

正式进行测试。命令如下：
```bash
#在hdfs根目录下新建test目录
bin/hdfs dfs -mkdir /test

#查看hdfs根目录下的目录结构
bin/hdfs dfs -ls /

#将本地文件上传至/test/目录下
bin/hdfs dfs -put /home/user/words.txt /test/

#运行wordcount
bin/hadoop jar share/hadoop/mapreduce/hadoop-mapreduce-examples-2.7.1.jar wordcount /test/words.txt /test/out

#在/test/目录下生成了一个名为out的文件目录，查看一下/out/目录下的文件
bin/hdfs dfs -ls /test/out

#结果保存在part-r-00000，查看一下运行结果
bin/hdfs fs -cat /test/out/part-r-00000
```


## HDFS的常用操作命令


```bash
#常用操作：
#HDFS shell
#查看帮助
hadoop fs -help <cmd>
#上传
hadoop fs -cat <hdfs上的路径>
#查看文件列表
hadoop fs -ls /
#下载文件
hadoop fs -get <hdfs上的路径> <linux上文件>
```

下一篇：{% post_link 在Hadoop基础上Hive的安装 %}