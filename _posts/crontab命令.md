---
title: crontab命令
date: 2017-08-24 08:44:45
tags: [Linux, crontab]
categories: [奇技淫巧]
---
Linux下crontab命令简单使用及介绍

<!-- more -->


```bash
# 查看当前用户的定时任务，也可以使用crontab -uroot -l查看指定用户的定时任务
>>>crontab -l

# 编辑crontab定时任务，貌似是使用vim的编辑命令
>>>crontab -e

# 其实基本上知道这两个就可以处理定时任务了

# 删除当前用户的所有定时任务，不知道为什么会有这种设定（手贱按下过一次，恢复起来真的惨）
>>>crontab -r
# 如果按到了，就去日志文件里面找记录，尽可能恢复命令吧
```

下面讲讲编辑任务的格式

```bash
# 基本格式

  *  *  *  *  * command

# 分 时 日 月 周 需要执行的命令
```

可视化测试在线网站，https://crontab.guru/

几个栗子

```bash
# 每分钟执行一次command命令，空格表示间隔
* * * * * command

# 每小时执行一次，‘/’表示频率
0 */1 * * * command

# 每小时的10分和40分各执行一次，‘,’表示并列
10,40 * * * * command

# 每个星期一的10-14点的10，40执行，‘-’表述区间
10,40 10-14 * * 1 command

# 每天的8，13，20点执行，注意此时表示分的那个*要替换为0
0 8,13,20 * * * command
```

日志文件一般存放在`/var/log/cron*`中，不过我的Ubuntu好像没有开启cron日志，可以在`/etc/rsyslog.d/50-default.conf`中进行设置（Centos在`/etc/rsyslog.conf`中设置）
```bash
> sudo vim /etc/rsyslog.d/50-default.conf
# 找到cron*, 去掉前面的#注释即可
# 重启rsyslog
> sudo service rsyslog  restart
```

crontab相关目录介绍

`/etc/anacrontab`: 这个文件存着系统级的任务。它主要用来运行每日的(daily)，每周的(weekly),每月的(monthly)的任务。一般不在此文件安装自己的任务

`/etc/cron.d/`: 此目录下存放的是系统级任务的任务文件。

`/var/spool/cron/`: 此目录下存放各个用户的任务文件。各个用户的任务存放在以自已用户名为文件名的任务文件中。此文件中的指令行没有用户域。

`/etc/crontab`: crontab的主要配置文件，执行了每小时、每日、每周、每月的脚本文件

`/etc/cron.hourly`、`/etc/cron.daily`、`/etc/cron.weekly`、`/etc/cron.monthly`：这四个目录下分别对应着每隔一段时间需要执行的脚本任务，脚本使用shell命令编写

[](https://segmentfault.com/a/1190000002628040)

[每天一个linux命令（50）：crontab命令](http://www.cnblogs.com/peida/archive/2013/01/08/2850483.html)

[Linux 下执行定时任务 crontab 命令详解](https://segmentfault.com/a/1190000002628040)


