---
title: 用ADB调试安卓手机
date: 2018-01-25 13:55:11
tags: [ADB, Python]
categories: [Android]
---

使用adb工具实现对安卓手机的调试，包括安装应用、查看运行App、录屏、截图等功能。本文详细介绍了adb工具的安装、配置和使用方法，并列出了一些实用的命令，同时还介绍了无线网调试设备的方法。

<!-- more -->

写在开头，一些推荐阅读的博客文章

[通过adb获取手机信息](http://blog.csdn.net/fasfaf454/article/details/51438743?locationNum=14)

[Android adb你真的会用吗?](https://www.jianshu.com/p/5980c8c282ef)

[ADB调试命令大全](http://blog.csdn.net/qq_15364915/article/details/52369266)

### 介绍

adb是安卓开发调试工具，接触到这个还是因为微信跳一跳的外挂原因。看文档说明之后得知需要adb工具，进行简单尝试后，发现这玩意还真的是挺好用的。填补了自己用代码控制安卓手机的空缺（之前也玩过按键精灵，但是需要Root权限）。

[**adb官方说明**](https://developer.android.com/studio/command-line/adb.html?hl=zh-cn)：

Android 调试桥 (adb) 是一个通用命令行工具，其允许您与模拟器实例或连接的 Android 设备进行通信。它可为各种设备操作提供便利，如安装和调试应用，并提供对 Unix shell（可用来在模拟器或连接的设备上运行各种命令）的访问。该工具作为一个客户端-服务器程序，包括三个组件：

### 安装

可参考文章:  [How to Install ADB on Windows, macOS, and Linux](https://www.xda-developers.com/install-adb-windows-macos-linux/)

#### Windows

这里我使用的是win10。如果以下操作不能成功配置，可以百度其他的安装教程

1. 下载。[下载链接](https://dl.google.com/android/repository/platform-tools-latest-windows.zip)
2. 移动目录

   1. 解压压缩包
   2. 移动文件夹你想让它放置的位置
3. 环境配置

   1. 添加环境变量，这样可以在cmd或者powershell中直接使用adb
   2. 右键"计算机"->"属性"->左边的"高级系统设置"->"环境变量">-在上面用户变量中的"Path"进行编辑->"添加"刚刚放置的文件夹目录，确认保存即可
      ![](https://raw.githubusercontent.com/wnma3mz/blog_posts/master/imgs/adb/20180125135305187.png)
   3. 这样再打开cmd或者powershell，就可以直接使用adb命令了

#### Linux

Linux系统安装adb工具还是很容易的，我使用的是Ubuntu17.10

```bash
# 下载
> sudo apt updagte
> sudo apt upgrade
> sudo apt install android-tools-adb
```

### 配置

手机需要打开USB调试模式，在实验的时候，市面上流行的大部分安卓手机USB调式模式开关默认关闭。当然如果手机默认开启了USB调试开关，自然可以省去下面一步。这里简单介绍一下如何手动打开USB调试模式开关，手机“设置”—>"关于手机"—>点击"版本号"7次。每个手机开启的方式大同小异，如果有不同，可以百度“xx手机如何开启USB调试”。

在USB调试开关开启之后，进入USB调试，打开USB调试开关，如下图。

<img src="https://raw.githubusercontent.com/wnma3mz/blog_posts/master/imgs/adb/20180125135319543.png" width="300px" />

之后使用USB数据线连接到电脑。

连接成功后，手机需要信任电脑。电脑打开cmd/powershell（Windows）或者终端（Linux），输入 `adb devices`。查看输出。

如果有输出设备名且无警告，则表示连接成功，否则需要进行下面一步的配置。

**Windows**

首先需要连接手机，确保手机能连接到电脑且打开了USB调试。

第二步，右键"计算机"->"属性"->左边的"设备管理器"->"便携设备"->"详细信息"->"属性"下拉选择"硬件Id"，查看"值"中的内容。

第三步，进入C盘用户目录下的隐藏文件夹".android"，编辑或者新建文件"adb_usb.ini"，用记事本打开，在里面写上内容。我这里需要写的是"0x2A45"，0x是前缀不需要更改，后面的字符串对应的更改"硬件Id"里面"值"的VID后面的四个字符。保存退出即可

![](https://raw.githubusercontent.com/wnma3mz/blog_posts/master/imgs/adb/20180125135348748.png)

第四步，断开USB连接，重新连接，再输入 `adb devices`。正常情况下应该会出现设备的成功连接的信息

**Linux**

```bash
# 先连接手机，输入下面命令，找到关于连接手机的那一行
> lsusb
# 输出形如下面，最后的"Google Inc"表示的是设备名，这里需要找到匹配自己手机的设备，并记录好ID
...
Bus 001 Device 017: ID 18d1:d002 Google Inc.
...
# 在目录/etc/udev/rules.d/下，添加文件70-android.rules(名字貌似无所谓？？？)
> sudo vim /etc/udev/rules.d/70-android.rules
# 写入下面内容，idVendor表示ID冒号前的数字，idProduct表示的是ID冒号后的数字，MODE固定为"0666"，其他不变
SUBSYSTEM=="usb", ATTRS{idVendor}=="18d1", ATTRS{idProduct}=="d002",MODE="0666"
# 保存退出，修改权限
> sudo chmod 644   /etc/udev/rules.d/70-android.rules
> sudo chown root. /etc/udev/rules.d/70-android.rules
# 重启服务
> sudo service udev restart
> sudo killall adb

# 输入adb devices查看输出，是否有误
> adb devices

```

### 使用

详细可参考这篇博客: [ADB调试命令大全](http://blog.csdn.net/qq_15364915/article/details/52369266)

在这里贴出一些我认为比较有意思且实用的功能

```bash
# 查看所有连接设备
> adb devices

# 进行截图保存在sd卡的根目录下，名字为screen.png
> adb shell screencap -p /sdcard/screen.png
# 将截图发送到本地（当前目录下），也可以发送其他文件
> adb pull /sdcard/screen.png
# 删除本地文件
> adb shell rm /sdcard/screen.png
# 发送电脑里的文件到设备
> adb shell push screen.png /sdcard/


# 进入手机的交互环境，操作类似linux终端，exit或者Ctrl+C退出
> adb shell

# 点击手机屏幕(1000,1000)的位置
> adb shell input tap 1000 1000
# 输入字符串"helloworld"，此处不能直接输入中文，且字符串不能有空格
> adb shell input text helloworld
# 滑动屏幕，从（100, 100)到（1000,1000)，经历10s(也可以当作长按屏幕来使用)
> adb shell input swipe 100 100 1000 1000 10

# 查看当前运行的App, 这里Windows没有grep所以会运行失败，可以进入先进入交互环境再输入下面去掉"adb shell"命令
> adb shell dumpsys window | grep mCurrentFocus
# 或者
> adb shell dumpsys activity activities | grep mFocusedActivity

# 按下电源键
> adb shell input keyevent 26
# 按下返回键
> adb shell input keyevent 4
# 按下HOME健
> adb shell input keyevent 3
# 点亮屏幕
> adb shell input keyevent 224
# 熄灭屏幕
> adb shell input keyevent 223

# 查看手机安装了哪些App，输出按行输出App的包名
> adb shell pm list packages
# 加"-s"表示只输出系统应用
# 加"-3"表示只输出第三方应用
# 加字符串表示过滤应用名称，当然也可以使用grep

# 安装apk
> adb install <packagename>
# 卸载apk
> adb uninstall <packagename>


# 从桌面启动app
> adb shell monkey -p <packagename> -c android.intent.category.LAUNCHER 1
# 关闭app
> adb shell am force-stop <packagename>
```

### 使用无线网调试设备

确保手机和电脑在同一局域网内

如何查看手机的ip，由于每个设备的方法不大相同，在此不进行说明，请读者自行百度。

```bash
# 使用USB连接设备，启动手机5555端口
> adb tcpip 5555
# adb连接命令，需要知道手机的ip
> adb connect 192.168.1.111
# 输入如下则表示连接成功
connected to 192.168.1.111:5555
# 这里要注意的是，如果接下来的步骤未断掉数据线，则电脑是连接了两台设备的，进行操作会产生错误，所以可以拔掉数据线或者断开网络或者指定设备运行
# 断开无线网连接
> adb disconnect 192.168.1.111
```
