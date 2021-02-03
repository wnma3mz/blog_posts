---
title: ICS在线课表制作
date: 2020-09-16 18:41:23
tags: [note, Python, ics]
categories: [奇技淫巧]
---
如何制作在线的日历ics文件，最后介绍各个设备如何使用

<!-- more -->

## 是什么

ICS为iCalendar的文件名，它是“日历数据交换”的标准。大多数日历本质创建事项都是通过ics这种格式来生成、解析。详见[Wikipedia](https://zh.wikipedia.org/wiki/ICalendar)

## 为什么

一般来说，对于临时会议/事件，就直接在日历添加事件即可。但如果是一些周期性、某个时间段持续的事件活动，如果手动添加就很麻烦，比如最常见的就是课表。还有一些如学术会议、球赛、定期活动、节假日等。这类往往是通过他人制作好ics文件，本地导入即可。

## 怎么做

我的目标就是制作自己的课表（在线）

### 格式

```
BEGIN:VCALENDAR
VERSION:2.0
PRODID:-//hacksw/handcal//NONSGML v1.0//EN
BEGIN:VEVENT
UID:uid1@example.com
DTSTAMP:19970714T170000Z
ORGANIZER;CN=John Doe:MAILTO:john.doe@example.com
DTSTART:19970714T170000Z
DTEND:19970715T035959Z
SUMMARY:Bastille Day Party
END:VEVENT
END:VCALENDAR
```

以行为分隔，每行以冒号分隔。

第一行，每个ics文件的开头，对应最后一行

第二行，版本（默认）

第三行：可自定义，文件说明

第四行：定义一个事件名称，对应倒数第二行

第五行：唯一的id

第六行：创建这个事件的时间

第七行：可忽略

第八行：开始时间

第九行：结束时间

第十行：事件名称

### 我的示例

```
BEGIN:VCALENDAR
VERSION:2.0
PRODID:-//JH-L//JH-L Calendar//
BEGIN:VEVENT
SUMMARY:课程A
DTSTART;TZID="UTC+08:00";VALUE=DATE-TIME:20200914T083000
DTEND;TZID="UTC+08:00";VALUE=DATE-TIME:20200914T101000
DTSTAMP;VALUE=DATE-TIME:20200915T000946Z
UID:c39b3578-f6e7-11ea-bd58-525400eb2034
DESCRIPTION:老师A
LOCATION:上课地点B
END:VEVENT
BEGIN:VEVENT
SUMMARY:课程B
DTSTART;TZID="UTC+08:00";VALUE=DATE-TIME:20200916T083000
DTEND;TZID="UTC+08:00";VALUE=DATE-TIME:20200916T101000
DTSTAMP;VALUE=DATE-TIME:20200915T000946Z
UID:c39b3af0-f6e7-11ea-bd58-525400eb2034
DESCRIPTION:老师B
LOCATION:上课地点A
END:VEVENT
```

相比于原始格式，多添加了一个事件，并且添加了地点`LOCATION`和事件描述`DESCRIPTION`

### 生成

既然知道格式，那么生成就很简单了，可以自己按照这种格式手写进一个文档，后缀名为ics即可。这里我借助了Python的icalendar来完成这个工作。

以下是一个函数，借助这个函数可以生成一个事件，生成后可以添加至日历中。这里uid因为是要求唯一的，所以借助了uuid来完成。这里有一个小坑，由于我们要是北京时间，所以添加时间是，需要加入`tz_utc_8`这个变量`tz_utc_8 = timezone(timedelta(hours=8))`

```python
def cread_event(lesson_name, classroom, teacher, start, end):
    # 创建事件/日程
    event = Event()
    event.add('summary', lesson_name)

    dt_now = datetime.now(tz=tz_utc_8)
    event.add('dtstart', start)
    event.add('dtend', end)
    # 创建时间
    event.add('dtstamp', dt_now)
    event.add('LOCATION', classroom)
    event.add('DESCRIPTION', '教师：' + teacher)

    # UID保证唯一
    event['uid'] = str(uuid.uuid1()) + '/wnma3mz@gmail.com'

    return event
```

下一个问题就变为，开始时间与结束时间该怎么写。这里需要使用datetime函数，利用datetime来生成时间格式。

```python
    for lesson in cls_lst:
        # 课程名字，教师，教室
        # 课程开始时间(s1小时，s2分钟)，课程结束时间(e1小时，e2分钟)
        # name, teacher, room = f'{lesson["name"]}-{lesson["room"]}', lesson['teacher'], lesson['room']
        name, teacher, room = lesson['name'], lesson['teacher'], lesson['room']
        s1, s2 = lesson['time'][0][0]
        e1, e2 = lesson['time'][-1][-1]
        for week in lesson['week']:
            # 第N周
            week_delta = timedelta(days=(week - 1) * 7)
            for day in lesson['day']:
                # 周N
                day_delta = timedelta(days=(day - 1))
                new_date = begin_date + week_delta + day_delta
                # 上课的年月日
                new_year, new_month, new_day = new_date.year, new_date.month, new_date.day
                ymd = [new_year, new_month, new_day]
                # 课程开始时间和结束时间
                start = datetime(*ymd, s1, s2, tzinfo=tz_utc_8)
                end = datetime(*ymd, e1, e2, tzinfo=tz_utc_8)

                cal.add_component(cread_event(name, room, teacher, start, end))
```

这里需要提前约定好格式。学期开始的年月日、上课的时间及课程上课的周数、时间等。ics是支持固定周期添加事件的，但是由于我的课很乱（上课周数不确定、有的课是1-3节，有的课是2-3节，有的课是1-2节），所以我这里没有使用周期这个功能。

```python
time_dict = {
    1: [(8, 30), (9, 20)],
    2: [(9, 20), (10, 10)],
    3: [(10, 30), (11, 20)],
    4: [(11, 20), (12, 10)],
    5: [(13, 30), (14, 20)],
    6: [(14, 20), (15, 10)],
    7: [(15, 30), (16, 20)],
    8: [(16, 20), (17, 10)],
    9: [(18, 10), (19, 00)],
    10: [(19, 00), (19, 50)],
    11: [(20, 10), (21, 00)],
    12: [(21, 00), (21, 50)],
}
begin_year = 2020
begin_month = 9
begin_day = 7

cls_lst = [
        {
            'name': '课程A',
            'teacher': '教师名称',
            'room': '教室',
            'time': [time_dict[1], time_dict[2]], # 第一节课-第二节课
            'week': [2, 3, 4, 5, 6, 7, 8, 9, 10], # 2-10周
            'day': [1, 3] # 周一、周三
        },
]
```

完整代码见[https://github.com/wnma3mz/Tools/blob/master/others/myics.py](https://github.com/wnma3mz/Tools/blob/master/myics.py)

### 部署

生成ics文件后，如果不需要更新的话，那么可以直接导入到本地的日历中。IOS系统可直接导入，安卓手机部分不支持，且如果导入后删除可能需要手动一个一个删除。。。

注：小米、VIVO经测试不可导入，因为厂商阉割了此功能；魅族、荣耀、三星、苹果、华为均可导入，其中苹果导入的方式是用邮件发送到ios设备上的已绑定的邮件中。

故基于此，最终考虑还是使用在线部署的方式进行导入。为方便IOS导入，所以需要配置好https。

```python
server {
    listen       port;
    listen       [::]:port ipv6only=on;
    server_name  ics.test.com; # ios导入必须使用域名
    root         /ics_dir;;
    index        my.ics;
    rewrite ^(.*)$ https://$host$1 permanent;
    
    location / {
        root  /ics_dir;
        index my.ics;
	    access_log off;
	    expires 1d;
    }
    location ~* ^.+\.(ics) {
        root /ics_dir;
 	    access_log off;
 	    expires 1d;
    }
}

server {
       listen       443 ssl http2;
       listen       [::]:443 ssl http2;
       server_name  ics.test.com;
       ssl_certificate /etc/nginx/ics.test.com/Nginx/1_ics.test.com_bundle.crt;
       ssl_certificate_key /etc/nginx/ics.test.com/Nginx/2_ics.test.com.key;
       location / {
           tcp_nodelay on;
           proxy_set_header Host $host;
           proxy_set_header X-Real-IP $remote_addr;
           proxy_set_header X-Forwarded-For $proxy_add_x_forwarded_for;
           root         /ics_dir;
           index        my.ics;
       }
       location ~* ^.+\.(ics) {
	         root /ics_dir;
	         access_log off;
		       expires 1d;
       }
     }
```

部署成功后，直接访问ics.test.com会直接该ics文件

### 各个设备配置

Windows端（浏览器）：Outlook、谷歌日历均有在线导入ics文件的入口

IOS/MacOS端：添加账号中，导入ics文件的域名即可

安卓端：由于原生日历不支持该项功能，故使用OneCalendar。当然，也可以在Windows端配置好后，下载对应APP亦可。

## 写到最后

制作完成之后发现，我校居然自动生成了ics文件。。。（但感觉还是不如自己做的好。。。。）