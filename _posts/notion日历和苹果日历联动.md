---
title: Notion 日历和苹果日历联动
date: 2023/12/03 00:31:01
tags: [Notion, Apple, Calendar]
categories: []
mathjax: false
---

通过一台服务器，同步 Notion 的事项至苹果手机的日历。

<!-- more -->

## 需求描述

Notion 作为一个强大的笔记软件，吸引笔者使用的点在于跨平台，功能齐全。因此，在将 Notion 作为笔记软件使用的同时，也可以将其作为 TODO 事项管理软件使用。但是，移动端的 Notion 并没有实用的小组件视图。因此，结合之前的文章想到用 ics 文件来管理 Notion 的 TODO 事项。

## 数据获取

Notion 支持以 API 的方式获取 Page 的信息，尤其是当 Page 中主要以表格形式展示时，甚至可以用一系列过滤条件。首先，需要创建一个工具，以能够用 API 的方式获取 Notion 中的 Page 信息。https://www.notion.so/my-integrations，创建过程很简单，主要保存下 Secrets 即可。

![](https://raw.githubusercontent.com/wnma3mz/blog_posts/master/imgs/notion_cal_apple/1701535383410.png)

创建完成后，就可以在 Notion 的 Page 中 Add connections，如下图所示，可以把刚刚创建的工具，添加进来。比如这里是 123。

![](https://raw.githubusercontent.com/wnma3mz/blog_posts/master/imgs/notion_cal_apple/1701535461302.png)

同时，保存 Page 页面的 url，比如 https://www.notion.so/AAA，访问页面时的 url，AAA 就是 dataset_id。至此，有两个信息，一个是 Secrets，一个是 dataset_id。将其粘贴至下面的 Python 代码中。

Python 的示例代码如下：

```python
import requests
from datetime import datetime, timedelta, timezone

key = ...
dataset_id = ...
url = f"https://api.notion.com/v1/databases/{dataset_id}/query"


today = datetime(datetime.now().year, datetime.now().month, datetime.now().day)
yesterday = today - timedelta(days=1)  # 计算前一天的时间

payload = {
    "page_size": 100,
    "filter": {}, # 增加一系列过滤条件，也可以事后用 Python 再过滤
}

headers = {
    "accept": "application/json",
    "Notion-Version": "2022-06-28",
    "content-type": "application/json",
    "Authorization": f"Bearer {key}",
}


if __name__ == "__main__":
    response = requests.post(url, headers=headers, json=payload)
    data_lst = response.json()["results"]
```

至此，Page 上的信息已经存到了 data_lst 中。

## ics 文件生成

接下来，就是生成 ics 文件的生成。ics 文件的格式需要日程标题 `summary`、日程描述 `description`（可为空）、开始时间 `dtstart`、结束时间 `dtend`。

```python
from datetime import datetime, timedelta, timezone
from icalendar import Calendar, Event

if __name__ == '__main__':
    cal = Calendar()
    cal["version"] = "2.0"
    cal["prodid"] = "-//ABC ICS//ZH"

    for item in data_lst:
        # 处理每条信息，将其转换为如下信息
        summary = ...
        description = ...
        dtstart = ...
        dtend = ...

        event = Event()
        event.add("summary", summary)
        event.add("description", description)
        event.add("dtstart", dtstart)
        event.add("dtend", dtend)
        cal.add_component(event)

    with open("my.ics", "wb") as f:
        f.write(cal.to_ical())
```

## ics 部署

将生成的 ics 文件部署到服务器上，以便于服务器定期更新日程信息。同时，苹果可以订阅该 ics，以更新日程信息。

对于 tailscale 这一点而言，非常简单，只需要在服务器上运行如下命令即可。否则，需要使用域名 + nginx + https，这样才能将 ics 文件暴露出去，且苹果只接受 https 的 ics 文件。

```bash
tailscale funnel --bg https+insecure://localhost --set-path /ics /data/my.ics
```

另外，需要在服务器上，设置一个定时任务，定时运行上面的 python 脚本，以更新 notion 上的日程信息至 ics，最后苹果手机再接收到 ics 文件对日历进行更新。

苹果手机的订阅步骤如下：

设置 -> 日历 -> 账户 -> 添加账户 -> 其他 -> 添加已订阅的日历 -> 输入 ics 文件的 url 即可（比如这里应该是 https://A/ics）。这样就完成了所有的设置
