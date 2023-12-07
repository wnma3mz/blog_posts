# -*- encoding: utf-8 -*-
'''
@File    :   download_imgs.py
@Time    :   2023/11/26 19:00:36
@Author  :   wnma3mz
@Contact :   wnma3mz@gmail.com
'''

from typing import *
import json
import requests
import re

def scrapy():
    pass


if __name__ == "__main__":
    with open(r"D:\hexo\source\_posts\Tracing-Model-Outputs-to-the-Training-Data.md", "r", encoding="utf-8") as f:
        text_lst = f.readlines()

    for text in text_lst:
        if "http" in text and ".png" in text and "![img]" in text:
            urls = re.findall(r'\bhttps?://\S+', text)
            for url in urls:
                url = url[:-1]
                fname = url.split("/")[4] + ".png"
                print(url, fname)
                # 保存图片
                r = requests.get(url)
                with open(fname, 'wb') as f:
                    f.write(r.content)