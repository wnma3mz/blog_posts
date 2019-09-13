---
title: 《Python自然语言处理》阅读笔记（一）
date: 2018-05-13 23:04:23
tags: [Python, NLP, note]
categories: [学习笔记]
---
NLP基本知识的介绍及NLTK模块的使用。

<!-- more -->

## 写在开头

在这里主要是对NLP的相关知识做一个整理和对NLTK模块的介绍，书中Python基础内容在这里不做介绍，如书中有我认为值得介绍的Python写法，我会进行说明。

由于书中nltk是老版本的问题，语法上存在一些变动，在这里也会进行修正。可能由于书是译本的原因（当然也可能原作者自己的失误），部分代码有些错误，在这里我也进行了校正。如果您发现了我的文章中有需要改正或改进的地方，欢迎在评论区提出。

最后十分感谢原作者的贡献和译者的翻译，强烈推荐读者亲自阅读此书。

nltk版本是3.2.3，[官网](http://www.nltk.org/)

python版本3.6.1

数据、PDF和一些重要的代码已经放在了Github上了。[github地址](https://github.com/wnma3mz/Nltk_Study)

## 第一章：语言处理与Python

NLTK库的安装，在这里不做介绍，强烈建议直接使用Anaconda环境。[官网下载链接](https://www.anaconda.com/download/)。本文是直接使用了Anaconda环境。

数据的获取，这里nltk有官方提供的文本数据，可以直接使用`nltk.download()`打开图形界面，下载语料集`book`。由于使用这个方法下载速度比较慢的原因，我在这里Github上提供了数据集`nltk_data`，下载之后，移动到`nltk.download()`原本的下载目录下，之后再运行`nltk.download()`就不需要下载数据集而是直接解压数据集，速度会快上很多。[下载链接](https://github.com/wnma3mz/Nltk_Study)

```python
# 导入nltk模块
import nltk
# 导入基本语料集(不需要额外下载)，包含text1到text9变量，可以直接输出这些变量
from nltk.book import *

# 搜索文本。这里表示找到"monstrous"所包含的句子，并且输出上下文
text1.concordance("monstrous")

# 搜索文本出现在相似的上下文中
text1.similar("monstrous")

# 搜索两个及两个以上共同词的上下文
text2.common_contexts(["monstrous", "very"])

# 画一张离散图表示这些词出现在文本中的位置，输出见下图
text4.dispersion_plot(["citizens", "democracy", "freedom", "duties", "America"])

# 基于文本，随机生成一些文本
text3.generate()
```

![1-1.png](https://raw.githubusercontent.com/wnma3mz/Nltk_Study/master/imgs/1-1.png)

```python
# 有序字典，按词频从高到低排序
fdist1 = FreqDist(text1)
# 选出词频最高的50个词
fdist1.keys()[:50]
# 某个词出现的频数
fdist1['whale']

# text1中词频最高的50个单词，进行绘图，输出见下图
fdist1.plot(50, cumulative=True)
# text1中只出现过一次的单词
fdist1.hapaxes()
```

![1-3.png](https://raw.githubusercontent.com/wnma3mz/Nltk_Study/master/imgs/1-3.png)

```python
# 词语搭配，双连词(bigrams)
nltk.bigrams(['more','is', 'said', 'than', 'done'])
# 输出
[('more','is'),('is','said'),('said', 'than'), ('than','done')]

# 文本中单个词的频率预期得到的更频繁出现的双连词
text4.collocations()
```



```python
# 与nltk自带的聊天机器人系统对话
# 导入模块
from nltk.chat import chatbots
# 选择一个聊天机器人并开始对话
chatbots()
```

补充部分：

| text  |                            文本名                            |
| :---: | :----------------------------------------------------------: |
| text1 |       《白鲸记》（Moby Dick by Herman Melville 1851）        |
| text2 | 《理智与情感》（Sense and Sensibility by Jane Austen 1811）  |
| text3 |              《创世纪》（The Book of Genesis）               |
| text4 |        《就职演说语料库》（Inaugural Address Corpus）        |
| text5 |               《NPS聊天语料库》（Chat Corpus）               |
| text6 |      《巨蟒与圣杯》（Monty Python and the Holy Grail）       |
| text7 |            《华尔街日报》（Wall Street Journal）             |
| text8 |               《个人文集》（Personals Corpus）               |
| text9 | 《周四的男人》（The Man Who Was Thursday by G . K . Chestert） |

概念补充：

- 词类型：一个词在一个文本中独一无二的出现形式或拼写。也就是说，这个词在词汇表中是唯一的。
- 频率分布：

![1-2.png](https://raw.githubusercontent.com/wnma3mz/Nltk_Study/master/imgs/1-2.png)

- FreqDist的API

![1-4.png](https://raw.githubusercontent.com/wnma3mz/Nltk_Study/master/imgs/1-4.png)

![1-5.png](https://raw.githubusercontent.com/wnma3mz/Nltk_Study/master/imgs/1-5.png)

- 词意消歧：需要算出特定上下文中的词被赋予的是哪个意思。单词可能存在相同/相近的含义，此时需要根据上下文来推断单词在此情景下的含义。
- 指代消解（anaphora resolution）：确定代词或名词短语指的是什么
- 先行词：代词可能代表的对象
- 语义角色标注（semantic role labeling）：确定名词短语如何与动词相关联（如施事，受事，工具等）
- 自动生成语言：需要解决自动语言理解。弄清楚词的含义、动作的主语以及代词的先行词是理解句子含义的步骤
- 机器翻译（MT）：难点，一个给定的词可能存在几种不同的解释。
- 文本对齐：根据一个网站发布的多种语言版本，来自动配对组成句子
- 人机对话系统：图灵测试。
- 文本含义识别（Recognizing Textual Entailment 简称 RTE）：根据假定的一些条件，来推断给出的一句话是否正确。

## 第二章：获得文本语料和词汇资源

[古腾堡语料库](http://www.gutenberg.org/)

![2-1.png](https://raw.githubusercontent.com/wnma3mz/Nltk_Study/master/imgs/2-1.png)

```python
# 导入nltk的古腾堡语料库
from nltk.corpus import gutenberg

# 查看这个语料库中的所有txt文件名
gutenberg.fileids()

# 选中其中的一个
emma = gutenberg.words('austen-emma.txt')
# 查看它包含的词汇个数
len(emma)

# 如果要使用第一章中concordance()这样的命令，就必须要将数据放到nltk.Text对象中
emma = nltk.Text(emma)
emma.concordance("suprprize")

# 使用words获取的是经过分隔成标识符的文本。使用sents函数则可以获取以句子进行划分后的数据。具体区别可以运行看看，请读者自行对比。
emma_sents = gutenberg.sents('austen-emma.txt')

# 获取原始文本。一个字符串
emma_raw = gutenberg.raw('austen-emma.txt')
```

```python
# 网络和聊天文本
from nltk.corpus import webtext
from nltk.corpus import nps_chat

# 布朗语料库
from nltk.corpus import brown

# 布朗语料库中所有的类别
brown.categories()
# words函数根据类别选择数据
brown.words(categories='news')
['The', 'Fulton', 'County', 'Grand', 'Jury', 'said', ...]
# words函数根据文件名选择数据
brown.words(fileids=['cg22'])
['Does', 'our', 'society', 'have', 'a', 'runaway', ',', ...]
# sents函数根据类别选择数据
brown.sents(categories=['news', 'editorial', 'reviews'])
```

```python
import nltk
from nltk.corpus import brown
# 带条件的频率分布函数

# 取出brown语料库中所有类别中的所有词。每个类别对应该类别下的所有词。(类别，所有词)
cfd = nltk.ConditionalFreqDist(
        (genre, word)
        for genre in brown.categories()
        for word in brown.words(categories=genre))

# 指定一部分文本类别(genres)
genres = ['news', 'religion', 'hobbies', 'science_fiction', 'romance', 'humor']
# 指定一些情态动词(modals)
modals = ['can', 'could', 'may', 'might', 'must', 'will']

# 输出一张表格。genres与modals的对应数量关系
cfd.tabulate(conditions=genres, samples=modals)
# 输出如下
"""
                  can could   may might  must  will
           news    93    86    66    38    50   389
       religion    82    59    78    12    54    71
        hobbies   268    58   131    22    83   264
science_fiction    16    49     4    12     8    16
        romance    74   193    11    51    45    43
          humor    16    30     8     8     9    13
"""
```

通过输出，我们很容易发现，在`news`类别里面用的最多的情态动词是`will`。当然，还有一些结论就由读者自行发现了。


关于其他语料库的信息挖掘，在这里就不赘述了。书中也只是浅显的用了之前用过的一些方法。下面展示一些我认为比较有意思的代码及输出。

官网提供了如何访问NLTK语料库的其他例子，[链接](http://www.nltk.org/howto/corpus.html)

```python
import nltk
from nltk.corpus import inaugural

# 先选出inaugural语料库中的所有字段的对应所有词汇，再统计出文本中以`america`和`citizen`开头的词在每个年份（filed）出现次数。（`america`或`citizen`, 年份）
cfd = nltk.ConditionalFreqDist(
        (target, fileid[:4])
        for fileid in inaugural.fileids()
        for w in inaugural.words(fileid)
        for target in ['america', 'citizen']
        if w.lower().startswith(target))
# 绘图
cfd.plot()
```
以`america`或`citizen`开始的词随时间（年份）的演变趋势

![2-2.png](https://raw.githubusercontent.com/wnma3mz/Nltk_Study/master/imgs/2-2.png)

```python
# udhr语料库中不同语言版本的字长差异（彩色图）
import nltk
from nltk.corpus import udhr

# 选取比较的语言
languages = ['Chickasaw', 'English', 'German_Deutsch', 'Greenlandic_Inuktikut', 'Hungarian_Magyar', 'Ibibio_Efik']
# 提取不同语言的数据
cfd = nltk.ConditionalFreqDist(
        (lang, len(word))
        for lang in languages
        for word in udhr.words(lang + '-Latin1'))

cfd.plot(cumulative=True)
```

累积字长分布： “世界人权宣言”的6个翻译版本
![2-3.png](https://raw.githubusercontent.com/wnma3mz/Nltk_Study/master/imgs/2-3.png)

```python
# 载入自己的语料库，文本文件
from nltk.corpus import PlaintextCorpusReader
# 设置语料库的路径
corpus_root = '/usr/share/dict'
# 解析路径下的所有文件
wordlists = PlaintextCorpusReader(corpus_root, '.*')
# 语料库中所有的文件名
wordlists.fileids()
# 语料库中`connectives`中的单词
wordlists.words('connectives')

# 本地个人的语料库
from nltk.corpus import BracketParseCorpusReader
corpus_root = r"C:\corpora\penntreebank\parsed\mrg\wsj"
# 使用正则匹配
file_pattern = r".*/wsj_.*\.mrg"
ptb = BracketParseCorpusReader(corpus_root, file_pattern)
ptb.fileids()
```

双连词运用
```python
# 依次连续输出最有可能跟在上一个词后面的词
def generate_model(cfdist, word, num=15):
    for _ in range(num):
        print(word)
        word = cfdist[word].max()
# 选中语料库，提取文本
text = nltk.corpus.genesis.words('english-kjv.txt')
# 生成双连词
bigrams = nltk.bigrams(text)
# 创建双连词频数对象
cfd = nltk.ConditionalFreqDist(bigrams)
# 找出最有可能在`living`后面的单词。
print(cfd['living'])
generate_model(cfd, 'living')
```


```python
# 加载停用词语料库
from nltk.corpus import stopwords
# 加载英语中的停用词
stopwords.words('English')

# 加载单词列表
from nltk.corpus import words
words.words()

# 加载人名表
from nltk.corpus import names

# 加载发音词典
from nltk.corpus import cmudict
cmudict.entries()

# 加载比较词表
from nltk.corpus import swadesh
# 比较词表的标识码
swadesh.fileids()
# 输出比较词汇中的`en`文本资料
swadesh.words('en')

# 加载词汇工具
from nltk.corpus import toolbox
# 输出罗托卡特语（Rotokas）的词典文本资料
toolbox.entries('rotokas.dic')
```

```python
# 加载WordNet
from nltk.corpus import wordnet as wn
# 输出`motocar`的同义词
wn.synsets('motorcar')
# 输出`car`名词的第一个意义中的同义词集
wn.synset('car.n.01').lemma_names
# 输出`car`名词的第一个意义的定义
wn.synset('car.n.01').definition
# 输出同义词集的所有词条
wn.synset('car.n.01').lemmas
# 输出特定的词条
wn.lemma('car.n.01.automobile')
# 输出一个词条对应的同义词集
wn.lemma('car.n.01.automobile').synset
# 输出一个词条的"名字"
wn.lemma('car.n.01.automobile').name
# 输出所有包含词`car`的词条
wn.lemmas('car')

# 获取cat词性为名词的意义
motorcar = wn.synset('car.n.01')
# 获取motorcar的下位词列表
types_of_motorcar = motorcar.hyponyms()
# 获取motorcar的上位词列表
motorcar.hypernyms()
# 获取motorcar上位词的路径列表
paths = motorcar.hypernym_paths()
# 得到motorcar的根上位（最一般的上位）同义词集
motorcar.root_hypernyms()
# 打开图形化WordNet的浏览器，更加便捷探索关系
nltk.app.wordnet()

# 更多词汇关系
# 整体到部分：想象WordNet网络关系是一棵树
# 得到树的一部分，如树干、树冠等
wn.synset('tree.n.01').part_meronyms()
# 得到树的实质，即心材和边材
wn.synset('tree.n.01').substance_meronyms()
# 得到许多树构成的集合，即森林
wn.synset('tree.n.01').member_holonyms()
# 得到mint所有名词意义的列表
wn.synsets('mint', wn.NOUN)

# 想象走路的动作
# 找出走路的一部分动作：抬脚
wn.synset('walk.v.01').entailments()

# 反义词
wn.lemma('supply.n.02.supply').antonyms()

# 使用dir()查看词汇关系和同义词集上的定义
dir(wn.synset('harmony.n.02'))

# 语义相似度
# 初始化一个同义词集中的两个单词
right = wn.synset('right_whale.n.01')
minke = wn.synset('minke_whale.n.01')
# 如果它们有共同的上位词（且层次结构较低），则它们的关系一定十分密切
right.lowest_common_hypernyms(minke)
# 查找同义词集深度量化关系，输出为数值。越小表示离根节点越近
wn.synset('baleen_whale.n.01').min_depth()
# 两个词 基于上位词层次结构中相互连接的概念之间的最短路径在0-1范围的打分（二者没有关系就返回1）
right.path_similarity(minke)

# 其他相似性度量方法，可以通过help(wn)获取。比如:VerbNet
```

概念补充：

- 文本语料库的结构：

    - 孤立的无特别组织的文本集合

    - 按文体等分类组织结构

    - 分类重叠，主题类别（路透社语料库）

    - 随时间变化语言语言用法的改变（就职演说语料库）

-  NLTK中定义的基本语料库函数

    ![2-4.png](https://raw.githubusercontent.com/wnma3mz/Nltk_Study/master/imgs/2-4.png)

- 条件频率分布

    **条件频率分布是频率分布的集合，每个频率分布有一个不同的“条件”。这个条件通常是文本的类别。**

    当语料文本被分为几类（文体、主题、作者等）时，我们可以计算每个类别独立的频率分布，研究类别之间的系统性差异。

    ![2-5.png](https://raw.githubusercontent.com/wnma3mz/Nltk_Study/master/imgs/2-5.png)

- 条件和事件

    对于NLP来说，文本出现的词汇就是事件。条件频率分布需要给每个事件关联一个条件，所以处理的是一个配对序列。每对形式是：（条件，事件）。举例就是，（文本类别，文本中的一个词汇）

- FreqDist()以一个简单的链表作为输入，ConditionalFreqDist()以一个配对链表作为输入。

- 停用词： 如`the`、`to`这种高频词汇。这种词通常没有什么词汇内容，但是它们又会让文本分析变得困难。所以在需要的情况下，我们就需要从文档中过滤掉他们。

- WordNet：面向语义的英语词典，类似与传统辞典，但具有更丰富的结构。每个节点对应一个**同义词集**，边表示上/下位词关系（上下级概念与从属概念的关系）

    ![2-6.png](https://raw.githubusercontent.com/wnma3mz/Nltk_Study/master/imgs/2-6.png)图片

- synset：同义词集，意义想相同的词（或“词条”）的集合。如car.n.01是car的第一个名词的意义，就是同义词集。

- 词条：同义词集和词的配对。

下一篇：[《Python自然语言处理》阅读笔记（二）](https://wnma3mz.github.io/hexo_blog/2018/05/23/《Python自然语言处理》阅读笔记（二）/)
