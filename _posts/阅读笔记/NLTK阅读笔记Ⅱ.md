---
title: NLTK阅读笔记Ⅱ
date: 2018-05-23 11:23:24
tags: [笔记, Python, NLTK]
categories: [NLP]
---
NLP基本知识的介绍及NLTK模块的使用。

接 {% post_link 阅读笔记/NLTK阅读笔记Ⅰ %}

<!-- more -->

## 第三章：加工原料文本

[古腾堡项目](http://www.gutenberg.org/catalog/)

可以从这个项目中获取感兴趣的文本，之后进行分析。

```python
import nltk
# 将原始文本字符串转换为list，以\r\n标点符号等为分隔符进行切割
tokens = nltk.word_tokenize(raw)
# 创建NLTK文本
text = nltk.Text(tokens)
```

处理文本部分：

`requests`： 数据获取

`Beautiful Soup`、`re`、`xpath`： 文本处理

`feedparser`：处理RSS订阅

`os`：处理本地文件

字符串基本操作在这里请见下图。注：字符串是不可改变的。

![](https://raw.githubusercontent.com/wnma3mz/blog_posts/master/imgs/nltk/3-2.png)

关于编码这个大坑，在Python3中做的已经比Python2好太多，在这里也不做介绍了。介绍一些关于编码处理的库。

`codecs`、`unicodedata`

正则表达式部分，推荐自主学习。难度系数不算高。搜索关键词，Python, 正则表达式, re即可。值得一提的是，`re`是目前我了解处理文本速度最快最优雅的一种方式。



英文文本是高度冗余的，忽略掉词内部的元音仍然可以很容易的阅读，有些时候这很明显。例如：declaration 变成 dclrtn，inalienable 变成 inlnble，保留所有词首或词尾的元音序列 。

```python
import nltk
import re

regexp = r'^[AEIOUaeiou]+|[AEIOUaeiou]+$|[^AEIOUaeiou]'
# 压缩元音字符串
def compress(word):
	pieces = re.findall(regexp, word)
	return ''.join(pieces)

# 获取语料库数据
english_udhr = nltk.corpus.udhr.words('English-Latin1')
# 进行压缩
nltk.tokenwrap(compress(w) for w in english_udhr[:75])

# 正则结合条件频率，提取辅音-元音序列
rotokas_words = nltk.corpus.toolbox.words('rotokas.dic')
cvs = [cv for w in rotokas_words for cv in re.findall(r'[ptksvr][aeiou]', w)]
cfd = nltk.ConditionalFreqDist(cvs)
cfd.tabulate()

    a   e   i   o   u
k 418 148  94 420 173
p  83  31 105  34  51
r 187  63  84  89  79
s   0   0 100   2   1
t  47   8   0 148  37
v  93  27 105  48  49
```

### 词干提取器

```python
# 查找词干，laptop与laptops其实只是单复数的区别。词干是相同的
def stem(word):
    regexp = r'^(.*?)(ing|ly|ed|ious|ies|ive|es|s|ment)?$'
    stem, suffix = re.findall(regexp, word)[0]
    return stem

# nltk自带的词干提取器
# 索引一些文本和使搜索支持不同词汇形式的话，使用Porter词干提取器
porter = nltk.PorterStemmer()
lancaster = nltk.LancasterStemmer()
# 编译一些文本的词汇或想要一个有效词条（或中心词）列表，使用WordNet词形归并器
wnl = nltk.WordNetLemmatizer()
```

### 分词：

```python
# 使用正则进行分词
re.findall(r"\w+(?:[-']\w+)|'|[-.(]+|\S\w", raw)

# 使用nltk自带的函数进行分词，效率更高且不需要特殊处理符号
nltk.regexp_tokenize()

pattern = r'''(?x) # 去除嵌入的空白字符和注释
			([A-Z]\.)+ # 设定允许的标识符（单词）
			| \w+(-\w+)* # 可选的内部连字符
			| \$?\d+(\.\d+)?%? # 货币与百分比
			| \.\.\. # 省略符
			| [][.,;"'?():-_`] # 额外的标志
		'''
nltk.regexp_tokenize(text, pattern)
```

已经分词好的数据举例：

《华尔街日报》原始文本（`nltk.corpus.treebank_raw.raw()`）和分好词的版本（`nltk.corpus.treebank.words()`）

分词的最后一个问题是缩写的存在，如“didn't”。如果我们想分析一个句子的意思，将这种形式规范化为两个独立的形式：“did”和“n't”(不是 not)可能更加有用。我们可以通过查表来做这项工作。

链表与字符串部分不做介绍，是Python字符串的常规操作

补充部分：

- 建立词汇表流程

  ![](https://raw.githubusercontent.com/wnma3mz/blog_posts/master/imgs/nltk/3-1.png)

- 编码点：每个字符分配一个编号

- 解码：通过一些机制来将文本翻译成 Unicode的过程

- 编码：将Unicode转化为其它编码的过程

  ![](https://raw.githubusercontent.com/wnma3mz/blog_posts/master/imgs/nltk/3-3.png)

- 从 Unicode的角度来看，字符是可以实现一个或多个 字形的抽象的实体。只有字形可以出现在屏幕上或被打印在纸上。一个字体是一个字符到字形映射。

- 规范化文本：如将所有的文本小写保存，使用`lower()`这个函数

- ​断句：在将文本分词之前，我们需要将它分割成句子

- 分词（重点内容）：

    1. 找到一种方法来分开文本内容与分词标志。给每个字符标注一个布尔值来指示这个字符后面是否有一个分词标志

    2. 找到将文本字符串正确分割成词汇的字位串。根据一个字典，来根据字典中词的序列来重构源文本。定义一个目标函数，通过字典大小与重构源文本所需的信息量尽力优化它的值。

    3. 找到最大化目标函数值的0和1的模式。使用**模拟退火算法**的非确定性搜索。

## 第四章：编写结构化程序

大量关于Python代码，不做介绍。此部分适宜阅读人群，使用Python时间不超过一年（不够熟练）

文档说明模块：` docstring`

调试技术：`pdb`

绘图：`matplotlib`

图的绘制：`networkx`

数据统计表：`csv`（这里个人推荐有使用`pandas`，虽然它很”重“）

数值运算：`numpy`

算法设计：

1. 递归
2. 权衡空间与时间
3. 动态规划

## 第五章：分类和标注词汇

NLP基本技术：

- 序列标注
- N-gram 模型
- 回退和评估

将词汇按它们的**词性**（parts-of-speech，POS）分类以及相应的标注它们的过程被称为**词性标注**（part-of-speech tagging, POS tagging）或干脆简称**标注**。词性也称为**词类**或**词汇范畴**。用于特定任务的标记的集合被称为一个**标记集**。

### 词性标注器

```python
import nltk
text = nltk.word_tokenize("And now for something completely different")
# 标注出text中每个词的词性
nltk.pos_tag(text)
'''
[('And', 'CC'),
 ('now', 'RB'),
 ('for', 'IN'),
 ('something', 'NN'),
 ('completely', 'RB'),
 ('different', 'JJ')]
'''
text = nltk.Text(word.lower() for word in nltk.corpus.brown.words())
# 为一个词找出所有它的上下文
text.similar('woman')
'''
man time day year car moment world house family child country boy
state job place way war girl work word
'''
```

### 标注语料库

```python
# 创建一个有标识的字符串
tagged_token = nltk.tag.str2tuple('fly/NN')
tagged_token
'''
tagged_token
('fly', 'NN')
'''
tagged_token[0]
tagged_token[1]

# 批量标注
sent = '''The/AT grand/JJ jury/NN commented/VBD on/IN a/AT number/NN of/INother/AP topics/NNS ,/, AMONG/IN them/PPO the/AT Atlanta/NP and/CCFulton/NP-tl County/NN-tl purchasing/VBG departments/NNS which/WDT it/PPsaid/VBD ``/`` ARE/BER well/QL operated/VBN and/CC follow/VB generally/Raccepted/VBN practices/NNS which/WDT inure/VB to/IN the/AT best/JJTinterest/NN of/IN both/ABX governments/NNS ''/'' ./.'''
[nltk.tag.str2tuple(t) for t in sent.split()]
'''
[('The', 'AT'),
 ('grand', 'JJ'),
 ('jury', 'NN'),
 ('commented', 'VBD'),
 ('on', 'IN'),
 ('a', 'AT'),
 ('number', 'NN'),
 ('of/INother', 'AP'),
 ('topics', 'NNS'),
 (',', ','),
 ('AMONG', 'IN'),
 ('them', 'PPO'),
 ('the', 'AT'),
 ('Atlanta', 'NP'),
 ('and/CCFulton', 'NP-TL'),
 ('County', 'NN-TL'),
 ('purchasing', 'VBG'),
 ('departments', 'NNS'),
 ('which', 'WDT'),
 ('it/PPsaid', 'VBD'),
 ('``', '``'),
 ('ARE', 'BER'),
 ('well', 'QL'),
 ('operated', 'VBN'),
 ('and', 'CC'),
 ('follow', 'VB'),
 ('generally/Raccepted', 'VBN'),
 ('practices', 'NNS'),
 ('which', 'WDT'),
 ('inure', 'VB'),
 ('to', 'IN'),
 ('the', 'AT'),
 ('best/JJTinterest', 'NN'),
 ('of', 'IN'),
 ('both', 'ABX'),
 ('governments', 'NNS'),
 ("''", "''"),
 ('.', '.')]
 '''
```

### 获取已标注的语料库

```python
# 获取标注好的语料库
nltk.corpus.brown.tagged_words()
#
nltk.corpus.brown.tagged_words(tagset=True)

nltk.corpus.nps_chat.tagged_words()
nltk.corpus.conll2000.tagged_words()
```

```python
# 未简化的标记

# 最频繁的名词标记
def findtags(tag_prefix, tagged_text):
	cfd = nltk.ConditionalFreqDist((tag, word) for (word, tag) in tagged_text if tag.startswith(tag_prefix))
	return dict((tag, cfd[tag].keys()[:5]) for tag in cfd.conditions())

tagdict = findtags('NN', nltk.corpus.brown.tagged_words(categories='news'))
for tag in sorted(tagdict):
	print tag, tagdict[tag]

'''
$：名词所有格,
S：复数名词（因为复数名词通常以 s结尾），
P：专有名词
-NC：引用
-HL：标题中的词
-TL：标题（布朗标记的特征）
'''

# 探索已标注的语料库

# 使用POS标记寻找三词短语
from nltk.corpus import brown
def process(sentence):
	for (w1,t1), (w2,t2), (w3,t3) in nltk.trigrams(sentence):
         # 动词+TO+动词
		if (t1.startswith('V') and t2 == 'TO' and t3.startswith('V')):
			print(w1, w2, w3)
for tagged_sent in brown.tagged_sents():
	process(tagged_sent)
```

Python字典部分不做介绍。

```python
# 使用默认字典可以防止使用未定义的key报错
# nltk的默认字典,需要提前定义类型（int、float、str、list、dict、tuple）。
frequency = nltk.defaultdict(int)
# 创建具有默认值的字典
pos = nltk.defaultdict(lambda: 'N')
```

### 自动标注

```python
import nltk
from nltk.corpus import brown

brown_tagged_sents = brown.tagged_sents(categories='news')
brown_sents = brown.sents(categories='news')

# 默认标注器
tags = [tag for (word, tag) in brown.tagged_words(categories='news')]
nltk.FreqDist(tags).max()
# 将所有词都标注为NN的标注器
raw = 'I do not like green eggs and ham, I do not like them Sam I am!'
tokens = nltk.word_tokenize(raw)
default_tagger = nltk.DefaultTagger('NN')
default_tagger.tag(tokens)
# 评判标注正确性
default_tagger.evaluate(brown_tagged_sents)

# 正则标注
patterns = [
	(r'.*ing$', 'VBG'), # gerunds
	(r'.*ed$', 'VBD'), # simple past
	(r'.*es$', 'VBZ'), # 3rd singular present
	(r'.*ould$', 'MD'), # modals
	(r'.*\'s$', 'NN$'), # possessive nouns
	(r'.*s$', 'NNS'), # plural nouns
	(r'^-?[0-9]+(.[0-9]+)?$', 'CD'), # cardinal numbers
	(r'.*', 'NN') # nouns (default)
]
regexp_tagger = nltk.RegexpTagger(patterns)
regexp_tagger.tag(brown_sents[3])
regexp_tagger.evaluate(brown_tagged_sents)

# 查询标注器
# 找出100个最频繁的词
fd = nltk.FreqDist(brown.words(categories='news'))
cfd = nltk.ConditionalFreqDist(brown.tagged_words(categories='news'))
most_freq_words = fd.keys()[:100]
likely_tags = dict((word, cfd[word].max()) for word in most_freq_words)
# 使用上面这个信息作为"查找标注器"
baseline_tagger = nltk.UnigramTagger(model=likely_tags)
baseline_tagger.evaluate(brown_tagged_sents)

# 使用信息作为标注器，如果没有找到就使用默认标注器。这个过程称为"回退"
baseline_tagger = nltk.UnigramTagger(model=likely_tags, backoff=nltk.DefaultTagger('NN'))
```

### 评估

使用**黄金标准**测试数据。一个已经手动标注并作为自动系统评估标准而被接收的语料库。

如果标注器对词的标记与黄金标准标记相同，那么标注器就被认为是正确的。当然这只是相对于黄金标准这个测试数据而言。

关于开发一个已标注的语料库，这是一个庞大的任务，其中涉及到了许多方面。



可以通过`nltk.app.concordance()`来可视化查找某个语料库中某个单词的词性


| 标记 | 含义           |
| ---- | -------------- |
| ADJ  | 形容词         |
| ADV  | 动词           |
| CNJ  | 连词           |
| CC   | 并列连词       |
| DET | 限定词 |
| EX | 存在量词 |
| FW | 外来词 |
| MOD | 情态动词 |
| RB   | 副词           |
| IN   | 介词           |
| N | 名词 |
| NP | 专有名词 |
| NUM | 数词 |
| PRO | 代词 |
| P | 介词 |
| TO | 词to |
| UH | 感叹词 |
| V | 动词 |
| VD | 过去式 |
| VG | 现在分词 |
| VN | 过去分词 |
| WH | Wh限定词 |
| NN   | 名词           |
| JJ   | 形容词         |
| VBP  | 一般现在时动词 |

### N-gram标注

1-gram标注器：一元标注器（unigram tagger）。用于标注**一个**标识符的上下文的只是标识符本身。

2-gram 标注器：二元标注器（bigram taggers）

3-gram 标注器：三元标注器（trigram taggers）

N-gram标注不考虑句子边界的上下文。

```python
from nltk.corpus import brown
brown_tagged_sents = brown.tagged_sents(categories='news')
brown_sents = brown.sents(categories='news')
# 标注
unigram_tagger = nltk.UnigramTagger(brown_tagged_sents)
bigram_tagger = nltk.BigramTagger(brown_tagged_sents)
trigram_tagger = nltk.TrigramTagger(brown_tagged_sents)
# 组合标注
t0 = nltk.DefaultTagger('NN')
t1 = nltk.UnigramTagger(train_sents, backoff=t0)
t2 = nltk.BigramTagger(train_sents, backoff=t1)
```

存储可以使用python自带的存储模块`pickle`。

效果：

1. 根据经验来进行判断——一般方法

   ```python
   cfd = nltk.ConditionalFreqDist(((x[1], y[1], z[0]), z[1]) for sent in brown_tagged_sents for x, y, z in nltk.trigrams(sent))
   ambiguous_contexts = [c for c in cfd.conditions() if len(cfd[c]) > 1]
   sum(cfd[c].N() for c in ambiguous_contexts) / cfd.N()
   ```

2. 研究标注器的错误——混淆矩阵

   ```python
   test_tags = [tag for sent in brown.sents(categories='editorial') for (word, tag) in t2.tag(sent)]
   gold_tags = [tag for (word, tag) in brown.tagged_words(categories='editorial')]
   print nltk.ConfusionMatrix(gold, test)
   ```

n-gram标注器的问题

1. 模型大小与标注器性能之前的平衡关系。如果使用回退标注器`n-gram`可能存储trigram和bigram表，这将会是很大的稀疏矩阵。
2. 使用上下文中的词的其他特征作为条件标记是不切实际的。

Brill标注只使用一小部分n-gram标注器。猜每个词的标记，然后返回和修复错误的。从大方面下手，再勾勒细节。规则是语言学可解释的。

```python
nltk.tag.brill.demo()
```

下一篇：{% post_link 阅读笔记/NLTK阅读笔记Ⅲ %}


