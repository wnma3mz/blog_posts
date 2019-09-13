---
title: 《Python自然语言处理》阅读笔记（三）
date: 2018-08-05 18:57:24
tags: [Python, NLP, note]
categories: [学习笔记]
---
NLP基本知识的介绍及NLTK模块的使用。

接[《Python自然语言处理》阅读笔记（二）](https://wnma3mz.github.io/hexo_blog/2018/05/23/%E3%80%8APython%E8%87%AA%E7%84%B6%E8%AF%AD%E8%A8%80%E5%A4%84%E7%90%86%E3%80%8B%E9%98%85%E8%AF%BB%E7%AC%94%E8%AE%B0%EF%BC%88%E4%BA%8C%EF%BC%89/)

<!-- more -->

## 第六章：学习分类文本

目标：

1. 找语言数据中的分类特征
2. 构建模型
3. 从模型中学习知识

分类学习，一般分为有监督学习、无监督学习、半监督学习。

分类的定义：将给定的输入选择正确（合适）的类标签。这里值得注意的是，输入之间是相互独立的，标签集是预先定义好的。延伸的任务，每个实例（输入）可以划分为多个标签；分类问题不一定预先定义标签集，类似K-Means聚类；序列分类中，一个输入链表可以作为一个整体分类。

在这里，我们预先得到一些包含正确标签的语料（实例/输入），根据这些数据建立得到的模型进行分类，我们称为有监督分类。有监督分类中，在训练过程中，需要将每个一个输入值转换为特征集（特征提取器）。特征集对应着输出类标签。



```python
# 特征提取器，提取输入字符串中最后一个字母。
def gender_features(word):
	# 返回特征集
    return {'last_letter': word[-1]}
# 导入姓名语料库, 获取男女姓名列表
from nltk.corpus import names
import random
import nltk

# 随机打乱男女姓名列表
random.shuffle(names)

# 划分训练集和测试集
featuresets = [(gender_features(n), g) for (n,g) in names]
train_set, test_set = featuresets[500:], featuresets[:500]
# 利用贝叶斯分类来训练模型
classifier = nltk.NaiveBayesClassifier.train(train_set)

# 进行预测 
classifier.classify(gender_features('Neo'))
classifier.classify(gender_features('Trinity'))

# 准确率
nltk.classify.accuracy(classifier, test_set)

# 最有效的五个特征（字母）
classifier.show_most_informative_features(5)
'''
Most Informative Features
             last_letter = 'k'              male : female =     43.4 : 1.0
             last_letter = 'a'            female : male   =     32.9 : 1.0
             last_letter = 'f'              male : female =     16.1 : 1.0
             last_letter = 'p'              male : female =     12.6 : 1.0
             last_letter = 'v'              male : female =     10.6 : 1.0
'''                   
# 输出的比率称为 似然比，可以用于比较不同特征-结果关系

# 处理大型语料库时，构建一个包含每一个实例的特征的单独的链表会使用大量的内存。这种情况下，可以使用下面的函数，使返回一个行为像一个链表而不会在内存存储所有特征集的对象
from nltk.classify import apply_features
train_set = apply_features(gender_features, names[500:])
test_set = apply_features(gender_features, names[:500])
```

### 选取特征

过拟合：用于一个给定的学习算法的特征的数目是有限的——如果提供太多的特征，那么该算法将高度依赖训练数据的特性而一般化到新的例子的效果不会很好。

过拟合当运作在小训练集上时尤其会有问题。

```python
# 特征提取器，过拟合提取特征。这里请对比gender_features2与上一版本的区别
def gender_features2(name):
    features = {}
    features["firstletter"] = name[0].lower()
    features["lastletter"] = name[–1].lower()
    for letter in 'abcdefghijklmnopqrstuvwxyz':
        features["count(%s)" % letter] = name.lower().count(letter)
        features["has(%s)" % letter] = (letter in name.lower())
    return features

featuresets = [(gender_features2(n), g) for (n,g) in names]
train_set, test_set = featuresets[500:], featuresets[:500]
classifier = nltk.NaiveBayesClassifier.train(train_set)
# 对比上一版本的预测结果，基本可以发现精确度下降了
nltk.classify.accuracy(classifier, test_set)
```

一旦初始特征集被选定，完善特征集的一个非常有成效的方法是**错误分析**。首先，我们选择一个 开发集，包含用于创建模型的语料数据。然后将这种开发集分为 训练集和 开发测试集。训练集用于训练模型，开发测试集用于进行错误分析，测试集用于系统的最终评估。

```python
# 划分训练集、开发集、开发测试集
train_names = names[1500:]
devtest_names = names[500:1500]
test_names = names[:500]
# 进行训练预测
train_set = [(gender_features(n), g) for (n,g) in train_names]
devtest_set = [(gender_features(n), g) for (n,g) in devtest_names]
test_set = [(gender_features(n), g) for (n,g) in test_names]
classifier = nltk.NaiveBayesClassifier.train(train_set)
nltk.classify.accuracy(classifier, devtest_set)

# 错误分析
errors = []
for (name, tag) in devtest_names:
    guess = classifier.classify(gender_features(name))
    if guess != tag:
        errors.append((tag, guess, name))
        
# 输出错误分类列表中，猜错的类别及名字
for (tag, guess, name) in sorted(errors):
	print('correct=%-8s guess=%-8s name=%-30s' % (tag, guess, name))
    
# 观察输出进行分析，得到一些结果。
# yn结尾的名字显示以女性为主，虽然n结尾的名字往往是男性
# 以ch结尾的名字通常是男性，虽然h结尾的名字倾向于是女性
# 根据结论，调整特征提取器
def gender_features(word):
	return {'suffix1': word[-1:], 'suffix2': word[-2:]}

# 进行训练预测
train_set = [(gender_features(n), g) for (n,g) in train_names]
devtest_set = [(gender_features(n), g) for (n,g) in devtest_names]
test_set = [(gender_features(n), g) for (n,g) in test_names]
classifier = nltk.NaiveBayesClassifier.train(train_set)
nltk.classify.accuracy(classifier, devtest_set)
# 最终结果理论上会进行提高
```

不断地进行错误分析改善特征提取器，可以不断提高预测的准确度。每次错误分析应该选取不同的开发测试/训练进行分割，以检查新改进分类器可能产生的新的错误模式。

### 文档分类

```python
from nltk.corpus import movie_reviews
import random
import nltk

# 根据电影评论分类（'neg', 'pos'）获取电影评论
documents = [(list(movie_reviews.words(fileid)), category) for category in movie_reviews.categories() for fileid in movie_reviews.fileids(category)]
# 随机打乱数据
random.shuffle(documents)

# 建立文档分类的特征提取器
# 使用FreqDist会比一般的list更快
all_words = nltk.FreqDist(w.lower() for w in movie_reviews.words())
word_features = list(all_words.keys())[:2000]
# 每个词是否在一个给定的文档中
def document_features(document):
    document_words = set(document)
    features = {}
    for word in word_features:
    	features['contains(%s)' % word] = (word in document_words)
    return features

# 训练文档分类模型
featuresets = [(document_features(d), c) for (d,c) in documents]
train_set, test_set = featuresets[100:], featuresets[:100]
classifier = nltk.NaiveBayesClassifier.train(train_set)
# 检验模型
nltk.classify.accuracy(classifier, test_set)
classifier.show_most_informative_features(5) 
'''
Most Informative Features
 contains(unimaginative) = True              neg : pos    =      7.7 : 1.0
    contains(schumacher) = True              neg : pos    =      7.5 : 1.0
        contains(suvari) = True              neg : pos    =      7.1 : 1.0
          contains(mena) = True              neg : pos    =      7.1 : 1.0
        contains(shoddy) = True              neg : pos    =      7.1 : 1.0
'''
# 显然可以发现，`unimaginative`这个词是负面的概率是正面的7.7倍
```

### 词性标注

```python
from nltk.corpus import brown
import nltk

# 为词选择词性标记，通过词的后缀来选择
suffix_fdist = nltk.FreqDist()
for word in brown.words():
    word = word.lower()
    suffix_fdist[word[-1:]] += 1
    suffix_fdist[word[-2:]] += 1
    suffix_fdist[word[-3:]] += 1
common_suffixes = list(suffix_fdist.keys())[:100]

# 现在通过特征提取器，计算哪个后缀最有信息量（最具代表性)
def pos_features(word):
    features = {}
    for suffix in common_suffixes:
        features['endswith(%s)' % suffix] = word.lower().endswith(suffix)
    return features

tagged_words = brown.tagged_words(categories='news')
# 利用特征提取器选取特征
featuresets = [(pos_features(n), g) for (n,g) in tagged_words]
size = int(len(featuresets) * 0.1)
train_set, test_set = featuresets[size:], featuresets[:size]
# 利用决策树方法来训练模型
classifier = nltk.DecisionTreeClassifier.train(train_set)

nltk.classify.accuracy(classifier, test_set)
classifier.classify(pos_features('cats'))
# 将nltk运算过程，以伪代码方式输出。设定决策树深度为4
classifier.pseudocode(depth=4)


# 根据上下文语境，更新特征提取器
def pos_features(sentence, i):
    features = {"suffix(1)": sentence[i][-1:],
        "suffix(2)": sentence[i][-2:],
        "suffix(3)": sentence[i][-3:]
    }
    if i == 0:
    	features["prev-word"] = "<START>"
    else:
    	features["prev-word"] = sentence[i-1]
    return features

pos_features(brown.sents()[0], 8)

# 检验新的词性标注器性能
tagged_sents = brown.tagged_sents(categories='news')
featuresets = []
for tagged_sent in tagged_sents:
	untagged_sent = nltk.tag.untag(tagged_sent)
	for i, (word, tag) in enumerate(tagged_sent):
		featuresets.append((pos_features(untagged_sent, i), tag))
size = int(len(featuresets) * 0.1)
train_set, test_set = featuresets[size:], featuresets[:size]
classifier = nltk.NaiveBayesClassifier.train(train_set)
nltk.classify.accuracy(classifier, test_set)
```

以上是在结合整个词集做的标注器，但是往往针对某些特殊情况，如形容词后面有很大概率上是名词。对于这种情况，我们的词性标注器目前还没有进行独立对待。

### 序列分类

联合分类器：捕捉相关的分类任务之间的依赖关系，收集有关输入，选择适当的标签

连续分类（贪婪序列分类）：为第一个输入找到最有可能的类标签，然后使用这个问题的答案帮助找到下一个输入的最佳的标签。

```python
def pos_features(sentence, i, history):
    features = {"suffix(1)": sentence[i][-1:],
    	"suffix(2)": sentence[i][-2:],
    	"suffix(3)": sentence[i][-3:]}
	if i == 0:
        features["prev-word"] = "<START>"
        features["prev-tag"] = "<START>"
    else:
        features["prev-word"] = sentence[i-1]
        features["prev-tag"] = history[i-1]
	return features
class ConsecutivePosTagger(nltk.TaggerI):
    def __init__(self, train_sents):
        train_set = []
        for tagged_sent in train_sents:
        	untagged_sent = nltk.tag.untag(tagged_sent)
            history = []
            for i, (word, tag) in enumerate(tagged_sent):
                featureset = pos_features(untagged_sent, i, history)
                train_set.append( (featureset, tag) )
                history.append(tag)
        self.classifier = nltk.NaiveBayesClassifier.train(train_set)
    def tag(self, sentence):
        history = []
        for i, word in enumerate(sentence):
            featureset = pos_features(sentence, i, history)
            tag = self.classifier.classify(featureset)
            history.append(tag)
        return zip(sentence, history)
    
tagged_sents = brown.tagged_sents(categories='news')
size = int(len(tagged_sents) * 0.1)
train_sents, test_sents = tagged_sents[size:], tagged_sents[:size]
tagger = ConsecutivePosTagger(train_sents)
tagger.evaluate(test_sents)
```

### 其他序列模型

以上模型有一个致命问题就是无法修复已经标注错误的词性。另一种方案**隐马尔科夫模型**可以为所有的可能序列进行打分，选择得分最高的序列。这种模型的缺点就是计算量相当大，可以采用动态规划来解决这种问题。基于这种模型，产生了**最大熵马尔可夫模型**和**线性链条件随机场模型**，二者为可能序列打分的算法不同。

### 句子分割

```python
sents = nltk.corpus.treebank_raw.sents()
# 单独的句子标识符的合并链表
tokens = []
# 所有句子边界标识索引
boundaries = set()
offset = 0
for sent in sents:
    tokens.extend(sent)
    offset += len(sent)
    boundaries.add(offset-1)
    
# 特征提取器
def punct_features(tokens, i):
	return {'next-word-capitalized': tokens[i+1][0].isupper(),
            'prevword': tokens[i-1].lower(),
            'punct': tokens[i],
            'prev-word-is-one-char': len(tokens[i-1]) == 1}

featuresets = [(punct_features(tokens, i), (i in boundaries))
				for i in range(1, len(tokens)-1)
				if tokens[i] in '.?!']
# 进行训练，检验模型准确率
size = int(len(featuresets) * 0.1)
train_set, test_set = featuresets[size:], featuresets[:size]
classifier = nltk.NaiveBayesClassifier.train(train_set)
nltk.classify.accuracy(classifier, test_set)
```

### 识别对话行为类型

```python
# 提取即时消息会话语料库
posts = nltk.corpus.nps_chat.xml_posts()[:10000]
# 特征提取器，每个帖子包含什么词
def dialogue_act_features(post):
    features = {}
    for word in nltk.word_tokenize(post):
    	features['contains(%s)' % word.lower()] = True
    return features
# 构建模型，判断是否为对话行为
featuresets = [(dialogue_act_features(post.text), post.get('class')) for post in posts]
size = int(len(featuresets) * 0.1)
train_set, test_set = featuresets[size:], featuresets[:size]
classifier = nltk.NaiveBayesClassifier.train(train_set)
nltk.classify.accuracy(classifier, test_set)
```

### 识别文字蕴含

识别文字蕴含（Recognizing textual entailment，RTE）是判断文本 T 的一个给定片段是否蕴含着另一个叫做“假设”的文本。

```python
def rte_features(rtepair):
    #  RTEFeatureExtractor 类建立了一个除去一些停用词后在文本和假设中都有的词汇包，然后计算重叠和差异
    extractor = nltk.RTEFeatureExtractor(rtepair)
    features = {}
    features['word_overlap'] = len(extractor.overlap('word'))
    features['word_hyp_extra'] = len(extractor.hyp_extra('word'))
    features['ne_overlap'] = len(extractor.overlap('ne'))
    features['ne_hyp_extra'] = len(extractor.hyp_extra('ne'))
    return features

# 原文中rte识别率只是刚刚超过了58%，所以不多做介绍
nltk.classify.rte_classify 
```

### 涉及到的定义

文中介绍一些关于评估模型、决策树、朴素贝叶斯分类器的知识。由于笔者已掌握这些基础模型知识所以在这里略去，以下介绍一些基本的定义。

测试集：选择适当的比例，防止过拟合

准确度：测量测试集上分类器正确标注的输入的比例。

精确度（Precision），表示我们发现的项目中有多少是相关的，TP/(TP+ FP)。

召回率（Recall），表示相关的项目中我们发现了多少，TP/(TP+ FN)。

F- 度量值（F-Measure）（或 F-得分，F-Score），组合精确度和召回率为一个单独的得分，被定义为精确度和召回率的调和平均数(2 × Precision × Recall)/(Precision+Recall)。

混淆矩阵：其中每个 cells[i,j]表示正确的标签 i 被预测为标签 j 的次数 。因此，对角线项目（即 cells[i,i]）表示正确预测的标签，非对角线项目表示错误。nltk中可以使用` nltk.ConfusionMatrix`函数

交叉验证：在不同的测试集上执行多个评估，然后组合这些评估的得分

信息增益：用给定的特征分割输入值，衡量数据变得更有序的程度

朴素贝叶斯假设（独立性假设）：每个输入值是通过首先为那个输入值选择一个类标签，然后产生每个特征的方式产生的 ，每个特征与其他特征完全独立。

零计数：如果训练集中有特征从来没有和给定标签一起出现，导致给定标签的标签可能性为 0。

平滑：给定一个初值，解决零计数问题。`nltk.probability`提供了多种平滑技术

Heldout估计使用一个heldout 语料库计算特征频率与特征概率之间的关系。

非二元特征：如果标签是1,2,4,5这种多元标签，可以转换0<x<3, 3<x<6这种二元标签

双重计数：在训练过程中特征的贡献被分开计算，但当使用分类器为新输入选择标签时，这些特征的贡献被组合。解决方案就是为每一个特征贡献设定一个权重。

最大熵分类器：与朴素贝叶斯类似，使用搜索技术找出一组将最大限度地提高分类器性能的参数。避免使用广义
迭代缩放（Generalized Iterative Scaling，GIS）或改进的迭代缩放（Improved Iterative Scaling，IIS），这两者都比共轭梯度（Conjugate Gradient，CG ）和 BFGS 优化方法慢很多

联合特征：每个接收它自己的参数的标签和特征的组合。联合特征是有标签的的值的属性，而（简单）特征是未加标签的值的属性。

一般情况下， 最大熵原理是说在与我们所知道的一致的的分布中，我们会选择熵最高（分布最均匀）的。

对于每个联合特征，最大熵模型计算该特征的“经验频率”——即它出现在训练集中的频率。然后，它搜索熵最大的分布，同时也预测每个联合特征正确的频率。

一般情况下，生成式模型确实比条件式模型强大。但是前者所需的参数会大于后者。



