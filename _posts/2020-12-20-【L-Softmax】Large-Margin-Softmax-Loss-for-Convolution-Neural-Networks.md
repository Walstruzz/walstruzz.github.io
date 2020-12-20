---
title: 【L-Softmax】Large-Margin Softmax Loss for Convolution Neural Networks
date: 2020-12-20 14：53：00
categories:

- 人脸识别
---

链接：[Paper](https://arxiv.org/pdf/1612.02295.pdf), [Github](https://github.com/wy1iu/LargeMargin_Softmax_Loss)

这篇文章主要提出了Large-Margin Softmax Loss，是后续NormFace、SphereFace、CosFace、ArcFace等一系列文章的基础。

# 动机

作者发现当前的CNN学习目标倾向于最大化类内紧凑(intra-class compactness)和类间稀疏性(inter-class separability)，比如Triplet Loss，而Triplet Loss计算复杂度较高，不利于大数据训练，而如此简单好用又如此合适的Softmax Loss却少有这方面的研究。Softmax Loss的格式为


$$
\begin{aligned}
L &= \frac{1}{N}\sum_{i}{L_{i}}\\
  &= \frac{1}{N}\sum_{i}{-\log{(\frac{e^{f_{y_{i}}}}{\sum_{j}{e^{f_{j}}}})}} 
  &= \parallel W \parallel
\end{aligned}
\tag{1}
$$


一般Softmax Loss前一层为权重为$$W$$的全连接层，忽略偏置，则上述公式(1)可以写作


$$
\begin{aligned}
L &= \frac{1}{N}\sum_{i}{L_{i}}\\
  &= \frac{1}{N}\sum_{i}{-\log{(\frac{e^{f_{y_{i}}}}{\sum_{j}{e^{f_{j}}}})}}\\
  &= \frac{1}{N}\sum_{i}{-\log{(\frac{e^{W_{y_{i}}x_{i}}}{\sum_{j}{e^{W_{j}x{i}}}})}}\\
  &= \frac{1}{N}\sum_{i}{-\log{(\frac{e^{\parallel W_{y_{i}} \parallel \parallel x_{i} \parallel \cos(\theta_{y{i}})}}{\sum_{j}{e^{\parallel W_{j} \parallel \parallel x_{i} \parallel \cos(\theta_{j})}}})}}
\end{aligned}
\tag{2}
$$


我们知道$$f_{y_{i}}$$即logit，我们最终希望对于样本对$$\{x_{i}, y_{i}\}$$，模型输出的logit中$y_{i}$对应的项大于其它项，即


$$
\begin{aligned}
f_{y_{i}} &\gt f_{y_{j}} \\
\parallel W_{y_{i}} \parallel \parallel x_{i} \parallel \cos(\theta_{y{i}}) &\gt  \parallel W_{y_{j}} \parallel \parallel x_{i} \parallel \cos(\theta_{y{j}}) \\
\end{aligned}
$$


讨论区间$$[0, \pi]$$，由于$$\cos$$函数在这个区间内单调递减，所以