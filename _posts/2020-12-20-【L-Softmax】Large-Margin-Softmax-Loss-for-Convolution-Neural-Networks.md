<head>
    <script src="https://cdn.mathjax.org/mathjax/latest/MathJax.js?config=TeX-AMS-MML_HTMLorMML" type="text/javascript"></script>
    <script type="text/x-mathjax-config">
        MathJax.Hub.Config({
            tex2jax: {
            skipTags: ['script', 'noscript', 'style', 'textarea', 'pre'],
            inlineMath: [['$','$']]
            }
        });
    </script>
</head>

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
L=\frac{1}{N}\Sigma_{i}{L_{i}}=\frac{1}{N}\Sigma_{i}{-\log{(\frac{e^{f_{y_{i}}}}{\Sigma_{j}{e^{f_{j}}}})}}
$$




