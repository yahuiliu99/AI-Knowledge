# 最大似然和贝叶斯估计

## 0. 引言

由贝叶斯决策理论，已经知道如何根据先验概率$$p(w_i)$$和类条件概率$$p(x|w_i)$$来设计最优分类器。然而在实际应用中，通常得不到有关问题的概率结构的全部知识。

解决办法：利用训练样本估计先验概率和条件概率密度函数，再把估计的结果当成实际的先验概率和条件概率密度函数。

关于条件密度函数，可以参数化。例如，可以假设$$p(x|w_i)$$是一个多元正态分布，均值$$\mu$$和协方差矩阵$$\Sigma_i$$为参数，只需估计这两个值。

![](https://gitee.com/liuyh9909/note-imgs/raw/master/img/20220104135847.png)

* #### 最大似然和贝叶斯参数估计的区别
  * <mark style="color:blue;">最大似然估计把待估计的参数看作是确定性的量</mark>，取似然度最大
  * <mark style="color:blue;">贝叶斯估计把待估计的参数看成是符合某种先验概率分布的随机变量</mark>

## 1. 最大似然估计

&#x20;

![](https://gitee.com/liuyh9909/note-imgs/raw/master/img/20220104140245.png)

![](https://gitee.com/liuyh9909/note-imgs/raw/master/img/20220104140406.png)

### 高斯密度函数的最大似然估计

![](https://gitee.com/liuyh9909/note-imgs/raw/master/img/20220104140550.png)

## 2. 贝叶斯参数估计

![](https://gitee.com/liuyh9909/note-imgs/raw/master/img/20220104141919.png)
