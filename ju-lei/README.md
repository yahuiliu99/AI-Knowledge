# 聚类

## 1. 聚类基本概念

* 聚类是针对给定的样本，依据它们<mark style="color:blue;">特征的相似度或距离</mark>，将其归并到若干个“类”或“簇”的数据分析问题。<mark style="color:purple;">聚类属于无监督学习</mark>。
* 聚类的结果取决于对<mark style="color:blue;">度量标准</mark>的选择 [\[\[距离度量\]\]](ju-li-du-liang.md)
*   如果一个聚类方法假定一个样本只能属于一个类（类的交集为空），称为硬聚类（hard clustering）

    否则，如果一个样本可以属于多个类（类的交集不为空），称为软聚类（soft clustering）
* 类的特征
  *   类的均值

      $$
      \overline{x}_G=\frac{1}{n_G}\sum\limits_{i=1}^{n_G}x_i
      $$
  *   类的直径

      $$
      D_G=\max\limits_{x_i,x_j\in G}d_{ij}
      $$
  *   类的样本散布矩阵$$A_G$$和样本协方差矩阵$$S_G$$

      $$
      A_G=\sum\limits_{i=1}^{n_G}(x_i-\overline{x}_G)(x_i-\overline{x}_G)^T\\ S_G=\frac{1}{n_G-1}\sum\limits_{i=1}^{n_G}(x_i-\overline{x}_G)(x_i-\overline{x}_G)^T
      $$
* 类与类之间的距离
  *   最短距离：定义两个类中最近的两个样本的距离为类间距离

      $$
      d_{\min }\left(D_{i}, D_{j}\right)=\min _{\mathbf{x} \in D_{i} \atop \mathbf{x}^{\prime} \in D_{j}}\left\|\mathbf{x}-\mathbf{x}^{\prime}\right\|
      $$
  *   最长距离：定义两个类中最远的两个样本的距离为类间距离

      $$
      d_{\max }\left(D_{i}, D_{j}\right)=\max _{\mathbf{x} \in D_{i} \atop \mathbf{x}^{\prime} \in D_{j}}\left\|\mathbf{x}-\mathbf{x}^{\prime}\right\|
      $$
  *   中心距离：定义两个类的中心之间的距离为类间距离

      $$
      d_{\text {mean }}\left(D_{i}, D_{j}\right)=\left\|\mathbf{m}_{i}-\mathbf{m}_{j}\right\|
      $$
  *   平均距离：定义两个类中任意两个样本的距离的平均值为类间距离

      $$
      d_{\text {avg }}\left(D_{i}, D_{j}\right)=\frac{1}{n_{i} n_{j}} \sum_{\mathbf{x} \in D_{i}} \sum_{\mathbf{x}^{\prime} \in D_{j}}\left\|\mathbf{x}-\mathbf{x}^{\prime}\right\|
      $$

## 2. K-Means Clustering

> 每个样本属于一个类，所以k均值聚类是硬聚类

*   #### 算法

    首先选择k个类的中心，将样本逐个指派到与其最近的中心的类中，得到一个聚类结果；然后更新每个类的样本的均值，作为类的新的中心；重复以上步骤，直到收敛为止。

    *   目标函数：

        $$
        L_{k-\text { means }}=\sum_{j \in[k]} \sum_{i \in S_{j}}\left\|x_{i}-\mu_{j}\right\|^{2}
        $$
    * 算法复杂度：$$O(knm)$$，其中k是类别数，n是样本数，m是样本维数

    ![image-20211224211148468](https://gitee.com/liuyh9909/note-imgs/raw/master/img/20211224211148.png)



    *   **E-Step:**

        对于每个样本$$x_j$$，寻找距离样本最近的聚类中心$$k^*$$

        $$
        k^{*}=\underset{k}{\arg \min }\left\|x_{j}-u_{k}\right\|^{2}, \quad \text { 设置 } r_{j, k}= \begin{cases}1, & k=k^{*} \\ 0, & k \neq k^{*}\end{cases}
        $$
    *   **M-Step:**

        使用每个类的样本的均值重置其聚类中心的位置

        $$
        \mu_{k}^{\text {new }}=\frac{\sum_{j=1}^{N} r_{j, k} x_{j}}{\sum_{j=1}^{N} r_{j, k}}
        $$
* #### 特点
  * 收敛性：k均值聚类属于启发式方法，不能保证收敛到全局最优
  * 初始中心选择
  * 类别数k的选择
    * Use trick of cross validation to select k
    * Let domain expert look at the clustering and decide if they like it
    *   The "knee" solution

        ![image-20211224213231610](https://gitee.com/liuyh9909/note-imgs/raw/master/img/20211224213231.png)
  * 缺点：
    * 一个样本只能属于一个类（改良：软聚类probability）
    * 每个类的样本数目要接近（改良：高斯混合模型GMM）
    * K-Means以欧氏距离平方表示样本之间距离，所以趋向于圆形（改良：流形学习，核方法，谱聚类）
* #### 改进
  *   K-Medians：把中心换为中位数

      ![image-20211224213523695](https://gitee.com/liuyh9909/note-imgs/raw/master/img/20211224213523.png)
  *   Frequency Sensitive Competitive Learning：频率敏感竞争学习

      ![image-20211224213628223](https://gitee.com/liuyh9909/note-imgs/raw/master/img/20211224213628.png)
  *   Rival Penalized Competitive Learning：对手惩罚竞争学习

      ![image-20211224213811577](https://gitee.com/liuyh9909/note-imgs/raw/master/img/20211224213811.png)
  *   Mean-Shift Clustering

      ![image-20211224214145752](https://gitee.com/liuyh9909/note-imgs/raw/master/img/20211224214145.png)
  *   DBSCAN

      Density-Based Spatial Clustering of Applications with Noise (DBSCAN) is a density-based clustered algorithm similar to mean-shift, but with a couple of notable advantages.

      ![image-20211224214411199](https://gitee.com/liuyh9909/note-imgs/raw/master/img/20211224214411.png)

## 3. Gaussian Mixture Models

> K-Means是高斯混合模型的特例
>
> The primary difference is that in K-means, the $$r_{j,⋅}$$ is a probability distribution that gives zero probability to all but one cluster, while EM for GMMs gives non-zero probability to every cluster.

高斯混合模型是一种期望最大化(EM)算法，其数据点假设为高斯(正态)分布。它需要均值$$\mu$$和协方差$$\Sigma$$两个参数来描述每个聚类的位置和形状。

$$
N(x| \mu, \Sigma)=\frac{1}{(2 \pi \Sigma)^{1 / 2}} \exp \left(-\frac{1}{2}(x-\mu)^{T} \Sigma^{-1}(x-\mu)\right)
$$

*   #### E-Step：

    对于每个样本$$j$$，评估其对于每个类$$k$$的隶属度

    $$
    r_{j, k}=\frac{\pi_{k} \mathcal{N}\left(x_{j} \mid \mu_{k}, \sigma_{k}\right)}{\sum\limits_{i=1}^{K} \pi_{i} \mathcal{N}\left(x_{j} \mid \mu_{i}, \sigma_{i}\right)}\\ \begin{aligned}\text{where}\quad \pi_k & =\text{weight of each cluster (mixing coefficients)}\\ K & =\text{number of clusters}\end{aligned}
    $$
*   #### M-Step:

    对于每个类$$k$$，重新计算参数$$\mu_k,\Sigma_k,\pi_k$$

    $$
    \begin{aligned} \mu_{k}^{\text {new }} &=\frac{1}{\sum_{j=1}^{N} r_{j, k}} \sum_{j=1}^{N} r_{j, k} x_{j} \\ \Sigma_{k}^{\text {new }} &=\frac{1}{\sum_{j=1}^{N} r_{j, k}} \sum_{j=1}^{N} r_{j, k}\left(x_{j}-\mu_{k}^{\text {new }}\right)\left(x_{j}-\mu_{k}^{\text {new }}\right)^{T} \\ \pi_{k}^{\text {new }} &=\frac{1}{N}\sum_{j=1}^{N} r_{j, k} \end{aligned}
    $$

## 4. Hierarchical Clustering

> 每个样本属于一个类，所以层次聚类是硬聚类

*   #### 自底向上（聚合）的分级聚类

    ![image-20211225121053555](https://gitee.com/liuyh9909/note-imgs/raw/master/img/20211225122659.png)
*   #### 系统树

    ![image-20211225121453963](https://gitee.com/liuyh9909/note-imgs/raw/master/img/20211225121659.png)
*   #### 例子

    ![image-20211225122533699](https://gitee.com/liuyh9909/note-imgs/raw/master/img/20211225122633.png) ![image-20211225122445783](https://gitee.com/liuyh9909/note-imgs/raw/master/img/20211225122445.png)

## 5. Spectral Clustering

> K-Means in spectrum space

广义上讲，任何在学习过程中<mark style="color:blue;">应用到矩阵特征值分解的方法均叫谱学习方法</mark>，比如主成分分析（PCA）、线性判别成分分析（LDA）、流形学习中的谱嵌入方法、谱聚类、等等。

* #### 谱聚类
  * 谱聚类算法建立在<mark style="color:purple;">图论的谱图理论</mark>基础之上，其本质是将聚类问题转化为一个<mark style="color:purple;">图上的关于顶点划分的最优问题</mark>。
  * 谱聚类算法建立在<mark style="color:blue;">点对亲和性</mark>基础之上，理论上能对任意分布形状的样本空间进行聚类。
* #### 图论基本概念
  * 图$$G(V,E)$$
    *   邻接矩阵$$A$$ 或者相似度矩阵$$W$$（也称亲和度矩阵，其元素$$w_{ij}$$表示顶点$$x_i$$与$$x_j$$之间的亲和程度）

        > 邻接矩阵元素为0或1，1表示两个顶点相连，0表示不相连；
        >
        > 相似度矩阵元素为$$[0,1]$$，反映点对亲和性，也即边的权重，如果两个顶点不相连，则权重为零。
    * 度矩阵$$D$$：对角矩阵，主对角元素为邻接矩阵$$A$$或相似度矩阵$$W$$每一行元素之和
    * 拉普拉斯矩阵$$L=D-A$$或者$$L=D-W$$
  * 子图
    * 子图$$A \subset V$$的势$$|A|$$：等于其所包含的顶点个数
    * 子图$$A \subset V$$的体积$$vol(A)$$：等于其中所有顶点的度之和
    * 子图相似度：连接两个子图所有边的权重之和$$W(A,B)=\sum\limits_{i\in A,j\in B}w_{ij}$$
    * 子图之间的切割：$$cut(A,B)=W(A,B)=\sum\limits_{i\in A,j\in B}w_{ij}$$
  *   图构造

      ![image-20211225170845800](https://gitee.com/liuyh9909/note-imgs/raw/master/img/20211225170845.png)
*   #### 算法

    ![image-20211225171910107](https://gitee.com/liuyh9909/note-imgs/raw/master/img/20211225171910.png)
*   #### 与K-Means的区别

    * **从优化目标的角度来看**
      * K-means致力于==最小化类内的平均距离== （采用欧氏距离度量）
      * 谱聚类致力于==最大化类内的平均相似度== （采用高斯函数计算相似度）
    * **从图的观点来看**
      * K-means采用的是==全图==，即任意两个样本之间的距离都没有被忽略（global）
      * 谱聚类采用的是==近邻图==，一般只计算每个样本与其近邻之间的相似度（local）

    ![image-20211225172236700](https://gitee.com/liuyh9909/note-imgs/raw/master/img/20211225172236.png)

## 6. Kernel Clustering

> K-Means in kernel space

*   #### Motivation

    <mark style="color:blue;">在低维空间线性不可分的数据在高维空间线性可分</mark>

    所谓Kernel就是把数据从低维变换到高维
*   #### Kernel

    ![image-20211225173216932](https://gitee.com/liuyh9909/note-imgs/raw/master/img/20211225173217.png)

    * **Polynomial：$\mathcal{K}(x,y)=(x^Ty)^p$**
    * **Gaussian：$\mathcal{K}(x,y)=\exp(-\lambda|x-y|\_2^2)$**
*   #### Kernel K-Means

    Replace the Euclidean distance / similarity computations in K-Means by the kernelized versions.
*   #### Kernel PCA + K-Means

    ![image-20211225173826040](https://gitee.com/liuyh9909/note-imgs/raw/master/img/20211225173826.png)

## 7. Deep Clustering

> K-Means in deep space

*   #### Motivations

    ![image-20211225174031630](https://gitee.com/liuyh9909/note-imgs/raw/master/img/20211225174031.png) ![image-20211225174013159](https://gitee.com/liuyh9909/note-imgs/raw/master/img/20211225174013.png) ![image-20211225174045034](https://gitee.com/liuyh9909/note-imgs/raw/master/img/20211225174045.png)
* #### 一些前沿工作
  *   **DeepCluster**

      Deep Clustering for Unsupervised Learning of Visual Features (ECCV 2018)
  *   **SwAV**

      Unsupervised Learning of Visual Features by Contrasting Cluster Assignments (NeurIPS 2020)

## Reference

* \[1] UCAS-AI模式识别2021-13-聚类01.pdf
* \[2] UCAS-AI模式识别2021-14-聚类02.pdf
* \[3] [K-Means and GMM](https://towardsdatascience.com/clustering-out-of-the-black-box-5e8285220717)
* \[4] [Difference between K-Means and GMM](https://stats.stackexchange.com/questions/489459/in-cluster-analysis-how-does-gaussian-mixture-model-differ-from-k-means-when-we/489537#489537)
* \[5] [Spectral Clustering](https://towardsdatascience.com/spectral-clustering-aba2640c0d5b)
* \[6] [谱聚类](https://www.cnblogs.com/pinard/p/6221564.html)
* \[6] [聚类准确度评估 Clustering Evaluation](https://smorbieu.gitlab.io/accuracy-from-classification-to-clustering-evaluation/)
