# 特征提取

## 1. 引言

### 特征表示的重要性

![](https://gitee.com/liuyh9909/note-imgs/raw/master/img/20211225192505.png)

### 特征提取的目的

* 减少噪声影响
* 提高稳定性
* 提取观测数据的内在特性

### 特征变换的目的

* 降低特征空间的维度，便于分析和减少后续步骤的计算量
* 减少特征之间可能存在的相关性
* 有利于分类

### 根据特征变换关系不同

*   **线性特征变换**

    采用线性映射将原特征变换至一个新的空间（通常维度更低）：

    * PCA、LDA、ICA
*   **非线性特征变换**

    采用非线性映射将原特征变换至一个新的空间（通常性能更好）：

    * KPCA、KLDA、Isomap、LLE、 HLLE、 LSTA、…

## 2. 特征提取

> 特征提取的最终形式都是使用向量来表示数据样本，便于分析

* #### 语音特征提取
* #### 文本特征提取
* #### 视觉特征提取
  * 局部二值模式（LBP）
  * Gabor特征提取
  * 尺度不变特征变换（SIFT）
  * 视觉词袋（bag of visual words）
  * 哈尔特征（Haar）
  * 梯度方向直方图（HoG）

## 3. 特征变换

* #### 维数灾难
  * 随着维数的增加，计算量呈指数倍增长
  * 随着维数的增加，具有相同距离的两个样本其相似程度可以相差很远
  * 当维度增加时，空间的体积增加得很快，可用数据变得稀疏
*   #### 维数缩减

    缓解维数灾难的一个重要途径是降维(dimensionality reduction)，即通过某种数学变换将原始高维特征空间 变换至某个低维“子空间”。在该子空间中，样本密度大幅度提高，距离计算也变得更为容易。

    *   **为什么能降维？**

        在很多时候，人们观测或收集到的数据虽然是高维的，但<mark style="color:blue;">与学习任务密切相关的特征通常位于某个低维分布上，即高维空间中的一个低维“嵌入”(embedding)</mark>。

## 4. 主成分分析（PCA）

![](https://gitee.com/liuyh9909/note-imgs/raw/master/img/20211225195948.png)

### 原理

PCA的目标是寻找主元，即最大程度地去除冗余和噪音的干扰。主元分析的原则是 1)<mark style="color:blue;">最小化变量冗余，对应于协方差矩阵的非对角元素尽量小</mark>；2)<mark style="color:blue;">最大化信号，对应于协方差矩阵对角线上的元素尽可能大</mark>。假设原数据集和转换后的新数据集分别用$$X$$和$$Y$$表示，$$P$$为线性变换矩阵，有$$PX=Y$$。所以优化的目标矩阵$$C_Y$$应该是一个对角阵。

### 模型

PCA问题可以被描述为寻找一组正交基组成的$$P$$，有$$PX=Y$$，使得$$C_Y$$是对角阵，则$$P$$的行向量就是数据$$X$$的主元向量。

$$
\begin{aligned} C_Y & =\frac{1}{n-1}YY^T\\ & =\frac{1}{n-1}(PX)(PX)^T\\ & =\frac{1}{n-1}PXX^TP^T\\ & =P\frac{1}{n-1}XX^TP^T\\ & =PC_XP^T \end{aligned}
$$

其中$$C_X=\frac{1}{n-1}XX^T$$，$$C_X$$为对称阵，故对$$C_X$$进行对角化，得到

$$
C_X=Q\Lambda Q^{-1}=Q\Lambda Q^T
$$

其中$$Q$$为正交矩阵，令$$P=Q^T$$

$$
\begin{aligned} C_Y & =PC_XP^T\\ & = PP^T\Lambda PP^T\\ & =\Lambda \end{aligned}
$$

此时$$P$$就是我们需要的变换基，至此已得到PCA的结果：

* X的主元即是$$C_X$$的特征向量，即矩阵Q的列向量，也就是变换矩阵P的行向量
* 矩阵$$C_Y$$对角线上第$$i$$个元素是数据$$X$$在方向$$P_i$$的方差

### 算法

* 求均值，$$\bar{x}=\dfrac{1}{n}\sum\limits_{i=1}^{n}x_i$$
* 计算协方差矩阵，$$C_X=\dfrac{1}{n}\sum\limits_{i=1}^{n}(x_i - \bar{x})(x_i - \bar{x})^T$$
* 计算协方差矩阵的特征值和特征向量
* 选取前$$m$$大的特征值及对应的特征向量组成投影矩阵$$W=[w_1,w_2,\cdots,w_m]\in \mathbb{R}^{d\times m}$$
* 利用投影矩阵对数据进行降维，$$y_i=W^Tx_i\in\mathbb{R}^m$$

### 进一步分析

![](https://gitee.com/liuyh9909/note-imgs/raw/master/img/20211225201407.png)

## 5. 线性判别分析（LDA）

![](https://gitee.com/liuyh9909/note-imgs/raw/master/img/20211225205138.png)

### 原理

LDA的目标是寻找一组投影方向，使得同类样本的投影点尽可能接近，不同类样本的投影点尽可能互相远离。即最小化类内方差，最大化类间均值。

### 模型

令$$x_{jk}$$表示第$$j$$类的第$$k$$个样本，$$n_{j}$$表示属于第$$j$$类的样本个数，第$$j$$类样本均值$$\mu_j=\dfrac{1}{n_j}\sum\limits_{x\in X_j}x$$，所有样本均值$$\mu=\dfrac{1}{n}\sum\limits_{i=1}^{n}x_i$$，则有：

*   类内散度矩阵

    $$
    S_w=\sum\limits_{j=1}^{c}\sum\limits_{k=1}^{n_j}(x_{jk}-\mu_j)(x_{jk}-\mu_j)^T
    $$
*   全局散度矩阵

    $$
    \begin{aligned} S_t & =\sum\limits_{i=1}^{n}(x_{i}-\mu)(x_{i}-\mu)^T\\ & =\sum\limits_{j=1}^{c}\sum\limits_{k=1}^{n_j}(x_{jk}-\mu)(x_{jk}-\mu)^T\\ & =\sum\limits_{j=1}^{c}\sum\limits_{k=1}^{n_j}(x_{jk}-\mu_j+\mu_j-\mu)(x_{jk}-\mu_j+\mu_j-\mu)^T\\ & =\sum\limits_{j=1}^{c}\sum\limits_{k=1}^{n_j}\left[(x_{jk}-\mu_j)(x_{jk}-\mu_j)^T +(x_{jk}-\mu_j)(\mu_j - \mu)^T\right.\\ & \qquad \left.+(\mu_j-\mu)(x_{jk}-\mu_j)^T+(\mu_j -\mu)(\mu_j-\mu)^T\right]\\ & =\sum\limits_{j=1}^{c}\sum\limits_{k=1}^{n_j}(x_{jk}-\mu_j)(x_{jk}-\mu_j)^T+ \sum\limits_{j=1}^{c}(\mu_j - \mu)^T\sum\limits_{k=1}^{n_j}(x_{jk}-\mu_j)\\ & \qquad + \sum\limits_{j=1}^{c}(\mu_j - \mu)\sum\limits_{k=1}^{n_j}(x_{jk}-\mu_j)^T+ \sum\limits_{j=1}^{c}\sum\limits_{k=1}^{n_j}(\mu_j -\mu)(\mu_j-\mu)^T\\ & =S_w + \sum\limits_{j=1}^{c}n_j(\mu_j -\mu)(\mu_j-\mu)^T \end{aligned}
    $$
*   类间散度矩阵

    $$
    S_b = S_t - S_w = \sum\limits_{j=1}^{c}n_j(\mu_j -\mu)(\mu_j-\mu)^T
    $$

则LDA可以描述为：

$$
\begin{aligned} & \max \frac{w^TS_bw}{w^TS_ww}, \quad s.t. \quad w^Tw=I\\ & \\ \Longrightarrow & \max w^TS_bw, \quad s.t. \quad w^TS_ww=I \end{aligned}
$$

根据拉格朗日乘子法，

$$
L(w,\lambda)=w^TS_bw+\lambda(I-w^TS_ww)
$$

$$
\begin{aligned} \frac{\partial L}{\partial w} & = S_bw+S_b^Tw - \lambda S_ww - \lambda S_w^Tw\\ & =2(S_bw-\lambda S_ww) \end{aligned}
$$

​ 原问题转化为

$$
S_bw=\lambda S_ww \quad \Longrightarrow \quad S_w^{-1}S_bw=\lambda w
$$

​ 所以$$w$$是$$S_w^{-1}S_b$$的特征向量。

### 算法

* 求类内散度矩阵$$S_w$$和类间散度矩阵$$S_b$$
* 计算$$S_w^{-1}S_b$$的特征值和特征向量
* 选取前$$m$$大的特征值及对应的特征向量组成投影矩阵$$W=[w_1,w_2,\cdots,w_m]\in \mathbb{R}^{d\times m}$$
* 利用投影矩阵对数据进行降维，$$y_i=W^Tx_i\in\mathbb{R}^m$$

## 6. 独立成分分析（ICA）

## 7. 典型关联分析（CCA）

## 8. 多维缩放（MDS）

![](https://gitee.com/liuyh9909/note-imgs/raw/master/img/20211225212037.png)

![](https://gitee.com/liuyh9909/note-imgs/raw/master/img/20211225212208.png)

### 算法步骤

* 给定数据的距离矩阵$$D\in \mathbb{R}^{n\times n}$$
* 构造矩阵$$D_2$$（元素为距离平方）
* 构造矩阵$$B=-\dfrac{1}{2}H^TD_2H$$（降维后样本的内积矩阵）
* 对矩阵$$B$$进行特征值分解：$$B=U\Lambda U^T$$
*   $$Z=\Lambda_m^{1/2}U_m^T\in \mathbb{R}^{m\times n}$$

    其中$$\Lambda_m^{1/2}$$表示由矩阵$$B$$的前m个最大的特征值开根号后对应的对角矩阵；$$U_m$$由前m个最大的特征值对应的特征向量组成。$$\Lambda_m^{1/2}=diag (\sqrt{\lambda_1},\sqrt{\lambda_2},\cdots,\sqrt{\lambda_m}) \in \mathbb{R}^{m\times m},\quad U_m=[u_1,u_2,\cdots,u_m]\in \mathbb{R}^{n\times m}$$

## 9. 流形学习（Manifold Learning）

* #### What are manifolds?
  *   在数学上，流形用于描述一个几何形体，它在==局部具有欧氏空间的性质==。即可以应用欧氏距离来描述局部区域，但在==全局欧氏距离不成立==。

      > 通过线性投影将高维数据降到低维将难以展开非线性结构！
* #### 非线性维数缩减
  * **基本思想**：<mark style="color:blue;">高维空间相似的数据点，映射到低维空间距离也是相似的</mark>

### 经典算法

#### **LLE（Locally linear embedding）**

#### **Isomap**

#### **LE**

## Reference
