# 深度前馈网络

## 前馈神经网络

![](https://gitee.com/liuyh9909/note-imgs/raw/master/img/20220308184625.png)

![](https://gitee.com/liuyh9909/note-imgs/raw/master/img/20220308184709.png)

![](https://gitee.com/liuyh9909/note-imgs/raw/master/img/20220308185217.png)

### 设计决策

* 需要选择优化器、损失函数和输出形式
* 选择激活函数
* 结构设计（网络层数等）

### 总结

* 设计和训练一个神经网络与训练任何其他具有梯度下降的机器学习模型没有太大区别
* 最显著的差异：许多损失函数变成非凸函数
* 与凸优化不同，收敛性并不能够保证
* 应用梯度下降：需要指定损失函数，和输出表达

## 损失函数

度量两个分布之间的相似度：KL散度

![](https://gitee.com/liuyh9909/note-imgs/raw/master/img/20220308194055.png)

## 基本结构单元

### 输出单元

#### 线性单元

![](https://gitee.com/liuyh9909/note-imgs/raw/master/img/20220308195811.png)

#### Sigmoid

![](https://gitee.com/liuyh9909/note-imgs/raw/master/img/20220308195838.png)

#### Softmax

![](https://gitee.com/liuyh9909/note-imgs/raw/master/img/20220308195917.png)

优点：

![](https://gitee.com/liuyh9909/note-imgs/raw/master/img/20220308195953.png)

### 隐藏单元

#### Rectified Linear Units(ReLU)

优点：

![](https://gitee.com/liuyh9909/note-imgs/raw/master/img/20220308200114.png)

* 提供数值较大且恒定的梯度（不会饱和）
* 迅速收敛，收敛速度比Sigmoid或Tanh快得多

缺点：

![](https://gitee.com/liuyh9909/note-imgs/raw/master/img/20220308200230.png)

* 非零中心输出
* Units “die”：即不活跃的单元永远不会被更新

#### Generalized ReLU

![](https://gitee.com/liuyh9909/note-imgs/raw/master/img/20220308200540.png)

![](https://gitee.com/liuyh9909/note-imgs/raw/master/img/20220308200614.png)

#### Exponential Linear Units (ELUs)

![](https://gitee.com/liuyh9909/note-imgs/raw/master/img/20220308200711.png)

#### Sigmoid

![](https://gitee.com/liuyh9909/note-imgs/raw/master/img/20220308200804.png)

#### Tanh

![](https://gitee.com/liuyh9909/note-imgs/raw/master/img/20220308200842.png)

### 总结

* 前馈网络中尽量不要使用Sigmoid
* 必须使用Sigmoid时，用Tanh代替
* 默认采用ReLU，但要谨慎选择学习率
* 尝试其他的ReLU方式等，也许会获得性能提升

## 网络结构设计

### MLP构成通用分类器

* MLPs可以拟合任何分类边界
* 单层MLP可以对任何分类边界进行建模
* MLP是通用分类器

### MLP用来回归

* 单层MLP可以对单个输入的任意函数建模

### MLP构造连续值函数

* 可以组成任意维数的任意函数
  * 只需要一层，由缩放和平移的圆柱体的和构成
  * 通过使得圆柱体变“瘦”，实现任意精度
* 利用MLP簇构成高维空间的任意函数
* MLP是一个通用逼近器

### 深度的优势

![](https://gitee.com/liuyh9909/note-imgs/raw/master/img/20220308203714.png)

![](https://gitee.com/liuyh9909/note-imgs/raw/master/img/20220308203741.png)

> 对照实验表明：卷积神经网络比全连接层的参数量小很多

## 反向传播算法

### 示例

![](https://gitee.com/liuyh9909/note-imgs/raw/master/img/20220308204412.png)

![](https://gitee.com/liuyh9909/note-imgs/raw/master/img/20220308204447.png)

### 多维输出

![](https://gitee.com/liuyh9909/note-imgs/raw/master/img/20220308204602.png)

### 随机反向传播

![](https://gitee.com/liuyh9909/note-imgs/raw/master/img/20220308204744.png)

![](https://gitee.com/liuyh9909/note-imgs/raw/master/img/20220308204813.png)

### Mini- batch随机梯度下降

![](https://gitee.com/liuyh9909/note-imgs/raw/master/img/20220308204902.png)

### Batch反向传播

![](https://gitee.com/liuyh9909/note-imgs/raw/master/img/20220308205015.png)

### 总结

* 随机学习
  * 梯度的估计是有噪声的，在每次迭代中权值可 能不会沿着梯度精确地向下移动
  * 比批量学习更快，特别是当训练数据有冗余的 时候
  * 噪声通常会产生更好的结果
  * 权值是波动的，它可能不会最终收敛到局部极 小值
* 批学习
  * 收敛条件很好理解
  * 一些加速技术只适用于批量学习
  * 权值变化规律和收敛速度的理论分析相对简单

\
