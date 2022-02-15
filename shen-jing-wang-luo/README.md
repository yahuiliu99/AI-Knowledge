# 神经网络

[\[\[Neural Network\]\]](neural-network.md)

## 1. 径向基函数网络（RBF）

与多层神经网络相似，RBF可以对任意连续的非线性函数进行近似

### 解决的典型问题

![](https://gitee.com/liuyh9909/note-imgs/raw/master/img/20220102213358.png)

### 矩阵形式

![](https://gitee.com/liuyh9909/note-imgs/raw/master/img/20220102213429.png)

### 网络结构

![](https://gitee.com/liuyh9909/note-imgs/raw/master/img/20220102213531.png)

### 网络结构简化与普遍化

![](https://gitee.com/liuyh9909/note-imgs/raw/master/img/20220102213656.png)

## 2. 自组织映射（SOM）

### 结构

![](https://gitee.com/liuyh9909/note-imgs/raw/master/img/20220102201218.png)

### 邻近结点相互作用

![](https://gitee.com/liuyh9909/note-imgs/raw/master/img/20220102201522.png)

### 功能描述

![](https://gitee.com/liuyh9909/note-imgs/raw/master/img/20220102201619.png)

### 学习算法的原理

* 通过自动寻找样本中的内在规律和本质属性，自组织、自适应地改变网络参数与结构
* 其自组织功能是通过竞争学习来实现的
  * 竞争学习规则—Winner-Take-All（胜者为王） • 网络的输出神经元之间相互竞争并期望被激活 • 在每一时刻只有一个输出神经元被激活 • 被激活的神经元称为竞争获胜神经元，其它神经元的状态被抑制
*

![](https://gitee.com/liuyh9909/note-imgs/raw/master/img/20220102202037.png)

### 学习步骤

![](https://gitee.com/liuyh9909/note-imgs/raw/master/img/20220102204948.png)

![](https://gitee.com/liuyh9909/note-imgs/raw/master/img/20220102205114.png)

### 邻域函数

![](https://gitee.com/liuyh9909/note-imgs/raw/master/img/20220102205254.png)

### 算法步骤
