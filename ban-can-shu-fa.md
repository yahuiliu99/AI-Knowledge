# 半参数法

## 1. 期望最大法（EM算法）

根据已有的数据递归估计似然函数

![](https://gitee.com/liuyh9909/note-imgs/raw/master/img/20220104143024.png)

### 高斯混合密度的最大似然估计(EM算法)

![](https://gitee.com/liuyh9909/note-imgs/raw/master/img/20220104143249.png)

![](https://gitee.com/liuyh9909/note-imgs/raw/master/img/20220104143410.png)

## 2. 隐马尔科夫模型

![](https://gitee.com/liuyh9909/note-imgs/raw/master/img/20220104143712.png)

其中，

$$A={a_{ij}}$$表示状态转移矩阵，元素$$a_{ij}$$表示隐状态之间的转移概率；

$$B={b_{j}(k)}$$表示观测矩阵，元素$$b_{j}(k)$$表示发出可见状态的概率；

$$\pi$$表示初始状态。

![](https://gitee.com/liuyh9909/note-imgs/raw/master/img/20220104144204.png)
