# 特征检测与匹配

## 图像底层特征提取

> 特征提取是计算机视觉的基本步骤

### 提取边缘

#### 灰度图象边缘提取思路：

* 抑制噪声（低通滤波、平滑、去噪、模糊）
* 边缘特征增强（高通滤波、锐化）
* 边缘定位

#### 使用微分滤波器提取边缘

**一阶微分滤波器：梯度算子**

* Sobel
* Prewitt
* Roberts

**二阶微分滤波器：LoG (Laplacian of Gaussian)**

* 首先用Gauss函数对图像进行平滑，抑制噪声
* 然后对经过平滑的图像使用Laplacian算子
* 利用卷积的性质LoG算子等效于：Gaussian平滑 + Laplacian 二阶微分

![](https://gitee.com/liuyh9909/note-imgs/raw/master/img/20220228152629.png)

#### Canny算子

![](https://gitee.com/liuyh9909/note-imgs/raw/master/img/20220228152715.png)

NMS：

![](https://gitee.com/liuyh9909/note-imgs/raw/master/img/20220307134448.png)

双阈值：

![](https://gitee.com/liuyh9909/note-imgs/raw/master/img/20220228153048.png)

### 图像特征点提取

#### Harris角点检测

![](https://gitee.com/liuyh9909/note-imgs/raw/master/img/20220314203459.png)

**Harris角点检测基本思想**

* 从图像局部的小窗口观察图像特征
* 角点定义 ：窗口向任意方向的移动都导致图像灰度的明显变化

**Harris角点检测**

![image-20220228154848456](https://gitee.com/liuyh9909/note-imgs/raw/master/img/20220228154848.png)

**Harris角点的性质**

* 旋转不变性
* 对于图像灰度的仿射变化具有部分的不变性
* 对于图像几何尺度变化不具有不变性
* 随尺度变化，Harris角点检测的性能下降

#### FAST

> features from accelerated segment test

**假设**

若该点的灰度值比其周围领域内足够多的像素点的灰度值大或者小，则该点可能为角点

**算法步骤**

![](https://gitee.com/liuyh9909/note-imgs/raw/master/img/20220228155828.png)

![](https://gitee.com/liuyh9909/note-imgs/raw/master/img/20220307135255.png)

![](https://gitee.com/liuyh9909/note-imgs/raw/master/img/20220307135341.png)

![](https://gitee.com/liuyh9909/note-imgs/raw/master/img/20220228160258.png)

#### SIFT

> Scale Invariant Feature Transform

**性质**

![](https://gitee.com/liuyh9909/note-imgs/raw/master/img/20220307135612.png)

**算法流程**

![](https://gitee.com/liuyh9909/note-imgs/raw/master/img/20220307135743.png)

**高斯差分尺度空间(DoG)**

![](https://gitee.com/liuyh9909/note-imgs/raw/master/img/20220307140638.png)

相邻尺度空间做差值

**高斯金字塔**

![](https://gitee.com/liuyh9909/note-imgs/raw/master/img/20220307140541.png)

连续的尺度空间进行特征点提取：尺度不变

![](https://gitee.com/liuyh9909/note-imgs/raw/master/img/20220307141112.png)

**DoG尺度空间极值点检测**

![](https://gitee.com/liuyh9909/note-imgs/raw/master/img/20220307141453.png)

**梯度方向直方图**

![](https://gitee.com/liuyh9909/note-imgs/raw/master/img/20220307145134.png)

![](https://gitee.com/liuyh9909/note-imgs/raw/master/img/20220307145206.png)

**SIFT描述子构造**

![](https://gitee.com/liuyh9909/note-imgs/raw/master/img/20220307145423.png)

![](https://gitee.com/liuyh9909/note-imgs/raw/master/img/20220307145755.png)

SIFT描述子的不变性：

![](https://gitee.com/liuyh9909/note-imgs/raw/master/img/20220307145925.png)

**SIFT特征匹配**

![](https://gitee.com/liuyh9909/note-imgs/raw/master/img/20220307150052.png)

#### ORB描述子

特征：

![](https://gitee.com/liuyh9909/note-imgs/raw/master/img/20220307150749.png)

**oFAST：带方向的FAST**

> 比计算梯度方向直方图快得多

![](https://gitee.com/liuyh9909/note-imgs/raw/master/img/20220307152445.png)

**BRIEF**

![](https://gitee.com/liuyh9909/note-imgs/raw/master/img/20220307151742.png)

**rBRIEF**

![](https://gitee.com/liuyh9909/note-imgs/raw/master/img/20220307152100.png)

**Time and Speed:**

![](https://gitee.com/liuyh9909/note-imgs/raw/master/img/20220307152227.png)

![](https://gitee.com/liuyh9909/note-imgs/raw/master/img/20220307152315.png)

## 图像特征匹配

> 匹配：确定不同图像中对应空间同一物体的投影的过程，匹配是基于多幅图像视觉问题的的基本步骤

#### 点匹配

> 利用图像点周围的信息来描述点，如灰度信息，颜色信息，梯度信息等，然后进行相似性度量

**Cross-correlation**

利用相关函数，评价两幅图像特征点邻域的灰度相似性以确定对应点

**特性**

* 基于图像灰度
* 如何确定窗口大小和形状是最大的问题
* 没有旋转不变性
* 对光照变化敏感
* 计算代价大

**Iterative Closest Point (ICP)**

> 两组点集之间的匹配

**研究背景**

* 对齐两个相互交叠的曲面
* 对齐两个相互交叠的曲线
* 三维数据点对应

**ICP算法**

* 点集选取（selecting）
  * 选择所有点
  * 均匀采样（Uniform Sampling）
  * 随机采样（Random Sampling）
  * 法方向空间均匀采样（Normal-space Uniform Sampling）
* 点集匹配（matching）
  * 最近邻点（Closest Point）
  * 法方向最近邻点（Normal shooting）
  * 投影法（Projection）
*   点集对应权重（weighting）

    ![image-20220314211650568](https://gitee.com/liuyh9909/note-imgs/raw/master/img/20220314211650.png)

**ICP算法流程**

![](https://gitee.com/liuyh9909/note-imgs/raw/master/img/20220314135253.png)

#### 直线匹配

#### 曲线匹配

#### 区域匹配

## 鲁棒估计

### RANSAC

> RANdom SAmple Consensus 随机一致性采样

![](https://gitee.com/liuyh9909/note-imgs/raw/master/img/20220314135358.png)

## References

* [Introduction to Harris Corner Detector](https://medium.com/@deepanshut041/introduction-to-harris-corner-detector-32a88850b3f6)
* [Introduction to FAST (Features from Accelerated Segment Test)](https://medium.com/@deepanshut041/introduction-to-fast-features-from-accelerated-segment-test-4ed33dde6d65)
* [Introduction to SIFT (Scale-Invariant Feature Transform)](https://medium.com/@deepanshut041/introduction-to-sift-scale-invariant-feature-transform-65d7f3a72d40)
* [Introduction to ORB (Oriented FAST and Rotated BRIEF)](https://medium.com/@deepanshut041/introduction-to-orb-oriented-fast-and-rotated-brief-4220e8ec40cf)
* [计算机视觉基本原理--RANSAC](https://zhuanlan.zhihu.com/p/45532306)
