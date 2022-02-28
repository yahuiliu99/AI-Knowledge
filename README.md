# 深度学习初步知识与特征检测\_

## 深度学习与卷积神经网络

### Convolution Summary

![](https://gitee.com/liuyh9909/note-imgs/raw/master/img/20220228142639.png)

> Convolution Example
>
> ![](https://gitee.com/liuyh9909/note-imgs/raw/master/img/20220228142903.png)

### Max Pooling Summary

![](https://gitee.com/liuyh9909/note-imgs/raw/master/img/20220228143032.png)

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

![](https://gitee.com/liuyh9909/note-imgs/raw/master/img/20220228153048.png)

### 图像特征点提取

#### 角点检测

**Harris角点**

**Harris角点检测基本思想**

* 从图像局部的小窗口观察图像特征
* 角点定义 ：窗口向任意方向的移动都导致图像灰度的明显变化

**Harris角点检测**

![](https://gitee.com/liuyh9909/note-imgs/raw/master/img/20220228154848.png)

**Harris角点的性质**

* 旋转不变性
* 对于图像灰度的仿射变化具有部分的不变性
* 对于图像几何尺度变化不具有不变性
* 随尺度变化，Harris角点检测的性能下降

**FAST (features from accelerated segment test)**

**假设**

若该点的灰度值比其周围领域内足够多的像素点的灰度值大或者小，则该点可能为角点

**算法步骤**

![](https://gitee.com/liuyh9909/note-imgs/raw/master/img/20220228155828.png)

![](https://gitee.com/liuyh9909/note-imgs/raw/master/img/20220228160258.png)
