# SVM

## 0. Large Margin and VC Dimension

分类器：<mark style="color:orange;">大间隔能够带来较小的VC维</mark>

![](https://gitee.com/liuyh9909/note-imgs/raw/master/img/20220103140402.png)

## 1. Hard-Margin SVM

![](https://gitee.com/liuyh9909/note-imgs/raw/master/img/20220103134408.png)

![](https://gitee.com/liuyh9909/note-imgs/raw/master/img/20220103134656.png)

## 2. Soft-Margin SVM

![](https://gitee.com/liuyh9909/note-imgs/raw/master/img/20220103140115.png)

![](https://gitee.com/liuyh9909/note-imgs/raw/master/img/20220103140029.png)

## 3. Lagrange 变换

### 对偶与KKT

#### **Lagrange Multipliers**

![](https://gitee.com/liuyh9909/note-imgs/raw/master/img/20220103142219.png)

#### **Karush-Kuhn-Tucker conditions**

![](https://gitee.com/liuyh9909/note-imgs/raw/master/img/20220103142314.png)

### Hard-Margin Case

![](https://gitee.com/liuyh9909/note-imgs/raw/master/img/20220103142612.png)

![](https://gitee.com/liuyh9909/note-imgs/raw/master/img/20220103143024.png)

### Soft-Margin Case

![](https://gitee.com/liuyh9909/note-imgs/raw/master/img/20220103151200.png)

![](https://gitee.com/liuyh9909/note-imgs/raw/master/img/20220103151506.png)

> Hard-Margin和Soft-Margin在对偶空间唯一区别就是Soft-Margin有个上界，即 $$0\leq \alpha_i \leq C$$

## 4. KKT条件（Soft-Margin）

![](https://gitee.com/liuyh9909/note-imgs/raw/master/img/20220103152005.png)

## 5. Support Vectors

### Hard-Margin Case

![](https://gitee.com/liuyh9909/note-imgs/raw/master/img/20220103143212.png)

### Soft-Margin Case

![](https://gitee.com/liuyh9909/note-imgs/raw/master/img/20220103151040.png)

![](https://gitee.com/liuyh9909/note-imgs/raw/master/img/20220216155811.png)

## 6. Kernel Methods

![](https://gitee.com/liuyh9909/note-imgs/raw/master/img/20220103154856.png)

### Examples of Kernel Function

![](https://gitee.com/liuyh9909/note-imgs/raw/master/img/20220103155614.png)

### Kernel Trick in SVM

![](https://gitee.com/liuyh9909/note-imgs/raw/master/img/20220103154942.png)

![](https://gitee.com/liuyh9909/note-imgs/raw/master/img/20220103155830.png)

![](https://gitee.com/liuyh9909/note-imgs/raw/master/img/20220103155855.png)

## 7. Model Selection

### The “C” Problem

![](https://gitee.com/liuyh9909/note-imgs/raw/master/img/20220103161359.png)

### Overfitting and Underfitting

![](https://gitee.com/liuyh9909/note-imgs/raw/master/img/20220103161516.png)

### Model Selection

![](https://gitee.com/liuyh9909/note-imgs/raw/master/img/20220103161625.png)
