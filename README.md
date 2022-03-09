# 概率论基本知识

## 常见分布

### 二项分布$$X\sim B(n,p)$$

* 分布密度：$$P(X=j)=C_n^jp^jq^{n-j}, \quad 0\leq j\leq n$$
* 均值：$$E(X)=np$$
* 方差：$$Var(X)=npq$$
* 二阶矩：$$E(X^2)=npq+(np)^2$$

### 几何分布

* 分布密度：$$P(X=j)=pq^{j-1}, \quad j=1,2,\cdots$$
* 均值：$$E(X)=\dfrac{1}{p}$$
* 方差：$$Var(X)=\dfrac{q}{p^2}$$
* 二阶矩：$$E(X^2)=\dfrac{q+1}{p^2}$$

### 均匀分布$$X\sim U(a,b)$$

* 分布密度：$$f(x)=\dfrac{1}{b-a}, \quad x\in (a,b)$$
* 均值：$$E(X)=\dfrac{a+b}{2}$$
* 方差：$$Var(X)=\dfrac{(b-a)^2}{12}$$
* 二阶矩：$$E(X^2)=\dfrac{b^3-a^3}{3(b-a)}$$

### 正态分布$$X\sim N(\mu,\sigma^2)$$

* 分布密度：$$f(x)=\dfrac{1}{\sqrt{2\pi}\sigma}e^{-\frac{(x-\mu)^2}{2\sigma^2}}$$
* 均值：$$E(X)=\mu$$
* 方差：$$Var(X)=\sigma^2$$
* 二阶矩：$$E(X^2)=\sigma^2+\mu^2$$

### 泊松分布$$Poission(\lambda)$$

* 分布密度：$$P(X=k)=\dfrac{\lambda^k}{k!}e^{-\lambda}, \quad k=0,1,\cdots$$
* 均值：$$E(X)=\lambda$$
* 方差：$$Var(X)=\lambda$$
* 二阶矩：$$E(X^2)=\lambda+\lambda^2$$

### 指数分布$$E(\lambda)$$

* 分布密度：$$f(x)=\lambda e^{-\lambda x},\quad x>0$$
* 均值：$$E(X)=\dfrac{1}{\lambda}$$
* 方差：$$Var(X)=\dfrac{1}{\lambda^2}$$
* 二阶矩：$$E(X^2)=\dfrac{2}{\lambda^2}$$

## 全概率公式

* #### 离散型

$$
P(A)=\sum\limits_i P\{A|B_i\}P\{B_i\}
$$

* #### 连续型

$$
P(A)=\int_{-\infin}^{\infin} P\{A|Y=y\}f_Y(y)dy
$$

## 条件数学期望

* #### 离散型情形

$$
E\{E\{X|Y\}\}=\sum \limits_j E\{X|Y=y_j\}P\{Y=y_j\}=E\{X\}
$$

* #### 连续型情形

$$
E\{E\{X|Y\}\}=\int _{-\infin}^{\infin} E\{X|Y=y\}f_Y(y)dy=E\{X\}
$$

<mark style="color:red;">【注】</mark>：条件数学期望$$E\{X|Y\}$$是随机变量Y的函数

* #### 条件数学期望的性质

![](https://gitee.com/liuyh9909/note-imgs/raw/master/img/20211223161146.png)

<mark style="color:red;">【注】</mark>：常用的计算式子

![](https://gitee.com/liuyh9909/note-imgs/raw/master/img/20211223161413.png)
