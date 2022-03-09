# 纯不连续马氏过程(参数连续状态离散)

## 1. 纯不连续马氏过程的定义$$\{X(t),t\geq0\}$$

$$
\begin{aligned} &P\left\{X\left(t_{n+1}\right)=i_{n+1} \mid X\left(t_{1}\right)=i_{1}, X\left(t_{2}\right)=i_{2}, \cdots, X\left(t_{n}\right)=i_{n}\right\} \\ &=P\left\{X\left(t_{n+1}\right)=i_{n+1} \mid X\left(t_{n}\right)=i_{n}\right\} \end{aligned}
$$

## 2. 纯不连续马氏过程的转移概率

$$
p_{i j}\left(t_{1}, t_{2}\right) \hat{=} P\left\{X\left(t_{2}\right)=j \mid X\left(t_{1}\right)=i\right\}
$$

若$$p_{i j}\left(t\right)=p_{i j}\left(t_{1}, t_{2}\right)$$，即转移概率仅为时间差$$t=t_2-t_1$$的函数，而与$$t_1$$和$$t_2$$的值无关，则称此纯不连续马氏过程为齐次的。此时有

$$
\left\{\begin{array}{l} p_{i j}\left(t\right) \geq 0 \\ \sum\limits_{j \in S} p_{i j}\left(t\right)=1 \quad i \in S \end{array}\right.
$$

{% hint style="warning" %}
以下主要讨论齐次纯不连续马氏过程
{% endhint %}

## 3. 纯不连续马氏过程的C-K方程

### 一般情形

$$
\begin{aligned} &P\left\{X\left(t_{3}\right)=j \mid X\left(t_{1}\right)=i\right\}= \\ &\quad=\sum_{k \in S} P\left\{X\left(t_{3}\right)=j \mid X\left(t_{2}\right)=k\right\} P\left\{X\left(t_{2}\right)=k \mid X\left(t_{1}\right)=i\right\} \\ &\quad\left(t_{1}<t_{2}<t_{3}, \quad i, j \in S\right) \end{aligned}
$$

### 齐次情形

$$
p_{i j}(t+\tau)=\sum_{k \in S} p_{i k}(t) p_{k j}(\tau), \quad(i, j \in S, t>0, \tau>0)
$$

### 连续性条件

$$
\lim _{t \rightarrow 0} p_{i j}(t)=\delta_{i j}= \begin{cases}1, & i=j \\ 0, & i \neq j\end{cases}
$$

满足连续性条件的马氏过程称为随机连续的马氏过程。

## 4. 无穷小转移率$$q_{ij}$$及转移率矩阵（Q矩阵）

### 无穷小转移率或跳跃强度

{% hint style="info" %}
转移概率在0处的变化率
{% endhint %}

$$
q_{i j}=\lim _{\Delta t \rightarrow 0} \frac{p_{i j}(\Delta t)-p_{i j}(0)}{\Delta t}
$$

由连续性条件，显然有

$$
q_{i j}= \begin{cases}\lim \limits_{\Delta t \rightarrow 0} \dfrac{p_{i j}(\Delta t)}{\Delta t}, & i \neq j \\ \lim \limits_{\Delta t \rightarrow 0} \dfrac{p_{i i}(\Delta t)-1}{\Delta t}, & i=j\end{cases}
$$

即有

$$
q_{i j} \geq 0,(i \neq j), \quad q_{i j} \leq 0,(i=j)\\ \sum\limits_{j\in S}q_{ij}=0
$$

### 转移率矩阵（Q矩阵）

> 1.  相当于马氏链中一步转移矩阵
>
>     $$
>     Q=\left(q_{ij}\right)_{s\times s}
>     $$
> 2.  P矩阵与Q矩阵的关系(<mark style="color:purple;">转移概率在0处的变化率</mark>)
>
>     $$
>     \begin{array}{c|r} \dfrac{dP(t)}{dt} & _{t=0} \end{array}=Q
>     $$

转移率矩阵或Q矩阵定义如下：

$$
Q=\left(\begin{array}{ccccc} q_{00} & q_{01} & q_{02} & \cdots & q_{0 n} \\ q_{10} & q_{11} & q_{12} & \cdots & q_{1 n} \\ \vdots & \vdots & \vdots & & \vdots \\ q_{n 0} & q_{n 1} & q_{n 2} & \cdots & q_{n n} \end{array}\right)_{(n+1) \times(n+1)}
$$

#### **性质1**

* 对角线上元素$$\leq 0$$
* 对角线外元素$$\geq 0$$

#### **性质2**

* 每一行元素之和为零，$$\sum\limits_{j\in S}q_{ij}=0$$

## 5. 前进方程

$$
\left\{\begin{array}{c} \dfrac{d P(t)}{d t}=P(t) Q \\ P(0)=I_{(n+1) \times(n+1)} \end{array}\right.
$$

## 6. 后退方程

$$
\left\{\begin{array}{c} \dfrac{d P(t)}{d t}=QP(t) \\ P(0)=I_{(n+1) \times(n+1)} \end{array}\right.
$$

## 7. Fokker-Planck方程

令$$p_{j}(t)=P\{X(t)=j\}$$

过程的初始分布为

$$
\vec{p}(0)=\left(p_{0}(0), p_{1}(0), \cdots, p_{n}(0)\right)
$$

设在t 时刻时，过程所处各状态的概率分布为

$$
\vec{p}(t)=\left(p_{0}(t), p_{1}(t), \cdots, p_{n}(t)\right)
$$

即有

$$
\vec{p}(t)=\vec{p}(0) P(t)
$$

因此有

$$
\begin{aligned} \frac{d \vec{p}(t)}{d t}& =\vec{p}(0) \frac{d P(t)}{d t}\\ & \\ & =\vec{p}(0) P(t) Q\\ & \\ & =\vec{p}(t) Q \end{aligned}
$$

此即Fokker-Planck方程。

## 8. 纯不连续马氏过程的极限性质

### 一些结论与定义

*   $$\forall t \geq 0,i\in S$$，有$$p_{ii}(t)>0$$

    > 因此对于纯不连续马氏过程，每一个状态都是非周期的，无需引入周期的概念
* 若$$\int_0^{+\infin} p_{ii}(t)dt=+\infin$$，则称状态$$i$$为常返状态，否则为非常返状态
* 设$$i$$为常返状态，若$$\lim\limits_{t\rightarrow \infin}p_{ii}(t)>0$$，则称状态$$i$$为正常返状态；若$$\lim\limits_{t\rightarrow \infin}p_{ii}(t)=0$$，则称状态$$i$$为零常返状态
*   若概率分布$$\mathbf{\pi}=(\pi_i,i\in S)$$，满足

    $$
    \pi=\pi P(t), \quad \forall t \geq 0
    $$

    则称$$\pi$$为$$\{X(t),t\geq 0\}$$的平稳分布。
* 若对$$\forall i \in S, \lim\limits_{t \rightarrow \infin}p_i(t)=\pi_i^*$$存在，则称$$\pi^*\hat{=}\{\pi_i^*,i\in S\}$$为$$\{X(t),t\geq 0\}$$的极限分布。
* 与马氏链的讨论类似，我们有：<mark style="color:blue;">不可约纯不连续马氏过程是正常返的充分必要条件是它存在平稳分 布，且此时的平稳分布就是极限分布</mark>。

### 极限性质

![](https://gitee.com/liuyh9909/note-imgs/raw/master/img/20211228172338.png)

![](https://gitee.com/liuyh9909/note-imgs/raw/master/img/20211228172359.png)

## 9. 纯不连续马氏过程的例子

### Poission过程

### 纯增殖过程

### 生灭过程

#### **定义**

![](https://gitee.com/liuyh9909/note-imgs/raw/master/img/20211228203845.png)

#### **正灭矩阵**

![](https://gitee.com/liuyh9909/note-imgs/raw/master/img/20211228204050.png)
