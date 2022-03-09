# 马尔可夫链（参数离散状态离散）

## 1.Markov链的定义$${X(n);n\geq0}$$

$$
\begin{gathered} P\left\{X(n+1)=i_{n+1} \mid X(0)=i_{0}, X(1)=i_{1}, \cdots, X(n)=i_{n}\right\} \\ =P\left\{X(n+1)=i_{n+1} \mid X(n)=i_{n}\right\} \end{gathered}
$$

此时表明随机过程将来所处各个状态的可能性只与现在随机过程所处的状态有关，而与过去无关。称此特性为Markov性或无后效性，简称马氏性。Markov链也称马氏链。

## 2.一步转移概率

### 马氏链在n时刻的一步转移概率

$$
P\{X(n+1)=j \mid X(n)=i\} \hat{=} p_{i j}(n)
$$

若$$p_{ij}(n)\equiv p_{ij}$$，即上面式子的右边与时刻n无关，则称此马氏链为齐次（或时齐的）马氏链。

{% hint style="info" %}
一步转移概率满足：

* $$p_{i j}(n) \geq 0 \quad(i, j \in S)$$
* $$\sum_{j \in S} p_{i j}(n)=1 \quad i \in S$$
{% endhint %}

### (转移矩阵)齐次马氏链的一步转移概率矩阵

$$
P=\left(p_{ij}\right)
$$

马氏链初始分布$$\{\pi_i(0),i\in S\}$$，其中$$\pi_i(0)=P\{X_0=i\}$$

## 3.m步转移概率

### 马氏链在n时刻的m步转移概率

$$
P\{X(n+m)=j \mid X(n)=i\}\hat{=}p_{i j}^{(m)}(n)
$$

在齐次马氏链的情况下，$$p_{ij}^{(m)}(n)\equiv p^{(m)}_{ij}$$

{% hint style="info" %}
m步转移概率满足：

* $$p_{i j}^{(m)}(n) \geq 0 \quad(i, j \in S)$$
* $$\sum_{j \in S} p_{i j}^{(m)}(n)=1 \quad i \in S$$
* 规定：$$p_{i j}^{(0)}(n)=\delta_{i j}= \begin{cases}1 & i=j \\ 0 & i \neq j\end{cases}$$
{% endhint %}

### 齐次马氏链的m步转移概率矩阵

$$
P^{(m)}=\left(p^{(m)}_{ij}\right)
$$

### C-K方程

$$
p_{i j}^{(m+r)}(n)=\sum_{k \in S} p_{i k}^{(m)}(n) p_{k j}^{(r)}(n+m) \quad(i, j \in S)
$$

对于齐次马氏链，此方程为

$$
p_{i j}^{(m+r)}=\sum_{k \in S} p_{i k}^{(m)} p_{k j}^{(r)} \quad(i, j \in S)
$$

对于齐次马氏链的情形，写成矩阵的形式

$$
P^{(m+r)}=P^{(m)}P^{(r)}
$$

$$
P^{(m)}=P^{(m-1)} P^{(1)}=\cdots=(P)^{m}=P^{m}
$$

因此，利用一步转移矩阵$$P$$及初始分布$$\pi(0)$$就可以完全确定齐次马氏链的统计性质。

## 4.马氏链的例子

* #### 随机游动
* #### 排队模型
* #### 离散分支过程
* #### Polya模型

## 5.Markov链状态的分类

### 到达与相通

#### **到达**

![](https://gitee.com/liuyh9909/note-imgs/raw/master/img/20211224144349.png)

#### **相通**

![](https://gitee.com/liuyh9909/note-imgs/raw/master/img/20211224144423.png)

> 可到达和相通都具有传递性。

### 首达时间和首达概率

#### **首达时间：从状态i出发首次到达状态j的时间**

$$
T_{i j}(\omega) \hat{=} \min \left\{n: X_{0}=i, X_{n}(\omega)=j, n \geq 1\right\}
$$

#### **首达概率：系统在0时从状态i出发，经n步首次到达状态j的概率**

$$
f_{i j}^{(n)} \hat{=} P\left\{T_{i j}=n \mid X_{0}=i\right\}
$$

*   **系统在0时从状态i出发，经有限步转移后迟早到达状态j的概率**

    $$
    f_{i j}=\sum_{1 \leq n<\infty} f_{i j}^{(n)}=\sum_{1 \leq n<\infty} P\left\{T_{i j}=n \mid X_{0}=i\right\}=P\left\{T_{i j}<\infty\right\}
    $$
*   **系统在0时从状态i出发，经有限步转移后不能到达状态j的概率**

    $$
    P\left\{T_{i j}=\infty\right\}\hat{=}f_{i j}^{(\infin)}=1-f_{ij}
    $$
*   **系统在0时从状态i出发，首次到达状态j的平均转移步数（时间）**

    $$
    \mu_{i j} \hat{=} E\left\{T_{i j} \mid X_{0}=i\right\}=\sum_{n=1}^{\infty} n f_{i j}^{(n)}\\ \mu_i\hat{=}\mu_{ii}
    $$
*   **系统在0时从状态i出发，至少到达状态j的次数为m次的概率**

    $$
    g_{i j}(m)=P\left\{Y(j) \geq m \mid X_{0}=i\right\}
    $$

    其中$$Y(i)$$表示马氏链$$\{X_n;n\geq 1\}$$处于状态$$i$$的次数
*   **系统在0时从状态i出发，无限次访问状态j的概率**

    $$
    g_{i j}=\lim _{m \rightarrow \infty} g_{i j}(m)=P\left\{Y(j)=+\infty \mid X_{0}=i\right\}\\ g_{ij}=f_{ij}g_{jj},\quad g_{ii}=\lim\limits_{n \rightarrow \infty}\left(f_{i i}\right)^{n}
    $$

#### 首达概率的基本性质

![](https://gitee.com/liuyh9909/note-imgs/raw/master/img/20211224150251.png)

### 状态的分类

#### **常返态**$$f_{ii}=1，g_{i i}=\lim\limits_{n \rightarrow \infty}\left(f_{i i}\right)^{n}=1$$****

* **正常返**$$\mu_i<\infin$$****
* **零常返**$$\mu_i=\infin$$****

#### **非常返态**$$f_{ii}<1，g_{i i}=\lim\limits_{n \rightarrow \infty}\left(f_{i i}\right)^{n}\rightarrow 0$$****

### 常返态和非常返态的判别

![](https://gitee.com/liuyh9909/note-imgs/raw/master/img/20211224151837.png)

{% hint style="info" %}
若$$i$$为常返状态，且$$i\rightarrow j$$，则有$$i\leftrightarrow j$$，且$$j$$为常返状态
{% endhint %}

### 闭集和周期

#### **闭集的定义**

设C 是状态空间S 的一个子集，如果从C 内任何一个状态i 不能到达C 外的任何状态，则称C 是一个闭集。如果单个状态i 构成的集{i} 是闭集，则称状态i 是吸收态。如果闭集C 中不再含有任何非空闭的真子集，则称C 是不可约的。闭集是存在的，因为整个状态空间 S 就是一个闭集，当S 不可约时，则称此马氏链不可约，否则称此马氏链可约。

#### **闭集的性质**

![](https://gitee.com/liuyh9909/note-imgs/raw/master/img/20211224165314.png)

#### **周期的定义**

![](https://gitee.com/liuyh9909/note-imgs/raw/master/img/20211224165618.png)

{% hint style="info" %}
定理：设$${X_n;n\geq 0}$$为马氏链，状态空间为S ，对于$$\forall i,j \in S$$，若$$i \leftrightarrow j$$，则$$i$$与$$j$$具有相同的周期
{% endhint %}

#### **周期状态的判别**

![](https://gitee.com/liuyh9909/note-imgs/raw/master/img/20211224170131.png)

### 常返、非常返、周期状态的分类特性

![](https://gitee.com/liuyh9909/note-imgs/raw/master/img/20211224171206.png)

{% hint style="warning" %}
只有状态空间为无穷大的马氏链才会存在零常返
{% endhint %}

## 6.状态空间的分解

### 分解定理

![](https://gitee.com/liuyh9909/note-imgs/raw/master/img/20211224171523.png)

### 有限状态马氏链的性质

1. 所有非常返状态组成的集合不可能是闭集；（无限状态马氏链不一定）
2. 没有零常返状态；
3. 必有正常返状态；
4. <mark style="color:blue;">不可约有限马氏链只有正常返态</mark>；（不可约马氏链中所有状态具有相同的状态类型）
5. 状态空间可以分解为：

$$
S=D \bigcup C_{1} \cup C_{2} \cup \cdots \cup C_{k}
$$

​ 其中，每个$$C_n,n=1,2,\cdots,k$$均是由正常返状态组成的有限不可约闭集，$$D$$是非常返态集。

{% hint style="info" %}
#### 无限状态马氏链都是常返态
{% endhint %}

## 7.马氏链的极限性态

### $$P^n$$的极限性态

#### ****$$j\in S$$**是非常返状态或零常返状态**

$$
\lim \limits_{n\rightarrow \infin}p_{ij}^{(n)}=0
$$

#### $$j\in S$$**是非周期正常返状态（遍历态）**

$$
\lim \limits_{n\rightarrow \infin}p_{ij}^{(n)}=\frac{f_{ij}}{\mu_j}
$$

对于不可约的遍历链，则对于任意的$$i,j \in S$$，有$$\lim \limits_{n\rightarrow \infin}p_{ij}^{(n)}=\frac{1}{\mu_j}$$

&#x20;

![](https://gitee.com/liuyh9909/note-imgs/raw/master/img/20211224174556.png)

![](https://gitee.com/liuyh9909/note-imgs/raw/master/img/20211224174710.png)

{% hint style="info" %}
随机矩阵：每一个元素位于$$0\sim 1$$之间，且每一行元素之和为1
{% endhint %}

![](https://gitee.com/liuyh9909/note-imgs/raw/master/img/20211224175337.png)

## 8.马氏链的平稳分布

![](https://gitee.com/liuyh9909/note-imgs/raw/master/img/20211224180215.png)

![](https://gitee.com/liuyh9909/note-imgs/raw/master/img/20211224180257.png)

{% hint style="info" %}
对于不可约遍历链，极限分布$$\mathbf{\pi}^*=\pi$$存在，且就是等于平稳分布

极限分布定义：
{% endhint %}

![](https://gitee.com/liuyh9909/note-imgs/raw/master/img/20211224180913.png)

## 9.非常返态分析

### 计算从状态i出发进入状态子集$$C_k$$的概率$$P\{C_k|i\}$$

![](https://gitee.com/liuyh9909/note-imgs/raw/master/img/20211224181407.png)

### 非常返态进入常返态所需的平均时间

![](https://gitee.com/liuyh9909/note-imgs/raw/master/img/20211224181553.png)

* **吸收时间的概率分布**
* **非常返态进入常返态所需的平均时间**
