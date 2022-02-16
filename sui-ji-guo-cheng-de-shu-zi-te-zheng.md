# 随机过程的数字特征

## 1.单个随机过程的情形$${X(t);t\in T}$$

### 均值函数

$$
\mu_X(t)\hat{=}m(t)=E\{X(t)\}
$$

### 方差函数

$$
\sigma^2_X(t)\hat{=}D_X(t)=E\{[X(t)-\mu_X(t)]^2\}
$$

### (自)协方差函数

$$
C_X(s,t)\hat{=}E\{[X(s)-\mu_X(s)][[X(t)-\mu_X(t)]\}
$$

### (自)相关函数

$$
R_X(s,t)\hat{=}E\{X(s)X(t)\}
$$

### 数字特征之间的关系

$$
C_X(s,t)=R_X(s,t)-\mu_X(s)\cdot \mu_X(t)\\ D_X(t)=C_X(t,t)=R_X(t,t)-[\mu_X(t)]^2
$$

## 2.两个随机过程的情形$${X(t);t\in T}、{Y(t);t\in T}$$

### 互协方差函数

$$
C_{XY}(s,t)\hat{=}E\{[X(s)-\mu_X(s)][[Y(t)-\mu_Y(t)]\}
$$

### 互相关函数

$$
R_{XY}(s,t)\hat{=}E\{X(s)Y(t)\}
$$

### 数字特征之间的关系

$$
C_{XY}(s,t)=R_{XY}(s,t)-\mu_X(s)\cdot \mu_Y(t)
$$

如果有$$C_{XY}(s,t)=0$$或$$R_{XY}(s,t)=\mu_X(s)\cdot \mu_Y(t)$$，则称随机过程$${X(t);t\in T}、{Y(t);t\in T}$$是不相关的。

## 3.两个随机过程的独立性${X(t);t\in T}、{Y(t);t\in T}$

联合分布函数

$$
F_{XY}(x_1,\cdots,x_n;t_1,\cdots,t_n;y_1,\cdots,y_m;t_1',\cdots,t_m')\\ =P\{X(t_1)\leq x_1,\cdots,X(t_n)\leq x_n,Y(t_1')\leq y_1,\cdots,Y(t_m')\leq y_m\}
$$

若满足下式，则称两个随机过程独立

$$
F_{XY}(x_1,\cdots,x_n;t_1,\cdots,t_n;y_1,\cdots,y_m;t_1',\cdots,t_m')\\ =F_{X}(x_1,\cdots,x_n;t_1,\cdots,t_n)\cdot F_Y(y_1,\cdots,y_m;t_1',\cdots,t_m')
$$

\==【注】==：两个随机过程独立即不相关，但反之不对。（只有正态过程的独立和不相关是等价的）

### 4.母函数

*   #### 母函数定义

    ![image-20211223142547728](https://gitee.com/liuyh9909/note-imgs/raw/master/img/20211223143321.png)
*   #### 常见分布的母函数

    ![image-20211223150615017](https://gitee.com/liuyh9909/note-imgs/raw/master/img/20211223150615.png)
*   #### 母函数的基本性质

    ![image-20211223151448564](https://gitee.com/liuyh9909/note-imgs/raw/master/img/20211223151448.png)
*   #### 母函数的应用

    ![image-20211223152128662](https://gitee.com/liuyh9909/note-imgs/raw/master/img/20211223152128.png)

### 5.特征函数

*   #### 特征函数定义

    ![image-20211223152631413](https://gitee.com/liuyh9909/note-imgs/raw/master/img/20211223152631.png)
*   #### 常见分布的特征函数

    ![image-20211223152948968](https://gitee.com/liuyh9909/note-imgs/raw/master/img/20211223152949.png)
*   #### 特征函数的性质

    ![image-20211223153049458](https://gitee.com/liuyh9909/note-imgs/raw/master/img/20211223153049.png)
*   #### 特征函数与分布函数的关系

    ![image-20211223153207543](https://gitee.com/liuyh9909/note-imgs/raw/master/img/20211223153207.png)
