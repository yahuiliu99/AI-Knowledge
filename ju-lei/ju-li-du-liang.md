# 距离度量

*   欧氏距离(Euclidean distance)

    $$
    d = \sqrt{\sum_{k=1}^n(x_{1k}-x_{2k})^2}
    $$
*   曼哈顿距离(Manhattan distance)

    $$
    d = \sum_{k=1}^n \mid {x_{1k}-x_{2k}} \mid
    $$
*   切比雪夫距离(Chebyshev distance)

    $$
    d = max(\mid x_{1k}-x_{2k} \mid)
    $$
*   闵可夫斯基距离(Minkowski distance)

    $$
    d = \sqrt[^p]{\sum_{k=1}^n \mid x_{1k}-x_{2k} \mid ^p}
    $$

    * 当$$p=1$$时，就是曼哈顿距离
    * 当$$p=2$$时，就是欧式距离
    * 当$$p \to \infty$$时，就是切比雪夫距离
*   余弦距离

    $$
    cos(\theta) = \frac{\sum_{k=1}^n x_{1k}x_{2k}}{\sqrt{\sum_{k=1}^n x_{1k}^2} \sqrt{\sum_{k=1}^n x_{2k}^2}}
    $$
*   马氏距离(Mahalanobis distance)

    $$
    d = \sqrt{(x_{1}-x_{2})^TS^{-1}(x_{1}-x_{2})}
    $$

    其中$$S$$为协方差矩阵，$$x_1=(x_{11},\cdots,x_{1k},\cdots,x_{1n})^T, \quad x_2=(x_{21},\cdots,x_{2k},\cdots,x_{2n})^T$$

    * 当$$S$$为单位矩阵时，退化为欧氏距离（单位圆）
    * 当$$S$$为对角矩阵时，退化为特征加权欧氏距离（椭圆）

    > $$S$$为单位矩阵，即样本数据的各个分量互相独立且各个分量的方差为1
