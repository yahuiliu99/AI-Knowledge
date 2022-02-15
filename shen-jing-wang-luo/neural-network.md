# Neural Network

## 1. Neurons

**perceptrons** ：binary (0 or 1)

**step function**

$$\text { output }=\left\{\begin{array}{ll} 0 & \text { if } \sum_{j} w_{j} x_{j} \leq \text { threshold } \\ 1 & \text { if } \sum_{j} w_{j} x_{j}>\text { threshold } \end{array}\right.$$

$$\text { output }=\left\{\begin{array}{ll} 0 & \text { if } w \cdot x+b \leq 0 \\ 1 & \text { if } w \cdot x+b>0 \end{array}\right.  ,$$ where $$b\equiv -threshold$$

**sigmoid neurons** : continuously ranging values between 0 and 1 &#x20;

**sigmoid function：**

$$
\sigma (z) = \dfrac{1}{1+e^{-z}}
$$

![Sigmoid Function](https://gitee.com/liuyh9909/note-imgs/raw/master/img/20220215181724.png)

This shape is a smoothed out version of a step function. So, sigmoid neuron is a smoothed out perceptron. The smoothness of $$\sigma$$ means that small changes $$\Delta w_j$$ in the weights and $$\Delta b$$ in the bias will produce a small change $$\Delta output$$ in the output from the neuron. That's the crucial fact which will allow a network of sigmoid neurons to learn.

$$
\Delta output \approx \sum_j \dfrac{\partial\ output}{\partial\ w_j}\ \Delta w_j + \dfrac{\partial\ output}{\partial\ b}\ \Delta b
$$

Δoutput is a _linear function_ of the changes Δ$$w_j$$ and Δb in the weights and bias.

**other models of artificial neuron**

#### tanh function :&#x20;

$$
\tanh (z) \equiv \dfrac{e^{z}-e^{-z}}{e^{z}+e^{-z}}
$$

![](https://gitee.com/liuyh9909/note-imgs/raw/master/img/20220215181730.png)

$$\sigma(z)=\frac{1+\tanh (z / 2)}{2}$$, $$\tanh$$ is just a rescaled version of the sigmoid function. One difference between tanh neurons and sigmoid neurons is that the output from tanh neurons ranges from -1 to 1, not 0 to 1.

**rectified linear neuron (rectified linear unit)**

#### relu function :&#x20;

$$
\max (0,z)
$$

![](https://gitee.com/liuyh9909/note-imgs/raw/master/img/20220215181734.png)

## 2. The Architecture of Neural Network

![img](https://gitee.com/liuyh9909/note-imgs/raw/master/img/20220215181744.png)

> called _**multilayer perceptrons**_ or **MLPs**, despite being made up of sigmoid neurons, not perceptrons.

design heuristics:

* ...
* ...

**deep neural networks** : many-layer structure - two or more hidden layers

**feedforward neural networks** : the output from one layer is used as input to the next layer.

**recurrent neural networks** : have feedback loops

## 3. Gradient Descent

### **cost function** :

$$
C(w,b)\equiv \dfrac{1}{2n}\sum_x \parallel y(x) - a \parallel^2
$$

$$w$$ denotes the collection of all weights in the network, $$b$$ all the biases, $$n$$ is the total number of training inputs, $$a$$ is the vector of outputs from the network when $$x$$ is input, and $$y(x)$$ denotes the desired output.

$$C$$ is called _**mean squared error**_ or **MSE**.

<mark style="color:blue;">**learning**</mark> <mark style="color:blue;"></mark><mark style="color:blue;">(training algorithm): find which weights and biases minimize a certain cost function by using an algorithm known as gradient descent.</mark>

### **Gradient Descent**

The process of repeatedly nudging an input of a function by some multiple of the negative gradient.

$$
\Delta C \approx \nabla C \cdot \Delta v
$$

$$
\begin{aligned} w_{k} \rightarrow w_{k}^{\prime} &=w_{k}-\eta \frac{\partial C}{\partial w_{k}} \\ b_{l} \rightarrow b_{l}^{\prime} &=b_{l}-\eta \frac{\partial C}{\partial b_{l}} \end{aligned}
$$

$$\eta$$ is called _**learning rate**_ .

#### **Stochastic Gradient Descent**

Randomly shuffle the training data, and then divide it into a whole bunch of mini-batches (each mini-batch in size of m) . Then estimate the overall gradient by computing gradients just for the randomly chosen mini-batch.

$$
\nabla C = \dfrac{1}{n} \sum_x \nabla C_x \approx \dfrac{1}{m}\sum_{j=1}^{m} \nabla C_{X_j}
$$

$$
\begin{aligned} w_{k} \rightarrow w_{k}^{\prime} &=w_{k}-\frac{\eta}{m} \sum_{j} \frac{\partial C_{X_{j}}}{\partial w_{k}} \\ b_{l} \rightarrow b_{l}^{\prime} &=b_{l}-\frac{\eta}{m} \sum_{j} \frac{\partial C_{X_{j}}}{\partial b_{l}} \end{aligned}
$$

where the sums are over all the training examples $$X_j$$ in the current mini-batch. Then we pick out another randomly chosen mini-batch and train with those. And so on, until we've exhausted the training inputs, which is said to complete an _**epoch**_ of training.

#### **Variations on Stochastic Gradient Descent**

*   **Hessian technique (Hessian optimization)**

    to consider the abstract problem of minimizing a cost function $$C$$ which is a function of many variables, $$w=w_1,w_2,…$$, so $$C=C(w)$$. By _**Taylor's theorem**_, the cost function can be approximated near a point $w$ by

    $$
    C(w+\Delta w)= C(w)+\sum_{j} \frac{\partial C}{\partial w_{j}} \Delta w_{j} +\frac{1}{2} \sum_{j k} \Delta w_{j} \frac{\partial^{2} C}{\partial w_{j} \partial w_{k}} \Delta w_{k}+\ldots
    $$

$$
C(w+\Delta w)=C(w)+\nabla C \cdot \Delta w+\frac{1}{2} \Delta w^{T} H \Delta w+\ldots
$$

$$
C(w+\Delta w) \approx C(w)+\nabla C \cdot \Delta w+\frac{1}{2} \Delta w^{T} H \Delta
$$

using _**Newton method**_, the expression can be minimized by choosing:

$$
\Delta w=-H^{-1} \nabla C
$$

$$
w+\Delta w=w-H^{-1} \nabla C
$$

![](https://gitee.com/liuyh9909/note-imgs/raw/master/img/20220215181753.png)

> **Jacobian Matrix**

$$
\left[\begin{array}{ccc}\frac{\partial y_{1}}{\partial x_{1}} & \cdots & \frac{\partial y_{1}}{\partial x_{n}} \\ \vdots & \ddots & \vdots \\ \frac{\partial y_{m}}{\partial x_{1}} & \cdots & \frac{\partial y_{m}}{\partial x_{n}}\end{array}\right]
$$

> **Hessian Matrix**

$$
\left[\begin{array}{cccc}\frac{\partial^{2} f}{\partial x_{1}^{2}} & \frac{\partial^{2} f}{\partial x_{1} \partial x_{2}} & \cdots & \frac{\partial^{2} f}{\partial x_{1} \partial x_{n}} \\ \frac{\partial^{2} f}{\partial x_{2} \partial x_{1}} & \frac{\partial^{2} f}{\partial x_{2}^{2}} & \cdots & \frac{\partial^{2} f}{\partial x_{2} \partial x_{n}} \\ \vdots & \vdots & \ddots & \vdots \\ \frac{\partial^{2} f}{\partial x_{n} \partial x_{1}} & \frac{\partial^{2} f}{\partial x_{n} \partial x_{2}} & \cdots & \frac{\partial^{2} f}{\partial x_{n}^{2}}\end{array}\right]
$$

*   **momentum-based gradient descent**

    \--similar to the notion of velocity from physics

    introduce velocity variables $$v=v_1,v_2,…$$, one for each corresponding $$w_j$$variable.

    $$
    \begin{array}{c} v \rightarrow v^{\prime}=\mu v-\eta \nabla C \\ w \rightarrow w^{\prime}=w+v^{\prime} \end{array}
    $$

    $$\mu$$ is a hyper-parameter which controls the amount of damping or friction in the system. It's called the _**momentum co-efficient**_.
* **Adagrad**
* **RMSProp**
* **Adam**

## 4. Backpropagation Algorithm

backpropagation gives us detailed insights into how changing the weights and biases changes the overall behavior of the network.

$$
a^{l}=\sigma\left(w^{l} a^{l-1}+b^{l}\right)
$$

$$
z^{l} \equiv w^{l} a^{l-1}+b^{l}
$$

$$
a^{l}=\sigma\left(z^{l}\right)
$$

$$
C_0=(a^l - y)^2
$$

we call $$z_l$$ the weighted input to the neurons in layer $$l$$.

$$
\left.\begin{array}{lll}w^{l-1}  \\ a^{l-2} \\b^{l-1} \end{array}\right\} \rightarrow z^{l-1} \rightarrow \left.\begin{array}{lll}w^l \\ a^{l-1} \\b^l\end{array}\right\} \rightarrow z^l \rightarrow a^l \rightarrow C
$$

$$
\dfrac{\partial C_0}{\partial w^l}= \dfrac{\partial z^l}{\partial w^l}\cdot \dfrac{\partial a^l}{\partial z^l}\cdot \dfrac{\partial C_0}{\partial a^l} =a^{l-1}\cdot \sigma'(z^l) \cdot 2(a^l-y)
$$

$$
\dfrac{\partial C}{\partial w^l}=\dfrac{1}{n}\sum_{k=0}^{n-1}\dfrac{\partial C_k}{\partial w^l}
$$

Similarly,

$$
\dfrac{\partial C_0}{\partial b^l}= \dfrac{\partial z^l}{\partial b^l}\cdot \dfrac{\partial a^l}{\partial z^l}\cdot \dfrac{\partial C_0}{\partial a^l} =1\cdot \sigma'(z^l) \cdot 2(a^l-y)
$$

$$
\dfrac{\partial C_0}{\partial a^{l-1}}= \dfrac{\partial z^l}{\partial a^{l-1}}\cdot \dfrac{\partial a^l}{\partial z^l}\cdot \dfrac{\partial C_0}{\partial a^l} =w^l\cdot \sigma'(z^l) \cdot 2(a^l-y)
$$

![](https://gitee.com/liuyh9909/note-imgs/raw/master/img/20220215181758.png)

$$
a_k^{l-1}\stackrel{w_{jk}^l}\longrightarrow \ a_j^l
$$

where the sum is over all neurons $$k$$ in the $$(l−1)^{th}$$ layer. Define a _weight matrix_ $$w^l$$ with $$j$$ rows and $$k$$ columns for each layer $$l$$ . The entry in the $$j^{th}$$ row and $$k^{th}$$ column is $$w_{jk}^{l}$$. Similarly, for each layer $$l$$ we define a _bias vector_, $$b_l$$ .

$$
a_{j}^{l}=\sigma\left(\sum_{k} w_{j k}^{l} a_{k}^{l-1}+b_{j}^{l}\right)
$$

$$
z_j^l = \sum_k w_{jk}^la_k^{l-1} + b_j^l
$$

$$
a_j^l=\sigma(z_j^l)
$$

$$
C_0 = \sum_{j=0}^{n_L - 1}(a_j^l-y_j)^2
$$

$$
\dfrac{\partial C_0}{\partial w_{jk}^l}= \dfrac{\partial z_j^l}{\partial w_{jk}^l}\cdot \dfrac{\partial a_j^l}{\partial z_j^l}\cdot \dfrac{\partial C_0}{\partial a_j^l} =a_k^{l-1}\cdot \sigma'(z_j^l) \cdot 2(a_j^l-y_j)
$$

$$
\dfrac{\partial C_0}{\partial b_j^l}= \dfrac{\partial z_j^l}{\partial b_j^l}\cdot \dfrac{\partial a_j^l}{\partial z_j^l}\cdot \dfrac{\partial C_0}{\partial a_j^l} =1\cdot \sigma'(z_j^l) \cdot 2(a_j^l-y_j)
$$

$$
\dfrac{\partial C_0}{\partial a_k^{l-1}}=\sum_{j=0}^{n_L-1} \dfrac{\partial z_j^l}{\partial a_k^{l-1}}\cdot \dfrac{\partial a_j^l}{\partial z_j^l}\cdot \dfrac{\partial C_0}{\partial a_j^l} =\sum_{j=0}^{n_L-1}w_{jk}^l\cdot \sigma'(z_j^l) \cdot 2(a_j^l-y_j)
$$

So,

$$
\dfrac{\partial C_0}{\partial w_{jk}^l}=a_k^{l-1}\cdot \sigma'(z_j^l) \cdot \dfrac{\partial C_0}{\partial a_j^l}=a_k^{l-1}\cdot \sigma'(z_j^l)\cdot\sum_{j=0}^{n_{L+1}-1}w_{jk}^{l+1}\cdot \sigma'(z_j^{l+1}) \cdot\dfrac{\partial C_0}{\partial a_j^{l+1}}=\cdots
$$

you can see why it's called backpropagation.

> **Hadamard Product** : elementwise product of two vectors, $$s \odot t$$
>
> $$\left[\begin{array}{l}1 \\ 2\end{array}\right] \odot\left[\begin{array}{l}3 \\ 4\end{array}\right]=\left[\begin{array}{l}1 * 3 \\ 2 * 4\end{array}\right]=\left[\begin{array}{l}3 \\ 8\end{array}\right]$$

**backpropagation** : the algorithm for determining how a single training example would like to nudge the weights and biases.

Hebbian theory--"neurons that fire together wire together"

## 5. The Cross-Entropy Cost Function

_**--optimization of quadratic cost function**_

* **a single sigmoid neuron**

$$
C=-\frac{1}{n} \sum_{x}[y \ln a+(1-y) \ln (1-a)]
$$

$$
\begin{aligned} \frac{\partial C}{\partial w_{j}} &=\frac{\partial C}{\partial a}\cdot \frac{\partial a}{\partial z}\cdot \frac{\partial z}{\partial w_j} \\ &=-\frac{1}{n} \sum_{x}\left(\frac{y}{\sigma(z)}-\frac{(1-y)}{1-\sigma(z)}\right) \sigma^{\prime}(z) x_{j}\\ &=\frac{1}{n} \sum_{x} \frac{\sigma^{\prime}(z) x_{j}}{\sigma(z)(1-\sigma(z))}(\sigma(z)-y) \end{aligned}
$$

where $$\sigma^{\prime}(z)=\dfrac{e^{-z}}{(1+e^{-z})^2}=\sigma(z)(1-\sigma(z))$$

$$
\frac{\partial C}{\partial w_{j}}=\frac{1}{n} \sum_{x} x_{j}(\sigma(z)-y)
$$

It tells us that the rate at which the weight learns is controlled by $$\sigma (z) - y$$. When we use the cross-entropy, the $$\sigma'(z)$$term gets canceled out, and we no longer need worry about it being small.

$$
\begin{aligned}\frac{\partial C}{\partial b} &=\frac{\partial C}{\partial a}\cdot \frac{\partial a}{\partial z}\cdot \frac{\partial z}{\partial b} \\&=-\frac{1}{n} \sum_{x}\left(\frac{y}{\sigma(z)}-\frac{(1-y)}{1-\sigma(z)}\right) \sigma^{\prime}(z)\\&=\frac{1}{n} \sum_{x} \frac{\sigma^{\prime}(z)}{\sigma(z)(1-\sigma(z))}(\sigma(z)-y)\\&=\frac{1}{n} \sum_{x}(\sigma(z)-y)\end{aligned}
$$

*   **many-neuron multi-layer networks**

    $$
    C=-\frac{1}{n} \sum_{x} \sum_{j}\left[y_{j} \ln a_{j}^{L}+\left(1-y_{j}\right) \ln \left(1-a_{j}^{L}\right)\right]
    $$

$$
\begin{aligned} \frac{\partial C}{\partial w_{j k}^{L}} &=\frac{1}{n} \sum_{x} a_{k}^{L-1}\left(a_{j}^{L}-y_{j}\right) \\ \frac{\partial C}{\partial b_{j}^{L}} &=\frac{1}{n} \sum_{x}\left(a_{j}^{L}-y_{j}\right) \end{aligned}
$$

*   **linear neurons**

    the outputs are simply $$a_j^l = z_j^l$$, just like "$$\sigma(z) = z \ , \ \sigma '(z) = 1$$"

    so quadratic cost is the same as cross-entropy cost function.

### **Where does the cross-entropy come from?**

*   Step 1:

    $$
    \frac{\partial C}{\partial b}=\frac{\partial C}{\partial a} \sigma^{\prime}(z)
    $$
* Step 2: Using $$\sigma '(z) = \sigma(z)(1-\sigma (z))=a(1-a)$$

$$
\frac{\partial C}{\partial b}=\frac{\partial C}{\partial a} a(1-a)\tag{1}
$$

* Step 3: our goal is to make the $$\sigma '(z)$$term disappear :

$$
\frac{\partial C}{\partial b}=(a-y) \tag{2}
$$

* Step 4: compare equation (1) and equation (2)&#x20;

$$
\frac{\partial C}{\partial a}=\frac{a-y}{a(1-a)}
$$

* Step 5:

$$
C=-[y \ln a+(1-y) \ln (1-a)]+\text { constant }
$$

> **Cross Entropy** :
>
> $$
> H(p, q)=H(p)+D_{\mathrm{KL}}(p \| q)
> $$
>
> $$
> H(p, q)=-\sum_{x \in \mathcal{X}} p(x) \log q(x)
> $$
>
> $$
> H(p)=-\sum_{x \in \mathcal{X}} p(x) \log p(x)
> $$

## 6. Softmax

_**--optimization of sigmoid function**_

$$
z_{j}^{L}=\sum_{k} w_{j k}^{L} a_{k}^{L-1}+b_{j}^{L}
$$

$$
a_{j}^{L}=\frac{e^{z_{j}^{L}}}{\sum_{k} e^{z_{k}^{L}}}
$$

Show that $$\dfrac{\partial a_j^L}{\partial z_k^L}$$ is positive if $$j=k$$ and negative if $$j≠k$$.

$$
\sum_{j} a_{j}^{L}=\frac{\sum_{j} e^{z_{j}^{L}}}{\sum_{k} e^{z_{k}^{L}}}=1
$$

the output from the softmax layer can be thought of as a _**probability distribution**_. We can interpret $$a_j^L$$ as the network's estimated probability that the correct digit classification is $$j$$.

You can think of softmax as a way of rescaling the $$z_j^L$$, and then squishing them together to form a probability distribution.

## 7. Overfitting and Regularization

*   regularized cross-entropy

    $$
    C=-\frac{1}{n} \sum_{x j}\left[y_{j} \ln a_{j}^{L}+\left(1-y_{j}\right) \ln \left(1-a_{j}^{L}\right)\right]+\frac{\lambda}{2 n} \sum_{w} w^{2}
    $$

    where $$\lambda > 0$$ is known as the _**regularization parameter**_.
*   regularized quadratic cost

    $$
    C=\frac{1}{2 n} \sum_{x}\left\|y-a^{L}\right\|^{2}+\frac{\lambda}{2 n} \sum_{w} w^{2}
    $$

In both cases we can write the regularized cost function as

$$
C=C_{0}+\frac{\lambda}{2 n} \sum_{w} w^{2}
$$

$$
\begin{aligned} \frac{\partial C}{\partial w} &=\frac{\partial C_{0}}{\partial w}+\frac{\lambda}{n} w \\ \frac{\partial C}{\partial b} &=\frac{\partial C_{0}}{\partial b} \end{aligned}
$$

gradient descent:

$$
b \rightarrow b-\eta \frac{\partial C_{0}}{\partial b}
$$

$$
\begin{aligned} w & \rightarrow w-\eta \frac{\partial C_{0}}{\partial w}-\frac{\eta \lambda}{n} w \\ &=\left(1-\frac{\eta \lambda}{n}\right) w-\eta \frac{\partial C_{0}}{\partial w} \end{aligned}
$$

rescale the weight $$w$$ by a factor $$1 - \dfrac{\eta \lambda}{n}$$. This rescaling is sometimes referred to as _**weight decay**_.

stochastic gradient descent:

$$
b \rightarrow b-\frac{\eta}{m} \sum_{x} \frac{\partial C_{x}}{\partial b}
$$

$$
w \rightarrow\left(1-\frac{\eta \lambda}{n}\right) w-\frac{\eta}{m} \sum_{x} \frac{\partial C_{x}}{\partial w}
$$

where the sum is over training examples x in the mini-batch.

> **regularization is a way to reduce overfitting and to increase classification accuracies.**

### **Other techniques for regularization**

*   L1 regularization

    $$
    C=C_{0}+\frac{\lambda}{n} \sum_{w}|w|
    $$

$$
\frac{\partial C}{\partial w}=\frac{\partial C_{0}}{\partial w}+\frac{\lambda}{n} \operatorname{sgn}(w)
$$

$$
w \rightarrow w^{\prime}=w-\frac{\eta \lambda}{n} \operatorname{sgn}(w)-\eta \frac{\partial C_{0}}{\partial w}
$$

*   **Dropout**

    Unlike L1 and L2 regularization, dropout doesn't rely on modifying the cost function. Dropout modify the network.

    Dropout refers to randomly (and temporarily) ignoring half the hidden neurons in the network, while leaving the input and output neurons untouched. By “_**ignoring**_”, these units are not considered during a particular forward or backward pass.



    This technique reduces complex co-adaptations of neurons, since a neuron cannot rely on the presence of particular other neurons. It is, therefore, forced to learn more robust features that are useful in conjunction with many different random subsets of the other neurons.

![](https://gitee.com/liuyh9909/note-imgs/raw/master/img/20220215181812.png)

#### Optimization of Parameters

* weight initialization $$w$$
* variable learning rate schedule $$\eta$$
* early stopping
* regularization parameter $$\lambda$$
* Mini-batch size

## 8. Universality Theorem

![](https://gitee.com/liuyh9909/note-imgs/raw/master/img/20220215181816.png)

**Proof**

## 9. Train Deep Networks

> _**The gradient in deep neural networks is unstable, tending to either explode or vanish in earlier layers.**_

### **The vanishing gradient problem**

phenomenon:

* more hidden layers make the accuracy drop
* later hidden layers learn faster than earlier hidden layers. (or early hidden layers learn much more slowly than later hidden layers.)
* the gradient tends to get _**smaller**_ as we move _**backward**_ through the hidden layers.

denote $$\delta_j^l = \dfrac{\partial C}{\partial b_j^l}$$ as the gradient (actually as error) for the $$j$$th neuron in the $$l$$th layer.

the length $$| \delta^1 |$$ measures the speed at which the first hidden layer is learning, while the length $$| \delta^2 |$$ measures the speed at which the second hidden layer is learning.

**What's causing the vanishing gradient problem?**

![](https://gitee.com/liuyh9909/note-imgs/raw/master/img/20220215181821.png)

$$
\dfrac{\partial C}{\partial b_1}=\dfrac{\partial C}{\partial a_4}\cdot \dfrac{\partial a_4}{\partial z_4}\cdot \dfrac{\partial z_4}{\partial a_3}\cdot \dfrac{\partial a_3}{\partial z_3}\cdot \dfrac{\partial z_3}{\partial a_2}\cdot \dfrac{\partial a_2}{\partial z_2}\cdot \dfrac{\partial z_2}{\partial a_1}\cdot \dfrac{\partial a_1}{\partial z_1}\cdot \dfrac{\partial z_1}{\partial b_1}
\\= \dfrac{\partial C}{\partial a_4}\cdot \sigma'(z_4)\cdot w_4\cdot \sigma'(z_3)\cdot w_3\cdot \sigma'(z_2)\cdot w_2\cdot \sigma'(z_1)\cdot 1
\\=\sigma^{\prime}\left(z_{1}\right)\cdot w_{2} \cdot\sigma^{\prime}\left(z_{2}\right)\cdot w_{3} \cdot\sigma^{\prime}\left(z_{3}\right) \cdot w_{4}\cdot \sigma^{\prime}\left(z_{4}\right) \cdot \frac{\partial C}{\partial a_{4}}
$$

&#x20;this expression is a product of terms of the form $$w_j\sigma'(z_j)$$.&#x20;

![](https://gitee.com/liuyh9909/note-imgs/raw/master/img/20220215181825.png)

The derivatives reaches a maximum at $$\sigma'(0)=1/4$$. If we choose the weights using a Gaussian with mean 0 and standard deviation 1. So the weights will usually satisfy $$|w_j|<1$$. And then $$|w_j\sigma'(z_j)| <1/4$$.

$$
\frac{\partial C}{\partial b_{1}}=\sigma'(z_{1}) \overbrace{w_{2} \sigma'(z_{2})}^{< \frac{1}{4}} \overbrace{w_{3} \sigma'(z_{3})}^{< \frac{1}{4}} \overbrace{w_{4} \sigma^{\prime}(z_{4})}^{< \frac{1}{4}} \frac{\partial C}{\partial a_{4}}
$$

The more terms, the smaller the product will be. This is the essential origin of the vanishing gradient problem.

> Indeed, if the terms get large enough greater than 1, then we will no longer have a vanishing gradient problem. Instead, the gradient will actually grow exponentially as we move backward through the layers. We'll have an exploding gradient problem.

### **The exploding gradient problem**

Contrary to the vanishing gradient problem, sometimes the gradient gets much lager in earlier layers.

**The unstable gradient problem**

<mark style="color:blue;">**It's that the gradient in early layers is the product of terms from all the later layers.**</mark> <mark style="color:blue;"></mark><mark style="color:blue;"></mark> When there are many layers, that's an intrinsically unstable situation.

**The prevalence of the vanishing gradient problem**

To avoid the vanishing gradient problem we need $$|w\sigma'(z)|\geqslant 1$$. when we make $$w$$ large, we need to be careful that we're not simultaneously making $$\sigma′(wa+b)$$ small. That turns out to be a considerable constraint. The only way to avoid this is if the input activation falls within a fairly narrow range of values.

**Unstable gradients in more complex networks**

$$
\delta^{l}=\Sigma^{\prime}(z^{l})(w^{l+1})^{T} \Sigma^{\prime}(z^{l+1})(w^{l+2})^{T} \ldots \Sigma^{\prime}(z^{L}) \nabla_{a} C
$$

Here, $$\Sigma^{\prime}(z^{l})$$ is a diagonal matrix whose entries are the $$\sigma'(z)$$ values for the weighted inputs to the $$l$$th layer. The $$w^l$$ are the weight matrices for the different layers. And $$\nabla_a C$$ is the vector of partial derivatives of $$C$$ with respect to the output activations.

## **10.** Optimization for Training Deep Networks

### **Batch Normalization**

## 11. Convolutional Neural Networks

Convolutional neural networks use three basic ideas: _**local receptive fields**_, _**shared weights**_, and _**pooling**_. Let's look at each of these ideas in turn.

**Local receptive fields**

That region in the input image is called the _local receptive field_ for the hidden neuron. It's a little window on the input pixels.

![](https://gitee.com/liuyh9909/note-imgs/raw/master/img/20220215181832.png)

**Shared weights and biases**

we sometimes call the map from the input layer to the hidden layer a _**feature map**_. We call the weights defining the feature map the _**shared weights**_. And we call the bias defining the feature map in this way the _**shared bias**_. The shared weights and bias are often said to define a _**kernel**_** or **_**filter**_.

![In the example shown, there are 3 feature maps. Each feature map is defined by a set of 5×5 shared weights, and a single shared bias.](https://gitee.com/liuyh9909/note-imgs/raw/master/img/20220215181837.png)

**Pooling layers**

Pooling layers are usually used immediately after convolutional layers. What the pooling layers do is simplify the information in the output from the convolutional layer.

* _**max-pooling**_ : a pooling unit outputs the maximum activation in the 2×2 input region

![Note that since we have 24×24 neurons output from the convolutional layer, after pooling we have 12×12 neurons.](https://gitee.com/liuyh9909/note-imgs/raw/master/img/20220215181856.png)

* _**L2 pooling**_ : a pooling unit outputs the square root of the sum of the squares of the activations in the 2×2 region.

**putting it all together**

![](https://gitee.com/liuyh9909/note-imgs/raw/master/img/20220215181900.png)

a convolutional layer using a 5×5 local receptive field and 3 feature maps. The result is a layer of 3×24×24 hidden feature neurons. The next step is a max-pooling layer, applied to 2×2 regions, across each of the 3 feature maps. The result is a layer of 3×12×12 hidden feature neurons.

The final layer of connections in the network is a _**fully-connected layer**_. This layer connects _every_ neuron from the max-pooled layer to every one of the output neurons.

**How to avoid gradient problems?**

(1) Using convolutional layers greatly reduces the number of parameters in those layers, making the learning problem much easier;

(2) Using more powerful regularization techniques (notably dropout and convolutional layers) to reduce overfitting, which is otherwise more of a problem in more complex networks;

(3) Using rectified linear units instead of sigmoid neurons, to speed up training - empirically, often by a factor of 3-5;

(4) Using GPUs and being willing to train for a long period of time.

## Appendix

Over the long run it's possible the biggest breakthrough in machine learning won't be any single conceptual breakthrough. Rather, the biggest breakthrough will be that machine learning research becomes profitable, through applications to data science and other areas.
