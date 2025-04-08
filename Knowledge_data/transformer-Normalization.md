---
title: transformer-Normalization
date: 2025-04-04 17:04:59
tags: LLM
---

<script>
MathJax = {
  tex: {
    inlineMath: [['$', '$'], ['\\(', '\\)']],
    processEscapes: true,
  }
};
</script>
<script src="https://cdn.jsdelivr.net/npm/mathjax@3/es5/tex-chtml.js"></script>

# Normalization

[Transformer学习笔记三：为什么Transformer要用LayerNorm/Batch Normalization & Layer Normalization （批量&层标准化)](https://zhuanlan.zhihu.com/p/456863215)

[深入理解Batch Normalization原理与作用](https://blog.csdn.net/litt1e/article/details/105817224)

Normalization又称为标准化或者规范化，是在深度学习中经常使用的一种算法。其核心思想是将输入数据的分布进行改变，改变数据的均值和方差，方便训练的进行。

## 为什么需要Normalizaion
在机器学习中，一般将采集到的真实场景的数据分为训练集和测试集，并假设训练集和测试集符合独立同分布，即将训练集和测试集看作两个随机变量**X**和**Y**，二者分布相同但却相互独立，这样才能保证训练集上表现良好的模型同样使用于测试集，即真实的场景。

但是人们发现有时候模型在训练集上表现出色，但是用在测试集上的时候，模型的表现却很糟糕。究其原因，是因为测试集和训练集的数据分布不同。假如对于一个分类模型，其一般基于一个条件分布假设，即：

$$
P(y | \mathbf{x}) \tag{1.1}
$$

其中**x**即为数据分布，y是标签。我们一般认为分类任务是x导致y，那么如果x发生改变，y的预测则也会发生改变。所以我们需要在模型中添加标准化层，期望当输入数据发生一定程度的偏移时，模型仍然能够work。

## 分布偏移类型

### 协变量偏移(Covariate Shift)
**协变量**是统计学中的一个概念，通常出现在回归分析中，表示可能会导致因变量发生变化的自变量中不是你重点关注的自变量。听起来很绕口，举个例子，假如你研究收入和受教育年限的关系，其中收入是因变量y，受教育年限是自变量或者解释变量x。直觉上来判断，受教育年限应该和收入的关系是正相关的，但是年龄也是一个重要的影响因变量收入的因素对吧，因为显然年龄40的人的收入普遍比年龄20的人的收入高，但这不是我们要重点研究的方向。所以我们要控制年龄这个变量防止其影响我们的实验结果，方法即为我们在同一年龄段内去收集样本。

这里的年龄即为协变量，它不是你研究的对象但却能影响实验结果。

那么在深度学习中，一般使用神经网络模型去做回归或者分类。神经网络的核心公式如下：
$$
Y = \sigma(X \cdot W + b) = \sigma(W \cdot X) \tag{1.2}
$$

将b参数也添加到权重矩阵中后，则上式中的W可以视为自变量，因为我们在训练中不断调整神经网络的权重参数才能达到目的，Y视为因变量，那么数据X则为协变量。所以数据的分布发生改变则叫做协变量偏移。

举个例子，假如给定一个显示中的猫狗图像数据集去训练一个分类模型，200只猫，300只狗，但是测试的时候却使用一个动漫的猫狗数据集，同样是200只猫，300只狗，这很考验模型的迁移学习能力。如果模型没有Normalization，还能不能work呢？有空可以做一个实验。

### 标签偏移
标签偏移和协变量偏移正好相反，即标签的分布发生了改变。例如，预测患者的疾病，我们可能根据症状来判断，即使疾病的相对流行率随着时间的推移而变化。标签偏移在这里是恰当的假设，因为疾病会引起症状。在另一些情况下，标签偏移和协变量偏移假设可以同时成立。例如,当标签是确定的，即使y导致x，协变量偏移假设也会得到满足。有趣的是，在这些情况下，使用基于标签偏移假设的方法通常是有利的。 这是因为这些方法倾向于包含看起来像标签（通常是低维）的对象，而不是像输入（通常是高维的）对象。

## Batch Normalization

Batch Normalization顾名思义是对batch层进行归一化和标准化。使用它的原因有以下几个：
- 1.通常我们需要将输入数据进行标准化，限制数据的大小在一个固定的量级，这样方便处理数据，加快模型收敛。
- 2.神经网络模型通常有很多层，每个层的输出在训练中可能会有不同的变化范围。假如一个层的输出的值的变化范围是另一个层的100倍，那么显然我们需要为输出变化范围较大的层，设置较小的学习率，以避免更新幅度过大。所以我们需要在每个层的输出加一个BN层，从而避免这种情况的发生。
- 3.对于深层的网络，如果输入数据的分布方差较大，在通过激活层的时候，容易发生梯度消失，即进入激活函数的梯度饱和区，进而影响模型的收敛速度。
- 4.通过BN层，可以将模型的不同的特征的分布都限定在一个范围内，从而减少内部协变量偏移(Internal Covariate Shift)

### BN层思路

- 对于二维数据，其形状一般为**B,C**，那么我们会在Batch维度上求解均值和方差，再让数据减去均值除以标准差，这样对于这一个批量的数据，它们在每一个特征维度上，都是均值为0，方差为1的。举个例子，比如15个样本，每个样本有10个特征，则数据形状为(15,10)。求得均值和方差的形状都为(1,10)，再减去均值除以方差，将15个样本的每个特征维度看作一个随机变量的话，那么这10个随机变量都服从均值为0，方差为1的分布。
- 对于图像数据，一般是三维**B,C,H*W**或者四维**B,C,H,W**,这时我们会去计算通道维度上均值和方差，如图1所示，蓝色部分表示要qiu。因为在图像的处理中，通道维C可以看作图像的特征。在卷积神经网络中，一个RGB三通道的图像，在经过多个卷积层之后，通道数在增加，图像尺寸是在减少。**因为我们希望用更多的通道来捕捉模型更抽象的特征**。举个例子，对于10张**H,W**大小的彩色图像，在经过BN层时，我们会计算R通道的所有像素值再除以**10 * H * W**，这样便得到了R通道的均值，R通道所有的值都减去均值再平方，再对结果求和除以**10 * H * W**，变得到了方差。同理，其他通道也一样。所以对于图像数据，我们是保证每个通道维度上的数据的分布都服从均值为0，方差为1的分布。
- 当然，如果单纯的把输入限制为标准分布也不合理，我们还会给数据乘以一个拉伸参数$\gamma$和偏移参数$\beta$，这两个参数是可学习的参数，让模型自己去学习该输出什么样的分布。

<div style="text-align: center;">
  <img src="transformer-Normalization/01.png" alt="图1：BN层" style="width: 30%; height: auto;">
</div>

代码如下：

```python
def batch_norm(X, gamma, beta, moving_mean, moving_var, eps, momentum):
    # 通过is_grad_enabled来判断当前模式是训练模式还是预测模式
    if not torch.is_grad_enabled():
        # 如果是在预测模式下，直接使用传入的移动平均所得的均值和方差
        X_hat = (X - moving_mean) / torch.sqrt(moving_var + eps)
    else:
        assert len(X.shape) in (2, 4)
        if len(X.shape) == 2:
            # 使用全连接层的情况，计算特征维上的均值和方差，mean和var的形状: (1, C)
            mean = X.mean(dim=0)
            var = ((X - mean) ** 2).mean(dim=0)
        else:
            # 使用二维卷积层的情况，计算通道维上（axis=1）的均值和方差。
            # 这里我们需要保持X的形状以便后面可以做广播运算
            # mean 和 var的形状为：(1, C, 1, 1)
            mean = X.mean(dim=(0, 2, 3), keepdim=True)
            var = ((X - mean) ** 2).mean(dim=(0, 2, 3), keepdim=True)
        # 训练模式下，用当前的均值和方差做标准化
        X_hat = (X - mean) / torch.sqrt(var + eps)
        # 更新移动平均的均值和方差
        moving_mean = momentum * moving_mean + (1.0 - momentum) * mean
        moving_var = momentum * moving_var + (1.0 - momentum) * var
    Y = gamma * X_hat + beta  # 缩放和移位
    return Y, moving_mean.data, moving_var.data
```

由代码可以看出，BN层在训练和推理阶段的行为是不同的，体现在对mean和var的选取上。在训练时，我们使用当前这一个小批次的数据的均值和方差来做归一化，同时使用滑动平均的方法来保存当前批次的均值和方差。在推理时，我们使用训练时保存的滑动平均的均值和方差来对数据进行归一化。使用滑动平均的方法可以保证均值和方差的更新更多的依赖历史数据，有助于减少小批量数据的波动，提供更稳定的估计。其中**momentum**越大，表示均值估计越依赖历史数据，越小，则越快的反映最近的小批量的统计信息，可能导致估计不够稳定。

那为什么要使用小批量数据来训练模型呢？
- 一个显而易见的回答是内存限制。如果数据过多不可能一次输入到模型中进行计算，那么显然我们需要小批量来处理数据。
- 在梯度下降算法中，梯度是损失函数关于模型参数的导数，指导模型参数更新的方向。一个样本可以视为高维空间中的一个点，如果使用数据集中所有样本的梯度的均值来指导模型去更新参数，理论上是最好的，但却不现实。如果只使用一个样本来计算梯度去指导模型更新，那么这个梯度大小非常不稳定，模型不能很好的收敛。所以小批量梯度更新是一种折中的办法。此外根据大数定律，当样本足够大时，样本的均值等于总体均值。

所以在推理时，滑动平均的均值和方差可以看作包含了整个数据集的均值和方差。此外推理时批量大小可能为1，如果这时仍然使用小批量均值，那么样本值的特征值都会变成0，显然是不对的。

## Layer Normalization

BN层提出来后，被广泛应用于CNN任务上来处理图像并取得了很好的效果。但是针对文本任务时，由于一个批量的文本数据的长度可能是不一致的，所以我们需要使用LN，有以下几个原因：
- 如果我们将不同文本的同一个位置的token看作特征，去计算这一个位置的均值做标准化的话，某些句子由于比较短，很可能这个位置上就没有token，这时会被0代替就是我们常用的padding。那么这样求出来的均值就会有误差，再用这个均值去归一化，每个位置的token的分布就不再是标准分布。
- 不同的句子的同一位置的token到底能不能看作同一类特征？我觉得不一定。因为不同的句子中的词语的语义不是由它的位置所决定的，而是由这个词所处的上下文所决定的。所以，token的语义向量才可以被看作特征。
- 我们都知道在注意力机制中，注意力分数的计算可以看作不同token之间的交流，通过这种交流得到的权重，每个token都会去更新自己的语义向量。也就是说，这种交流引起的语义向量的改变只发生在注意力计算中，而其他层比如LN层和MLP层，都只是在token的维度上对语义向量进行操作，比如归一化，升维，降维，点乘等
  
所以，在图像处理任务中，LN是指对一整张图片进行标准化处理，即在一张图片所有channel的pixel范围内计算均值和方差。而在NLP的问题中，LN是指在一个句子的一个token的范围内进行标准化。如下图所示。

<div style="display: flex; justify-content: space-between;">
  <img src="transformer-Normalization/02.jpg" alt="CNN中" style="width: 70%; height: auto;">
  <img src="transformer-Normalization/03.jpg" alt="远程衰减" style="width: 70%; height: 80%;">
</div>

代码如下：

```python
class CustomLayerNorm(nn.Module):
    def __init__(self, normalized_shape, eps=1e-5):
        super(CustomLayerNorm, self).__init__()
        """
        初始化 LayerNorm 层。

        参数:
        - normalized_shape: 需要归一化的特征维度。可以是一个整数（单个维度）或一个元组（多个维度）。
        - eps: 用于数值稳定性的小常数，防止除以零。
        """
        self.normalized_shape = normalized_shape
        self.eps = eps

        # 定义可学习的参数 gamma 和 beta
        self.gamma = nn.Parameter(torch.ones(normalized_shape))  # 缩放参数
        self.beta = nn.Parameter(torch.zeros(normalized_shape))  # 平移参数

    def forward(self, x):
        """
        前向传播函数。

        参数:
        - x: 输入张量，形状为 (batch_size, *normalized_shape)

        返回:
        - 归一化后的张量
        """
        # 计算均值和方差
        mean = x.mean(dim=-1, keepdim=True)  # 沿最后一个维度计算均值
        var = x.var(dim=-1, keepdim=True, unbiased=False)  # 沿最后一个维度计算方差

        # 归一化
        x_norm = (x - mean) / torch.sqrt(var + self.eps)

        # 应用缩放和平移
        x_norm = self.gamma * x_norm + self.beta

        return x_norm
```

## RMSnorm

- 是layernorm的一种，相比于layernorm，rmsnorm在归一化时不再计算均值，而是只计算均方根，即只保留layernorm层对输入的缩放不变性，放弃平移不变性
- 有研究表明，Layernorm之所以有效，更多的是因为其缩放不变性，而不是平移不变性

代码如下：

```python
class RMSNorm(nn.Module):
    def __init__(self, hidden_size, eps=1e-6):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(hidden_size))
        self.varance_eps = eps

    def forward(self, hidden_states):
        input_dtype = hidden_states.dtype
        ## 计算平方的均值
        variance = hidden_states.to(torch.float32).pow(2).mean(-1, keepdim=True)
        hidden_states = hidden_states * torch.rsqrt(variance + self.varance_eps)
        ## weight是可训练参数
        return (self.weight * hidden_states).to(input_dtype)
```


