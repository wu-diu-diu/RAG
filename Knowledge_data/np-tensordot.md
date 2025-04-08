---
title: np.tensordot
date: 2024-07-25 11:00:45
tags: Numpy
---
## tensordot使用方法
### 主要是对张量进行点乘，对于多维张量来说，有时候需要指定维度进行点乘
。`numpy.tensordot(a, b, axes=0,1,2)`axes=1的话，则结果是两个数组以矩阵相乘的形式计算.实际为**a的最后一个维度和b的第一个维度进行点乘**

若a，b是二维数组，此时即为矩阵乘法。numpy中的轴分布如下图：


axis=0和axis=2的情况参考numpy的[官方解释](https://numpy.org/doc/stable/reference/generated/numpy.tensordot.html)

对于多维数组，axis可以指定多个轴进行点乘，例如（3， 4， 5）数组和（5， 4， 2）数组，可以指定第一个数组的1， 2轴和第二个数组的1， 0轴相点乘。代码为：

	np.random.seed(10)
	A = np.random.randint(0,9,(3,4,5))
	B = np.random.randint(0,9,(5,4,2))
	result5 = np.tensordot(A, B, [(1,2), (1,0)])
	result5
	array([[181, 142],
       [251, 190],
       [259, 235]])
意为A的第1轴和第2轴，与B的第1轴和第0轴点积，注意要点积的轴尺寸是一样的，这是点积的要求。(**这里B的取出来后会转置一下**)

对A来说，即要取第0轴的元素，有3个，对于B来说，要取第2轴的元素，有2个，故最后的结果为(3,2)。可手动取出来测试一下：

	np.sum(B[:,:,0].T * A[0])
	181

[参考链接](https://blog.csdn.net/weixin_28710515/article/details/90230842)
