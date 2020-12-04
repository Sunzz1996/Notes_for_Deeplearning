#### Cross Entropy (交叉熵损失的理解)

#### 1. 二分类交叉熵损失的定义

对于某一个样本 $i$, 其交叉熵损失为：

   

$$
CE_{i}=-\left(y \cdot \log p+\left(1-y\right) \cdot \log \left(1-p\right)\right) \tag{1}\label{eq1}
$$

$$
-CE_{i} =\left(y-0\right) \cdot \log p+\left(1-y\right) \cdot \log \left(1-p\right)
$$

$$
-CE_{i} =\left(y-0\right) \cdot \log p_{1}+\left(1-y\right) \cdot \log p_{0}
$$

其中 $y$ 为样本 $i$ 的 标签， $p$ 为模型预测样本$i$ 是  $y$ 的概率 。

$p_{1}$为模型预测样本为 1 类的概率， $p_{0}$ 为模型预测样本为 0 类的概率。

对于二分类，其关系如下表：

| 1类     | 0类              |
| ------- | ---------------- |
| $p_{1}$ | $p_{0}= 1-p_{1}$ |

可以看出，当样本$i$的真实类别为 1 类时，其交叉熵损失为：(后半部分没有了，即与 $p_{0}$无关)
$$
-CE_{i} =\left(y-0\right) \cdot \log p_{1} = \log p_{1}
$$
当样本的真实类别为 0 类时， 其交叉熵损失为：(前半部分没有了，即与 $p_{1}$无关)
$$
-CE_{i} =\left(1-y\right) \cdot \log p_{0} = \log p_{0}
$$
由此可知，一个样本$i$的交叉熵最后的形式只与其本身的概率（模型输出的）$p_{i}$有关， 即  $-\log p_{i}$

但是，要写成统一的形式, 如式$\ref{eq1}$所示。而写成统一的形式，为什么叫“交叉”熵呢?

![](..\Figures\CE loss2.png)

即， 在构建统一形式时， 如果输入样本的类别为 1 ，则减去 0 类 ； 如果输入样本类别为 0 则 减去 1 类。总之， 要确保最后计算输入样本 $i$ 的交叉熵损失时， 只剩下  $-CE_{i} = \log p_{i}$

#### 2. 多分类交叉熵损失的定义

多分类交叉熵损失 相比与二分类 ，无非就是 从两个类拓展到了多个类。对于每一个输入样本，其真实的类别只可能有一个标签，因此，对于其交叉熵应该还是
$$
-CE_{i} = \log p_{i}
$$
对于$N$个样本写成统一的形式:
$$
CE =\frac{1}{N} \sum_{i}CE_{i}=-\frac{1}{N} \sum_{i} y_{i}\log p_{i}
$$
其中 $y_{i}$为指示变量 （target）， 只有当模型的预测类别与标签类别（真实）相同时，为 1 ；否则为 0.  仍然是 输入样本 $i$ 的交叉熵  只与其自身的模型预测概率有关。

#### 3. 交叉熵的物理意义

通俗讲，就是模型使用了交叉熵之后会产生什么作用。

对于二分类：

就是使得 模型预测 为 类别 1 的概率 $p_{1}$ 在 类别 $y=1$ 时越来越大； 模型预测 为 类别 0 的概率 $p_{0}$ 在 类别 $y=0$ 时越来越大；



#### 4. 那么 现在需要模型的输出概率 $S_{i}$ 在  类别为 $y_{i}$ 时越来越大，输出概率 $S_{i+1}$ 在  类别为 $y_{i+1}$ 时越来越大, 怎么写统一形式的二分类交叉熵？

对于输入样本 $i$:
$$
-CE_{i}=\left\{
\begin{aligned}
&\log S_{i},&& y = y_{i} \\
&\log S_{i+1}, && y = y_{i+1}  \\
\end{aligned}
\right.
$$
统一形式的交叉熵：
$$
-CE_{i} = \left(y_{i+1} -y\right)\log S_{i} + \left(y -y_{i}\right)\log S_{i+1}
$$
其中， 约束  $y_{i}\leq y \leq y_{i+1}$. 如此， 计算出来的 交叉熵损失和原生的交叉熵损失只相差一个  常量系数 $y_{i+1} -y_{i}$.  如令  $y_{i+1} -y_{i} = 1$，则两者等价。

具体例子见 **Distribution Focal Loss (DFL)** 
$$
\operatorname{DFL}\left(\mathcal{S}_{i}, \mathcal{S}_{i+1}\right)=-\left(\left(y_{i+1}-y\right) \log \left(\mathcal{S}_{i}\right)+\left(y-y_{i}\right) \log \left(\mathcal{S}_{i+1}\right)\right)
$$
