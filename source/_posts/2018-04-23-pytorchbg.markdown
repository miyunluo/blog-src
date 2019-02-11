---
layout:     post
title:      "pytorch自用笔记"
tags:
    - python
---

> “防止以后踩坑”

pytorch 0.4 较之前api有了很大变化

## 0.0 Tensor

在pytorch 0.4 中 Tensor 和 Variable 合并

Tensor可以理解为一个多维矩阵，可以使用 `.size()` 显示大小，同时也可以和 numpy.array 转换

![](/images/in-post/post-blog-torch0.png)

由于和Variable合并，`requires_grad` 和 `autograd` 成了 `Tensor` 的属性。

`Tensor` 包含的属性，`.data` 获得Tensor的数据 `.item` 获得 Tensor的数值大小，`.grad` 获得梯度，比如

![](/images/in-post/post-blog-torch1.png)

现在Tensor要求导必须是 floating point dtype

以及一些其他操作

```python
# 沿行取最大值
max_value, max_idx = torch.max(x, dim=1)
# 沿行求和
sum_x = torch.sum(x, dim=1)

# 增减维度
x = x.unsqueeze(0) # 在第一维增加
x = x.unsqueeze(1) # 在第二维增加

x = x.squeeze(0)   # 减少第一维度
x = x.squeeze()    # tensor中所有维度为1全部去掉

# 维度交换
x = torch.randn(3, 4, 5)
# permute 重新排列
x = x.permute(1, 0, 2) # x : torch.Size([4, 3, 5])
# transpose 交换 tensor 中两个维度
x = x.transpose(0, 2)  # x : torch.Size([5, 3, 4])

# view 对 tensor 进行 reshape
x = torch.randn(3, 4, 5)
x = x.view(-1, 5) # -1 表示任意大小，5 表示第二维变成 5
				  # x : torch.Size([12, 5])
x = x.view(3, 20) # x : torch.Size([3, 20])

# inplace 操作，直接对tensor操作而不需要另外开辟内存空间。一般是操作符后加_
x.unsqueeze_(0)     # unsqueeze 进行 inplace
x.transpose_(1, 0)  # transpose 进行 inplace

x = torch.ones(3, 3)
y = torch.ones(3, 3)
x.add_(y)	# add 进行 inplace
```

## 0.1 自动求导

- **多次自动求导**

  在调用过 backward 后自动计算一次导数，但是再次调用会报错，因为pytorch默认做完一次自动求导后，计算图被丢弃，两次求导需要手动设置。

![](/images/in-post/post-blog-torch2.png)

y对x两次求导，第一次保留计算图，第二次不保留。第二期求导后，梯度变为两次梯度的和，也就是8.2+8.2



## 1.0 线性模型与梯度下降

最简单的线性模型$y=x*w+b$，计算误差函数为$\frac{1}{n}\sum^n_{i=1}(\widehat{y}_i-y_i)^2$

数分里都学过梯度的意义在于，沿着梯度函数变化最快，为了尽快找到误差的最小值，需要沿着梯度方向更新$w,b$，二者的梯度分别为

$$\frac{\partial}{\partial w}=\frac{2}{n}\sum^n_{i=1}x_i(wx_i+b-y_i)$$

$$\frac{\partial}{\partial b}=\frac{2}{n}\sum^n_{i=1}(wx_i+b-y_i)$$

```Python
# 多次更新参数
for e in range(10): # 进行 10 次更新
    y_ = linear_model(x_train)
    loss = get_loss(y_, y_train)
    
    w.grad.zero_() # 记得归零梯度
    b.grad.zero_() # 记得归零梯度
    loss.backward()
    
    w.data = w.data - 1e-2 * w.grad.data # 更新 w
    b.data = b.data - 1e-2 * b.grad.data # 更新 b 
    print('epoch: {}, loss: {}'.format(e, loss.data))
```

需要注意的是，每一次求梯度的时候，需要手动设置 **梯度归零**，因为pytorch默认对梯度进行累加。（但是每次bp后都会将计算图丢弃，为什么循环里没有丢弃？）

code 见 [sec1.0.py](https://github.com/miyunluo/pytorch-beginner/blob/master/basics/sec1.0.py)

## 1.1 逻辑回归

Logistic 回归处理的是一个分类问题 (二分类)，形式与线性回归一样，都是 $y=wx+b$，但是它使用 Sigmod 函数将结果变到 0 ~ 1 之间。对于任意输入一个数据，经过 Sigmoid 之后的结果记为 $\widehat{y}$

+ 损失函数

$$loss=-(ylog(\widehat{y})+(1-y)log(1-\widehat{y}))$$

$y$ 表示真实 label，取值 {0, 1}。如果 $y$ 是 0，表示属于第一类，则希望 $\widehat{y}$ 越小越好，这时 $loss$ 函数为，$loss=-(log(1-\widehat{y}))$，根据函数单调性，最小化 $loss$ 也就是最小化 $\widehat{y}$，与要求一致。

如果 $y$ 是 1，表示属于第二类，则希望 $\widehat{y}$ 越大越好，这时 $loss$ 函数为，$loss=-(log(\widehat{y}))$，最小化 $loss$ 是最大化 $\widehat{y}$。

使用 `torch.optim` 更新参数，需要配合另一个数据类型 `nn.Parameter` Parameter 默认要求梯度。

pytorch 提供了 `nn.BCEWithLogitsLoss()`，将 sigmoid 和 loss 写在了一起，直接使用

```python
for e in range(1000):
    # 前向传播
    y_pred = logistic_reg(x_data)
    loss = criterion(y_pred, y_data)
    # 反向传播
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
```

与前面一样，每次bp之前需要梯度归零，code见 [sec1.1.py](https://github.com/miyunluo/pytorch-beginner/blob/master/basics/sec1.1.py)

一般套路是 定义一个criterion，一个optimizer，然后bp的时候先optimizer.zero_grad()，然后loss.backward()，optimizer.step()



## 1.2 MLP, Sequential, Module

**MLP**

神经网络使用的激活函数都是非线性的，sigmode函数 $\sigma(x)=\frac{1}{1+e^{-x}}$，tanh函数 $tanh(x)=2\sigma(2x)-1$，ReLU函数 $ReLU(x)=max(0,x)$。其中ReLU使用的较多，因为计算简单，可以加速梯度下降的收敛速度。

+ 为什么要使用激活函数

  假设一个两层的网络，不使用激活函数，结构为$y=w_2(w_1x)=(w_2w_1)x=wx$，实际和一个一层的网络一样。

  通过使用激活函数，网络可以通过改变权重实现任意形状，成为非线性分类器。

**Sequential**

Sequential 用来构建序列化模型

```python
# sequential
seq_net = nn.Sequential(
	nn.Linear(2,4),
    nn.Tanh(),
    nn.Linear(4,1)
)
# 序列化模型可以通过索引访问每一层
seq_net[0]
# 打印第一层权重
w0 = seq_net[0].weight
print(w0)
# 通过parameters获得模型参数
param = seq_net.parameters()
# 定义优化器
optim = torch.optim.SGD(param, 1.)
```

保存模型

```python
# 将参数和模型保存在一起
torch.save(seq_net, 'save_seq_net.pth')
# 读取保存的模型
seq_net1 = torch.load('save_seq_net.pth')
```

```python
# 只保存模型参数
torch.save(seq_net.state_dict(), 'save_seq_net_params.pth')
# 要重新读入模型的参数，首先我们需要重新定义一次模型，接着重新读入参数
seq_net2 = nn.Sequential(
    nn.Linear(2, 4),
    nn.Tanh(),
    nn.Linear(4, 1)
)
seq_net2.load_state_dict(torch.load('save_seq_net_params.pth'))
```

**Module**

```python
# module模板
class 网络名字(nn.Module):
    def __init__(self, 一些定义的参数):
        super(网络名字, self).__init__()
        self.layer1 = nn.Linear(num_input, num_hidden)
        self.layer2 = nn.Sequential(...)
        ...

        定义需要用的网络层

    def forward(self, x): # 定义前向传播
        x1 = self.layer1(x)
        x2 = self.layer2(x)
        x = x1 + x2
        ...
        return x
    
# 访问模型中的某层可以直接通过名字
l1 = 网络名字.layer1
```

## 1.3 多分类网络

**softmax**

设多分类网络输出为 $z_1,z_2,…,z_k$，首先取指数变成 $e^{z_1},e^{z_2},…,e^{z_k}$，每一项都除以求和

$$\frac{e^{z_i}}{\sum^k_{j=1}e^{z_j}}$$

于是所有项求和等于1，每一项分别表示其中某一种的概率。

**交叉熵**

多分类问题使用更加复杂的loss函数，交叉熵。

$$cross\_entropy(p,q)=E_p\[-logq\]=-\frac{1}{m}p(x)logq(x)$$

这里需要back up一下。所谓 **熵** 是用来反应信息量的。

+ 信息量

  假设 $X$ 是一个离散随机变量，概率分布为 $p(x)=Pr(X=x)$，事件 $X=x_0$的信息量为 $I(x_0)=-log(p(x_0))$。从表达式可以看出，概率越大，信息量越小。概率越大，大家越会认为这一事件的发生几乎是确定的，这一事件的发生并不会引入太多信息量。

+ 熵

  假设李华考试结果为一个0-1分布，{0不及格，1及格}，根据先验知识，由于李华不好好学习，考试及格(记为事件A)概率是$p(x_A)=0.1$，不及格的概率是0.9，则所有可能结果带来的额外信息的期望为

  $$H_A(x)=-[p(x_A)log(p(x_A))+(1-P(x_A))log(1-p(x_A))]=0.469$$

  如果还有一个张华成绩不好不坏，考试及格(记为时间B)概率是 $p(x_B)=0.5$

  $$H_B(x)=-[p(x_B)log(p(x_B))+(1-P(x_B))log(1-p(x_B))]=1$$

  说明在成绩出来前，猜出张华的结果比李华要难。熵越大，变量取值越不稳定。

+ 交叉熵

  两个分布$p,q$，$CEH(p,q)=E_p\[-logq\]$



对于二分类问题，loss可以写为

$$-\frac{1}{m}\sum^m_{i=1}(y^ilog\, sigmod(x^i)+(1-y^i)log(1-sigmod(x^i)))$$

```python
# pytorch 提供了交叉熵函数
criterion = nn.CrossEntropyLoss()
```

## 1.4 参数初始化

首先，使用Sequential构建模型，可以直接访问参数

![](/images/in-post/post-blog-torch3.png)

那么就可以直接从numpy赋值 来初始化参数

![](/images/in-post/post-blog-torch4.png)

使用

```python
net1[0].weight.data = torch.from_numpy(np.random.uniform(3, 5, size=(4, 3)))
```

初始化了第一层的权重，可以看到weight的数值发生了变化，bias没有变。

要对每一个层都初始化，使用循环

```python
for layer in net1:
    if isinstance(layer, nn.Linear): # 判断是否是线性层
        paran_shape = layer.weight.shape
        layer.weight.data = torch.form_numpy(np.random.normal(0, 0.5, size=param_shape))
        # 正太分布
```



pythorch 提供了 `torch.nn.init` 用于初始化

```python
from torch.nn import init
init.xavier_uniform(net1[0].weight)
```

Xavier初始化方式来源于一篇论文，证明使用这种初始化方法可以使得每层的输出方差尽可能相等


























