---
layout:     post
title:      "pytorch自用笔记"
tags:
    - python
---

> “防止以后踩坑”

第一眼看到pytorch就十分喜欢，就像第一眼看到golang。

pytorch 0.4 发布，api改了。。。

### 0.0 Tensor and Variable

- **Tensor**

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

- **Variable**

  Variable 是对 tensor 的封装，每个 Variabel都有三个属性，Variable 中的 tensor本身`.data`，对应 tensor 的梯度`.grad`以及这个 Variable 是通过什么方式得到的`.grad_fn`

```python
x_tensor = torch.randn(10, 5)
y_tensor = torch.randn(10, 5)

# 将 tensor 变成 Variable
x = Variable(x_tensor, requires_grad=True) 
# 默认 Variable 是不需要求梯度的，所以申明需要对其进行求梯度
y = Variable(y_tensor, requires_grad=True)

z = torch.sum(x + y)
```

```python
# 求 x 和 y 的梯度
z.backward()

print(x.grad)
# Variable containing:
#     1     1     1     1     1
#     1     1     1     1     1
#     1     1     1     1     1
#     1     1     1     1     1
#     1     1     1     1     1
#     1     1     1     1     1
#     1     1     1     1     1
#     1     1     1     1     1
#     1     1     1     1     1
#     1     1     1     1     1
# [torch.FloatTensor of size 10x5]

print(y.grad) 
```

### 0.1 自动求导

- **多次自动求导**

  调用 backward 后自动计算一次导数，再次调用会报错，因为pytorch默认做完一次自动求导后，计算图被丢弃，两次求导需要手动设置。

```python
x = Variable(torch.FloatTensor([3]), requires_grad=True)
y = x * 2 + x ** 2 + 3

y.backward(retain_graph=True)
print(x.grad)
# Variable containing:
#  8
# [torch.FloatTensor of size 1]

y.backward(retain_graph=True)
print(x.grad) 
# 输出 16，因为做了两次自动求导，所以将第一次的梯度 8 和第二次的梯度 8 相加得到 16

y.backward() # 再做一次自动求导，这次不保留计算图
print(x.grad)
# 输出 24

y.backward() # 再做会报错，计算图已经丢弃
```

### 0.2 动态图与静态图

pytorch与python的写法基本一致，没有任何额外的学习成本。tensorflow需要先定义图，然后执行，不能直接使用while，需要使用tf.while_loop，有些反直觉。

### 1.0 线性模型与梯度下降

最简单的线性模型$y=x*w+b$，计算误差函数为$\frac{1}{n}\sum^n_{i=1}(\widehat{y}_i-y_i)^2$

数分里都学过梯度的意义在于，沿着梯度函数变化最快，为了尽快找到误差的最小值，需要沿着梯度方向更新$w,b$，二者的梯度分别为

$$\frac{\partial}{\partial w}=\frac{2}{n}\sum^n_{i=1}x_i(wx_i+b-y_i)$$

$$\frac{\partial}{\partial b}=\frac{2}{n}\sum^n_{i=1}(wx_i+b-y_i)$$

```Python
import torch
import numpy as np
from torch.autograd import Variable
torch.manual_seed(2018)
# 读入数据 x 和 y

# 读入数据 x 和 y
x_train = np.array([...], dtype=np.float32)
y_train = np.array([...], dtype=np.float32)

# 转换成 Tensor
x_train = torch.from_numpy(x_train)
y_train = torch.from_numpy(y_train)
# 定义参数 w 和 b
w = Variable(torch.randn(1), requires_grad=True) # 随机初始化
b = Variable(torch.zeros(1), requires_grad=True) # 使用0进行初始化
# 构建线性回归模型
x_train = Variable(x_train)
y_train = Variable(y_train)
def linear_model(x):
    return x * w + b
# 计算误差
def get_loss(y_, y):
    return torch.mean((y_ - y_train) ** 2)
loss = get_loss(y_, y_train)
# 自动求导
loss.backward()
# 查看 w 和 b 的梯度
print(w.grad)
print(b.grad)
# 更新一次参数
w.data = w.data - 1e-2 * w.grad.data
b.data = b.data - 1e-2 * b.grad.data
```

### 1.1 逻辑回归

Logistic 回归处理的是一个分类问题 (二分类)，形式与线性回归一样，都是 $y=wx+b$，但是它使用 Sigmod 函数将结果变到 0 ~ 1 之间。对于任意输入一个数据，经过 Sigmoid 之后的结果记为 $\widehat{y}$

+ 损失函数

$$loss=-(ylog(\widehat{y})+(1-y)log(1-\widehat{y}))$$

$y$ 表示真实 label，取值 {0, 1}。如果 $y$ 是 0，表示属于第一类，则希望 $\widehat{y}$ 越小越好，这时 $loss$ 函数为，$loss=-(log(1-\widehat{y}))$，根据函数单调性，最小化 $loss$ 也就是最小化 $\widehat{y}$，与要求一致。

如果 $y$ 是 1，表示属于第二类，则希望 $\widehat{y}$ 越大越好，这时 $loss$ 函数为，$loss=-(log(\widehat{y}))$，最小化 $loss$ 是最大化 $\widehat{y}$。



使用 `torch.optim` 更新参数，需要配合另一个数据类型 `nn.Parameter`，本质上与 Variable 一样，但是 Parameter 默认要求梯度。pytorch 也提供了 Sigmode 函数，通过导入 `torch.nn.functional` 使用。

pytorch 提供了 `nn.BCEWithLogitsLoss()`，将 sigmoid 和 loss 写在一层，有更快的速度与稳定性。所以使用它的话，就不需要再定义 Sigmod 函数了。

```python
# 不使用自带的loss
import torch.nn as nn
import torch.nn.functional as F
w = nn.Parameter(torch.randn(2, 1))
b = nn.Parameter(torch.zeros(1))
# 定义sigmod
def logistic_regression(x):
    return F.sigmoid(torch.mm(x, w) + b)
# 定义loss
def binary_loss(y_pred, y):
    logits = (y * y_pred.clamp(1e-12).log() + (1 - y) * (1 - y_pred).clamp(1e-12).log()).mean()
    return -logits
#- - - - - - - - - - - - - - - - - - - -
optimizer = torch.optim.SGD([w, b], lr=1.)
# 前向传播
y_pred = logistic_regression(x_data)
loss = binary_loss(y_pred, y_data) # 计算 loss
# 反向传播
optimizer.zero_grad() # 使用优化器将梯度归 0
loss.backward()
optimizer.step() # 使用优化器来更新参数
##########################################

# 使用自带的loss
import torch.nn as nn
criterion = nn.BCEWithLogitsLoss()
# 使用 torch.optim 更新参数
w = nn.Parameter(torch.randn(2, 1))
b = nn.Parameter(torch.zeros(1))
def logistic_reg(x):
    return torch.mm(x, w) + b
#- - - - - - - - - - - - - - - - - - - -
optimizer = torch.optim.SGD([w, b], lr=1.)
# 前向传播
y_pred = logistic_reg(x_data)
loss = criterion(y_pred, y_data)
# 反向传播
optimizer.zero_grad()
loss.backward()
optimizer.step()
# 计算正确率
mask = y_pred.ge(0.5).float()
acc = (mask == y_data).sum().data[0] / y_data.shape[0]
```

### 1.2 MLP, Sequential, Module

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

### 1.3 多分类网络

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

### 1.4 参数初始化

+ **使用numpy初始化**


