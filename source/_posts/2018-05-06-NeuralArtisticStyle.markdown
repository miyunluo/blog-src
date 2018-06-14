---
layout:     post
title:      "A Neural Algorithm of Artistic Style"
tags:
    - Python
---

> “Deep style transfer”

大概是两年前，有一篇照片风格迁移的论文火的不得了，名字记不得了，文中的例子是将图片迁移为梵高风格的画风，当时觉得这东西很厉害，而且还有一个叫Prisma的应用在移动端实现了图片的风格迁移，各种画风随意生成，也是厉害的不行。

今天这篇论文应该算是此类问题的开篇鼻祖，看完觉得基础原理其实也挺简单的。

## Overview

所谓风格迁移，就是一张图片A作为输入，一张图片B作为风格，比如梵高的抽象油画，然后生成一张图片C，就好像C是A根据B画出来的一样。

也就是说图片C看起来既像A又像B，根据这句朴素的话语，我们可以引出这篇论文的算法。我们定义两个距离，一个称作内容(content)距离 $D_c$，一个称为风格(style)距离 $D_s$，内容距离用来衡量图片C和内容A的差异，风格距离用来衡量图片C和风格B的差异。我们的最终目的是希望生成的C可以同时最小化 $D_c$ 和 $D_s$，这样就可以保证C在内容上和A接近同时风格上又和B接近。

## Content and Style Distance

记卷积网络为 $C_{nn}$，输入图像为 $X$，则 $C_{nn}(X)$ 表示网络的输入为 $X$。记以 $X$ 为输入，第 $L$ 层的feature map为 $F_{XL}$，这里将所有的feature拼接为一个一维向量。

+ **内容距离**

若 $Y$ 是另一个与 $X$ 同大小的输入，则内容距离定义如下：

$$D^L_C(X,Y)=||F_{XL}-F_{YL}||^2=\sum_i(F_{XL}(i)-F_{YL}(i))^2$$

，$F_{XL}(i))$ 表示 $F_{XL}$ 的第 $i$ 个元素。

+ **风格距离**

计算风格距离，首先要定义风格。输入 $X$ 在第 $L$ 层的风格由Gram矩阵计算得出。

记第 $L$ 层的feature map个数为 $K$，$F^k_{XL}$ 表示第$k$个feature map的向量化表示，则Gram矩阵的大小为 $K*K$，且

$$G_{XL}(k,l)=<F^k_{XL},F^l_{XL}>=\sum_iF^k_{XL}(i)*F^l_{XL}(i)$$

也就是说$G_{XL}$ 是第$L$层feature map的相关矩阵。

接下来是计算风格距离。记 $Y$ 是另一张任意大小的图片，第 $L$ 层的风格距离定义为：

$$D^L_S(X,Y)=||G_{XL}-G_{YL}||^2=\sum_{k,l}(G_{XL}(k,l)-G_{YL}(k,l))^2$$

## Train

作者采用的是VGG19 (16 Conv + 5 Pooling) 模型。当然Pytorch提供了VGG19的模型，直接下载使用就可以了。

VGG19有19层，都进行上述计算的话，计算量是很大的，我们可以选择在想要的层计算上述两个距离。

$$\nabla_{total}(X,C,S)=\sum_{L_C}w_{CL_C}\nabla_{content}^{L_C}(X,C)+\sum_{L_S}w_{SL_S}\nabla^{L_S}_{style}(X,S)$$

$L_C$ 和 $L_S$分别是想要计算内容距离和风格距离的层，$w_{CL_C}$和 $w_{SL_S}$是所对应层的参数。然后对 $X$ 进行梯度下降

$$X \leftarrow X-\alpha \nabla_{total}(X,C,S)$$

## Result

CPU真的跑的好慢啊…(ಥ_ಥ) 

| ![Pic](/images/in-post/post-blog-DNNartstyle3.jpg) | ![Pic](/images/in-post/post-blog-DNNartstyle1.png) | ![Pic](/images/in-post/post-blog-DNNartstyle4.png) |
| -------------------------------------------------- | -------------------------------------------------- | -------------------------------------------------- |
| ![Pic](/images/in-post/post-blog-DNNartstyle0.jpg) | ![Pic](/images/in-post/post-blog-DNNartstyle1.png) | ![Pic](/images/in-post/post-blog-DNNartstyle2.png) |
| ![Pic](/images/in-post/post-blog-DNNartstyle0.jpg) | ![Pic](/images/in-post/post-blog-DNNartstyle5.jpg) | ![Pic](/images/in-post/post-blog-DNNartstyle6.png) |

用于训练图片源于网络，侵删

## Code

[github](https://github.com/miyunluo/pytorch-beginner/tree/master/neural-artistic-style)