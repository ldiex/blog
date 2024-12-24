+++
title = 'More on Fisher Information'
date = 2024-12-24T12:07:07+08:00
+++

# What's Fisher Information?
![](/blog/Fisher-Information.jpg)
早上看到这张讲Fisher信息量的图, 感觉这个解释视角非常生动直观, 但是仔细一看好像又有一些小问题. 想到过一周也要考概统了, 就顺便来复习一下这个知识点.

我们回忆**一元参数**情况下*Fisher信息量*的定义为
$$
I(\theta) = \mathrm{E}\left[ \left( \frac{ \partial \log f(x; \theta) }{ \partial \theta } \right)^{2}   \right] 
$$
如果这个PDF$f(x;\theta)$的性质足够好, 使得我们能交换积分和求导的顺序, 那么就有Fisher信息量的等价定义
$$
I(\theta) = - \mathrm{E} \left[ \frac{ \partial^{2}  }{ \partial \theta^{2} } \log f(x;\theta )  \right] 
$$
一般来说, 我们碰到的绝大部分的$f(x;\theta)$都满足这个性质, 而且求完二阶导后通常比较好算, 所以一般选取这个形式的Fisher信息量.

不过图中所用的Fisher信息量实际上是我们从某个样本$x$**观察到**的Fisher信息量, 这个*Observed Fisher Information*定义为
$$
\mathcal I_x(\theta) = - \frac{ \partial^{2} }{ \partial \theta^{2}}  \log f(x;\theta)
$$
也就是说, 我们之前定义的Fisher信息量(Expected Fisher Information)$I(\theta)$是对不同样本$x$**观察**到的$\mathcal I_x(\theta)$求期望得到的. (所以求$I(\theta)$的时候注意不要跟求MLE的过程搞混, $I(\theta)$是一个理论量, 不应该包含实际的样本)

现在我们可以来看这个图, 它给出了$\mathcal I_x(\theta)$的一个直观解释: 对于给定的样本$x$, 我们得到的分布$f(x;\theta \mid x) = f_x(\theta)$只是$\theta$的函数, 它对应的似然函数(likelihood)$\ell_x(\theta)$也只是$\theta$的函数. 似然函数达到最大值的$\hat{\theta}_{\mathrm{MLE}}$就是$\theta$的一个极大似然估计(MLE). 图中给出了这样一个结论
- Fisher信息量$\mathcal I_x(\theta)$表示$\hat{\theta}_{\mathrm{MLE}}$处$\ell_x(\theta)$的曲率, 也就是该处曲率圆半径的导数
- 曲率圆的半径对应在大样本量下$\hat{\theta}_{\mathrm{MLE}}$的方差, 也就是这个估计量的*有效性*(回忆[C-R不等式](https://ldiex.github.io/quartz/Academic-Notes/Mathematics/Uniformly-Minimum-Variance-Unbiased-Estimate,-UMVUE)?)

简单来说就是
- Fisher信息量越大, 似然函数曲线顶点的曲率半径越小, 曲线越尖锐, MLE估计越有效
- Fisher信息量越小, 似然函数曲线顶点的曲率半径越大, 曲线越平缓, MLE估计越不准确

这也直观理解了Fisher信息量为什么代表了数据中"信息"的多少. 更大的Fisher信息量意味着似然函数的导数, 也就是*得分函数(score function)* 变化得越快, 这个时候似然函数对参数的敏感性更高, 参数的轻微变化会导致似然发生较大变化, 这就表明数据对最大似然估计值的偏好更强. 这个意思是, 如果Fisher信息量比较高, 数据就能够非常精确地指出参数的位置. 当你能够"精确锁定"某个参数时, 不确定性就更低了, 那么这个数据含有的信息就越高. 正如Shannon所言: 

> 凡是在一种情况下能减少不确定性的任何事物都叫信息.  -- Claude Shannon

# More on Fisher Information
我第一次接触到Fisher还是在学习[DDPM](https://ldiex.github.io/quartz/Academic-Notes/Machine-Learning/Denoising-Diffusion-Probabilistic-Models-(DDPM))的时候, 暑假的时候对这玩意还是一知半解, 结果一学期过去了还是一知半解(摊手). 但是这不妨碍我们再小试牛刀地讨论一下Fisher Information在Diffusion模型中的形式

在Score Matching的范式下, DDPM的目标是优化这个*Fisher Divergence*
{{< katex >}}
$$
\mathbb{E}_{p(\boldsymbol x)} \left[ \Vert \boldsymbol s_{\boldsymbol \theta}(\boldsymbol x) - \nabla \log p(\boldsymbol x) \Vert^{2} \right] 
$$
{{< /katex >}}
也就是说我们要训练一个函数去拟合实际分布的得分函数, 为了简单起见我们只考虑一维的情况, 这时候的Fisher散度变成

{{< katex >}}
$$
\mathbb{E}_{p(x)} \left[ \left( s_\theta(x) - \frac{\partial}{\partial x} \log p(x) \right)^2 \right]
$$

{{< /katex >}}
直接展开

{{< katex >}}
$$
\begin{aligned}
\mathbb{E}_{p(x)} \left[ \left( s_\theta(x) - \frac{\partial}{\partial x} \log p(x) \right)^2 \right] &=
\mathbb{E}_{p(x)} \left[ s_\theta(x)^2 \right] \\
&\quad- 2 \mathbb{E}_{p(x)} \left[ s_\theta(x) \cdot \frac{\partial}{\partial x} \log p(x) \right] \\
&\quad+ \mathbb{E}_{p(x)} \left[ \left( \frac{\partial}{\partial x} \log p(x) \right)^2 \right].
\end{aligned}
$$
{{< /katex >}}
当我们训练的$s_\theta$足够好的时候, 第二项就可看成$-2I(\theta)$, 那么我们就得到


{{< katex >}}
$$
\mathbb{E}_{p(x)} \left[ \left( s_\theta(x) - \frac{\partial}{\partial x} \log p(x) \right)^2 \right] \approx \mathbb{E}_{p(x)} \left[ s_\theta(x)^2 \right] - I(\theta)
$$
{{< /katex >}}

目标损失包含两项: 得分函数的norm项和Fisher信息量的负项. 最小化损失也就是在最大化Fisher信息量

在DDPM中, 为了让反向扩散过程有效工作, 我们需要一个稳定且准确的得分函数. Fisher信息量隐式地惩罚分布中"平坦区域", 并鼓励更清晰的分布转变. 某一点的得分函数具有高 Fisher信息量, 就意味着概率分布在该点附近高度集中, 这使我们能够在对应的噪声水平下做出精确预测.






