+++
title = 'Low Rank Adaptation and all its variansts'
date = 2024-03-08T13:45:00Z

author = ["Yuchen Mao"]

draft = false
math             = 'katex'

+++
**This works is a wrapping up work that conclude all paper around LoRA I've seen to develop SeLoRA. The variants of LoRA included does not include**


# Low Rank Adaptation
[Low Rank Adaptation](https://arxiv.org/abs/2106.09685) (LoRA), a line of research under Parameter efficient Fine-Tuning (PEFT) methods, was initially designed for Large Language Models (LLMs), but has later been popularized by the Stable Diffusion community due to its fascinating ability of efficiently learning a style from only a few images. As a PEFT method, *LoRA* not only comes with the advantage of allowing effcient fine-tuning, but also adds no additional inference time. *LoRA* achieves this by freezing the entire model and injecting trainable low-rank decomposition matrices alongside with each linear layer. This is built upon the hypothesis that the weight update during fine-tuning often exists an **low intrinsic rank**, thus a low-rank decomposition matrices could potentially mimic the weight changes with a few trainable parameters. Mathematically, all the linear weight matrices is replaced by:
$$
W = W_{0} + \frac{\alpha}{r}AB
$$
where $W_{0}$ is the frozen original weight and $A$ and $B$ are the trainable low-rank decomposition matrices. $\alpha$ is a hyperparameter that scales the initial weight to match the magnitude of $W_{0}$, allowing the fine-tuning to be more efficient. Here, given a rank r, we have that $A \in \mathbb{R}^{d_{in}\times r}, B \in \mathbb{R}^{r \times d_{out}}$, where r is explicitly chosen to be small, such that the hypothesis of 'low intrinsic rank' is satisfied. In practice, a rank of $4$ will often be sufficient and since *LoRA* is often only injected to the 'q' and 'k' layer of the Attention layer, the total trainable parameter often only constitute a fraction of the total number of parameters.

Notice that when the fine-tuning is finished, the *LoRA*'s $A, B$ matrices could merge into the original weight by simple matrix product. Now the model looks just like the model went through a full fine-tuning, and thus it have the same inference time as the original model. When the inference is done, the model could again revert to it's original weights by simply subtracting the $AB$ matrix from the modified weight.

{{< figure src="low-rank-adaptation.png" align=center width=300px caption="Fig. 1. *LoRA* when injected into a linear layer. (Image source: [Hu, Edward J., et al. 2021](https://arxiv.org/abs/2106.09685))">}}

## History
Now that we've introduced LoRA, let's briefly delve into the history of Parameter-Efficient Fine-Tunin, toching on the innovative methods that have flourished in its wake. Tracing back the origins of PEFT methods, we first encounter *[BitFit](https://arxiv.org/abs/2106.10199)*, which exclusively fine-tunes the bias term of a pretrained model. Following this, there is *[Scaling & Shifting Your Features](https://arxiv.org/abs/2210.08823)*, where each operation of the pretrained model is scaled and shifted by two weight matrices, described by:
$$
y = \gamma \cdot x + \beta
$$

Subsequently, *[adapter](https://arxiv.org/abs/1902.00751)* methods were introduced, with a structure similar to *LoRA* but is injected after each chosen operations of the model. In contrast to *LoRA*, adapters introduced a non-linear activation function between the two low-rank matrices, preventing the merging of weights but potentially could enhance the performance through the introduced nonlinearity. The adapter method is also widely explore, and could potentially be one of the main priation for *LoRA*.

Another important line of research is prompt tuning, which often involves adjusting the embedding layer. Examples include *[visual prompt tuning](https://arxiv.org/abs/2203.12119)*, *[textual inversion](https://arxiv.org/abs/2208.01618)*, *[dream booth](https://arxiv.org/abs/2208.12242)*, and more. The history of PEFT methods is extensive, and the myriad groundbreaking methods we could mention are non-exhaustive. Here, we've provided just a glimpse by listing a few to offer a taste of the diverse landscape.

## Variants of LoRA

Now that we've introduced *LoRA* and discussed the history of PEFT methods, let's delve into some of the variants of *LoRA*. Essentially, further development of *LoRA* can be categorized into several lines of research:

- Further Reduction of Trainable Parameters
    - Alteration of Product Operation
    - Weight Sharing
    - Quantization

- Rank Pruning 

- Search Free Training (Sepecifically for Rank)

- Faster Inference with Multiple *LoRA* 

Each direction of *LoRA* will be briefly introduced to provide insight.

# Further Reduction of Trainable Parameters

While *LoRA* has already achieved a remarkable reduction in the number of trainable parameters, the pursuit of minimizing trainable parameters continues. Further reduction not only enhances memory efficiency during fine-tuning but also conserves storage space, especially when *LoRA* is employed as a personalization tool. For instance, when one *LoRA*  is stored for each user, the space required for storing every user's *LoRA*  can still be significant. Therefore, to address this challenge , several methods have been proposed.

## Alteration of Product Operation
This type of *LoRA* involves altering the product operation between the two low-rank weight matrices. By doing so, it often enables the multiplied matrices to explain more rank than the normal product operation would allow.

### KronA
Paper: [KronA: Parameter Efficient Tuning with Kronecker Adapter](https://arxiv.org/abs/2212.10650)

*KronA* employs Kronecker product to replace the matrix product. Given $A \in \mathbb{R}^{m \times n}, B \in \mathbb{R}^{p \times q}$ matrices, the kronecker product of them can be expressed as:
$$
A \otimes B = 
\begin{pmatrix}
A_{11} B & \cdots & A_{1n} B \\\\
\vdots   & \ddots & \vdots   \\\\
A_{m1}B  & \cdots & A_{mn}B
\end{pmatrix}
$$
The advantage of kronecker product is that it preserves the rank, so this allows *KronA* to achieve even lower number of trainable parameter with large rank. The paper proposed two type of KronA, namely $KronA^B$ and $KronA^B_{res}$, expressed as:
$$
W_{KronA^B}  = W_{0} + s[A\otimes B] 
$$
and
$$
W_{KronA^B_{res}} = W_{0} + s[A \otimes B] + I
$$
where $s$ is a hyperparameter, and $I$ is the identity matirx.
{{< figure src="KronA.png" align=center width=500px caption="Fig. 2. Two variants of KronA when injected into a linear layer. (Image source: [Edalati, Ali, et al. 2022](https://arxiv.org/abs/2212.10650))">}}
### LyCORIS
Paper: [Navigating Text-To-Image Customization: From LyCORIS Fine-Tuning to Model Evaluation](https://arxiv.org/abs/2309.14859)

LyCORIS utilized both the Kronecker product mentioned above and Hadamard product to construct its *LoRA* variants. The Hadamard product is the simple element wise product, so 
$$
A\odot B = \begin{pmatrix}
A_{11} B_{11} & \cdots & A_{1n} B_{1n} \\\\
\vdots   & \ddots & \vdots   \\\\
A_{m1}B_{m1}  & \cdots & A_{mn}B_{mn}
\end{pmatrix}
$$

Althoug Hadamard product seems to be simple, but this allows the upper bound on rank of the product of low rank matrix to increase from $2r$ to $r^2$. Notice that when $r > 2$, we have $2r < r^2$, which implies that the Hadamard product can again allow more rank to be expressed with a even lower number of trainable parmeter.

In the paper, two varainats of *LoRA* is proposed, *LoHA* and *LoKr*, which can be expressed as 
$$
W_{LoHA} = W_{0} + s[B_{1} A_{1}\odot B_{2}A_{2}]
$$
and 
$$
W_{LoKr} = W_{0} + s[C \otimes (BA)]
$$
Here,$A, A_{1}, A_{2}, B, B_{1}, B_{2}$ are all low rank matrices, where $A_{1}, A_{2}, B_{1}, B_{2}$ all have shape defined just like a typical LoRA. For the low rank matrices $A$ and $B$, it have the shape of $A \in \mathbb{R}^{\frac{d_{in}}{f}\times r}, B \in \mathbb{R}^{r \times \frac{d_{out}}{f}}$. where $f$ is a hyperparameter. With $f$ and $r$, the shape of C is defined as
$$
C \in \mathbb{R}^{\max\\{u \leq \min{f, \sqrt{d_{in}}} | d_{in} mod u = 0\\}\times \max\\{u \leq \min{f, \sqrt{d_{out}}} | d_{out} mod u = 0\\}}.
$$
Although it seems really complicated, it is just defining the shape of C to be such that when it is timed with the shape of $AB$, it is equal to the original weihght's shape.
{{< figure src="Lycoris.png" align=center width=800px caption="Fig. 3. LoHA and LoKr (Image source: [Yeh, Shih-Ying, et al.2024](https://arxiv.org/abs/2309.14859))">}}


## Weight Sharing
Weight Sharing reduces the number of parameter by using a weight matrices that is shared across *LoRA* in different layer, and often include a unique scaling or bias weight for each *LoRA*, just like  *[Scaling & Shifting Your Features](https://arxiv.org/abs/2210.08823)*, to allow *LoRA* distinguish itself.

### VeRA
Paper: [VeRA: Vector-based Random Matrix Adaptation](https://arxiv.org/abs/2310.11454)

*VeRA* initialize two **random** low-rank matrices that is shared across layer. Then, two scaling weights are inserted inbetween and after the two matrics. Mathematically, it can be expressed as:
$$
f_{VeRA}(x) = W_{0} x + \Lambda_{b}B\Lambda_{d}A x
$$
Here, $A, B$ are the randomly initialized low-rank matrices, and $\Lambda_{b} , \Lambda_{d}$ are the trainable diagonal weight matrices.

The Goal of *VeRA* is to use the scaling weights to adapt to these random low-rank Matrices, thereby mimic the weight update during fine-tuning. Considering the fact that adpating to random matrices could be hard, the paper proposed that the low-rank doesn't really needs to be low rank, and can be even as large as $256$. This comes from the benefit that the number of parameter doesn't scales up as quickly as the Original *LoRA* when the rank is increased. For each *LoRA* with rank of r, *LoRA* will need $r (d_{in} + d_{out})$ number of trainable parameters, but VeRA only need $r + d_{out}$ number of trainable parameters, which doesn't really differs much when r is increased.
{{< figure src="VeRA.png" align=center width=300px caption="Fig. 4. VeRA when injected into a linear layer. (Image source: [Kopiczko, Dawid J., et al. 2024](https://arxiv.org/abs/2212.10650))">}}

### Tied-LoRA
Paper: [Tied-Lora: Enhacing parameter efficiency of LoRA with weight tying](https://arxiv.org/abs/2311.09578)

Building upon *VeRA*, Tied-LoRA parametized its variant of LoRA as:
$$
W = W_{0} + vBuA
$$ 
where $A, B$ are the low rank weight matrices, and $u, v$ are the scaling weights. Tied-LoRA explored all combinations of applying techniques of **shared weight** and **frozen weight** on $A, B, u, v$. Notice that the original *LoRA* is just when the $u, v$ matrces are frozen and equals to identity matrix. *VeRA* is just when $A, B$ matrices are frozen and random. 

The result shows that the original design of *LoRA* always have the best performance, and LoRA with shared $A, B$ weigths have the secondary performance. More comprehensive result on other combination of **weight sharing** and **frozen weight** can be seen in the paper. 

## Quantization
Paper: [QLoRA: Efficient Finetuning of Quantized LLMs](https://arxiv.org/abs/2305.14314)

*QLoRA* achieves lower memory usage during fine-tuning through quantization. It follows the same structure as *LoRA*, but changed all weights except the original weight to **BF16** data type. For the original weight, it is further quantized into **4-bit NormalFloat** datatype, where **k-bit NormalFloat** data type is proposed in this paper. **k-bit NormalFloat** follows from the assumption that the all pre-trained weight follows a normal distribution, so we can estimate the any number in the original weight with one of the $2^k$ values of $q_{i}$ through 
$$
q_{i} = \frac{1}{2} \\left( Q_{X} \\left( \frac{i}{2^k+1} \\right) + Q_{X}\\left( \frac{i + 1}{2^k+1} \\right) \\right),
$$ 
where $Q_{X}$ is the quantile function of the standard normal distribution. Here, intuitively, we are just splitting the $\mathbb{R}$ line into $2^k$ sections and each section is assigned to a number in k-bit variable. Now, given a number in pre-trained weight that is also in $\mathbb{R}$, we can use a number in k-bit to encode it, and when we are decoding it, the proposed k-bit NormalFLoat Quantizaiton is the method that is the most accurate one.

Additionally, *QLoRA* used double quantization, where the quantization constants of the quantization used to encode the original weight is furthor quantized into k-bit NormalFloat data type.

Finally, *QLoRA* employs paged optimizer to avoid gradient checkpointing memory spikes when a mini-batch contains a large sequence.


# Rank Pruning
Rank Pruninng or Adaptive rank type of *LoRA* operates on the idea that the optimal rank across different layer may be different, and the optimal rank itself is hard to search for. Hence, these methods often start with an initial rank, and prunes the rank during or after training. Hence, it could find the optimal rank more efficeintly, without iteratively training over LoRA with different rank.

## DyLoRA
Paper:[DyLoRA: Parameter Efficient Tuning of Pre-trained Models using Dynamic Search-Free Low-Rank Adaptation](https://arxiv.org/abs/2210.07558)

*DyLoRA* is inspired by *Nested Dropout*, a variant of dropout. At each iteration, *Nested Dropout* samples a number $k$ and drops any element that its index is bigger then $k$. By doing so, *Nested Dropout* encourages the more important information to be kept in the first few indices. *DyLoRA* behaves similar to Nested Dropout, where at each iteration a number $b$ is sampled from $U[0, r]$, the rows and columns of the two low rank decomposition matrices with index larger then b is dropped for this iteration. During the backward propagation, *DyLoRA* introduced two mode:
1. **Frozen Mode**: Druing Frozen mode, only the row and column indexed with $b$ update the gradient, while all other weight are frozen. The author suggests that this could have a minor drawback on performance, but it can greatly boost the efficiency of *DyLoRA*.
2. **Normal Mode**: For Normal Mode, only the parameters that is used in the forward pass receives the gradients.

{{< figure src="DyLoRA.png" align=center width=500px caption="Fig. 5. DyLoRA in its forward and backward pass. (Image source: [Valipour, Mojtaba, et al. 2023](https://arxiv.org/abs/2212.10650))">}}

When the training is finished, *DyLoRA* can keep all important information in columns and rows that have a smaller index. Thus, when searching for the optimal rank, it does not need to re-train the model. Instead it can search the rank by removing the column and rows with largest index, evaluate it, and then repeat this procedure until the rank reaches 0. Then an optimal rank can be find.


## AdaLoRA
Paper:[AdaLoRA: Adaptive Budget Allocation for Parameter-Ef cficient Fine-Tuning](https://arxiv.org/abs/2303.10512)


*AdaLoRA* aims to answer the question of '**How can we allocate the parameter budget adaptively according to importance
of modules to improve the performance of parameter-efficient fine-tuning?**'. To solve this, *AdaLoRA* is initialized with an uniform initial rank, and is gradually pruned to reach the targeted average rank during the fine-tuning. To achive this idea, we have to answer three more question, namely: 'How to Prune?', 'Where to Prune?', and 'When to Prune?'

**How to Prune?**
*AdaLoRA* decomposes the 'low intrinsic rank' weight update matrix in the fashine of singular value decomposition (SVD), where it can be expressed as:
$$
W = W_{0} + A \Lambda B.
$$
$A, B$ are the low rank decomposition matrices as usual, and $\Lambda$ is a diagonal matrix with dimension of $r \times r$. Non-zero elements in $\Lambda$ is referred as singular value. Now, when pruning a rank $i$, the $i$th column of $A$, $i$th singular value, and $i$th row of $B$ are all been dropped.


**Where to Prune?**
To facilitate affective pruning, *AdaLoRA* assigns a importance score to each rank. The importance score can be calculated via:
1. **Magnitude of Singular Value**: This approach just takes the absolute of the singular values, and the smallest one out of all will be pruned. The paper suggested that this approach could minimize the deviation from the original matrix and further stabilize the training.
2. **Sensitive-based Importance**: For each rank $i$, the score is defined as
$$
S_{i} = s(\Lambda_{i}) + \frac{1}{d_{in}}\sum_{j = 0}^{d_{in}} A_{ji} + \frac{1}{d_{out}}\sum_{j = 0}^{d_{out}} B_{ij},
$$
where s is the importance score for each element of the weight matrices, and is defined as:
$$
s(w_{ij}) = |w_{ij}\Delta_{w_{ij}}\mathcal{L}|.
$$
Now, when the importance score is below a threshould, the corresponding rank will be pruned.

**When to Prune?**
The algorithm first starts with a few warmup rounds, and the rank is pruned to the targeted average rank through cubic schedule. After reaching the targeted average rank, the model is fine-tuned for another few rounds to make the model stabilize.


## SoRA
Paper : [Sparse Low-rank Adaptation of Pre-trained Language Models](https://arxiv.org/abs/2311.11696)

[Note]: Although termed with SoRA, this paper is not the viral text-to-vedio SoRA model from OpenAI. 

SoRA adds a gated vector in between the two low rank decomposition matrices, such that it forces the latent representation to be sparse. It is similar to AdaLoRA's formulation, but here the singular value is replaced by gated vector, and gated vector have its own update rule. 

Given a gated vector $g$, SoRA can be expressed as:
$$
W_{SoRA}= W_{0} + B(g\odot A).
$$
Notice that if we diagonalize $g$ it will just be the same as the singular value matrices $\Lambda$ in AdaLoRA. 

The gated vector $g$ have its own unique update rule, expressed as:

$$ 
g_{t+1} \leftarrow \mathcal{T_{\eta_{t} \cdot \lambda}} (g_{t} - \eta_{t} \Delta_{g}\mathcal{L_{0}}),
$$
where $T_{\eta_{t} \cdot \lambda}(\cdot)$ is a element wise soft threshould function:
$$
\mathcal{T_{\xi}}(x) = \begin{cases}
x - \xi , \quad x > \xi \\\\
0 , \quad -\xi < x \leq \xi \\\\
x + \xi, x \leq - \xi
\end{cases}
$$
where $\xi = \eta_{t} \cdot \lambda$, $\eta_{t}$ is the step-size in any epoch, and $\lambda$ is a hyper parameter promoting sparsity.

After the training is complete, all the columns and rows of A, B aligned with the zeros element in the gated vector is pruned for efficiency.

