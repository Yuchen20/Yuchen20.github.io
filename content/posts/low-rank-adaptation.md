+++
title = 'Low Rank Adaptation and all its variansts'
date = 2024-03-08T13:45:00Z

author = ["Yuchen Mao"]

draft = false
math             = 'katex'
ShowToc          = true
ShowBreadCrumbs  = true
ShowPastNavLinks = true
ShowSHareButtons = true
ShowReadingTime  = true
ShowWordCount    = true
+++
**This works is a wrapping up work that conclude all paper around LoRA I've seen to develop SeLoRA. The variants of LoRA included does not include**

# Low Rank Adaptation

[Low Rank Adaptation](https://arxiv.org/abs/2106.09685) (LoRA), initially designed for Large Language Models (LLMs), has later been popularized by the Stable Diffusion community due to its fascinating ability to efficiently learn a style from only a few images. Pre-trained foundational model with billions of parameters 

To adapt these Pre-trained foundational model

LoRA faciliate efficient learning by fine-tuning a pre-trained modle with minimum number of trainable parameters, which is a parameter-efficient fine-tuning (PEFT) methods.To fine

 LoRA is a line of reasearch in parameter-efficient fine-tuning (PEFT) methods characterized by it's ability to fine-tune a pre-trained foundational models with a minimum number of trainable parameters, was developed upon the hypothesis that 'the weight update of fine-tuning a downstream task on a pre-trained model exhibits a **low intrinsic rank**.' To mimic the low-rank weight update while also making the training parameter-efficient, the author of LoRA proposed to freeze the entire pre-trained model and inject trainable low-rank decomposition matrices alongside linear layers expressed as:
$$
W = W_0 + AB
$$ 
where $W_0$ is the frozen pre-trianed weight, $A $ and $B$ are the trainable low rank decomposition matrices. Here, given a rank r, we have that $A \in \mathbb{R}^{d_{in}\times r}, B \in \mathbb{R}^{r \times d_{out}}$, where r is explicitly to be chosen to be small to satisfy the hypothesis of 'low rank'. Notice that when the training is finished and before infering, the LoRA's $A, B$ matrices could merge back to the original weight by simple matrix product and addition, so now when infering, there is no latency introduced by LoRA. When the inference is finished, the mode could revert to it's original weight by simply subtracting the $AB$ matrix from the modified weight.

In practice, the rank of *LoRA* is chosen to be in one of $4, 8, 64$, and is injected alongside with the $q$ and $k$ layer of the attention module of LLMs.

{{< figure src="low-rank-adaptation.png" align=center width=300px >}}

## LoRA's history and branches
## LoRA's History and Branches

Now that we've introduced LoRA, let's briefly delve into the history of Parameter-Efficient Fine-Tuning (PEFT), highlighting the innovative methods that have emerged in its evolution. Tracing back the origins of PEFT methods, we first encounter [BitFit](https://arxiv.org/abs/2106.10199), which exclusively fine-tunes the bias term of a pretrained model. Following this, there is [Scaling & Shifting Your Features](https://arxiv.org/abs/2210.08823), where the output result of each operation of the pretrained model is scaled and shifted by two weight matrices, as described by the equation:
$$
y = \gamma \cdot x + \beta
$$

Subsequently, adapter methods were introduced, with a structure similar to LoRA injected after each selected operation of the model. However, in contrast to LoRA, adapters introduce a non-linear activation function between the two low-rank matrices, preventing the merging of weights but potentially enhancing performance through the introduced nonlinearity. The adapter approach has attracted numerous researchers, leading to the development of various methods within this framework.

Another significant line of research is prompt tuning, which often involves adjusting the embedding layer or introducing a trainable embedding before the feature. Examples include visual prompt tuning, textual inversion, dream booth, and more. The history of PEFT methods is extensive, and the myriad groundbreaking methods we could mention are non-exhaustive. Here, we've provided just a glimpse by listing a few to offer a taste of the diverse landscape.

Now, As we have introduced LoRA itself, I want to briefly touch on the history of Parameter efficeint fine-tuning, toching on the innovative methods that have flourished in its wake.Back tracking the PEFT methods, initially, [BitFit](https://arxiv.org/abs/2106.10199) was introduced, which only tunes the bias term of an pretrained model, then there is the [Scaling & Shifting Your Features](https://arxiv.org/abs/2210.08823) where the output result of each operation of the pretrained model is scaled and shifted by two weight matrices, described as 
$$
y = \gamma \cdot x + \beta
$$
Later on, there was the adapter methods, where a structure just like LoRA was introudced after each operation of the model, however, differ from LoRA, it introudces a non-linear actiavtion funciton inbetween the two low rank matrices, which prevents the mergeing of weight, but possbility improves the performance due to the nonlinearity introduced. Adapter was another line of reaserach that have attracted many researchers to it, and many methods was developed upon it. Another line would be prompt tuning, that often ajust the embedding layer, or introduce a trainable embedding before the fetature, these could be seen from visual prompt tuning, textual inversion, dream booth... The history of PEFT methods is rich, and the eye-brashjing methods that we could mention is non-exhaustive, but here we just only list a few to give a taste. 


Now, we bring back to our focus 'LoRA', and it's variants. In essense, Further development upon LoRA could be categroize into a few lines of research:

- Minimizing the trainable parameters
    - Change Product Operation
    - Quantization
    - Weight Sharing
    - Rank Pruning
- Rank - Search free
- Inference Improvement

Each of the direction of LoRA will be briefly introudced to have a view.

# Minimizing Trainable Parameters
The Goal of minimizing trainable parameter not only leads more efficient training, in terms of memory, but also minimize the storage needed to store each individual LoRA. Designs to reduce trainable parameters often aims to minimize the number of trainabke parameter while keeping the performance the same. 

## Product Rule
### KronA
Paper: [KronA: Parameter Efficient Tuning with Kronecker Adapter](https://arxiv.org/abs/2212.10650)
