## Controllable Artistic Text Style Transfer via Shape-Matching GAN

### Author

Shuai Yang, Zhangyang Wang, Zhaowen Wang, Ning Xu, Jiaying Liu and Zongming Guo

### Abstract

艺术字的风格转移的主要任务就是把一张source image的style转移到文字上从而形成艺术字。最近几年的风格转移方法都是在考虑texture control 从而提高使用性。然而呢，在艺术字的艺术程度上，就形变程度来说，仍旧是一个问题。这篇论文里面呢，就作者就提出了第一个文字风格转移network，这个network允许实时控制一些参数来控制艺术字的艺术程度。作者的主要idea就是bidirectional shape matching framework, 也就是双向的形状匹配。用这个framework来建立一个有效的字形风格匹配，在不同的形变level，还不用去看，去匹配真实值。基于这个主意，作者就提出scale-controllable module 来使得a single network持续地获取style image的形状features，然后把这些features转移到文字中。作者提出的这个方法在多样性上，可控性上，和高质量styled上，都比以前的方法好。

### 1	Introduction

艺术字的生成主要是把文字具象成为某一种参考的图像的style，一般应用在视觉效果上，比如说poster，广告设计啊等。取决于参考图片，文字呢，可以styled 成已经存在设计好的文字效果，或者是模仿图片中的feature，从而生成更大自由度的图片，后者更具有灵活性和创意性。

对于大多数的参考图来说，因为文字跟自然图片不一样，因为更加的structured，所以，对于文字来说，我们需要注重在文字的stroke shape。举个例子来说，我们需要通过操作风格程度或者是说文字风格形变度来得到类似的火焰字，看下面这张图。

<img src="https://user-images.githubusercontent.com/68700549/117521009-7c861e00-af79-11eb-9cbe-c0aafb73d111.png" alt="WeChat Screenshot_20210507211611" style="zoom:50%;" />

同时呢，在制造艺术性的同时，我们也得去维持文字的可读性，这种平衡是比较难自动做到的。因此呢，一个实用的方法就是用一个数值去控制这种平衡。还有就是users喜欢去尝试使用不同的背景来获得不同的效果，所以，实时的渲染与调整也是很重要的。

在以前的文献中，有些人已经解决了*fast scale-controllable style transfer*. 但是呢，他们比较focus on在图片的texture

比如说texture length，或者是texture pattern的size，把这些feature都放在forward network。但是呢，还没有人在做实时的文字渲染，对于文字风格转移，这似乎更为重要。

作者的主要任务就是实时对文字的艺术性进行调整。这可以让我们去挑选我们想要的效果。这个fast controllable artistic text style transfer的挑战有两个方面，一是，不跟以前一样，可以使用已经定义好的scale，比如说texture strength那些，都可以用超参数来调整，然而，文字变形程度会比较敏感，不能直接定义，使用参数也不是非常容易。二是不存在大量的训练集，就是已经相对应的变形程度的文字且已经有style transfer的图片。普遍来说，我们就有一张reference image是可以用的，就是某种style的image。所以，对于data-driven的model来说，是远远不够的。

这里呢，作者就提出一种新的方法来解决训练集不够的困难，方法就是shape-matching GAN。Idea就是双向形状匹配来建立起在source style and target glyphs之间的风格转移，双向就有forward and backward transfer。作者先展示出glyph deformation是可以用style image的粗细映射来建模，也就是文字形变可以用style image的mapping的粗细程度来决定。基于这个idea，作者就弄出了sketch module，这个module的作用就是简化这个style images到不同的粗细levels，方法就是backward transferring，transfer the features from text to style image, backward就是text->style。这个作用的结果就是会有更多的coarse-fine image pairs来作为我们的训练集。拿到这个训练集之后，作者就build scale-controllable module，Controllable ResBlock，这个module的作用就是使得network能够学到并且推断style features，应用在不同scale之中。最终呢，我们就可以把这些学到的features forward transfer到不同的scale的文字中，实现了scale-controllable style transfer。做个小总结

1.	对于fast controllable artistic text style transfer，提出新问题，就是文字形变程度，提出新的方法来解决，就是bidirectional shape matching framework。
2.	弄出了sketch module，把从backward transfer，从文字那里学过来的shape features map到style中，这样子就使得a single style有了多个在不同程度的文字形变上有了很多的训练数据，因此呢，也加强了glyph-style mappings
3.	作者提出了Shape-Matching GAN来制作这个text style，而且parameters还是可控的（用一个scale-controllable module），使得文字具有艺术多样性，还是实时的。

### 2	Related Work

#### Image style transfer



#### Artistic text style transfer



#### Multi-scale style control



### 3	Problem Overview

这里呢，我们举个maple style的例子，想想看，如果我们的艺术字仅仅只有maple leaves的texture，没有leaves-like shape的话，就会看起来很奇怪。我们可以看看下面这张图。这也就说明了我们需要进行shape deformation，而不仅仅只是transfer texture patterns。

![WeChat Screenshot_20210508145137](https://user-images.githubusercontent.com/68700549/117550193-f3202b80-b00c-11eb-9148-cadd4bf516da.png)

同时呢，在平衡可读性与艺术性时，这也取决于不同的文字与style image， （not to mention）更不用说人与人之间的主管差异了。看看下面这张图，我们可以看到如果笔画越复杂，那么形变就会对它影响越大。所以呢，用户就会更倾向于使用可以自由调节文字形变程度的工具，也不用说要去对每一个scale再去train一遍。简单来说，一个controllable artistic text style transfer会保证以下两点

1. artistry： 就是艺术性，就会生成文字的艺术性应该要模仿我们给出style image的features
2. controllability：就是可控性，形成程度的可控性

所以说，解决了这两个问题，作者的方法就比之前的人做的会更好，之前要么不能adjust the shape，要么不能不能做的很快。

作者提出的思路就是novel bidirectional shape matching strategy. 我们看看下面这张图，就是bidirectional 的简单介绍。先看左边的四幅图，就是backward的过程，左上角的那张图就是source image （这里的例子是maple leaves），通过backward的过程，从文字到source image，通过不同程度简化，得到不同程度coarse levels。左边的其他三张图就是不同程度的简化版本。接下来的步骤就是把通过backward得到的stylish images (这里就包含了shape characteristics of the text) forward到text中，text中又有不同level的形变程度，形变越大，就越coarser，我们看看右边的六张图，上面的三张表示的是三个差不多的stroke，下面表示的是不同程度的形变。从左到右是越来越大的形变，越大的形变，那么得到的图就是一定是越参差，崎岖不平的。

<img src="https://user-images.githubusercontent.com/68700549/117550859-a2aacd00-b010-11eb-9168-f7d0558ea319.png" alt="WeChat Screenshot_20210508151811" style="zoom:50%;" />

Artistry的目标就已经做到了，通过这些不同level的形变，controllability 也能通过forward network来达到。

做个小总结，作者就是已经做到了把style image map到不同的coarse levels (就是 forward)。但是呢，现在依旧面临两个问题，一是怎么做这个simplify的过程（backward）。二是怎么做多对一的匹配（就是backward会生成不同level 的 simplified的图，怎么匹配到单个text图中）又要保证model 不会collapse。下一个section就会讲述作者是怎么解决这个问题的。

### 4	Shape-Matching GAN

假使$Y$ and $I$ 分别表示style image and text image。作者去设计一个feed-forward stylization model $G$, 来渲染在不同形变程度的艺术字，形变程度有参数$l\in[0,1]$ 来表示,$l$的值越大，表示形变程度越大。作者把style transfer的步骤分为两个连续步骤，分别是structure transfer and texture transfer，用generator model $G_S$ and $G_T$来表示。分开这两个步骤的原因是可以分离texture对结果的影响，可以先focus on在形变问题上。作者把这个过程表示为$G=G_T \circ G_S$，用公式来表示就是

$I_l^Y = G_T(G_S(I,l)),I_l^Y \sim p(I_l^Y | I,Y,l)$, where the target statistic $p(I_l^Y)$ of stylized image $I_l^Y$ is characterized by the text image $I$, the style image $Y$ and the controllable parameter $l$.

作者提出的算法的整体的过程就是backward transfer (backward structure transfer)->forward transfer(structure transfer and texture transfer). 

下面这张图就是bidirectional shape matching framework，当我们拿到一张source image $Y$之后，通过PS或者matting算法之类的，取得structure $X$ 这张图，然后通过sketch module $G_B$, 得到一些不同形变的training pairs {${\tilde{X_l},X}$}, 这个$\tilde{X_l}$ 不同coarse version的图且包含着有来自text shape的信息。在forward的过程中，$G_S$ 就会把这里{${\tilde{X_l},X}$}学到的map到glyph with various deformation degrees.  

<img src="https://user-images.githubusercontent.com/68700549/117552389-ffaa8100-b018-11eb-8622-3854ec730457.png" alt="WeChat Screenshot_20210508161759" style="zoom: 67%;" />

Forward中有两个重要的components

1. Glyph network $G_S$: 在训练过程中，把$\tilde{X_l}$ （backward过来的图，有着各种不同的形变程度）map 到 $X$中。 在test过程中，把$X$ 的shape feature target text image $I$ 中，生成structure transfer result $I_l^X$. 
2. Texture network $G_T$: 把source image $Y$的texture渲染到由$G_S$生成的图 $I_l^X$, 从而生成最终艺术字 $I_l^Y$.

因为使用的是GAN，所以，有generator ($G_S, G_T$), 就会有discriminator ($D_S, D_T$), 这个discriminator 会帮助提升我们结果的质量，也就是adversarial learning. 接下来呢，作者就会详细地介绍bidirectional shape matching，还有controllable module (enables $G_S$ to to learn multi-scale glyph deformations). 还会介绍texture network。

#### 4.1	Bidirectional Structure Transfer ($G_S$)

##### Backward structure transfer

为了把文字的shape characteristics transfer到 $X$ 中，还得带有不同的coarse levels，作者提出的sketch module $G_B$ 包含着一个smoothness block and a transformation block. 下面这张图，就是sketch module的整体过程。

<img src="https://user-images.githubusercontent.com/68700549/117553160-e3f5a980-b01d-11eb-8fcc-4119cde14273.png" alt="WeChat Screenshot_20210508165305" style="zoom:67%;" />

Smoothness 的过程还是使用的是Gaussian kernel 作为一个convolutional layer，然后standard deviation就是$\sigma= f(l)$, $f$ is a linear function, $l$ is the deformation parameter. 作者的key idea就是连接source style domain and the target text domain，这样子的话，structure $X$ 和 text image $I$ 就会在相同的smooth domain。两张图的一些细节都会被消除，然后两张图的轮廓表示着相似的smoothness。 Backward structure transfer最终就是通过transformation block来实现的，把已经smoothed text images map back to the text domain，从而学习glyph characteristics。 作者提出的sketch module有两个优点，一是这个coarse level (deformation level) 可以很自然地用Gaussian parameter $\sigma$ 来表示。二是，$G_B$ 的training process仅仅只是需要一些简单的text images来做。一旦训练好后，就可以在在任意的text images使用了。

For training $G_B$, 作者随机地拿出一个text image $t$ from the text dataset, 然后取 parameter $l$ from [0,1]. $G_B$的作用就是生成其他的$t$, $G_B$ is tasked to reconstruct $t$: 

$L_B^{rec}=\mathbb{E}_{t,l}[||G_B(t,l)-t||_1]$. 

此外，作者还采用了conditional adversarial loss使$G_B$生成更多text-like的图片. 

$L_B^{adv}=\mathbb{E}_{t,l}[log D_B(t,l,\bar{t}_l)]+\mathbb{E}[log(1-D_B(G_B(t,l),l,\bar{t}_l))]$. $D_B$的作用就是决定input image的正确性，（1代表real image， 0 代表fake image），还决定是否match到给定的smoothed image $\bar{t}_l$ and the parameter $l$. 所以，我们的loss就是minmax function

$min_{G_B}max_{D_B}\lambda _B^{adv} L_B^{adv}+\lambda _B^{rec} L_B^{rec}$.

这些loss是在课上讲过的，以后还会出一篇DCGAN的论文。

最后呢，就把trained $G_B$ 用在X中 with various level $l$, 然后我们就可以得到sketchy shape $\tilde{X}_l=G_B(X,l)$. 我们看看下面这张图，我们会看到有两个对比，一个是用sigmoid的，一个是用transformation block的，$\tilde{X}_l$使用transformation block得到的，我们可以看到红色框框的图，就有 $t$ 的印记。使用transformation block会使network更加robust。

<img src="https://user-images.githubusercontent.com/68700549/117554317-e60f3680-b024-11eb-886c-0363a26157e7.png" alt="WeChat Screenshot_20210508174312" style="zoom:67%;" />

##### Forward structure transfer

当我们通过前面的backward structure transfer之后，我们会拿到${\tilde{X}_l},l\in [0,1]$, 我们就可以train这个glyph network $G_S$了，就是要把${\tilde{X}_l}$ map back to original $X$ 从而得到$X$的shape features，然后再把这些shape features transfer到text image中。我们要记住，map到$X$ 的过程是一个many to one的过程，我们只有一个$X$. 所以我们要非常小心地去设计我们的network，要不然model仅仅只是记住了ground truth $X$, 到时model collapse。也就是说，不管$l$的值怎么变化，都生成差不多的结果。

为了解决这个问题，作者使用了两个技巧，一个data augmentation,还有一个就是controllable ResBlock。第一步，就是制造一些sub-image pairs,（用剪切的方式）把$X,\tilde{X}_l$ 进行随机组合来得到training set。其次，作者build $G_S$, 这个是根据Feifei Li提出的styleNet进行改造的，作者提出一种简单但是有效的Controllable ResBlock来替换原来的ResBlock。这个ResBlock位于StyleNet的中间层。作者提出的controllable ResBlock是由两个ResBlock组合在一起的，而且是由 $l$ 来衡量的。我们看看下面这张图，左边的就是Feifei Li原来的ResBlock，右边的是作者新提出来的。当$l=1\space or\space 0$, $G_S$的结构就会跟原来的StyleNet是一样的，但是仅仅只是针对形变最大或者是形变最小的情况，这就会避免many-to-one problem. 与此同时，对于$l\in (0,1)$, $G_S$在两个极限值中保持平衡。

<img src="https://user-images.githubusercontent.com/68700549/117556575-5b383700-b038-11eb-91e5-6aa106dcd34d.png" alt="WeChat Screenshot_20210508200231" style="zoom:67%;" />

就loss来说，$G_S$ 的目标就是生成跟$X$相似的图片 in an $L_1$ sense, 并且confuse discriminator $D_S$:

$L_S^{rec}=\mathbb{E}_{x,l}[||G_S(\tilde{x}_l,l)-x||_1]$

$L_S^{adv}=\mathbb{E}_x[logD_S(x)]+\mathbb{E}_{x,l}[log(1-D_S(G_S(\tilde{x}_l,l)))]$.

对于一些style，又放了很大的$l$ 值，这就会导致可读性不高，于是，作者就提供给了一个loss选项，当发生这种情况的时候，用这个loss来保留$G_S(t,l)$ 生成结果的main stroke

$L_S^{gly}=\mathbb{E}_{t,l}[||(G_S(t,l)-t)\otimes M(t)||_1]$

这里$\otimes$表示的是the element-wise multiplication operator.就是进行每个像素的计算。$M(t)$是一个weighting map, 里面的值是要根据距离的大小来计算weight。如果是距离text contour point of t越近，那么这个pixel value就要越大。所以$G_S$的loss就是

$min_{G_S}max_{D_S}\lambda _S^{adv}L_S^{adv}+\lambda _S{rec}L_S^{rec}+\lambda_S^{gly}L_S^{gly}$.

#### 4.2	Texture Transfer ($G_T$)

当我们从$G_B$得到structure transfer的结果后，$I_l^X =G_S(I,l)$,接下来就要做texture transfer，这个方法就是image analogy problem。就是类似映射，such that $X:Y::I_l^X:I_l^Y$. 这个就可以用已经存在的算法来算，比如说greedy-based Image Analogy，或者是the optimization-based Neural Doodle.  作者为了建立一个端对端的 fast text stylization model1，就train了一个feed-forward network $G_T$ 来做texture rendering。 就跟train $G_S$一样，也随机（用剪切的方式，后面有讲）取一些sample {${x,y}$} from $X,Y$. 这样才会有足够的training set。 然后训练$G_T$, 还是使用传统GAN的loss

$L_T^{rec}=\mathbb{E}_{x,y}[||G_T(x)-y||_1]$

$L_T^{adv}=\mathbb{E}_{x,y}[logD_T(x,y)]+\mathbb{E}_{x,y}[log(1-D_T(x,G_T(x)))]$.

在考虑style rendering performance的时候，也要把style loss，$L_T^{style}$给考虑进去，这是Neural Style Transfer那篇论文提出来的，以后也会做一篇这个论文。所以，最终这个loss就是

$min_{G_T}max_{D_T}\lambda_T^{adv}L_T^{adv}+\lambda_T^{rec}L_T^{rec}+\lambda_{T}^{style}L_T^{style}$.

### 5	Experimental Results

#### 5.1	Implementation Details

##### Network architecture

作者采用的generator是从Encoder-Decoder architecture of StyleNet那里改编过来的，这里有6个ResBlock，除了$G_S$不一样，因为作者自己用了controllable ResBlock。 Discriminator用的也是别人的，是Patch-GAN。 因为structure map包含了狠毒的饱和区域，所以，作者就加了一些gaussian noises 到$G_S$ 和 $G_T$ 的input中。作者也发现，这样会work地更好。

##### Network training

作者随机crop 256*256的sub-images来做训练集。使用Adam optimizer and the learning rate = 0.0002. 为了稳定$G_S$的训练，作者逐渐地增加$G_S$的sample range。具体来说就是$G_S$是先在$l=1$时训练了一会，来学习最大程度的形变。然后把这个训练了一半的参数（放from controllable ResBlock）,用这个参数去继续训练另外的一半$l\in (0,1)$, 通过这样的方式来训练两个极端。最终呢，$G_S$ 在

$l\in${$i/K$}$_{i=0,...,K}$. $K=3$时继续训练，且取得的效果不错。$G_S$已经能够学习中间的范围了。对于linear function to control the standard deviation of the Gaussian kernel is $f(l)=16l+8$. 对于所有的experiments， we set $\lambda_B^{rec}=\lambda_S^{rec}=\lambda_T^{rec}=100,\lambda_B^{adv}=\lambda_T^{adv}=1,\lambda_S^{adv}=0.1,\lambda_T^{style}=0.01$.  



#### 5.2	Comparisons with State-of-the-Art Methods

##### Artistic text style transfer

##### Scale-controllable style transfer

#### 5.3	Ablation Study

##### Network architecture

为了分析我们的model每一个component的作用，做了一些实验

1. baseline： baseline model仅仅只有texture network，直接map $X$ to $Y$.
2. W/o CR: 就是有没有controllable ResBlock。 就是简单版本的glyph network and a texture network. 
3. W/o TN: 就是只有glyph network，没有texture network。直接map $\tilde{X}_l$ to $Y$. 
4. Full model: 齐全。

下面这张图就是结果的比较。我们可以看到如果没有structure transfer，结果图会很僵硬。如果没有CR，全部图都差不多。作者还发现，glyph network能同时做structure transfer and texture transfer，尽管结果会比较假一点。但是如果用了texture transfer效果会好很多。

<img src="https://user-images.githubusercontent.com/68700549/117557829-38604f80-b045-11eb-8bc5-034eb3c4ee7d.png" alt="WeChat Screenshot_20210508213439" style="zoom:67%;" />



##### Sketch module

这里就对比有使用transformation block和没有使用的区别。取消transformation block，就用sigmoid layer来替代。就是看到结果是使用了full model，h底下那个边，会圆点。

<img src="https://user-images.githubusercontent.com/68700549/117557911-08fe1280-b046-11eb-8512-03a4d737d94a.png" alt="WeChat Screenshot_20210508214031" style="zoom:67%;" />

##### Loss function

这里就测试上面所说的对于一些style，当形变很大的时候，可读性会不高，增加的一个loss， $L_S^{gly}$. 我们来看看下图，有加和没加的区别。加了效果会好一点。可以保持artistry and legibility的平衡。这里$l=0.75$.

<img src="https://user-images.githubusercontent.com/68700549/117557949-8de92c00-b046-11eb-9a28-fc823e42d155.png" alt="WeChat Screenshot_20210508214414" style="zoom:67%;" />

#### 5.4	Applications

接下来就是应用。比如说，看下图，海报的制作啊，dynamic typography design啊。

![WeChat Screenshot_20210508214702](https://user-images.githubusercontent.com/68700549/117557995-f20bf000-b046-11eb-8401-c4f7b83c5a97.png)

![WeChat Screenshot_20210508214833](https://user-images.githubusercontent.com/68700549/117558018-2e3f5080-b047-11eb-9e59-5b57555a5eee.png)

作者还提及了另外两个应用

##### Structure/texture mash-up

产生全新的 text styles，比如说，看下面这张图，可以分别使用不同的settings。

![WeChat Screenshot_20210508215015](https://user-images.githubusercontent.com/68700549/117558033-66df2a00-b047-11eb-9546-56000ef7f0b6.png)

##### Stroke-based art design

可以applied到其他一些简单的图形中，symbols or icons。可以跟人物啊，字体啊进行结合。

<img src="https://user-images.githubusercontent.com/68700549/117558084-b160a680-b047-11eb-9c2d-ea75a9831c4f.png" alt="WeChat Screenshot_20210508215223" style="zoom:67%;" />

###	6	Conclusion

结论就是作者提出了新的方法来渲染文字，可以更快更好。方法就是bidirectional shape matching，在训练过程中，还不惧怕many-to-one的过程。还用ablation study进行了验证。