# Seminar Computer Vision by Deep Learning (CS4245)

1) Read [this paper](https://ieeexplore.ieee.org/abstract/document/8489068)
2) Reproduce the aforementioned paper
3) Think about applications in crowd anomaly detection

Citation: <cite> Aytekin, C., Ni, X., Cricri, F., & Aksu, E. (2018, July). Clustering and unsupervised anomaly detection with l 2 normalized deep auto-encoder representations. In 2018 International Joint Conference on Neural Networks (IJCNN) (pp. 1-6). IEEE. <cite>


[USPS dataset](http://ieeexplore.ieee.org/document/291440/)
--- 
## Some takeaways from the paper

### Introduction
* Introduce multiple AE architectures
* Regardless of AE type, L2 normalization works
* Applying it during training improves anomaly detection

### Related work
* Assumption of mentioned work assumes that AE does not learn anomaly data,
  because of the ratio in the training data
* Assumption fails when 'normal' data entails multiple classes
* Proposed: **L2 constraint on activations** (parameter free)
* L2 norm reg is applied to weights, proposed method to activations

### Proposed method

#### Normalization
* **Autoencoder representation is L2 normalized** -> last layer = l2
  normalization layer
* L2 normalized representations should be more suitable for clustering
  purposes, especially when using Euclidean distance (k-means)
* Adding l2 normalizing the representations after training does not work at all
* Clusters become more compact

#### Anomaly detection
1. Train on full dataset, anomalies + 'normal' samples
2. After training the l2 normalized autoencoder representations are clustered
  using k-means
3. Use normality score: ![normality score](http://www.sciweavers.org/upload/Tex2Img_1587649594/render.png) on k-means clusters (C), representations (E(I))
4. Use threshold to separate 'normal' samples from anomalies

### Experimental results
* Evaluation metric: ![accuracy](http://www.sciweavers.org/upload/Tex2Img_1587649539/render.png) simply: the maximum number of correctly classified images according to each of the possible cluster (c) mappings (m) to classes (l). 
* Maximization of the evaluation metric is done according to the Hungarian
  algorithm
* Dense AEs are evaluated on MNIST, convolutional AEs on MNIST, USPS
* Dense AEs used: 
  * [IDEC](https://github.com/XifengGuo/IDEC) & [DEC](https://github.com/XifengGuo/DEC-keras): encoding layers: 500 - 500 - 2000 - 10, decoding layers: 2000 - 500 - 500 - d -> trained 100 epochs
* Convolutional AEs used:
  * [DCEC](https://github.com/XifengGuo/DCEC): encoding: 5x5 (32), 5x5 (64), 3x3 (128). 2x2 stride. decoding: 3x3
(64), 5x5 (32), 5x5 (1) -> trained 200 epochs

#### Proposed normalization vs other normalizations:
* Batch normalization: 70.67 (MNIST), 74.95 (USPS) vs 95.11, 91.35
* Layer normalization: 70.83 (MNIST), 75.26 (USPS) vs 95.11, 91.35

#### Anomaly detection
* Used AUC of ROC for evaluation
* Choose one class as anomaly, use DCEC with proposed L2 norm, k=9
* Baselines: 
  * DCEC without proposed L2 norm
  * DCEC with proposed L2 norm, but with reconstrution error per sample as
anomaly sore

## Tests:
The following is the template for the name of the test, each part of the name is
separated using an underscore:
* The first letters are the type of model: DCEC, IDEC or DEC. 
* The input shape, e.g. 500\_500
* The output shape, e.g. 200\_200 or 500\_500
* The number of parametrised layers in the encoder (CONV and DENSE layers)
* The number of parametrised layers in the decoder (CONV and DENSE layers)
* If the encoder is regularized add REGEN
* If the decoder is regularized add REGDE
* Size of embedding space
* Number of dense layers in the encoder
* Number of dense layers in the decoder

An example of a name would be: DCEC\_500\_500\_200\_200\_8\_5\_REGEN\_256\_1\_1
Which is a convolutional autoencoder, with an input shape of (500, 500) an
output shape of (200, 200), 7 conv layers and 1 dense layer in the encoder, 1
dense layer and 4 conv\_transpose layers in the decoder and regularisation on
the layers in the encoder. The embedding space size is 256-dimensional.

## Results
| Name                                           | Epochs | Tr-loss | Va-loss | Te-acc |
|------------------------------------------------|--------|---------|---------|--------|
|DCEC\_500\_500\_200\_200\_8\_5\_REGEN\_256\_1\_1| 200    |         |         |        |

Tr-loss = Training loss after training epochs
Va-loss = Validation loss after training epochs
Te-acc = The accuracy on the test set (with labels)
