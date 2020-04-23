# Seminar Computer Vision by Deep Learning (CS4245)

1) Read [this paper](https://ieeexplore.ieee.org/abstract/document/8489068)
2) Reproduce the aforementioned paper
3) Think about applications in crowd anomaly detection

Citation: <cite> Aytekin, C., Ni, X., Cricri, F., & Aksu, E. (2018, July). Clustering and unsupervised anomaly detection with l 2 normalized deep auto-encoder representations. In 2018 International Joint Conference on Neural Networks (IJCNN) (pp. 1-6). IEEE. <cite>

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
3. Use normality score: [!normality score](http://www.sciweavers.org/upload/Tex2Img_1587649594/render.png) on k-means clusters (C), representations (E(I))
4. Use threshold to separate 'normal' samples from anomalies

### Experimental results
* Evaluation metric: ![accuracy](http://www.sciweavers.org/upload/Tex2Img_1587649539/render.png) simply: the maximum number of correctly classified images according to each of the possible cluster (c) mappings (m) to classes (l). 
* Maximization of the evaluation metric is done according to the Hungarian
  algorithm
* Dense AEs are evaluated on MNIST, convolutional AEs on MNIST, USPS
* Dense AEs used: 
..* (I)DEC: encoding layers: 500 - 500 - 2000 - 10, decoding layers: 2000 - 500 - 500 - d -> trained 100 epochs
* Convolutional AEs used:
..* DCEC: encoding: 5x5 (32), 5x5 (64), 3x3 (128). 2x2 stride. decoding: 3x3
(64), 5x5 (32), 5x5 (1) -> trained 200 epochs

#### Proposed normalization vs other normalizations:
* Batch normalization: 70.67 (MNIST), 74.95 (USPS) vs 95.11, 91.35
* Layer normalization: 70.83 (MNIST), 75.26 (USPS) vs 95.11, 91.35

#### Anomaly detection
* Used AUC of ROC for evaluation
* Choose one class as anomaly, use DCEC with proposed L2 norm, k=9
* Baselines: 
..* DCEC without proposed L2 norm
..* DCEC with proposed L2 norm, but with reconstrution error per sample as
anomaly sore

