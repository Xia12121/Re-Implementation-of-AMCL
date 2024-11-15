# Re-Implementation of Adaptive Multi-head Contrastive Learning

This project contains my comprehension of this paper and the re-implementation of the experimental code. My code is various from original work because the lib version is different from the source code. This project is based on [Wang's work](https://github.com/LeiWangR/cl).
## Content of this page
[I. Research Background and Motivation](#research-background-and-motivation)

[II. Baseline Models and their principles](#Baseline-Models-and-their-principles)

[III. Methodology of this paper](#Methodology-of-this-paper)

[IV. Re-Implementation of Experiments](#Re-Implementation-of-Experiments)

## I. Research Background and Motivation
In contrastive learning, two views generated by various augmentations from same original image are considered Positive Pair. And two view generated from different images are considered Negative Pair. For contrastive learning, people hope the similarity between positive pair to be high and similarity between negative pair to be low.

![image](https://github.com/user-attachments/assets/96e29daf-868e-460d-909b-2bd0c395cd5e)

However, due to the diversity of augmentations views generated from same image may be quite different and views generated from various images may be similar with each other. (intra-sample similarity is low while inter-sample similarity is high)

![image](https://github.com/user-attachments/assets/e9bdb487-78ef-4e45-ad42-d4c690b3acd5)

Which result in expected result may be unattainable! To solve this problem, authour proposed an approach named Adaptive Multi-head Contrastive Learning. This approach has two major innovations:
- Multi-head Projection: Enhances feature diversity, enabling the model to capture richer features from multiple perspectives.
- Trainable Adaptive Temperature: Adjusts similarity requirements for positive and negative pairs, enhancing discrimination accuracy and optimizing contrastive learning stability.

The pipline chart is following:

![image](https://github.com/user-attachments/assets/b2f66081-ca9a-4c2b-8973-56b51d00fb0a)


## II. Baseline Models and their principles
Author declaim AMCL method can be applied to four mainstream SSL method: [SimCLR](#2.1-A-Simple-Framework-for-Contrastive-Learning-of-Visual-Representations (SimCLR)), [MoCo](#2.2-Momentum-Contrast-for-Unsupervised-Visual-Representation-Learning (MoCo)), [SimSiam](#2.1-A-Simple-Framework-for-Contrastive-Learning-of-Visual-Representations (SimCLR)), and [B.Twins[(#2.1-A-Simple-Framework-for-Contrastive-Learning-of-Visual-Representations (SimCLR)). This section will illustrate them all respectively.  

### 2.1 A Simple Framework for Contrastive Learning of Visual Representations (SimCLR) 

![image](https://github.com/user-attachments/assets/3b58d5f5-922a-463d-bf0e-efb093bf8fcd)

SimCLR is a kind of unsupervised learning method based on contrast learning, which is a classical model framework in contrast learning. The main idea is to learn useful feature representations by maximizing the similarity between different data-enhanced versions of the same image.

To extract features from two views, two independent encoders (encoder $$q$$ and encoder $$k$$) are utilized. $$q$$ and $$k$$ are features of query and target respectively. The similarity between $$q$$ and $$k$$ is calculated through the following equation:

$$
Similarity = \frac {q\cdot k}{\left|q\right|\cdot \left|k\right|}
$$

The model is optimized through the following equation:

$$
Loss = -\log \frac{exp(similarity(q,k^+))}{\sum_{i=0}^K{exp(similarity(q,k_i^-))}}
$$

Just like the following chart, the loss will propagate to both encoders!

### 2.2 Momentum Contrast for Unsupervised Visual Representation Learning (MoCo)

![image](https://github.com/user-attachments/assets/02a425f4-2d55-49d5-a337-b7cfc0957e6d)

MoCo is also a SSL framework, compared with SimCLR there are two major differences:
- The parameter update mode of the target encoder (encoder $$k$$) is different.
- The methods for storing negative samples are different.

As for the first difference, compared with SimCRL loss won't affect the target encoder directly. The parameter updated through Momentum-based parameter updating:

$$
𝜃_k^{new}=m\cdot 𝜃_k+(1-m)\cdot 𝜃_q
$$

Beside this, MoCo dynamically updated negative samples dictionary (When a new negative sample is added, one is removed in front, and the workflow is similar to a queue). 

### 2.3 Exploring Simple Siamese Representation Learning (SimSiam) 

![image](https://github.com/user-attachments/assets/886903ef-3e4d-41bf-ab4b-57eecd3acaea)

SimCLR and MoCo employ consine to calculate similarity, based on the similarity loss function is constructed. However, SimSiam employs cosine to calculate loss value directly:

$$
Loss=-cosine(x^q,x^{k+})
$$

Moreover, SimSiam proposed a novel mechanism named "gradient stop", which prevent gradient from flowing through the encoder.

### 2.4 Barlow Twins: Self-Supervised Learning via Redundancy Reduction (Twins) 

![image](https://github.com/user-attachments/assets/26a8f3b7-c9ee-4173-8539-08ec74e048da)

Similar to prior works, two encoders embed query view and target view respectively. However, the fusing approach is various. B.Twins leverage dot product to fuse features of query and target (In this method, only the values on the diagonal of the fusion matrix are the product of the query and its corresponding correct target, and subsequent optimization is based on this feature).

The loss function is following:

$$
Loss=\sum_i{(1-C_{ii})}^2+𝜆\sum_i\sum_{𝑗≠𝑖}C_{ij}^2
$$

Where $$C_{ii}$$ is the value on the diagonal of the fusion matrix.

## III. Methodology of this paper

## IV. Re-Implementation of Experiments

The re-implementation is conducted in Nvidia T4 GPU provided by Google Colab. The set up enviroment is following:

|              | Re-Implementation   | Author's set up |
|--------------|-----------------|---------------------|
| CUDA         | 12.2            |              |
| Pytorch      | 2.3.1+cu121     | 1.4.0              |
| Torchvision  | 0.18.1+cu121    | 0.5.0              |
| Pytorch-lightning| 1.6.0           | 1.3.8              |
| Lightly      | 1.0.8           | 1.0.8              |



