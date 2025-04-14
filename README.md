## BP-SGCN: Behavioral Pseudo-Label Informed Sparse Graph Convolution Network for Pedestrian and Heterogeneous Trajectory Prediction

This is the official implementation of our paper **BP-SGCN** [https://arxiv.org/abs/2502.14676](https://arxiv.org/abs/2502.14676).

This work has been accepted at the IEEE Transactions on Neural Networks and Learning Systems (TNNLS 2025).

![image](https://github.com/Carrotsniper/BP-SGCN/blob/master/overview.png)

## Abstract
Trajectory prediction allows better decision-making in applications of autonomous vehicles or surveillance by predicting the short-term future movement of traffic agents. It is classified into pedestrian or heterogeneous trajectory prediction. The former exploits the relatively consistent behavior of pedestrians, but is limited in real-world scenarios with heterogeneous traffic agents such as cyclists and vehicles. The latter typically relies on extra class label information to distinguish the heterogeneous agents, but such labels are costly to annotate and cannot be generalized to represent different behaviors within the same class of agents. In this work, we introduce the behavioral pseudo-labels that effectively capture the behavior distributions of pedestrians and heterogeneous agents solely based on their motion features, significantly improving the accuracy of trajectory prediction. To implement the framework, we propose the Behavioral Pseudo-Label Informed Sparse Graph Convolution Network (BP-SGCN) that learns pseudo-labels and informs to a trajectory predictor. For optimization, we propose a cascaded training scheme, in which we first learn the pseudo-labels in an unsupervised manner, and then perform end-to-end fine-tuning on the labels in the direction of increasing the trajectory prediction accuracy. Experiments show that our pseudo-labels effectively model different behavior clusters and improve trajectory prediction. Our proposed BP-SGCN outperforms existing methods using both pedestrian (ETH/UCY, pedestrian-only SDD) and heterogeneous agent datasets (SDD, Argoverse 1).

## Highlights
- We propose the novel concept of behavioral pseudo-labels to represent clusters of traffic agents with different movement behaviors, improving trajectory prediction without the need for any extra annotation.
- To implement the idea, we propose BP-SGCN, which introduces a cascaded training scheme to optimize the compatibility of its two core modules: the pseudo-label clustering module and the trajectory prediction module.
- We propose a deep unsupervised behavior clustering module to obtain behavioral pseudo-labels, tailoring the geometric feature representation and the loss to best learn the agents’ behaviors.
- We propose a pseudo-label informed goal-guided trajectory prediction module, which facilitates end-to-end fine-tuning with its prediction loss for better clustering and prediction, outperforming existing pedestrian and heterogeneous prediction methods.

The code is undergoing processing and will be available soon. 

## Cite this Work

If this work is useful, please consider citing the paper, and/or mentioning this repository:
```bibtex
@article{li2025bpsgcn,
  title={BP-SGCN: Behavioral Pseudo-Label Informed Sparse Graph Convolution Network for Pedestrian and Heterogeneous Trajectory Prediction},
  author={Li, Ruochen and Katsigiannis, Stamos and Kim, Tae-Kyun and Shum, Hubert PH},
  journal={IEEE Transactions on Neural Networks and Learning Systems},
  year={2025},
  publisher={IEEE}
}
```
