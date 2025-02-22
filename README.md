# BP-SGCN: Behavioral Pseudo-Label Informed Sparse Graph Convolution Network for Pedestrian and Heterogeneous Trajectory Prediction
This repository contains the official implementation of  **BP-SGCN: Behavioral Pseudo-Label Informed Sparse Graph Convolution Network for Pedestrian and Heterogeneous Trajectory Prediction**.

## Highlights
- We propose the novel concept of behavioral pseudo-labels to represent clusters of traffic agents with different movement behaviors, improving trajectory prediction without the need for any extra annotation.
- To implement the idea, we propose BP-SGCN, which introduces a cascaded training scheme to optimize the compatibility of its two core modules: the pseudo-label clustering module and the trajectory prediction module.
- We propose a deep unsupervised behavior clustering module to obtain behavioral pseudo-labels, tailoring the geometric feature representation and the loss to best learn the agents’ behaviors.
- We propose a pseudo-label informed goal-guided trajectory prediction module, which facilitates end-to-end fine-tuning with its prediction loss for better clustering and prediction, outperforming existing pedestrian and heterogeneous prediction methods.

## Our model
The architecture of our BP-SGCN trajectory prediction model
![image](https://github.com/Carrotsniper/BP-SGCN/blob/main/overview.png)

The code is undergoing processing and will be available soon. 
