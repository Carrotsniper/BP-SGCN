import os
import seaborn as sns
import pandas as pd 
import time
import pdb
from tqdm import *
import torch
from torch import nn
from torch.autograd import Variable
from torch.utils.data import DataLoader
from torch.nn import Parameter
from torchvision import transforms
from torchvision.datasets import MNIST
from torchvision.utils import save_image
from sklearn.cluster import KMeans
import numpy as np
from tqdm import *
from metrics import *
from sklearn.manifold import TSNE
from matplotlib import pyplot as plt
device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')


class ClusteringLayer(nn.Module):
	def __init__(self, n_clusters=6, hidden=10, cluster_centers=None, alpha=1.0):
		super(ClusteringLayer, self).__init__()
		self.n_clusters = n_clusters
		self.alpha = alpha
		self.hidden = hidden
		if cluster_centers is None:
			initial_cluster_centers = torch.zeros(
			self.n_clusters,
			self.hidden,
			dtype=torch.float
			).cuda()
			nn.init.xavier_uniform_(initial_cluster_centers)
		else:
			initial_cluster_centers = cluster_centers
		self.cluster_centers = Parameter(initial_cluster_centers)
	def forward(self, x):
		norm_squared = torch.sum((x.unsqueeze(1) - self.cluster_centers)**2, 2)
		numerator = 1.0 / (1.0 + (norm_squared / self.alpha))
		power = float(self.alpha + 1) / 2
		numerator = numerator**power
		t_dist = (numerator.t() / torch.sum(numerator, 1)).t() #soft assignment using t-distribution
		return t_dist


class DEC(nn.Module):
	def __init__(self, n_clusters=6, autoencoder=None, hidden=10, cluster_centers=None, alpha=1.0):
		super(DEC, self).__init__()
		self.n_clusters = n_clusters
		self.alpha = alpha
		self.hidden = hidden
		self.cluster_centers = cluster_centers
		self.autoencoder = autoencoder
		self.clusteringlayer = ClusteringLayer(self.n_clusters, self.hidden, self.cluster_centers, self.alpha)

	def target_distribution(self, q_):
		weight = (q_ ** 2) / torch.sum(q_, 0)
		return (weight.t() / torch.sum(weight, 1)).t()

	def forward(self, x):
		_  , _, h, z_t, enc_t, (all_enc_mean, all_enc_var), (all_dec_mean, all_dec_var)= self.autoencoder(x)
		return self.clusteringlayer(z_t)
	
	def visualize(self, epoch,x):
		
		_  , _, h, z_t, enc_t, (all_enc_mean, all_enc_var), (all_dec_mean, all_dec_var) = self.autoencoder(x)

		labels = self.clusteringlayer(z_t)
		label = np.argmax(labels.detach().cpu().numpy(), axis=1)

		x = z_t.detach().cpu().numpy()[:1300]
		x_embedded = TSNE(n_components=2, init="pca").fit_transform(x)
		df = pd.DataFrame()
		df["Cluster"] = label[:1300]
		df["dim-0"] = x_embedded[:,0]
		df["dim-1"] = x_embedded[:,1]
		

		fig = plt.figure(figsize=(16,10))
		ax = plt.subplot(111)
		sns.scatterplot(
			x="dim-0", y="dim-1",
			hue="Cluster",
			palette=sns.color_palette("Set2", 6),
			data=df,
			legend="full",
		)

		# plt.scatter(x_embedded[:,0], x_embedded[:,1],c = label[:1300], s=0.5, alpha = 0.5)
		fig.savefig('plots/SDD_{}.png'.format(epoch))
		plt.close(fig)
