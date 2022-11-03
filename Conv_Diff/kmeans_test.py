#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Aug 18 21:09:38 2022

@author: vgopakum
"""
# %%
import sklearn 
import numpy as np 

from sklearn.cluster import KMeans
import numpy as np

# %%
X = np.random.uniform(0,1,(1000,2))

# %%
kmeans = KMeans(n_clusters=2, random_state=0).fit(X)
# kmeans.labels_
# kmeans.predict([[0, 0], [12, 3]])

cluster_centres = kmeans.cluster_centers_


# %%
from sklearn.metrics import pairwise_distances
X = [[0, 1], [1, 1]]
# distance between rows of X

distances = pairwise_distances(X, cluster_centres, metric='euclidean')
