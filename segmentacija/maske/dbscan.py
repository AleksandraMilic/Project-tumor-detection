import matplotlib.pyplot as plt
import numpy as np
from sklearn.cluster import DBSCAN
from sklearn import metrics
from sklearn.datasets.samples_generator import make_blobs
 
centers = [[5, 5], [0, 0], [1, 5], [5, -1]]
X, labels_true = make_blobs(n_samples=500, n_features=5, centers=centers, cluster_std=0.9, center_box=(1, 10.0), shuffle=True, random_state=0)

# Compute DBSCAN
db = DBSCAN(eps=0.5, min_samples=10).fit(X)
#print(db)

#zeros_like :Return an array of zeros with the same shape and type as a given array., dtype will overrides the data type of the result.
core_samples_mask = np.zeros_like(db.labels_, dtype=bool)

#core_sample_indices_ : Attributes and it is index of core samples (array, shape = [n_core_samples])
core_samples_mask[db.core_sample_indices_] = True
labels = db.labels_

# Number of clusters in labels, ignoring noise if present.
n_clusters_ = len(set(labels)) - (1 if -1 in labels else 0)

#print results 
print('Estimated number of clusters: %d' % n_clusters_)
print("Homogeneity: %0.3f" % metrics.homogeneity_score(labels_true, labels))
print("Completeness: %0.3f" % metrics.completeness_score(labels_true, labels))
print("V-measure: %0.3f" % metrics.v_measure_score(labels_true, labels))
print("Adjusted Rand Index: %0.3f"% metrics.adjusted_rand_score(labels_true, labels))
print("Adjusted Mutual Information: %0.3f"% metrics.adjusted_mutual_info_score(labels_true, labels))
print("Silhouette Coefficient: %0.3f"% metrics.silhouette_score(X, labels))
      

# Drawing chart
# Black removed and is used for noise instead.
unique_labels = set(labels)
colors = plt.cm.Spectral(np.linspace(0, 1, len(unique_labels)))
for k, col in zip(unique_labels, colors):
    if k == -1:
        # Black used for noise.
        col = 'k'

    class_member_mask = (labels == k)

    xy = X[class_member_mask & core_samples_mask]
    plt.plot(xy[:, 0], xy[:, 1], 'o', markerfacecolor=col,
             markeredgecolor='k', markersize=14)

    xy = X[class_member_mask & core_samples_mask]
    plt.plot(xy[:, 0], xy[:, 1], 'o', markerfacecolor=col,
             markeredgecolor='k', markersize=6)

plt.title('Estimated number of clusters: %d' % n_clusters_)
plt.show()

print("X",type(X))
print("core_samples_mask", type(core_samples_mask))  
#print("core_sample_indices_", type(core_sample_indices))
#print("core_sample_indices_", type(core_sample_indices))
print("labels", labels)
