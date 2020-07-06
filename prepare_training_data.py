import h5py
import numpy as np
import pickle


def euclidean_dist_normalized(x1, x2=None, eps=1e-8):
    left = x1 / normalize_factor
    right = x2 / normalize_factor
    return np.sqrt(((left - right) ** 2).mean())

cluster_size = 100
query_size = 10000
min_threshold = 0.0
max_threshold = 0.5
slot = 0.01
queries_dimension = 784
normalize_factor = 255.0
distance_function = euclidean_dist_normalized
hidden_num = 256
output_num = cluster_size
dataset_id = 'fashion_mnist_784_euclidean'

f = h5py.File('fashion-mnist-784-euclidean.hdf5', 'r')

from sklearn.cluster import MiniBatchKMeans
import numpy as np

kmeans = MiniBatchKMeans(n_clusters=cluster_size,random_state=0,batch_size=1000).fit(f['train'])

queries = np.array(f['test'])
clusters = kmeans.predict(f['train'])
clusters_points = []
for cluster_id in range(0, cluster_size):
    clusters_points.append(np.array(f['train'])[(clusters == cluster_id).nonzero()])

print ('complete clusters, writing to file ... ...')
with open('clustered_points.pkl', 'wb') as f:
    pickle.dump(clusters_points, f)
print ('complete!')

import torch.nn.functional as F

from numpy import dot
from numpy.linalg import norm

from multiprocessing import Pool

def run_proc(clus):
    dataset = clusters_points[clus]
    ground_truth = []
    for idxx, q in enumerate(queries):
        if idxx % 100 == 0:
            print (clus, idxx)
        thresholds = np.arange(min_threshold, max_threshold, slot)
        count = [0 for _ in thresholds]
        for d in dataset:
            distance = distance_function(q, d)
            for idx, threshold in enumerate(thresholds):
                if distance < threshold + slot and distance >= threshold:
                    count[idx] += 1
        for idx, threshold in enumerate(thresholds):
            ground_truth.append((clus, idxx, threshold, threshold + slot, count[idx]))
    return ground_truth
#             print (threshold, threshold + slot, count[idx])
ground_truth_total = []
processes = []
pool = Pool(processes=30)
for clus in range(cluster_size):
    processes.append(pool.apply_async(run_proc, args=(clus,)))
pool.close()
pool.join()

for i in processes:
    ground_truth_total.append(i.get())

print ('complete ground truth prepare, writing to file ... ...')
with open('ground_truth_fashion_mnist_784_euclidean_0_0_0_5.pkl', 'wb') as f:
    pickle.dump(ground_truth_total, f)
print ('complete!')