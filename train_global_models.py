from __future__ import print_function
import argparse
import torch
from torch.autograd import Variable
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision import datasets, transforms

from numpy import dot
from numpy.linalg import norm
from scipy import spatial
import torch.utils.data

import h5py
import numpy as np
import pickle

data = h5py.File('fashion-mnist-784-euclidean.hdf5', 'r')
data_train = np.array(data['train'])
data_test = np.array(data['test'])
with open('clusters_fashion_mnist_784_euclidean.pkl', 'rb') as f:
    clusters = pickle.load(f)
with open('ground_truth_fashion_mnist_784_euclidean_0_0_0_5.pkl', 'rb') as f:
    ground_truth_total = pickle.load(f)


def euclidean_dist_normalized(x1, x2=None, eps=1e-8):
    left = x1
    right = x2
    return np.sqrt(((left - right) ** 2).mean())

cluster_size = len(clusters)
query_size = 10000
min_threshold = 0.0
max_threshold = 0.5
slot = 0.01
queries_dimension = 784
normalize_factor = 255.0
distance_function = euclidean_dist_normalized
hidden_num = 256
output_num = cluster_size

ground_truth_total_level = [[[] for _ in range(query_size)] for _ in range(cluster_size)]
for clus in range(cluster_size):
    for t in ground_truth_total[clus]:
        ground_truth_total_level[t[0]][t[1]].append(t)

centroids = []
for cluster in clusters:
    centroids.append(np.mean(cluster))

train_features = []
train_thresholds = []
train_distances = []
train_targets = []
train_cards = []
for query_id in range(int(query_size * 0.8)):
    cardinality = [0 for _ in range(cluster_size)]
    distances2centroids = []
    for cc in centroids:
        distances2centroids.append(distance_function(data_test[query_id] / normalize_factor, cc / normalize_factor))
    for threshold_id, threshold in enumerate(np.arange(min_threshold, max_threshold, slot)):
        indicator = []
        cards = []
        for cluster_id in range(cluster_size):
            cardinality[cluster_id] += ground_truth_total_level[cluster_id][query_id][threshold_id][-1]
            if cardinality[cluster_id] > 0:
                indicator.append(1)
            else:
                indicator.append(0)
            cards.append(cardinality[cluster_id])
        feature = data_test[query_id] / normalize_factor
        train_features.append(feature)
        train_distances.append(distances2centroids)
        train_thresholds.append([threshold+slot])
        train_targets.append(indicator)
        train_cards.append(cards)

test_features = []
test_thresholds = []
test_distances = []
test_targets = []
test_cards = []
for query_id in range(int(query_size * 0.8),query_size):
    cardinality = [0 for _ in range(cluster_size)]
    distances2centroids = []
    for cc in centroids:
        distances2centroids.append(distance_function(data_test[query_id] / normalize_factor, cc / normalize_factor))
    for threshold_id, threshold in enumerate(np.arange(min_threshold, max_threshold, slot)):
        indicator = []
        cards = []
        for cluster_id in range(cluster_size):
            cardinality[cluster_id] += ground_truth_total_level[cluster_id][query_id][threshold_id][-1]
            if cardinality[cluster_id] > 0:
                indicator.append(1)
            else:
                indicator.append(0)
            cards.append(cardinality[cluster_id])
        feature = data_test[query_id] / normalize_factor
        test_features.append(feature)
        test_distances.append(distances2centroids)
        test_thresholds.append([threshold+slot])
        test_targets.append(indicator)
        test_cards.append(cards)

batch_size = 128
train_loader = torch.utils.data.DataLoader(
        torch.utils.data.TensorDataset(torch.FloatTensor(train_features), torch.FloatTensor(train_thresholds), torch.FloatTensor(train_distances), torch.FloatTensor(train_targets), torch.FloatTensor(train_cards)), batch_size=batch_size, shuffle=True)
test_loader = torch.utils.data.DataLoader(
        torch.utils.data.TensorDataset(torch.FloatTensor(test_features), torch.FloatTensor(test_thresholds), torch.FloatTensor(test_distances), torch.FloatTensor(test_targets), torch.FloatTensor(test_cards)), batch_size=batch_size, shuffle=True)

class Model(nn.Module):
    def __init__(self):
        super(Model, self).__init__()
        self.nn1 = nn.Linear(queries_dimension, hidden_num)
        self.nn2 = nn.Linear(hidden_num, hidden_num)
#         self.nn3 = nn.Linear(hidden_num, hidden_num)
        
        self.dist1 = nn.Linear(cluster_size, hidden_num)
        self.dist2 = nn.Linear(hidden_num, hidden_num)
        
        self.nn4 = nn.Linear(hidden_num, hidden_num)
        self.nn5 = nn.Linear(hidden_num, output_num)
        
        self.thres1 = nn.Linear(1, hidden_num)
        self.thres2 = nn.Linear(hidden_num, 1)

    def forward(self, x, distances, thresholds):
        out1 = F.relu(self.nn1(x))
        out2 = F.relu(self.nn2(out1))
#         out3 = F.relu(self.nn3(out2))
#         print (distances.shape)
        distance1 = F.relu(self.dist1(distances))
        distance2 = F.relu(self.dist2(distance1))
        
        thresholds_1 = F.relu(self.thres1(thresholds))
        thresholds_2 = self.thres2(thresholds_1)

        out4 = F.relu(self.nn4((out2 + distance2) / 2))
        out5 = self.nn5(out2)
        
        probability = F.sigmoid(out5 + thresholds_2)
        return probability

def loss_fn(estimates, targets, cards):
    punish_idx = (estimates < 0.5).float()
#     return F.mse_loss(estimates, targets) - 0.01 * (estimates * cards).mean()
    min_v, _ = torch.min(cards, dim=1)
    max_v, _ = torch.max(cards, dim=1)
    min_v = min_v.unsqueeze(dim=1)
    max_v = max_v.unsqueeze(dim=1)
    range_v = max_v - min_v
#     print (min_v.shape,range_v.shape)
    normalized_cards = (cards - min_v) / (range_v + 0.01)
    loss = ((F.relu(estimates - targets) + F.relu(targets - estimates) * (normalized_cards + 1.0)) ** 2).sum(dim=1).mean()
    return loss
#     return F.mse_loss(estimates, targets)
#     return F.mse_loss(estimates, targets) - (estimates * cards).mean()
#     return F.mse_loss(estimates, targets) - (estimates * cards).mean() / F.mse_loss(estimates, targets) 
#     return F.mse_loss(estimates, targets) * torch.exp(1 + 1 / (estimates * cards + 0.1)).mean() 
#     return F.mse_loss(estimates, targets) - 0.2 * torch.log((estimates * cards).mean() + 1.0)
#     return F.mse_loss(estimates, targets) + 0.2 * ((0.5 - estimates) * cards * punish_idx).mean()
#     return F.mse_loss(estimates, targets) + 0.2 * torch.log((cards * punish_idx) + 1.0).mean()

def print_loss(estimates, targets, cards):
    true_positive = 0.0
    true_negative = 0.0
    false_positive = 0.0
    false_negative = 0.0
    num_elements = estimates.shape[1]
    for est, tar in zip(estimates, targets):
        for i in range(num_elements):
            if est[i] < 0.5 and tar[i] == 0:
                true_negative += 1
            elif est[i] < 0.5 and tar[i] == 1:
                false_negative += 1
            elif est[i] >= 0.5 and tar[i] == 0:
                false_positive += 1
            else:
                true_positive += 1
    if true_positive + false_positive > 0:
        precision = true_positive / (true_positive + false_positive)
    else:
        precision = 1.0
    if true_positive + false_negative > 0:
        recall = true_positive / (true_positive + false_negative)
    else:
        recall = 1.0
    total_card = cards.sum(dim=1)
#     print ('total_card: ', total_card.shape)
    miss_card = torch.FloatTensor([cards[i][((estimates[i] < 0.5).nonzero())].sum() for i in range(cards.shape[0])])
#     print ('miss_card: ', miss_card.shape)
    miss_rate = (miss_card / (total_card + 0.1)).mean()
    return precision, recall, miss_rate

model = Model()
opt = optim.Adam(model.parameters(), lr=2e-4)
for e in range(5):
    model.train()
    for batch_idx, (features, thresholds, distances, targets, cards) in enumerate(train_loader):
        x = Variable(features)
        y = Variable(targets.unsqueeze(1))
        z = Variable(thresholds)
        dists = Variable(distances)
        opt.zero_grad()
        estimates = model(x, dists, z)
        loss = loss_fn(estimates, targets, cards)
        if batch_idx % 100 == 0:
            print('Training: Iteration {0}, Batch {1}, Loss {2}'.format(e, batch_idx, loss.item()))
        loss.backward()
        opt.step()

    model.eval()
    test_loss = 0.0
    precision = 0.0
    recall = 0.0
    miss_rate = 0.0
    for batch_idx, (features, thresholds, distances, targets, cards) in enumerate(test_loader):
        x = Variable(features)
        y = Variable(targets.unsqueeze(1))
        z = Variable(thresholds)
        dists = Variable(distances)
        estimates = model(x, dists, z)
        loss = loss_fn(estimates, targets, cards)
        test_loss += loss.item()
        prec, rec, miss = print_loss(estimates, targets, cards)
        precision += prec
        recall += rec
        miss_rate += miss
        if batch_idx % 100 == 0:
            print ('Testing: Iteration {0}, Batch {1}, Loss {2}, Precision {3}, Recall {4}, Miss {5}'.format(e, batch_idx, loss.item(), prec, rec, miss))
    test_loss /= len(test_loader)
    precision /= len(test_loader)
    recall /= len(test_loader)
    miss_rate /= len(test_loader)
    print ('Testing: Loss {0}, Precision {1}, Recall {2}, Miss {3}'.format(test_loss, precision, recall, miss_rate))
    
torch.save(model.state_dict(), 'global_fashion_mnist_784_euclidean_punish_query_threshold_monotonic.model')
