from __future__ import print_function
import argparse
import torch
from torch.autograd import Variable
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision import datasets, transforms
import torch.utils.data

import h5py
import numpy as np
import pickle

from numpy import dot
from numpy.linalg import norm
from scipy import spatial

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
dataset_name = 'fashion_mnist_784_euclidean'

ground_truth_total_level = [[[] for _ in range(query_size)] for _ in range(cluster_size)]
for clus in range(cluster_size):
    for t in ground_truth_total[clus]:
        ground_truth_total_level[t[0]][t[1]].append(t)

centroids = []
for cluster in clusters:
    centroids.append(np.mean(cluster))

from numpy import dot
from numpy.linalg import norm
from scipy import spatial

test_features = []
test_thresholds = []
test_distances = []
test_targets = []
test_cards = []

for query_id in range(int(query_size * 0.8),query_size):
    cardinality = [0 for _ in range(cluster_size)]
    distances2centroids = []
    for cc in centroids:
        distances2centroids.append(distance_function(data_test[query_id], cc))
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
        feature = data_test[query_id].astype(float) / normalize_factor
        test_features.append(feature)
        test_distances.append(distances2centroids)
        test_thresholds.append([threshold+slot])
        test_targets.append(indicator)
        test_cards.append(cards)

batch_size = 128
test_loader = torch.utils.data.DataLoader(
        torch.utils.data.TensorDataset(torch.FloatTensor(test_features), torch.FloatTensor(test_thresholds), torch.FloatTensor(test_distances), torch.FloatTensor(test_targets), torch.FloatTensor(test_cards)), batch_size=batch_size, shuffle=False)

input_dimension = queries_dimension
cluster_dimension = cluster_size

class Model(nn.Module):
    def __init__(self):
        super(Model, self).__init__()
        self.nn1 = nn.Linear(input_dimension, hidden_num)
        self.nn2 = nn.Linear(hidden_num, hidden_num)
#         self.nn3 = nn.Linear(hidden_num, hidden_num)
        
        self.dist1 = nn.Linear(cluster_dimension, hidden_num)
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
    return F.mse_loss(estimates, targets) + 0.02 * torch.log(((0.5 - estimates) * cards * punish_idx).mean() + 1.0)

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
    precision = true_positive / (true_positive + false_positive)
    recall = true_positive / (true_positive + false_negative)
    total_card = cards.sum(dim=1)
#     print ('total_card: ', total_card.shape)
    miss_card = torch.FloatTensor([cards[i][((estimates[i] < 0.5).nonzero())].sum() for i in range(cards.shape[0])])
#     print ('miss_card: ', miss_card.shape)
    miss_rate = (miss_card / (total_card + 0.1)).mean()
    return precision, recall, miss_rate

class Threshold_Model(nn.Module):
    
    def __init__(self):
        super(Threshold_Model, self).__init__()
        self.fc1 = nn.Linear(1, hidden_num)
        self.fc2 = nn.Linear(hidden_num, 1)
    
    def forward(self, threshold):
        t1 = F.relu(self.fc1(threshold))
        t2 = self.fc2(t1)
        return t2

class CNN_Model(nn.Module):
    
    def __init__(self, in_channel, out_channel, kernel_size, stride, padding, pool_type, pool_size):
        super(CNN_Model, self).__init__()
        if pool_type == 0:
            pool_layer = nn.MaxPool1d(kernel_size=pool_size, stride=pool_size)
        elif pool_type == 1:
            pool_layer = nn.AvgPool1d(kernel_size=pool_size, stride=pool_size)
        else:
            print ('CNN_Model Init Error, invalid pool_type {}'.format(pool_type))
            return
        self.layer = nn.Sequential(
            nn.Conv1d(in_channel, out_channel, kernel_size=kernel_size, stride=stride, padding=padding), 
            nn.BatchNorm1d(out_channel),
            nn.ReLU(),
            pool_layer)
        
    def forward(self, inputs):
        hid = self.layer(inputs)
        return hid

class Output_Model(nn.Module):
    
    def __init__(self, inputs_dim):
        super(Output_Model, self).__init__()
        self.fc1 = nn.Linear(inputs_dim, hidden_num)
        self.fc2 = nn.Linear(hidden_num, 1)
    
    def forward(self, queries, threshold):
        out1 = F.relu(self.fc1(queries + threshold))
        out2 = self.fc2(out1)
        return out2

def loss_fn(estimates, targets, mini, maxi):
    est = unnormalize(estimates, mini, maxi)
    print (torch.cat((est, targets), 1))
    return F.mse_loss(est, targets)

def l1_loss(estimates, targets, eps=1e-5):
    return F.smooth_l1_loss(estimates, torch.log(targets))

def mse_loss(estimates, targets, eps=1e-5):
    return F.mse_loss(estimates, torch.log(targets))

def qerror_loss(preds, targets, mini, maxi):
    qerror = []
    preds = unnormal1ize_label(preds, mini, maxi)
    for i in range(len(targets)):
        if (preds[i] > targets[i]).cpu().data.numpy()[0]:
            qerror.append(preds[i]/targets[i])
        else:
            qerror.append(targets[i]/(preds[i] + 0.1))
    return torch.mean(torch.cat(qerror) ** 2)

def print_loss(estimates, targets):
    esti = torch.exp(estimates)
    qerror = []
    for i in range(esti.shape[0]):
        if esti[i] > targets[i] + 0.1:
            qerror.append((esti[i] / (targets[i] + 0.1)).item())
        else:
            qerror.append(((targets[i] + 0.1) / esti[i]).item())
    
    return F.mse_loss(esti, targets), np.mean(qerror), np.max(qerror)

class TunableParameters():
    
    def __init__(self, out_channel, kernel_size, stride, padding, pool_size, pool_type):
        self.out_channel = out_channel
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding
        self.pool_size = pool_size
        self.pool_type = pool_type
        
    def __repr__(self):
        return str(self.out_channel) +' '+ str(self.kernel_size) +' '+ str(self.stride) +' '+ str(self.padding) +' '+ str(self.pool_size) +' '+ str(self.pool_type)
 
    def __str__(self):
        return str(self.out_channel) +' '+ str(self.kernel_size) +' '+ str(self.stride) +' '+ str(self.padding) +' '+ str(self.pool_size) +' '+ str(self.pool_type)

import pickle
hyper_parameterss = []
with open('cnn_hyper_parameters.hyperpara', 'r') as handle:
    for paras in handle.readlines():
        hyper_parameters = []
        for para in paras.split(';'):
            para = para.split(' ')
            hyper_parameters.append(TunableParameters(int(para[0]), int(para[1]), int(para[2]),
                                                      int(para[3]), int(para[4]), int(para[5])))
        hyper_parameterss.append(hyper_parameters)

cnn_modelss = []
threshold_models = []
output_models = []
for idx in range(cluster_size):
    states = torch.load('local_'+dataset_name+'_cluster_' + str(idx) + '.model')
    hyper_para = hyper_parameterss[idx]
    cnn_models = []
    weights = [None for _ in range(len(hyper_para))]
    for key, value in states.items():
        if key != 'threshold_model_state_dict' and key != 'output_model_state_dict':
            layer_id = int(key.split('_')[-1])
            weights[layer_id] = value
    in_channel = 1
    in_size = queries_dimension
    for weight_idx, weight in enumerate(weights):
        hyper = hyper_para[weight_idx]
        cnn_model = CNN_Model(in_channel, hyper.out_channel, hyper.kernel_size,
                              hyper.stride, hyper.padding, hyper.pool_type, hyper.pool_size)
        in_size = int((int((in_size - hyper.kernel_size + 2*(hyper.padding)) / hyper.stride) + 1) / hyper.pool_size)
        in_channel = hyper.out_channel
        cnn_model.load_state_dict(weight)
        cnn_model.eval()
        cnn_models.append(cnn_model)
    cnn_modelss.append(cnn_models)
        
    threshold_model_state_dict = states['threshold_model_state_dict']
    threshold_model = Threshold_Model()
    threshold_model.load_state_dict(threshold_model_state_dict)
    threshold_model.eval()
    threshold_models.append(threshold_model)
    
    output_model_state_dict = states['output_model_state_dict']
    output_model = Output_Model(in_size * in_channel)
    output_model.load_state_dict(output_model_state_dict)
    output_model.eval()
    output_models.append(output_model)

def get_local_cardinality(cnn_models, threshold_model, output_model, queries, thresholds):
    queries = queries.unsqueeze(2).permute(0,2,1)
    for model in cnn_models:
        queries = model(queries)
    threshold = threshold_model(thresholds)
    queries = queries.view(queries.shape[0], -1)
    estimates = output_model(queries, threshold)
    esti = torch.exp(estimates)
    return esti.detach()

def print_qerror(estimates, targets):
    qerror = []
    for i in range(estimates.shape[0]):
        left = estimates[i] + 1
        right = targets[i] + 1
        if left > right:
            qerror.append((left / right).item())
        else:
            qerror.append((right / left).item())
    return qerror

import time

model = Model()
model.load_state_dict(torch.load('global_'+dataset_name+'_punish_query_threshold_monotonic.model'))
model.eval()
test_loss = 0.0
precision = 0.0
recall = 0.0
miss_rate = 0.0
estimatesss = []
q_errors = []
start = time.time()
for batch_idx, (features, thresholds, distances, targets, cards) in enumerate(test_loader):
    if batch_idx % 100 == 0:
        print (batch_idx)
    current_batch_size = len(features)
    estimates = model(features, distances, thresholds)
    global_indicator = (estimates >= 0.5).float()
    local_estimates = []
    for cluster_id in range(cluster_size):
        local_estimates.append(get_local_cardinality(cnn_modelss[cluster_id], threshold_models[cluster_id],
                                                     output_models[cluster_id], features, thresholds))

    localss = torch.cat(local_estimates, dim = 1)
    cards_estimates = (localss * global_indicator).sum(dim=1).unsqueeze(1)
    cards = cards.sum(dim=1).unsqueeze(1)
    # print (torch.cat((cards_estimates, cards), dim=1))
    q_errors += print_qerror(cards_estimates, cards)
end = time.time()
mean = np.mean(q_errors)
percent90 = np.percentile(q_errors, 90)
percent95 = np.percentile(q_errors, 95)
percent99 = np.percentile(q_errors, 99)
median = np.median(q_errors)
maxi = np.max(q_errors)
print ('Testing: Mean Error {}, Median Error {}, 90 Percent {}, 95 Percent {}, 99 Percent {}, Max Percent {}, Latency {}'
       .format(mean, median, percent90, percent95, percent99, maxi, (end - start) / len(q_errors)))