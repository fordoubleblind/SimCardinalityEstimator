from __future__ import print_function
import h5py
import pickle
import argparse
from torch.autograd import Variable
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision import datasets, transforms
from numpy import dot
from numpy.linalg import norm
from scipy import spatial
import torch
import torch.utils.data
import numpy as np
import random
from multiprocessing import Pool

data = h5py.File('fashion-mnist-784-euclidean.hdf5', 'r')
data_train = np.array(data['train'])
data_test = np.array(data['test'])
with open('clusters_fashion_mnist_784_euclidean.pkl', 'rb') as f:
    clusters = pickle.load(f)
with open('ground_truth_fashion_mnist_784_euclidean_0_0_0_5.pkl', 'rb') as f:
    ground_truth_total = pickle.load(f)

def euclidean_dist_normalized(x1, x2=None, eps=1e-8):
    if np.isnan(np.sum(x2)):
        print (np.isnan(x2).sum(), x2)
        return 1.0
    left = x1
    right = x2
    return np.sqrt(((left - right) ** 2).mean())

cluster_size = len(clusters)
# cluster_size = 2
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

ground_truth_total_level = [[[] for _ in range(query_size)] for _ in range(cluster_size)]
for clus in range(cluster_size):
    for t in ground_truth_total[clus]:
        if t[1] >= query_size:
            break
        ground_truth_total_level[t[0]][t[1]].append(t)

centroids = []
for cluster in clusters:
    print (len(cluster))
    centroids.append(np.mean(cluster, 0))
    # print (type(centroids[-1]))

def prepare_for_cluster(cluster_id):
    batch_size = 128
    min_card = 1e10
    max_card = 0
    train_queries = []
    train_distances = []
    train_thresholds = []
    train_targets = []
    for query_id in range(int(query_size * 0.8)):
        cardinality = 0
        for threshold_id, threshold in enumerate(np.arange(min_threshold, max_threshold, slot)):
            cardinality += ground_truth_total_level[cluster_id][query_id][threshold_id][-1]
            # if random.random() < 0.02:
            #     print ('cluster: {}, query: {}, threshold: {}, cardinality: {}'.format(cluster_id,query_id,threshold,cardinality))
            if cardinality > 0:
                train_queries.append(data_test[query_id] / normalize_factor)
                train_distances.append([distance_function(data_test[query_id] / normalize_factor, centroids[cluster_id] / normalize_factor)])
                train_thresholds.append([threshold+slot])
                train_targets.append([cardinality])

    test_queries = []
    test_distances = []
    test_thresholds = []
    test_targets = []
    for query_id in range(int(query_size * 0.8),query_size):
        cardinality = 0
        for threshold_id, threshold in enumerate(np.arange(min_threshold, max_threshold, slot)):
            cardinality += ground_truth_total_level[cluster_id][query_id][threshold_id][-1]
            if cardinality > 0:
                test_queries.append(data_test[query_id] / normalize_factor)
                test_distances.append([distance_function(data_test[query_id] / normalize_factor, centroids[cluster_id] / normalize_factor)])
                test_thresholds.append([threshold+slot])
                test_targets.append([cardinality])
    
    train_loader = torch.utils.data.DataLoader(
        torch.utils.data.TensorDataset(torch.FloatTensor(train_queries), torch.FloatTensor(train_distances), torch.FloatTensor(train_thresholds), torch.FloatTensor(train_targets)), batch_size=batch_size, shuffle=True)
    test_loader = torch.utils.data.DataLoader(
        torch.utils.data.TensorDataset(torch.FloatTensor(test_queries), torch.FloatTensor(test_distances), torch.FloatTensor(test_thresholds), torch.FloatTensor(test_targets)), batch_size=batch_size, shuffle=True)
    
    return train_loader, test_loader, min_card, max_card

train_loaders = []
test_loaders = []
min_cards = []
max_cards = []
for cluster_id in range(cluster_size):
    print (cluster_id)
    train, test, min_card, max_card = prepare_for_cluster(cluster_id)
    train_loaders.append(train)
    test_loaders.append(test)
    min_cards.append(min_card)
    max_cards.append(max_card)

# queries_dimension = 200
# hidden_num = 128

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

def l1_loss(estimates, targets, eps=1e-5):
    estimates = torch.exp(estimates)
    qerror = 0.0
    for i in range(estimates.shape[0]):
        if estimates[i] > targets[i] + 0.1:
            qerror += ((estimates[i] / (targets[i] + 0.1)))
        else:
            qerror += (((targets[i] + 0.1) / estimates[i]))
    return qerror / estimates.shape[0]

def mse_loss(estimates, targets, eps=1e-5):
#     print (torch.cat((estimates, targets), 1))
    return F.mse_loss(estimates, torch.log(targets))

def print_loss(estimates, targets):
    esti = torch.exp(estimates)
#     print (torch.cat((estimates, esti, targets), 1))
    qerror = []
    for i in range(esti.shape[0]):
        if esti[i] > targets[i] + 0.1:
            qerror.append((esti[i] / (targets[i] + 0.1)).item())
        else:
            qerror.append(((targets[i] + 0.1) / esti[i]).item())
    
    return F.mse_loss(esti, targets), np.mean(qerror), np.max(qerror)

from random import sample

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
        

def only_test(cnn_models, threshold_model, output_model, test):
    for model in cnn_models:
        model.eval()
    threshold_model.eval()
    output_model.eval()
    q_errors = []
    mape_errors = []
    for batch_idx, (queries, _, thresholds, targets) in enumerate(test):
        queries = Variable(queries)
        thresholds = Variable(thresholds)
        targets = Variable(targets)
        queries = queries.unsqueeze(2).permute(0,2,1)
        for model in cnn_models:
            queries = model(queries)
        threshold = threshold_model(thresholds)
        queries = queries.view(queries.shape[0], -1)
        estimates = output_model(queries, threshold)

        loss = l1_loss(estimates, targets)
        
        esti = torch.exp(estimates)
        for i in range(esti.shape[0]):
            if esti[i] > targets[i] + 0.1:
                q_errors.append((esti[i] / (targets[i] + 0.1)).item())
            else:
                q_errors.append(((targets[i] + 0.1) / esti[i]).item())
        mape_errors.append((torch.abs(esti - targets) / (targets + 0.1)).mean().item())
    mean = np.mean(q_errors)
    percent90 = np.percentile(q_errors, 90)
    percent95 = np.percentile(q_errors, 95)
    percent99 = np.percentile(q_errors, 99)
    median = np.median(q_errors)
    maxi = np.max(q_errors)
    mape_mean = np.mean(mape_errors)
    print ('Testing: Mean Error {}, Median Error {}, 90 Percent {}, 95 Percent {}, 99 Percent {}, Max Percent {}, MAPE {}'
           .format(mean, median, percent90, percent95, percent99, maxi, mape_mean))
    return mean, mape_mean
    
    
def train_and_test(cnn_models, threshold_model, output_model, opt, train, test, episode, stop_batch=-1):
    print ('size: {}'.format(len(train)))
    test_errors = []
    q_errors_buffer = [100000.0 for i in range(3)]
    for e in range(episode):
        for model in cnn_models:
            model.train()
        threshold_model.train()
        output_model.train()
        for batch_idx, (queries, _, thresholds, targets) in enumerate(train):
    #         print (torch.cat((queries, thresholds), 1)[0])
            if stop_batch == batch_idx:
                break
            queries = Variable(queries)
            thresholds = Variable(thresholds)
            targets = Variable(targets)
    #         print (targets)
            opt.zero_grad()
            queries = queries.unsqueeze(2).permute(0,2,1)
            for model in cnn_models:
                queries = model(queries)
            threshold = threshold_model(thresholds)
            queries = queries.view(queries.shape[0], -1)
            estimates = output_model(queries, threshold)
            
            loss = l1_loss(estimates, targets)
            loss.backward()
            opt.step()
            if batch_idx % 100 == 0:
                print('Training: Iteration {0}, Batch {1}, Loss {2}'.format(e, batch_idx, loss.item()))
        for model in cnn_models:
            model.eval()
        threshold_model.eval()
        output_model.eval()
        test_loss = 0.0
        mse_error = 0.0
        q_mean = 0.0
        q_max = 0.0
        for batch_idx, (queries, _, thresholds, targets) in enumerate(test):
            queries = Variable(queries)
            thresholds = Variable(thresholds)
            targets = Variable(targets)
            
            queries = queries.unsqueeze(2).permute(0,2,1)
            for model in cnn_models:
                queries = model(queries)
            threshold = threshold_model(thresholds)
            queries = queries.view(queries.shape[0], -1)
            estimates = output_model(queries, threshold)
            
            loss = l1_loss(estimates, targets)
            mse, qer_mean, qer_max = print_loss(estimates, targets)
            test_loss += loss.item()
            mse_error += mse.item()
            q_mean += qer_mean
            if qer_max > q_max:
                q_max = qer_max
        test_loss /= len(test)
        mse_error /= len(test)
        q_mean /= len(test)
        test_errors.append(q_mean)
        print ('Testing: Iteration {0}, Loss {1}, MSE_error {2}, Q_error_mean {3}, Q_error_max {4}'.format(e, test_loss, mse_error, q_mean, q_max))
    return np.mean(test_errors[-3:]), test_errors

reload = True
if reload:
# TunableParameters(): def __init__(self, out_channel, kernel_size, stride, padding, pool_size, pool_type):
    print ('Loading Select Hyperparameters ...')
    next_cnn_parameterss = []
    with open('cnn_hyper_parameters.hyperpara', 'r') as handle:
        for paras in handle.readlines():
            hyper_parameters = []
            for para in paras.split(';'):
                para = para.split(' ')
                hyper_parameters.append(TunableParameters(int(para[0]), int(para[1]), int(para[2]),
                                                        int(para[3]), int(para[4]), int(para[5])))
            next_cnn_parameterss.append(hyper_parameters)
    next_cnn_modelss = []
    threshold_models = []
    next_output_models = []
    for idx in range(cluster_size):
        hyper_para = next_cnn_parameterss[idx]
        cnn_models = []
        weights = [None for _ in range(len(hyper_para))]
        in_channel = 1
        in_size = queries_dimension
        for weight_idx, weight in enumerate(weights):
            hyper = hyper_para[weight_idx]
            cnn_model = CNN_Model(in_channel, hyper.out_channel, hyper.kernel_size,
                                hyper.stride, hyper.padding, hyper.pool_type, hyper.pool_size)
            in_size = int((int((in_size - hyper.kernel_size + 2*(hyper.padding)) / hyper.stride) + 1) / hyper.pool_size)
            in_channel = hyper.out_channel
            cnn_model.train()
            cnn_models.append(cnn_model)
        next_cnn_modelss.append(cnn_models)
            
        threshold_model = Threshold_Model()
        threshold_model.train()
        threshold_models.append(threshold_model)
        
        output_model = Output_Model(in_size * in_channel)
        output_model.train()
        next_output_models.append(output_model)
else:
    raise Exception('Hyperparameter Selection is not unvailable for current version, please reload from given file.')  
print ('Starting Final Training ...')
q_errors = []
mapes = []
training_errors = []
for idx in range(cluster_size):
    print (idx)
    test = test_loaders[idx]
    train = train_loaders[idx]
    paras = [{"params": model.parameters()} for model in next_cnn_modelss[idx]]
    paras.append({"params": threshold_models[idx].parameters()})
    paras.append({"params": next_output_models[idx].parameters()})
    opt = optim.Adam(paras, lr=2e-4)
    episode = 10
    _, errors = train_and_test(next_cnn_modelss[idx], threshold_models[idx], next_output_models[idx], opt, train, test, episode)
    training_errors.append(errors)
    q_error, mape = only_test(next_cnn_modelss[idx], threshold_models[idx], next_output_models[idx], test)
    q_errors.append(q_error)
    mapes.append(mape)
print ('Completed')
for idx in range(cluster_size):
    states = {}
    for idd, cnn_model in enumerate(next_cnn_modelss[idx]):
        states['cnn_model_state_dict_' + str(idd)] = cnn_model.state_dict()
    states['threshold_model_state_dict'] = threshold_models[idx].state_dict()
    states['output_model_state_dict'] = next_output_models[idx].state_dict()
    torch.save(states, 'local_fashion_mnist_784_euclidean_cluster_' + str(idx) + '.model')
print ('Model Saved')