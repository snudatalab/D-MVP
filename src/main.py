"""
Accurate Graph-based Multi-Positive Unlabeled Learning via Disentangled Multi-view Feature Propagation
Authors:
- Junghun Kim (bandalg97@snu.ac.kr), Seoul National University
- Hoyoung Yoon (crazy8597@snu.ac.kr), Seoul National University
- Ka Hyun Park (kahyunpark@snu.ac.kr), Seoul National University
- U Kang (ukang@snu.ac.kr), Seoul National University
"""

import argparse
import data
import os
import random

import pandas as pd
import numpy as np

import torch
from torch import optim

import models

def to_device(gpu):
    if gpu is not None and torch.cuda.is_available():
        return torch.device('cuda:{}'.format(gpu))
    else:
        return torch.device('cpu')

def print_results(logs, best_epoch):
    rst = logs.loc[best_epoch:best_epoch]
    rst = rst.drop(rst.columns[[1]], axis=1)
    test_f1, test_acc = rst.iloc[0][['test_f1', 'test_acc']]
    print(f'Test F1: {test_f1} \t Test accuracy: {test_acc}')

def parse_args():
    """
    parser arguments to run program in cmd
    """
    parser = argparse.ArgumentParser()

    # Pre-sets before start the program
    parser.add_argument('--data', type=str, default='Cora')
    parser.add_argument('--gpu', type=int, default=0)
    parser.add_argument('--seed', type=int, default=0)
    parser.add_argument('--verbose', action='store_true')

    # Hyperparameters for training
    parser.add_argument('--epochs', type=int, default=500)
    parser.add_argument('--trn-ratio', type=float, default=0.2)
    parser.add_argument('--pos-class', type=int, default=3)
    parser.add_argument('--patience', type=int, default=1)
    parser.add_argument('--noise-ratio', type=float, default=0.0)

    # Hyperparameters for models
    parser.add_argument('--layers', type=int, default=1)
    parser.add_argument('--units', type=int, default=16)
    parser.add_argument('--iteration', type=int, default=10)
    parser.add_argument('--prior', type=str, default='unknown')

    return parser.parse_args()


def main():
    # initial setting for torch
    args = parse_args()
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    random.seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(args.seed)
        torch.cuda.manual_seed_all(args.seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    device = to_device(args.gpu)

    # Call the designated data
    features, labels, edges, trn_nodes, test_nodes, vld_nodes = data.read_data(
        args.data, args.trn_ratio, noise_ratio=args.noise_ratio, pos_class=args.pos_class, verbose=args.verbose)
    num_nodes = features.size(0)
    num_features = features.size(1)
    trn_labels = torch.zeros(num_nodes, dtype=torch.float)
    trn_labels[trn_nodes] = labels[trn_nodes].type(torch.float)

    # Initialize model
    model = models.HOPE(num_features, args.pos_class + 1, args.units, args.layers).to(device)
    model.preprocess(features, edges, device)
    optimizer = optim.Adam(model.parameters())

    RPN = labels.cpu().numpy()
    RPN = torch.from_numpy(np.delete(RPN, trn_nodes))
    real_PN = models.compute_prior(RPN, num_class=args.pos_class+1)

    if args.prior == 'unknown':
        prior = models.compute_prior(trn_labels, num_class=args.pos_class+1)
    else:
        prior = real_PN

    # Define loss
    loss = models.Hope_loss_with_contrastive(edges.t().numpy(), prior, 0.99, False, trn_labels,
                                             trn_labels, args.pos_class + 1)

    features = features.to(device)
    edges = edges.to(device)
    trn_labels = trn_labels.to(device)

    # Train the model
    best_epoch, logs, best_loss, best_model = models.train_model(
        model, features, edges, labels, test_nodes, loss, optimizer, trn_labels, args.epochs)

    # Iterations
    patience = 0
    cur_best_epoch, cur_logs, cur_best_loss, cur_best_model = best_epoch, logs, best_loss, best_model

    for i in range(args.iteration):
        with torch.no_grad():
            predictions = cur_best_model(features, edges)
            posterior, exp_labels = models.estimate_prior(trn_nodes, predictions, args.pos_class+1, trn_labels)
            edge_weights = models.generate_edge_weights(cur_best_model, features, edges)

        model = models.HOPE(num_features, args.pos_class + 1, args.units, args.layers).to(device)
        model.preprocess(features, edges, device)
        optimizer = optim.Adam(model.parameters())

        loss = models.Hope_loss_with_contrastive(edges.cpu().t().numpy(), posterior, 0.99, True,
                                                 exp_labels.cpu(), trn_labels.cpu(), args.pos_class + 1)

        cur_best_epoch, cur_logs, cur_best_loss, cur_best_model = models.train_model(
            model, features, edges, labels, test_nodes, loss, optimizer, trn_labels, args.epochs, edge_weights)

        if cur_best_loss > best_loss:
            patience += 1
            if patience > args.patience:
                break
        else:
            best_epoch, logs, best_loss, best_model = cur_best_epoch, cur_logs, cur_best_loss, cur_best_model
            patience = 0

    print_results(logs, best_epoch)

if __name__ == '__main__':
    main()
