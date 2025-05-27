"""
Accurate Graph-based Multi-Positive Unlabeled Learning via Disentangled Multi-view Feature Propagation
Authors:
- Junghun Kim (bandalg97@snu.ac.kr), Seoul National University
- Hoyoung Yoon (crazy8597@snu.ac.kr), Seoul National University
- Ka Hyun Park (kahyunpark@snu.ac.kr), Seoul National University
- U Kang (ukang@snu.ac.kr), Seoul National University
"""

"""
This file includes functions for loading data
"""

import os
from collections import defaultdict
import numpy as np

import torch
from torch_geometric import datasets


def compute_heterophilic_ratio(edge_index, node_labels):
    """
    Computes the heterophilic ratio, i.e., the proportion of edges connecting nodes with different labels.

    Args:
        edge_index (Tensor): Edge indices of shape [2, E].
        node_labels (Tensor): Labels for each node of shape [N].

    Returns:
        float: Heterophilic ratio in the graph.
    """
    src, tgt = edge_index

    label_diff = (node_labels[src] != node_labels[tgt]).float()
    heterophilic_ratio = label_diff.mean().item()

    return heterophilic_ratio

def add_random_edges(edge_index, num_nodes, noise_ratio):
    """
    Adds symmetric random edges as noise to the original edge index.

    Args:
        edge_index (Tensor): Original edge indices of shape [2, E].
        num_nodes (int): Total number of nodes in the graph.
        noise_ratio (float): Ratio of noise edges to be added.

    Returns:
        Tensor: Updated edge index with noise, shape [2, E'].
    """
    num_edges = edge_index.size(1)
    num_noise_edges = int(num_edges/2 * noise_ratio)

    random_source_nodes = torch.randint(0, num_nodes, (num_noise_edges,))
    random_target_nodes = torch.randint(0, num_nodes, (num_noise_edges,))

    mask = random_source_nodes != random_target_nodes
    random_source_nodes = random_source_nodes[mask]
    random_target_nodes = random_target_nodes[mask]

    noise_edges = torch.stack([random_source_nodes, random_target_nodes], dim=0)
    combined_edges = torch.cat([edge_index, noise_edges], dim=1)

    edges_np = combined_edges.numpy().transpose()

    edge_map = defaultdict(set)
    for src, dst in edges_np:
        if src != dst:
            edge_map[src].add(dst)
            edge_map[dst].add(src)

    sorted_edges = []
    for src in sorted(edge_map):
        for dst in sorted(edge_map[src]):
            sorted_edges.append((src, dst))

    sorted_edges_np = np.array(sorted_edges, dtype=np.int64).transpose()
    sorted_edge_index = torch.tensor(sorted_edges_np, dtype=torch.long)

    return sorted_edge_index

def preprocess_edges(edges):
    """
    Converts edge index into a sorted undirected edge list without duplicates or self-loops.

    Args:
        edges (Tensor): Edge indices of shape [2, E].

    Returns:
        ndarray: Preprocessed edge indices of shape [2, E_clean].
    """
    m = defaultdict(lambda: set())
    for src, dst in edges.t():
        src = src.item()
        dst = dst.item()
        if src != dst:
            m[src].add(dst)
            m[dst].add(src)

    edges = []
    for src in sorted(m):
        for dst in sorted(m[src]):
            edges.append((src, dst))
    return np.array(edges, dtype=np.int64).transpose()

def to_pu_setting(labels, pos_number):
    """
    Converts multi-class labels to PU labels by assigning top-k classes as positive.

    Args:
        labels (ndarray): Original class labels [N].
        pos_number (int): Number of positive classes to retain.

    Returns:
        ndarray: PU-style labels with 0 as negative and 1~k as positive classes.
    """
    count = np.bincount(labels)
    pu_labels = np.zeros_like(labels)
    for i in range(pos_number):
        positive_nodes = labels == count.argmax()
        pu_labels[positive_nodes] = i+1
        count[count.argmax()]=-1

    return pu_labels

def split_nodes(labels, trn_ratio, vld_ratio, pos_number=1, seed=0):
    """
    Splits the dataset into train, validation, and test nodes under the PU setting.

    Args:
        labels (ndarray): PU labels of shape [N].
        trn_ratio (float): Ratio of labeled positives used for training.
        vld_ratio (float): Ratio of positive and negative nodes used for validation.
        pos_number (int): Number of positive classes.
        seed (int): Random seed.

    Returns:
        Tuple[np.ndarray, np.ndarray, np.ndarray]:
            Indices of training nodes, test nodes, and validation nodes.
    """
    state = np.random.RandomState(seed)

    all_nodes = np.arange(labels.shape[0])
    pos_nodes = []
    for i in range(pos_number):
        pos_nodes.append(all_nodes[labels == i+1])
    neg_nodes = all_nodes[labels == 0]

    trn_nodes = []
    n_pos_nodes = len(all_nodes[labels != 0])
    n_trn_nodes = int(len(all_nodes) * trn_ratio)

    vld_nodes = []
    for pos_nodes_i in pos_nodes:
        trn_temp = state.choice(pos_nodes_i, size=int(pos_nodes_i.shape[0] * n_trn_nodes / n_pos_nodes), replace=False)
        pos_nodes_i = np.array(list(set(pos_nodes_i).difference(set(trn_temp))))
        vld_temp = state.choice(pos_nodes_i, size=int(pos_nodes_i.shape[0] * vld_ratio), replace=False)
        trn_nodes.append(trn_temp)
        vld_nodes.append(vld_temp)

    pos_nodes = np.concatenate(pos_nodes)
    trn_nodes = np.concatenate(trn_nodes)
    vld_nodes = np.concatenate(vld_nodes)
    test_nodes = np.array(list(set(pos_nodes).difference(set(trn_nodes)).difference(set(vld_nodes))))
    vld_neg_nodes = state.choice(neg_nodes, size=int(neg_nodes.shape[0] * vld_ratio), replace=False)
    test_neg_nodes = np.array(list(set(neg_nodes).difference(set(vld_neg_nodes))))
    test_nodes = np.concatenate([test_nodes, test_neg_nodes])
    vld_nodes = np.concatenate([vld_nodes, vld_neg_nodes])

    return trn_nodes, test_nodes, vld_nodes

def read_data(dataset, trn_ratio, vld_ratio=0.0, noise_ratio=0.0, pos_class=1, verbose=False):
    """
    Loads and processes a graph dataset for PU learning.

    Args:
        dataset (str): Dataset name. Supported: 'Cora', 'CiteSeer', 'Cora_ML', 'CiteSeer_full', 'chameleon'.
        trn_ratio (float): Proportion of positive nodes used in training.
        vld_ratio (float): Proportion of nodes used for validation.
        noise_ratio (float): Ratio of random noise edges added to the graph.
        pos_class (int): Number of positive classes (1 = binary PU, >1 = multi-PU).
        verbose (bool): Whether to print dataset statistics.

    Returns:
        Tuple[Tensor, Tensor, Tensor, np.ndarray, np.ndarray, np.ndarray]:
            - Node features (FloatTensor) [N, D]
            - PU labels (LongTensor) [N]
            - Edge index with noise (LongTensor) [2, E]
            - Training node indices (np.ndarray)
            - Test node indices (np.ndarray)
            - Validation node indices (np.ndarray)
    """
    root = '../data'
    root_cached = os.path.join(root, 'cached', dataset)

    if not os.path.exists(root_cached):
        if dataset in ['Cora', 'CiteSeer']:
            data = datasets.Planetoid(root, dataset)
        elif dataset in ['Cora_ML', 'CiteSeer_full']:
            dataset = dataset.replace('_full', '')
            data = datasets.CitationFull(root, dataset)
        elif dataset in ["chameleon"]:
            data = datasets.WikipediaNetwork(root, dataset)
        else:
            raise NotImplementedError

        node_x = data.data.x
        node_x[node_x.sum(dim=1) == 0] = 1
        node_x = node_x / node_x.sum(dim=1, keepdim=True)
        node_y = to_pu_setting(data.data.y, pos_number=pos_class)
        edges = preprocess_edges(data.data.edge_index)

        os.makedirs(root_cached, exist_ok=True)
        np.save(os.path.join(root_cached, 'x'), node_x)
        np.save(os.path.join(root_cached, 'y'), node_y)
        np.save(os.path.join(root_cached, 'edges'), edges)

    node_x = torch.from_numpy(np.array(np.load(os.path.join(root_cached, 'x.npy'), allow_pickle=True), dtype=np.float32))
    node_y = torch.from_numpy(np.array(np.load(os.path.join(root_cached, 'y.npy'), allow_pickle=True), dtype=np.int64))
    edges = torch.from_numpy(np.load(os.path.join(root_cached, 'edges.npy'), allow_pickle=True))
    edge_with_noise = add_random_edges(edges, node_x.shape[0], noise_ratio)

    trn_nodes, test_nodes, vld_nodes = split_nodes(node_y, trn_ratio, vld_ratio, pos_number=pos_class)

    if verbose:
        classes = ['N', 'P1', 'P2', 'P3']
        unique_values, counts = torch.unique(node_y, return_counts=True)

        print(f'----------------- Data statistics of {dataset} -----------------')
        print(f'Heterophilic ratio: {compute_heterophilic_ratio(edge_with_noise, node_y):.4f}')
        print('# of nodes:', node_x.size(0))
        print('# of features:', node_x.size(1))
        print(f'# of edges: {edges.size(1)} -> {edge_with_noise.size(1)}')
        print('# of instances for each class:', ', '.join(f"{cnt}({cls})" for cls, cnt in zip(classes, counts.tolist())))
        print(f'# of train / val / test instances: {len(trn_nodes)} / {len(vld_nodes)} / {len(test_nodes)}\n')

    return node_x, node_y, edge_with_noise, trn_nodes, test_nodes, vld_nodes
