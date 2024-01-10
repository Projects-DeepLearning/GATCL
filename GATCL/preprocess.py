import numpy as np
import scanpy as sc
import scipy.sparse as sp
import os
import torch
import random
from torch.backends import cudnn
from scipy.sparse.csc import csc_matrix
from scipy.sparse.csr import csr_matrix
from scipy.sparse import coo_matrix
from torch_geometric.utils import from_scipy_sparse_matrix

def permutation(feature, mean=0, std=1):
    # print(type(feature))
    mask = np.random.rand(*feature.shape) < 0.6
    mask = np.array(mask)
    feature = feature * mask
    ids = np.arange(feature.shape[0])
    ids = np.random.permutation(ids)
    feature_permutated = feature[ids]
    # noise = np.random.normal(mean, std, size=feature_permutated.shape)
    # print(type(feature_permutated))

    return feature_permutated


def calculate_distance(position):
    n_spot = position.shape[0]
    distance_matrix = np.zeros([n_spot, n_spot])

    for i in range(n_spot):
        x = position[i, :]
        for j in range(i+1, n_spot):
            y = position[j, :]
            d = np.sqrt(np.sum(np.square(x-y)))
            distance_matrix[i, j] = d
            distance_matrix[j, i] = d
            
    return distance_matrix 


def construct_interaction(adata, n_neighbors=3):

    position = adata.obsm['spatial']
    distance_matrix = calculate_distance(position)
    n_spot = distance_matrix.shape[0]

    adata.obsm['distance_matrix'] = distance_matrix  # 1

    # k-nearest neighbors
    interaction = np.zeros([n_spot, n_spot])
    for i in range(n_spot):
        vec = distance_matrix[i, :]
        distance = vec.argsort()
        for t in range(1, n_neighbors + 1):
            y = distance[t]
            interaction[i, y] = 1

    adata.obsm['graph_neigh'] = interaction

    adj = interaction
    adj = adj + adj.T
    adj = np.where(adj>1, 1, adj)

    adata.obsm['adj'] = adj


def preprocess(adata):
    sc.pp.highly_variable_genes(adata, flavor="seurat_v3", n_top_genes=3000)
    sc.pp.normalize_total(adata, target_sum=1e4)
    sc.pp.log1p(adata)
    sc.pp.scale(adata, zero_center=False, max_value=10)
    
def get_feature(adata, deconvolution=False):
    if deconvolution:
       adata_Vars = adata
    else:   
       adata_Vars = adata[:, adata.var['highly_variable']]   # 选取高可变基因  4226×3000

    if isinstance(adata_Vars.X, csc_matrix) or isinstance(adata_Vars.X, csr_matrix):
       feat = adata_Vars.X.toarray()[:, ]
    else:
       feat = adata_Vars.X[:, ] 

    feat_a = permutation(feat)

    adata.obsm['feat'] = feat
    adata.obsm['feat_a'] = feat_a
    
def add_contrastive_label(adata):
    n_spot = adata.n_obs
    one_matrix = np.ones([n_spot, 1])
    zero_matrix = np.zeros([n_spot, 1])
    label_CSL = np.concatenate([one_matrix, zero_matrix], axis=1)
    adata.obsm['label_CSL'] = label_CSL
    
def normalize_adj(adj):
    adj = sp.coo_matrix(adj)
    rowsum = np.array(adj.sum(1))
    d_inv_sqrt = np.power(rowsum, -0.5).flatten()
    d_inv_sqrt[np.isinf(d_inv_sqrt)] = 0.
    d_mat_inv_sqrt = sp.diags(d_inv_sqrt)
    adj = adj.dot(d_mat_inv_sqrt).transpose().dot(d_mat_inv_sqrt)
    return adj.toarray()

def preprocess_adj(adj):
    adj_normalized = normalize_adj(adj)+np.eye(adj.shape[0])
    return adj_normalized

def preprocess_adjtoGAT(adj):
    # from scipy.sparse import coo_matrix
    # from torch_geometric.utils import from_scipy_sparse_matrix

    adj_normalized = coo_matrix(adj)
    edge_index, edge_weight = from_scipy_sparse_matrix(adj_normalized)
    edge_index = torch.tensor(edge_index, dtype=torch.long)
    edge_weight = torch.tensor(edge_weight, dtype=torch.float)
    return edge_index, edge_weight

def fix_seed(seed):
    os.environ['PYTHONHASHSEED'] = str(seed)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    cudnn.deterministic = True
    cudnn.benchmark = False

    os.environ['PYTHONHASHSEED'] = str(seed)
    os.environ['CUBLAS_WORKSPACE_CONFIG'] = ':4096:8'
    
    
