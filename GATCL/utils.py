from sklearn.decomposition import PCA
import pandas as pd
import numpy as np
from sklearn import metrics


def mclust_R(adata, num_cluster, modelNames='EEE', used_obsm='emb_pca', random_seed=2020):
    """\
    Clustering using the mclust algorithm.
    The parameters are the same as those in the R package mclust.
    """
    np.random.seed(random_seed)
    import rpy2.robjects as robjects
    robjects.r.library("mclust")

    import rpy2.robjects.numpy2ri
    rpy2.robjects.numpy2ri.activate()
    r_random_seed = robjects.r['set.seed']
    r_random_seed(random_seed)
    rmclust = robjects.r['Mclust']

    res = rmclust(rpy2.robjects.numpy2ri.numpy2rpy(adata.obsm[used_obsm]), num_cluster, modelNames)
    mclust_res = np.array(res[-2])

    adata.obs['mclust'] = mclust_res
    adata.obs['mclust'] = adata.obs['mclust'].astype('int')
    adata.obs['mclust'] = adata.obs['mclust'].astype('category')
    return adata


def clustering(adata, n_clusters=7, radius=50, key='emb', threshold=0.06, refinement=True):
    pca = PCA(n_components=20, random_state=42) 
    
    # clustering 1
    embedding = pca.fit_transform(adata.obsm['emb'].copy())
    adata.obsm['emb_pca'] = embedding
    adata = mclust_R(adata, used_obsm='emb_pca', num_cluster=n_clusters)
    adata.obs['label'] = adata.obs['mclust']
    new_type = refine_label(adata, radius, key='label')
    adata.obs['label_refined'] = new_type
    
    # clustering 2
    embedding = pca.fit_transform(adata.obsm['emb_reg'].copy())
    adata.obsm['emb_reg_pca'] = embedding
    adata = mclust_R(adata, used_obsm='emb_reg_pca', num_cluster=n_clusters)
    adata.obs['label_reg'] = adata.obs['mclust']
    new_type = refine_label(adata, radius, key='label_reg')
    adata.obs['label_reg_refined'] = new_type
    
    # Silhouette
    SIL = metrics.silhouette_score(adata.obsm['emb_pca'], adata.obs['label'], metric='euclidean')
    SIL_reg = metrics.silhouette_score(adata.obsm['emb_reg_pca'], adata.obs['label_reg'], metric='euclidean')
    
    if abs(SIL-SIL_reg) > threshold and SIL_reg > SIL:
       if refinement: 
          adata.obs['domain'] = adata.obs['label_reg_refined']
       else:   
          adata.obs['domain'] = adata.obs['label_reg']
    else:
       if refinement: 
          adata.obs['domain'] = adata.obs['label_refined']
       else:
          adata.obs['domain'] = adata.obs['label'] 
       
def refine_label(adata, radius=50, key='label'):
    n_neigh = radius
    new_type = []
    old_type = adata.obs[key].values

    if 'distance_matrix' not in adata.obsm.keys():
        raise ValueError("Distance matrix is not existed!")
    distance = adata.obsm['distance_matrix'].copy()
           
    n_cell = distance.shape[0]
    
    for i in range(n_cell):
        vec  = distance[i, :]
        index = vec.argsort()
        neigh_type = []
        for j in range(1, n_neigh+1):
            neigh_type.append(old_type[index[j]])
        max_type = max(neigh_type, key=neigh_type.count)
        new_type.append(max_type)
        
    new_type = [str(i) for i in list(new_type)]
    
    return new_type





