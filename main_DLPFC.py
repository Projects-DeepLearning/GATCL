import os
import torch
import pandas as pd
import scanpy as sc
from sklearn import metrics
from GATCL import GATCL
from GATCL.utils import clustering

'''DLPFC 10x'''

device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
print(device)

# R path
os.environ['R_HOME'] = '/home/zsl/yes/envs/zsl/lib/R'


# Set the dataset and cluster number
dataset = ['151507', '151508', '151509', '151510', '151669', '151670', '151671', '151672', '151673', '151674', '151675','151676']
n_clusters = [7, 7, 7, 7, 5, 5, 5, 5, 7, 7, 7, 7]

# dataset = ['151674']
# n_clusters = [7]

i = 0
ARI_Average = 0


for dataset, n_clusters in zip(dataset, n_clusters):
    print(dataset, n_clusters)

    # read dataset
    file_fold = './Data/DLPFC/' + str(dataset)      # dataset path
    print(file_fold)
    adata = sc.read_visium(file_fold, count_file='filtered_feature_bc_matrix.h5', load_images=True)
    adata.var_names_make_unique()

    # model definition
    model = GATCL.GATCL(adata, device=device)

    # model training
    adata = model.train_GATCL()
    # print(adata)

    radius = 50
    clustering(adata, n_clusters, radius=radius, refinement=True)


    # ground truth
    df_meta = pd.read_csv(file_fold + '/metadata.tsv', sep='\t')  # ground truth
    df_meta_layer = df_meta['layer_guess']
    adata.obs['ground_truth'] = df_meta_layer.values
    adata = adata[~pd.isnull(adata.obs['ground_truth'])]

    # ARI
    ARI = metrics.adjusted_rand_score(adata.obs['domain'], adata.obs['ground_truth'])
    adata.uns['ARI'] = ARI

    print('Dataset:', dataset)
    print('ARI:', ARI)


    # save the image
    folder_name = "Result_DLPFC"
    # creat folder
    if not os.path.exists(folder_name):
        os.mkdir(folder_name)

    sc.pl.spatial(adata,
                  img_key="hires",
                  cmap="hot",
                  color=["ground_truth", "domain"],
                  title=["Ground truth", "ARI=%.4f"%ARI],
                  show=False,
                  save=dataset+"_ARI:" +str(round(ARI, 4)) + ".png")

    print('saved')

    # ARI is saved in Results_ARI.txt
    file_path = os.path.join(folder_name, "Results_ARI.txt")
    with open(file_path, 'a') as f:
        f.write(dataset+" ARI: " +str(round(ARI, 4)) + "\n")
    print('ARI was saved')

    i = i+1
    ARI_Average = ARI_Average + ARI
print(ARI_Average/i)


