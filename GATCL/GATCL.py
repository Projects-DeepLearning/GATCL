import torch
from GATCL.preprocess import preprocess_adj, preprocess, construct_interaction, add_contrastive_label, get_feature, fix_seed, preprocess_adjtoGAT
import numpy as np
from .model import Encoder
from tqdm import tqdm
from torch import nn
import torch.nn.functional as F



class GATCL():

    def __init__(self, adata, random_seed=50, add_regularization=True, device='cuda:0'):

       self.adata = adata

       self.random_seed = random_seed
       self.add_regularization = True
       self.device = device
       
       fix_seed(self.random_seed)
       preprocess(self.adata)
       construct_interaction(self.adata)
       add_contrastive_label(self.adata)
       
       self.adata_output = self.adata.copy()
    
    def train_GATCL(self):
       if self.add_regularization:
          adata = self.adata_output.copy()
          get_feature(adata)
          model = Train(adata, device=self.device)

          emb = model.train()
          self.adata_output.obsm['emb'] = emb
          
          fix_seed(self.random_seed)
          adata = self.adata_output.copy()
          get_feature(adata)
          model = Train(adata, add_regularization=True, device=self.device)
          emb_regularization = model.train()
          self.adata_output.obsm['emb_reg'] = emb_regularization
          
       else:
          model = Train(self.adata.copy())
          emb = model.train()
          self.adata_output.obsm['emb'] = emb
          
       return self.adata_output   


class Train():
    def __init__(self, 
            adata,
            device='cuda:0',
            learning_rate=0.001,
            learning_rate_sc = 0.01,
            weight_decay=0.00,
            epochs=5000,
            dim_input=3000,
            dim_output=64,
            random_seed = 50,
            alpha = 10,
            beta = 1,
            theta = 0.1,
            lamda1 = 10,
            lamda2 = 1,
            add_regularization = False
            ):

        self.adata = adata.copy()
        self.device = device
        self.learning_rate=learning_rate
        self.learning_rate_sc = learning_rate_sc
        self.weight_decay=weight_decay
        self.epochs=epochs
        self.alpha = alpha
        self.beta = beta
        self.theta = theta
        self.lamda1 = lamda1
        self.lamda2 = lamda2
        self.add_regularization = add_regularization
        
        self.features = torch.FloatTensor(adata.obsm['feat'].copy()).to(self.device)
        self.features_a = torch.FloatTensor(adata.obsm['feat_a'].copy()).to(self.device)
        self.label_CSL = torch.FloatTensor(adata.obsm['label_CSL']).to(self.device)
        self.adj = adata.obsm['adj']
        self.graph_neigh = torch.FloatTensor(adata.obsm['graph_neigh'].copy() + np.eye(self.adj.shape[0])).to(self.device)
        self.dim_input = self.features.shape[1]
        self.dim_output = dim_output

        self.adj = preprocess_adj(self.adj)

        self.edge_index, self.edge_weight = preprocess_adjtoGAT(self.adj)

        self.edge_index = torch.LongTensor(self.edge_index).to(self.device)
        self.edge_weight = torch.FloatTensor(self.edge_weight).to(self.device)


            
    def train(self):
        self.model = Encoder(self.dim_input, self.dim_output, self.graph_neigh).to(self.device)

        self.loss_CSL = nn.BCEWithLogitsLoss()
    
        self.optimizer = torch.optim.Adam(self.model.parameters(), self.learning_rate, 
                                          weight_decay=self.weight_decay)

        if not self.add_regularization:
           print('training...')
        self.model.train()


        try:
            with tqdm(range(self.epochs), dynamic_ncols=True) as t:
                for epoch in t:
                    self.model.train()

                    self.hiden_feat, self.emb, ret, ret_a = self.model(self.features, self.features_a, self.edge_index, self.edge_weight)

                    self.loss_sl_1 = self.loss_CSL(ret, self.label_CSL)
                    self.loss_sl_2 = self.loss_CSL(ret_a, self.label_CSL)
                    self.loss_feat = F.mse_loss(self.features, self.emb)


                    if self.add_regularization:
                       self.loss_norm = 0
                       for name, parameters in self.model.named_parameters():
                           if name in ['weight1', 'weight2']:
                              self.loss_norm = self.loss_norm + torch.norm(parameters, p=2)
                       loss =  self.alpha*self.loss_feat + self.beta*(self.loss_sl_1 + self.loss_sl_2) + self.theta*self.loss_norm
                    else:
                       loss =  self.alpha*self.loss_feat + self.beta*(self.loss_sl_1 + self.loss_sl_2)


                    self.optimizer.zero_grad()
                    loss.backward()
                    self.optimizer.step()


                if not self.add_regularization:
                   print("Completion of training")

                with torch.no_grad():
                     self.model.eval()
                     self.emb_rec = self.model(self.features, self.features_a, self.edge_index, self.edge_weight)[1].detach().cpu().numpy()
                     return self.emb_rec
        except KeyboardInterrupt:
            t.close()
            raise
        t.close()

