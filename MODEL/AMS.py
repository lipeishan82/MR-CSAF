# coding: utf-8
# @email: enoche.chow@gmail.com
r"""
FREEDOM: A Tale of Two Graphs: Freezing and Denoising Graph Structures for Multimodal Recommendation
# Update: 01/08/2022
"""

import numpy as np
import scipy.sparse as sp
import torch
import torch.nn as nn
import torch.nn.functional as F

from common.abstract_recommender import GeneralRecommender
import torch.nn.functional as F

    
class Selector(nn.Module):
    def __init__(self, input_dim, hidden_dim):
        super(Selector, self).__init__()
        self.modality_selector = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 2)  
        )

    def forward(self, image_embedding, text_embedding,image_adj,text_adj):
        item_embedding = torch.cat((image_embedding,text_embedding),dim = -1)
        modality_weights = self.modality_selector(item_embedding)
        modality_weights = F.softmax(modality_weights, dim=-1)
        image_weights = modality_weights[:, 0]
        text_weights = modality_weights[:, 1]
        
        mm_adj = image_weights * image_adj.to_dense() + text_weights * text_adj.to_dense()
        mm_adj = mm_adj.to_sparse()
        return mm_adj


