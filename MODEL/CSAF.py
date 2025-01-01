import torch
import torch.nn as nn
import torch.nn.functional as F

class CrossSelfAttentionFusion(nn.Module):
    def __init__(self, d_model):
        super(CrossSelfAttentionFusion, self).__init__()
        # Learnable weights for cross attention (text and image)
        self.Q_v = nn.Linear(self.embedding_dim, self.embedding_dim, bias=False)
        self.K_v = nn.Linear(self.embedding_dim, self.embedding_dim, bias=False)
        self.V_v = nn.Linear(self.embedding_dim, self.embedding_dim, bias=False)
        
        self.Q_t = nn.Linear(self.embedding_dim, self.embedding_dim, bias=False)
        self.K_t = nn.Linear(self.embedding_dim, self.embedding_dim, bias=False)
        self.V_t = nn.Linear(self.embedding_dim, self.embedding_dim, bias=False)

        # Self-attention layers
        self.self_attention = nn.MultiheadAttention(embed_dim=d_model, num_heads=1, batch_first=True)

        # Fusion with MLP
        self.fusion_mlp = nn.Sequential(
            nn.Linear(2 * d_model, d_model),
            nn.ReLU(),
            nn.Linear(d_model, d_model)
        )

    def forward(self, Hv, Ht):
        # Cross-attention: Text -> Image
        Qv = self.Q_v(Hv)
        Kv = self.K_v(Hv)
        Vv = self.V_v(Hv)
        
        # Linear projections for text
        Qt = self.Q_t(Ht)
        Kt = self.K_t(Ht)
        Vt = self.V_t(Ht)
        
        attn_weights_v_to_t =torch.matmul(Qt, Kv.T) / (self.embedding_dim ** 0.5)
        attn_weights_t_to_v = torch.matmul(Qv, Kt.T) / (self.embedding_dim ** 0.5)
        
        delta_Hv_to_t = F.softmax(attn_weights_v_to_t, dim=-1)
        delta_Ht_to_v = F.softmax(attn_weights_t_to_v, dim=-1)

        attn_weights_vt =torch.matmul(delta_Hv_to_t, Vv) 
        attn_weights_tv = torch.matmul(delta_Ht_to_v,Vt)


        # Update features with cross-attention outputs
        c_t = Ht + attn_weights_vt
        c_v = Hv + attn_weights_tv

        # Self-attention within each modality
        e_t_self, _ = self.self_attention(Ht, Ht, Ht)
        e_v_self, _ = self.self_attention(Hv, Hv, Hv)

        e_t_self = Ht + e_t_self
        e_v_self = Hv + e_v_self

        return c_t,c_v,e_t_self,e_v_self
