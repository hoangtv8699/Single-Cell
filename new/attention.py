import torch
import torch.nn.functional as F
import torch.nn as nn
import math

torch.random.manual_seed(6)
def scaled_dot_product(q, k, v, mask=None):
    d_k = q.size()[-1]
    attn_logits = torch.matmul(q, k.transpose(-2, -1))
    attn_logits = attn_logits / math.sqrt(d_k)
    if mask is not None:
        attn_logits = attn_logits.masked_fill(mask == 0, -9e15)
    attention = F.softmax(attn_logits, dim=-1)
    values = torch.matmul(attention, v)
    return values, attention

class AttentionHead(nn.Module): 
    def __init__(self, q_dim, k_dim, v_dim, embed_dim):
        super().__init__()
        self.q = nn.Linear(q_dim, embed_dim)
        self.k = nn.Linear(k_dim, embed_dim)
        self.v = nn.Linear(v_dim, embed_dim)
        
    def forward(self, q, k, v, mask=None):
        q = self.q(q)
        k = self.k(k)
        v = self.v(v)

        # Determine value outputs
        values, attention = scaled_dot_product(q, k, v, mask=mask)

        return values, attention




class MultiHeadAttention(nn.Module): 
    def __init__(self,  q_dim, k_dim, v_dim, embed_dim, num_heads) :
        super().__init__()
        assert embed_dim % num_heads == 0, "Embedding dimension must be 0 modulo number of heads."

        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.head_dim = embed_dim // num_heads

        self.heads = nn.ModuleList([AttentionHead(q_dim, k_dim, v_dim, self.head_dim) for _ in range(self.num_heads)])
        self.o = nn.Linear(embed_dim, embed_dim)


    def forward(self, q, k, v, mask=None, return_attention=False):
        values = []
        atts = []
        for head in self.heads:
            value, attention = head(q, k, v, mask=mask)
            values.append(value)
            atts.append(torch.unsqueeze(attention, 1))
        
        values = torch.cat(values, -1)
        atts = torch.cat(atts, 1)

        o = self.o(values)

        if return_attention:
            return o, attention
        else:
            return o


# multiHeadAttention = MultiHeadAttention(1, 8, 8, 8, 2)
# q = torch.randn((4, 1000, 1))
# k = torch.randn((4, 10000, 8))
# v = torch.randn((4, 10000, 8))
# ax = multiHeadAttention(q, k, v)
# print("out shape: " , ax.shape)

