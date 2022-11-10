import torch.nn as nn
import torch
import math

from .attention import MultiHeadedAttention
from .utils import SublayerConnection, PositionwiseFeedForward


class LearnablePositionalEmbedding(nn.Module):

    def __init__(self, d_model, max_len=512):
        super().__init__()

        # Compute the positional encodings once in log space.
        pe = torch.zeros(max_len, d_model).float().to(device='cuda')
        pe.require_grad = True
        pe = pe.unsqueeze(0)
        self.pe=nn.Parameter(pe)
        torch.nn.init.normal_(self.pe,std=0.02)

    def forward(self, x):
        return self.pe[:, :x.size(1)]


class Embedding(nn.Module):

    def __init__(self, input_dim, max_len, dropout=0.1):

        super().__init__()
        self.learnedPosition = LearnablePositionalEmbedding(d_model=input_dim,max_len=max_len)
        self.dropout = nn.Dropout(p=dropout)

    def forward(self, sequence):
        x = self.learnedPosition(sequence)+sequence
        return self.dropout(x)


class TransformerBlock(nn.Module):
    """
    Transformer = MultiHead_Attention + Feed_Forward with sublayer connection
    """

    def __init__(self, hidden, attn_heads, feed_forward_hidden, dropout):
        """
        :param hidden: hidden size of transformer
        :param attn_heads: head sizes of multi-head attention
        :param feed_forward_hidden: feed_forward_hidden, usually 4*hidden_size
        :param dropout: dropout rate
        """

        super().__init__()
        self.attention = MultiHeadedAttention(h=attn_heads, d_model=hidden)
        self.feed_forward = PositionwiseFeedForward(d_model=hidden, d_ff=feed_forward_hidden, dropout=dropout)
        self.input_sublayer = SublayerConnection(size=hidden, dropout=dropout)
        self.output_sublayer = SublayerConnection(size=hidden, dropout=dropout)
        self.dropout = nn.Dropout(p=dropout)

    def forward(self, x, mask):
        x = self.input_sublayer(x, lambda _x: self.attention.forward(_x, _x, _x, mask=mask))
        x = self.output_sublayer(x, self.feed_forward)
        return self.dropout(x)


class BERT(nn.Module):
    """
    BERT model : Bidirectional Encoder Representations from Transformers.
    """

    def __init__(self, input_dim, max_len, hidden=768, n_layers=12, attn_heads=12, dropout=0.1, mask_prob=0.9):

        super().__init__()
        self.hidden = hidden
        self.n_layers = n_layers
        self.attn_heads = attn_heads
        self.max_len=max_len
        self.input_dim=input_dim
        self.mask_prob=mask_prob        
        
        clsToken = torch.zeros(1,1,self.input_dim).float().cuda()
        clsToken.require_grad = True
        self.clsToken= nn.Parameter(clsToken)
        torch.nn.init.normal_(clsToken,std=0.02)
        
        # paper noted they used 4*hidden_size for ff_network_hidden_size
        self.feed_forward_hidden = hidden * 4

        # embedding for BERT
        self.embedding = Embedding(input_dim=input_dim, max_len=max_len+1)

        # multi-layers transformer blocks, deep network
        self.transformer_blocks = nn.ModuleList(
            [TransformerBlock(hidden, attn_heads, self.feed_forward_hidden, dropout) for _ in range(n_layers)])    
    
    def forward(self, input_vectors):
        batch_size=input_vectors.shape[0]
        sample=None
        if self.training:
            bernolliMatrix=torch.cat((torch.tensor([1]).float().cuda(), (torch.tensor([self.mask_prob]).float().cuda()).repeat(self.max_len)), 0).unsqueeze(0).repeat([batch_size,1])
            self.bernolliDistributor=torch.distributions.Bernoulli(bernolliMatrix)
            sample=self.bernolliDistributor.sample()
            mask = (sample > 0).unsqueeze(1).repeat(1, sample.size(1), 1).unsqueeze(1)
        else:
            mask=torch.ones(batch_size,1,self.max_len+1,self.max_len+1).cuda()

        # embedding the indexed sequence to sequence of vectors
        x = torch.cat((self.clsToken.repeat(batch_size,1,1),input_vectors),1)
        x = self.embedding(x)
        
        # running over multiple transformer blocks
        for transformer in self.transformer_blocks:
            x = transformer.forward(x, mask)
        
        return x, sample