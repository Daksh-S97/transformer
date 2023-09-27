import torch
import torch.nn as nn
import math

class InputEmbedding(nn.Module):

    def __init__(self, d_model:int, vocab_size:int)->None:
        super().__init__()
        self.d_model = d_model
        self.embedding = nn.Embedding(vocab_size, d_model)
        self.vocab_size = vocab_size

    def forward(self, x):
        return self.embedding(x) * math.sqrt(self.d_model)    

class PosEmbedding(nn.Module):

    def __init__(self, d_model: int, seq_len:int, dropout:float)->None:    
        super().__init__()
        self.d_model = d_model
        self.seq_len = seq_len
        self.dropout= nn.Dropout(dropout)

        # Matrix to store pos_embs (seq_len x d_model)
        pe = torch.zeros(seq_len, d_model)

        # vectors of shape (seq_len, 1) for sine and cosine calculations

        num = torch.arange(0, seq_len, dtype=torch.float).unsqueeze(1)
        denom = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0)/d_model))

        # apply sine function to even positions
        pe[:, 0::2] = torch.sin(num * denom)
        pe[:, 1::2] = torch.cos(num * denom)

        pe = pe.unsqueeze(0) #(1, seq_len, d_model) to accomodate batching

        self.register_buffer('pe', pe) # ensures the embeddings are stored in module params

    def forward(self, x): 
        x = x + (self.pe[:, :x.shape[1], :]).requires_grad_(False)
        return self.dropout(x)


class LayerNorm(nn.Module):
    
    def __init__(self, eps: float = 10 ** -6) -> None:
        super.__init__()
        self.eps = eps
        self.alpha = nn.Parameter(torch.ones(1)) 
        self.bias = nn.Parameter(torch.ones(1))
    
    def forward(self, x):
        mean = x.mean(dim = -1, keepdim=True)
        std = x.std(dim = -1, keepdim=True)

        # TODO: check if formula correct    
        return self.alpha * (x-mean) / (std + self.eps) + self.bias
    
class FFN(nn.Module):

    def __init__(self, d_model:int, hidden:int, dropout: float)->None:
        super.__init__()
        self.lin1 = nn.Linear(d_model, hidden)
        self.lin2 = nn.Linear(hidden, d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        
        # lin1 -> relu -> dropout -> lin2
        return self.lin2(self.dropout(torch.relu(self.lin1(x))))

class MultiHeadAttention(nn.Module):

    def __init__(self, d_model: int, h: int, dropout:int)->None:

        super.__init__()
        self.d_model = d_model
        self.heads = h
        self.dropout = nn.Dropout(dropout)

        assert d_model % h == 0, "d_model not divisible by h"
        self.d_k = d_model // h
        # Wq, Wk, Wv
        self.w_q = nn.Linear(d_model, d_model)
        self.w_k = nn.Linear(d_model, d_model)
        self.w_v = nn.Linear(d_model, d_model)

        #Wo
        self.w_o = nn.Linear(d_model, d_model)

    @staticmethod
    def Attention(q, k, v, mask:None, dropout:None):
        d_k = q.shape(-1)

        # (b x h x s x d) @ (b x h x d x s) = (b x h x s x s)
        attention_scores = (q @ k.transpose(-2,-1)) / math.sqrt(d_k)
        
        if mask:
            attention_scores.masked_fill_(mask == 0, -1e9)
        attention_scores = attention_scores.softmax(dim = -1)

        if dropout:
            attention_scores = dropout(attention_scores)

        # (x, att_scores)
        # x = (b x h x s x s) @ (b x h x s x d) = (b x h x s x d) 
        return (attention_scores @ v, attention_scores)    

    def forward(self, q, k, v, mask):
        q_prime = self.w_q(q)
        v_prime = self.w_v(v)
        k_prime = self.w_k(k)

        # split into heads
        # transpose because we want each head to be able to see the entire sequence
        # (b x s x d) -> (b x s x h x d_k) -> (b x h x s x d_k)
        query = q_prime.view(q_prime.shape[0], q_prime.shape[1], self.heads, self.d_k).transpose(1,2)
        key = k_prime.view(k_prime.shape[0], k_prime.shape[1], self.heads, self.d_k).transpose(1,2)
        value = v_prime.view(v_prime.shape[0], v_prime.shape[1], self.heads, self.d_k).transpose(1,2)

        x, attention_scores = MultiHeadAttention.Attention(query, key, value, mask, self.dropout)

        # (b x h x s x d) -> (b x s x h x d) -> (b x s x d_model)
        x = x.transpose(1,2)
        x = x.view(x.shape[0], x.shape[1], self.d_model)

        return self.w_o(x)

class ResidualConnection(nn.Module):

    def __init__(self, dropout:float) -> None:
        super().__init__()
        self.dropout = nn.Dropout(dropout)
        self.norm = LayerNorm()

    def forward(self, x, sublayer):
        # norm & add better than add & norm
        # TODO: unclear if sublayer is previous layer or aage ki
        return x + self.dropout(sublayer(self.norm(x)))        