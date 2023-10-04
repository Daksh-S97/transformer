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
        super().__init__()
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
        super().__init__()
        self.lin1 = nn.Linear(d_model, hidden)
        self.lin2 = nn.Linear(hidden, d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        
        # lin1 -> relu -> dropout -> lin2
        return self.lin2(self.dropout(torch.relu(self.lin1(x))))

class MultiHeadAttention(nn.Module):

    def __init__(self, d_model: int, h: int, dropout:int)->None:

        super().__init__()
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
        d_k = q.shape[-1]

        # (b x h x s x d) @ (b x h x d x s) = (b x h x s x s)
        attention_scores = (q @ k.transpose(-2,-1)) / math.sqrt(d_k)
        
        if mask is not None:
            attention_scores.masked_fill_(mask == 0, -1e9)
        attention_scores = attention_scores.softmax(dim = -1)

        if dropout:
            attention_scores = dropout(attention_scores)

        # (x, att_scores)
        # x = (b x h x s x s) @ (b x h x s x d) = (b x h x s x d) 
        return (attention_scores @ v), attention_scores    

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

        x, self.attention_scores = MultiHeadAttention.Attention(query, key, value, mask, self.dropout)

        # (b x h x s x d) -> (b x s x h x d) -> (b x s x d_model)
        x = x.transpose(1,2).contiguous().view(x.shape[0], -1, self.d_model)

        return self.w_o(x)

class ResidualConnection(nn.Module):

    def __init__(self, dropout:float) -> None:
        super().__init__()
        self.dropout = nn.Dropout(dropout)
        self.norm = LayerNorm()

    def forward(self, x, sublayer):
        # norm & add better than add & norm
        return x + self.dropout(sublayer(self.norm(x)))        
    
class EncoderBlock(nn.Module):

    def __init__(self, self_attention: MultiHeadAttention, ffn: FFN, dropout: float) -> None:
        super().__init__()
        self.self_attention_block = self_attention
        self.ffn = ffn
        self.res_con = nn.ModuleList([ResidualConnection(dropout) for _ in range(2)])

    def forward(self, x, src_mask):
        x = self.res_con[0](x, lambda x: self.self_attention_block(x, x, x, src_mask))
        x = self.res_con[1](x, self.ffn)
        return x
    
class Encoder(nn.Module):

    def __init__(self, layers: nn.ModuleList) -> None:
        super().__init__()
        self.layers = layers
        self.norm = LayerNorm()

    def forward(self, x, mask):

        for layer in self.layers:
            x = layer(x, mask)
        return self.norm(x)    


class DecoderBlock(nn.Module):

    def __init__(self, self_attention: MultiHeadAttention, cross_attention: MultiHeadAttention, 
                 ffn: FFN, dropout: float) -> None:
        super().__init__()
        self.self_attention = self_attention
        self.cross_attention = cross_attention
        self.ffn = ffn
        self.res_con = nn.ModuleList([ResidualConnection(dropout) for _ in range(3)])

    def forward(self, x, encoder_output, src_mask, tgt_mask):
        x = self.res_con[0](x, lambda x: self.self_attention(x, x, x, tgt_mask))
        x = self.res_con[1](x, lambda x: self.cross_attention(x, encoder_output,
                                                              encoder_output, src_mask))
        x = self.res_con[2](x, self.ffn)

        return x            

class Decoder(nn.Module):
    
    def __init__(self, layers: nn.ModuleList) -> None:
        super().__init__()
        self.layers = layers
        self.norm = LayerNorm()

    def forward(self, x, enc_outputs, src_mask ,tgt_mask):
        # each layer is a decoder block
        for layer in self.layers:
            x = layer(x, enc_outputs, src_mask, tgt_mask)
        return self.norm(x)
        
class FinalFFN(nn.Module):

    def __init__(self, d_model: int, vocab_size: int) -> None:
        super().__init__()
        self.lin = nn.Linear(d_model, vocab_size)

    def forward(self, x):
         return torch.log_softmax(self.lin(x), dim = -1)

class Transformer(nn.Module):

    def __init__(self, enc: Encoder, dec:Decoder, inp: InputEmbedding,
                  tgt: InputEmbedding, inp_pos: PosEmbedding, tgt_pos: PosEmbedding,
                  ffn: FinalFFN) -> None:
        super().__init__()
        # emebeddings are all layers and not actual embeddings
        self.enc = enc
        self.dec = dec
        self.inp_emb = inp
        self.tgt_embed = tgt
        self.inp_pos = inp_pos
        self.tgt_pos = tgt_pos
        self.ffn = ffn

    def encode(self, x, src_mask):
        x = self.inp_emb(x)
        x = self.inp_pos(x)
        return self.enc(x, src_mask)

    def decode(self, x, enc_output, src_mask, tgt_mask):
        x = self.tgt_embed(x)
        x = self.tgt_pos(x)
        return self.dec(x, enc_output, src_mask, tgt_mask)

    def project(self, x):
        return self.ffn(x) 
    

def build_tf(inp_vocab_size: int, tgt_vocab_size: int, max_inp_len: int, 
             max_tgt_len:int, d_model: int = 512, hidden: int = 2048, dropout: float = 0.1,
             layers: int = 6, heads: int = 8):
    
    # embedding layers
    src_emb = InputEmbedding(d_model, inp_vocab_size)
    tgt_emb = InputEmbedding(d_model, tgt_vocab_size)

    # pos enc layers
    src_pos = PosEmbedding(d_model, max_inp_len, dropout)
    tgt_pos = PosEmbedding(d_model, max_tgt_len, dropout)

    # encoder layers
    encoder_blocks = []
    for _ in range(layers):
        self_att = MultiHeadAttention(d_model, heads, dropout)
        ffn = FFN(d_model, hidden, dropout)
        enc_block = EncoderBlock(self_att, ffn, dropout)
        encoder_blocks.append(enc_block)

    # encoder
    enc = Encoder(nn.ModuleList(encoder_blocks))

    # decoder layers
    dec_blocks = []
    for _ in range(layers):
        self_att = MultiHeadAttention(d_model, heads, dropout)
        ffn = FFN(d_model, hidden, dropout)
        cross_att = MultiHeadAttention(d_model, heads, dropout)
        dec_block = DecoderBlock(self_att, cross_att, ffn, dropout)
        dec_blocks.append(dec_block)

    # decoder
    dec = Decoder(nn.ModuleList(dec_blocks))

    #projection layer
    proj = FinalFFN(d_model, tgt_vocab_size)

    # Transformer
    transformer = Transformer(enc, dec, src_emb, tgt_emb, src_pos, tgt_pos, proj)

    # intialize parameters with xavier uniform initialization
    # TODO: lookup wtf is that
    for p in transformer.parameters():
        if p.dim() > 1:
            nn.init.xavier_uniform_(p)

    return transformer                





