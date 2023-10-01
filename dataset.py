from typing import Any
import torch 
import torch.nn as nn
from torch.utils.data import Dataset

class BilingualTranslationDataset(Dataset):

    def __init__(self, ds, src_tokenizer, tgt_tokenizer, src_lang, tgt_lang, seq_len) -> None:
        super().__init__()
        self.ds  = ds
        self.src_tokenizer = src_tokenizer
        self.tgt_tokenizer = tgt_tokenizer
        self.src_lang = src_lang
        self.tgt_lang = tgt_lang
        self.seq_len = seq_len

        self.start = torch.Tensor([src_tokenizer.token_to_id(['SOS'])], dtype=torch.int64)
        self.eos = torch.Tensor([src_tokenizer.token_to_id(['EOS'])], dtype=torch.int64)
        self.pad = torch.Tensor([src_tokenizer.token_to_id(['PAD'])], dtype=torch.int64)

    def __len__(self):
        return len(self.ds)
    
    def __getitem__(self, index: Any) -> Any:
        raw = self.ds[index]
        src = raw['translation'][self.src_lang]
        tgt = raw['translation'][self.tgt_lang]

        enc_input = self.src_tokenizer.encode(src).ids
        dec_input = self.tgt_tokenizer.encode(tgt).ids

        # pad
        enc_num_pad = self.seq_len - len(enc_input) - 2
        dec_num_pad = self.seq_len - len(dec_input) - 1

        # TODO: implement truncation
        if enc_num_pad < 0 or dec_num_pad < 0:
            return ValueError('Sentence too long')
        
        # START + tokens + EOS + pad
        enc_input = torch.cat(
            [
                self.start,
                torch.tensor(enc_input, dtype=torch.int64),
                self.eos,
                torch.tensor([self.pad] * enc_num_pad, dtype=int.64)
            ]
        )

        # tokens + EOS + pad
        dec_input = torch.cat(
            [
                torch.tensor(dec_input, dtype=torch.int64),
                self.eos,
                torch.tensor([self.pad] * dec_num_pad, dtype=int.64)
            ]
        )

        # START + tokens + pad (ground truth/actual sentence)
        target = torch.cat(
            [
                self.start,
                torch.tensor(dec_input, dtype=torch.int64),
                torch.tensor([self.pad] * dec_num_pad, dtype=int.64)
            ]
        )

        assert enc_input.size(0) == self.seq_len
        assert dec_input.size(0) == self.seq_len
        assert target.size(0) == self.seq_len

        return {
            "enc_input": enc_input,
            "dec_input": dec_input,
            "enc_mask": (enc_input != self.pad).unsqueeze(0).unsqueeze(0).int(), # (1, 1, seq_len)
            "dec_mask": (dec_input != self.pad).unsqueeze(0).unsqueeze(0).int() & causal_mask(dec_input.size(0)),
            "label": target,
            "src_text": src,
            "tgt_text": tgt  
        }

def causal_mask(size):
    mask = torch.triu(torch.ones(1, size, size), diagonal=1).type(torch.int) # returns matrix with lower triangle
    return mask == 0    # because we want upper triangle 0


