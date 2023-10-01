import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader, random_split

from dataset import BilingualTranslationDataset, causal_mask
from model import build_tf

from datasets import load_dataset
from tokenizers import Tokenizer
from tokenizers.models import WordLevel
from tokenizers.trainers import WordLevelTrainer
from tokenizers.pre_tokenizers import Whitespace

from pathlib import Path

def get_sentences(ds, lang):
    """
    ds: dataset
    lang: Language
    """
    for item in ds:
        yield item['translation'][lang]



def get_or_build_tokenizer(config, ds, lang):
    """
    config: config file, contains tokenizer file info
    ds: Dataset
    lang: flag for checking which language the tpokenizer is being built for 
    """

    path = Path(config['tokenizer_file'],format(lang))
    if not Path.exists(path):
        tokenizer = Tokenizer(WordLevel(unk_token='[UNK]'))
        tokenizer.pre_tokenizer = Whitespace()
        trainer = WordLevelTrainer(special_tokens= ['[UNK]', '[PAD]', '[SOS]', '[EOS]'], min_frequency=2)
        tokenizer.train_from_iterator(get_sentences(ds, lang), trainer=trainer)
        tokenizer.save(str(path))
    else:
        tokenizer = Tokenizer.from_file(str(path))

    return tokenizer  

def get_ds(config):
    """
    config: config file of the model
    """
    ds_raw = load_dataset('opus_books', f'{config['lang_src']}-{config['lang_tgt']}', split='train')

    # tokenizers
    src_tokenizer = get_or_build_tokenizer(config, ds_raw, config['lang_src'])
    tgt_tokenizer = get_or_build_tokenizer(config, ds_raw, config['lang_tgt'])

    # split dataset
    train_ds_size = len(ds_raw) * 0.9
    val_ds_size  = len(ds_raw) - train_ds_size
    train, val = random_split(ds_raw, [train_ds_size, val_ds_size])

    train_ds = BilingualTranslationDataset(train, src_tokenizer, tgt_tokenizer, config['lang_src'], config['lang_tgt'], config['seq_len'])
    val_ds = BilingualTranslationDataset(val, src_tokenizer, tgt_tokenizer, config['lang_src'], config['lang_tgt'], config['seq_len'])

    # calc max seq_len
    max_len_src = 0
    max_len_tgt = 0

    for item in ds_raw:
        src = src_tokenizer.encode(item['translation'][config['lang_src']]).ids
        tgt = tgt_tokenizer.encode(item['translation'][config['lang_tgt']]).ids
        max_len_src = max(max_len_src, len(src))
        max_len_tgt = max(max_len_tgt, len(tgt))

    train_dataloader = DataLoader(train_ds, batch_size=config['batch_size'], shuffle=True)
    val_dataloader = DataLoader(val_ds, batch_size=1, shuffle=True)

    return train_dataloader, val_dataloader, src_tokenizer, tgt_tokenizer  

def get_model(config, src_vocab_len, tgt_vocab_len):
    model = build_tf(src_vocab_len, tgt_vocab_len, config['seq_len'], config['seq_len'], config['d_model'])
    return model